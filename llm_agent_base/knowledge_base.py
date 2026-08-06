import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from .llm_connection_config import LLMConnectionConfig

_CATALOG_VERSION = 2
_DEFAULT_EMBED_BATCH_SIZE = 100


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class KnowledgeBase:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

    def __init__(
        self,
        folder_path: str,
        llm_config: LLMConnectionConfig,
        index_dir: str = ".kb_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embed_batch_size: int = _DEFAULT_EMBED_BATCH_SIZE,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
                f"({chunk_size}); otherwise chunking cannot advance."
            )
        if embed_batch_size <= 0:
            raise ValueError(f"embed_batch_size must be positive, got {embed_batch_size}")

        self.folder_path = Path(folder_path)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_batch_size = embed_batch_size
        self.embedding_model = llm_config.embedding_model
        self._client = llm_config.build_client()

        self._chunks: list[DocumentChunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dim: Optional[int] = None
        self._manifest: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def ingest(self) -> int:
        """Parse all files in folder_path, chunk, embed, and build the index."""
        chunks: list[DocumentChunk] = []
        for path in self._source_files():
            text = self._parse_file(path)
            if not text:
                continue
            source = self._relative(path)
            for i, chunk_text in enumerate(self._chunk_text(text)):
                chunks.append(DocumentChunk(text=chunk_text, source=source, chunk_index=i))

        self._chunks = chunks
        self._manifest = self._fingerprint()

        if not chunks:
            self._index = None
            self._dim = None
            return 0

        vectors = self._embed_texts([c.text for c in chunks])
        self._build_index(np.array(vectors, dtype="float32"))
        return len(self._chunks)

    def sync(self) -> dict:
        """Re-embed only the files that changed since the last ingest.

        Unchanged chunks keep their existing vectors, so a one-file edit costs
        one file's worth of embedding calls instead of a full rebuild. Returns
        a summary dict of what changed.
        """
        current = self._fingerprint()

        # Nothing usable to reuse — fall back to a full ingest.
        if self._index is None or not self._manifest:
            count = self.ingest()
            return {
                "added": len(current), "updated": 0, "removed": 0,
                "reused_chunks": 0, "embedded_chunks": count, "total_chunks": count,
            }

        previous = self._manifest
        added = {p for p in current if p not in previous}
        updated = {p for p in current if p in previous and current[p] != previous[p]}
        removed = set(previous) - set(current)

        if not (added or updated or removed):
            return {
                "added": 0, "updated": 0, "removed": 0,
                "reused_chunks": len(self._chunks), "embedded_chunks": 0,
                "total_chunks": len(self._chunks),
            }

        stale = updated | removed
        stored = self._index.reconstruct_n(0, self._index.ntotal)
        kept_chunks: list[DocumentChunk] = []
        kept_vectors: list[np.ndarray] = []
        for chunk, vector in zip(self._chunks, stored):
            if chunk.source not in stale:
                kept_chunks.append(chunk)
                kept_vectors.append(vector)

        new_chunks: list[DocumentChunk] = []
        for source in sorted(added | updated):
            text = self._parse_file(self.folder_path / source)
            if not text:
                continue
            for i, chunk_text in enumerate(self._chunk_text(text)):
                new_chunks.append(DocumentChunk(text=chunk_text, source=source, chunk_index=i))

        new_vectors = self._embed_texts([c.text for c in new_chunks]) if new_chunks else []

        self._chunks = kept_chunks + new_chunks
        self._manifest = current

        rows = kept_vectors + [np.array(v, dtype="float32") for v in new_vectors]
        if rows:
            self._build_index(np.array(rows, dtype="float32"))
        else:
            self._index = None
            self._dim = None

        return {
            "added": len(added), "updated": len(updated), "removed": len(removed),
            "reused_chunks": len(kept_chunks), "embedded_chunks": len(new_chunks),
            "total_chunks": len(self._chunks),
        }

    def has_saved_index(self) -> bool:
        """True when a loadable index and catalog exist on disk."""
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "catalog.json").exists()

    def is_stale(self) -> bool:
        """True when the source files no longer match the indexed ones."""
        return self._fingerprint() != self._manifest

    def save(self):
        """Persist the FAISS index and chunk catalog to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        catalog = {
            "version": _CATALOG_VERSION,
            "dim": self._dim,
            "embedding_model": self.embedding_model,
            "manifest": self._manifest,
            "chunks": [asdict(c) for c in self._chunks],
        }
        with open(self.index_dir / "catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False)

    def load(self):
        """Restore a previously saved index and catalog from disk."""
        index_path = self.index_dir / "index.faiss"
        catalog_path = self.index_dir / "catalog.json"

        if not index_path.exists() or not catalog_path.exists():
            if (self.index_dir / "catalog.pkl").exists():
                raise FileNotFoundError(
                    f"{self.index_dir} holds a legacy pickle catalog, which is no longer "
                    f"read for security reasons. Delete {self.index_dir} and re-run "
                    f"ingest_knowledge() to rebuild it as JSON."
                )
            raise FileNotFoundError(f"No saved index found in {self.index_dir}")

        self._index = faiss.read_index(str(index_path))
        with open(catalog_path, encoding="utf-8") as f:
            data = json.load(f)

        self._dim = data.get("dim")
        self._manifest = data.get("manifest", {})
        self._chunks = [
            DocumentChunk(
                text=c["text"],
                source=c["source"],
                chunk_index=c["chunk_index"],
                metadata=c.get("metadata", {}),
                score=c.get("score", 0.0),
            )
            for c in data.get("chunks", [])
        ]

    def retrieve(self, query: str, top_k: int = 5, min_score: Optional[float] = None) -> list[DocumentChunk]:
        """Return the top_k most relevant chunks for the given query.

        Each returned chunk carries its cosine similarity in ``score``. When
        ``min_score`` is set, weaker matches are dropped rather than padding the
        result out to ``top_k``.
        """
        if self._index is None or self._index.ntotal == 0:
            return []
        vec = np.array([self._embed(query)], dtype="float32")
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results = []
        for score, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            if min_score is not None and score < min_score:
                continue
            results.append(replace(self._chunks[i], score=float(score)))
        return results

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _source_files(self) -> list[Path]:
        return [
            p for p in sorted(self.folder_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.folder_path).as_posix()

    def _fingerprint(self) -> dict[str, str]:
        """Map each source file to a hash of its contents."""
        fingerprint = {}
        for path in self._source_files():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            fingerprint[self._relative(path)] = digest
        return fingerprint

    def _build_index(self, matrix: np.ndarray) -> None:
        faiss.normalize_L2(matrix)
        self._dim = matrix.shape[1]
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)

    def _parse_file(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._parse_pdf(path)
        if ext == ".json":
            return self._parse_json(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _parse_json(path: Path) -> str:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        return [c.strip() for c in chunks if c.strip()]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts, batching them into as few requests as possible."""
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i:i + self.embed_batch_size]
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            # The API may return items out of order; index is authoritative.
            for item in sorted(response.data, key=lambda d: d.index):
                vectors.append(item.embedding)
        return vectors

    def _embed(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]
