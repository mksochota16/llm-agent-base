import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import faiss

from .llm_connection_config import LLMConnectionConfig


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class KnowledgeBase:
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".pdf"}

    def __init__(
        self,
        folder_path: str,
        llm_config: LLMConnectionConfig,
        index_dir: str = ".kb_index",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ):
        self.folder_path = Path(folder_path)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = llm_config.embedding_model
        self._client = llm_config.build_client()

        self._chunks: list[DocumentChunk] = []
        self._index: Optional[faiss.IndexFlatIP] = None
        self._dim: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def ingest(self) -> int:
        """Parse all files in folder_path, chunk, embed, and build the index."""
        self._chunks = []
        vectors = []

        for path in sorted(self.folder_path.rglob("*")):
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            text = self._parse_file(path)
            if not text:
                continue
            for i, chunk_text in enumerate(self._chunk_text(text)):
                chunk = DocumentChunk(
                    text=chunk_text,
                    source=str(path.relative_to(self.folder_path)),
                    chunk_index=i,
                )
                self._chunks.append(chunk)
                vectors.append(self._embed(chunk_text))

        if not vectors:
            return 0

        matrix = np.array(vectors, dtype="float32")
        self._dim = matrix.shape[1]
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)
        return len(self._chunks)

    def save(self):
        """Persist the FAISS index and chunk catalog to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        with open(self.index_dir / "catalog.pkl", "wb") as f:
            pickle.dump({"chunks": self._chunks, "dim": self._dim}, f)

    def load(self):
        """Restore a previously saved index and catalog from disk."""
        index_path = self.index_dir / "index.faiss"
        catalog_path = self.index_dir / "catalog.pkl"
        if not index_path.exists() or not catalog_path.exists():
            raise FileNotFoundError(f"No saved index found in {self.index_dir}")
        self._index = faiss.read_index(str(index_path))
        with open(catalog_path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._dim = data["dim"]

    def retrieve(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Return the top_k most relevant chunks for the given query."""
        if self._index is None or self._index.ntotal == 0:
            return []
        vec = np.array([self._embed(query)], dtype="float32")
        faiss.normalize_L2(vec)
        _scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))
        return [self._chunks[i] for i in indices[0] if i >= 0]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

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

    def _embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding
