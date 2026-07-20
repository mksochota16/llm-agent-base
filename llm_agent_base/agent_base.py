import base64
import mimetypes
from pathlib import Path
from typing import Callable, Optional, Union

from .knowledge_base import DocumentChunk, KnowledgeBase
from .llm_connection_config import LLMConnectionConfig
from .tool_calling import build_tool_schema, execute_tool_loop

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".xml", ".html", ".yaml", ".yml", ".py", ".js", ".ts"}
_SUPPORTED_EXTENSIONS = _IMAGE_EXTENSIONS | _TEXT_EXTENSIONS | {".pdf"}


def _read_file_as_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _build_user_content(prompt: str, files: Optional[list[str]]) -> Union[str, list]:
    if not files:
        return prompt
    parts: list = [{"type": "text", "text": prompt}]
    for file_path in files:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}' for '{path.name}'. Supported: {sorted(_SUPPORTED_EXTENSIONS)}")
        if ext in _IMAGE_EXTENSIONS:
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            data = base64.b64encode(path.read_bytes()).decode()
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
        else:
            text = _read_file_as_text(path)
            parts.append({"type": "text", "text": f"[{path.name}]\n{text}"})
    return parts


class AgentBase:
    def __init__(
        self,
        system_prompt: str,
        llm_config: LLMConnectionConfig,
        knowledge_folder_path: Optional[str] = None,
        knowledge_index_dir: str = ".kb_index",
        knowledge_top_k: int = 5,
        auto_load_or_ingest: bool = False,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        debug: bool = False,
    ):
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.knowledge_top_k = knowledge_top_k
        self.temperature = temperature
        self.response_format = response_format
        self.debug = debug
        self._client = llm_config.build_client()
        self._kb: Optional[KnowledgeBase] = None
        self._tools: dict[str, tuple[Callable, dict]] = {}
        self._conversation_messages: list[dict] = []

        if knowledge_folder_path:
            self._kb = KnowledgeBase(
                folder_path=knowledge_folder_path,
                llm_config=llm_config,
                index_dir=knowledge_index_dir,
            )
            self._register_knowledge_tool()
            if auto_load_or_ingest:
                self.load_or_ingest_knowledge()

    def register_tool(self, fn: Callable) -> Callable:
        """Register a Python function as a callable tool. Can be used as a decorator."""
        self._tools[fn.__name__] = (fn, build_tool_schema(fn))
        return fn

    def _register_knowledge_tool(self) -> None:
        def search_knowledge(query: str) -> str:
            """Search the knowledge base for information relevant to the query. Use this when you need to look up facts, context, or details that may be stored in the available knowledge."""
            chunks = self._kb.retrieve(query, top_k=self.knowledge_top_k)
            if not chunks:
                return "No relevant information found in the knowledge base."
            if self.debug:
                print(f"[debug] search_knowledge query={query!r} returned {len(chunks)} chunk(s)")
            return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)

        self._tools["search_knowledge"] = (search_knowledge, build_tool_schema(search_knowledge))

    def ingest_knowledge(self, save: bool = True) -> int:
        if self._kb is None:
            return 0
        count = self._kb.ingest()
        if save:
            self._kb.save()
        return count

    def load_knowledge(self):
        if self._kb is not None:
            self._kb.load()

    def load_or_ingest_knowledge(self) -> int:
        """Load a saved index if one exists, otherwise ingest and save. Returns the number of chunks ingested (0 if loaded)."""
        if self._kb is None:
            return 0
        if (self._kb.index_dir / "index.faiss").exists():
            self._kb.load()
            return 0
        count = self._kb.ingest()
        self._kb.save()
        return count

    def retrieve_knowledge(self, query: str) -> list[DocumentChunk]:
        if self._kb is None:
            return []
        if self.debug:
            print("[debug] Retrieving knowledge")
        return self._kb.retrieve(query, top_k=self.knowledge_top_k)

    def chat(self, message: str, files: Optional[list[str]] = None) -> str:
        """Send a message and get a response, maintaining conversation history across calls."""
        chunks = self.retrieve_knowledge(message)
        system = self.system_prompt
        if chunks:
            context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
            system = f"{system}\n\n<context>\n{context}\n</context>"

        messages = (
            [{"role": "system", "content": system}]
            + list(self._conversation_messages)
            + [{"role": "user", "content": _build_user_content(message, files)}]
        )

        response = execute_tool_loop(
            self._client,
            self.llm_config.model,
            messages,
            self._tools,
            debug=self.debug,
            temperature=self.temperature,
            response_format=self.response_format,
        )

        self._conversation_messages.append({"role": "user", "content": _build_user_content(message, files)})
        self._conversation_messages.append({"role": "assistant", "content": response})
        return response

    def reset_conversation(self) -> None:
        """Clear the stored conversation history."""
        self._conversation_messages = []

    def ask(self, prompt: str, files: Optional[list[str]] = None) -> str:
        """Single LLM call with optional knowledge retrieval but no tool calling."""
        chunks = self.retrieve_knowledge(prompt)
        system = self.system_prompt
        if chunks:
            context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
            system = f"{system}\n\n<context>\n{context}\n</context>"

        kwargs: dict = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": _build_user_content(prompt, files)},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        return self._client.chat.completions.create(**kwargs).choices[0].message.content

    def run(self, prompt: str, files: Optional[list[str]] = None) -> str:
        chunks = self.retrieve_knowledge(prompt)

        system = self.system_prompt
        if chunks:
            context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
            system = f"{system}\n\n<context>\n{context}\n</context>"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _build_user_content(prompt, files)},
        ]

        return execute_tool_loop(
            self._client,
            self.llm_config.model,
            messages,
            self._tools,
            debug=self.debug,
            temperature=self.temperature,
            response_format=self.response_format,
        )
