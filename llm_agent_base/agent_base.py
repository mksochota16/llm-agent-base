import base64
import mimetypes
from pathlib import Path
from typing import Callable, Literal, Optional, Union

from .knowledge_base import DocumentChunk, KnowledgeBase
from .llm_connection_config import LLMConnectionConfig
from .tool_calling import _MAX_ITERATIONS, build_tool_schema, execute_tool_loop

_DEFAULT_KNOWLEDGE_FILE_MAX_CHARS = 20_000

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
        knowledge_min_score: Optional[float] = None,
        auto_load_or_ingest: bool = False,
        knowledge_search_tool: bool = True,
        knowledge_file_tool: bool = True,
        knowledge_file_max_chars: int = _DEFAULT_KNOWLEDGE_FILE_MAX_CHARS,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        max_iterations: int = _MAX_ITERATIONS,
        max_history_messages: Optional[int] = None,
        debug: bool = False,
    ):
        self.system_prompt = system_prompt
        self.llm_config = llm_config
        self.knowledge_top_k = knowledge_top_k
        self.knowledge_min_score = knowledge_min_score
        self.knowledge_file_max_chars = knowledge_file_max_chars
        self.temperature = temperature
        self.response_format = response_format
        self.max_iterations = max_iterations
        self.max_history_messages = max_history_messages
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
            self._register_knowledge_tool(knowledge_search_tool, knowledge_file_tool)
            if auto_load_or_ingest:
                self.load_or_ingest_knowledge()

    def register_tool(self, fn: Callable) -> Callable:
        """Register a Python function as a callable tool. Can be used as a decorator."""
        self._tools[fn.__name__] = (fn, build_tool_schema(fn))
        return fn

    def _register_knowledge_tool(self, search_tool: bool = True, file_tool: bool = True) -> None:
        def search_knowledge(query: str) -> str:
            """Search the knowledge base for information relevant to the query. Use this when you need to look up facts, context, or details that may be stored in the available knowledge.

            Args:
                query: What to look for, phrased as a natural-language question or topic.
            """
            chunks = self._kb.retrieve(
                query,
                top_k=self.knowledge_top_k,
                min_score=self.knowledge_min_score,
            )
            if not chunks:
                return "No relevant information found in the knowledge base."
            if self.debug:
                print(f"[debug] search_knowledge query={query!r} returned {len(chunks)} chunk(s)")
            return "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)

        def read_knowledge_files(
            keywords: list[str],
            search_in: Literal["filename", "content", "both"] = "both",
            match_mode: Literal["any", "all", "min"] = "any",
            min_matches: Optional[int] = None,
        ) -> str:
            """Find files in the knowledge base and return their full text. Use this when you need complete file contents rather than a few relevant chunks.

            Args:
                keywords: Phrases to search for. Each entry is matched as a whole phrase, not split into words.
                search_in: Where to match keywords: 'filename', 'content', or 'both'.
                match_mode: How many keywords must match: 'any' (at least one), 'all' (every keyword), or 'min' (at least min_matches).
                min_matches: Minimum number of matching keywords, used only when match_mode is 'min'.
            """
            terms = [k.lower() for k in keywords if isinstance(k, str) and k.strip()]
            if not terms:
                return "No keywords provided."

            def _hits(text: str) -> int:
                lower = text.lower()
                return sum(1 for t in terms if t in lower)

            def _matches(text: str) -> bool:
                n = _hits(text)
                if match_mode == "all":
                    return n == len(terms)
                if match_mode == "min":
                    return n >= (min_matches or 1)
                return n >= 1

            matches: list[tuple[str, str]] = []
            for path in sorted(self._kb.folder_path.rglob("*")):
                if path.suffix.lower() not in self._kb.SUPPORTED_EXTENSIONS:
                    continue
                relative = str(path.relative_to(self._kb.folder_path))
                check_filename = search_in in ("filename", "both")
                check_content = search_in in ("content", "both")
                if check_filename and _matches(path.name):
                    matches.append((relative, _read_file_as_text(path)))
                    continue
                if check_content:
                    text = _read_file_as_text(path)
                    if _matches(text):
                        matches.append((relative, text))
            if not matches:
                return f"No files found matching: {keywords}"
            if self.debug:
                print(f"[debug] read_knowledge_files keywords={keywords!r} matched {len(matches)} file(s)")

            # Cap the payload: a broad keyword can otherwise return the whole
            # knowledge base in a single tool result and blow the context window.
            budget = self.knowledge_file_max_chars
            results = []
            omitted = 0
            for relative, text in matches:
                if budget <= 0:
                    omitted += 1
                    continue
                if len(text) > budget:
                    text = f"{text[:budget]}\n[... file truncated, {len(text) - budget} character(s) omitted]"
                budget -= len(text)
                results.append(f"[{relative}]\n{text}")
            if omitted:
                results.append(
                    f"[{omitted} more matching file(s) omitted: the {self.knowledge_file_max_chars}-character "
                    f"limit was reached. Narrow the search with more specific keywords, "
                    f'search_in="filename", or match_mode="all".]'
                )
            return "\n\n---\n\n".join(results)

        if search_tool:
            self._tools["search_knowledge"] = (search_knowledge, build_tool_schema(search_knowledge))
        if file_tool:
            self._tools["read_knowledge_files"] = (read_knowledge_files, build_tool_schema(read_knowledge_files))

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

    def sync_knowledge(self, save: bool = True) -> dict:
        """Re-embed only the source files that changed since the last ingest."""
        if self._kb is None:
            return {}
        summary = self._kb.sync()
        if save:
            self._kb.save()
        if self.debug:
            print(f"[debug] sync_knowledge {summary}")
        return summary

    def load_or_ingest_knowledge(self) -> int:
        """Load a saved index if one exists, otherwise ingest and save.

        A loaded index whose source files have changed is brought up to date
        incrementally, so editing a document no longer leaves a stale index.
        Returns the number of chunks embedded (0 when the index was reused
        unchanged).
        """
        if self._kb is None:
            return 0
        if self._kb.has_saved_index():
            self._kb.load()
            if not self._kb.is_stale():
                return 0
            if self.debug:
                print("[debug] knowledge index is stale; syncing changed files")
            summary = self._kb.sync()
            self._kb.save()
            return summary.get("embedded_chunks", 0)
        count = self._kb.ingest()
        self._kb.save()
        return count

    def retrieve_knowledge(self, query: str) -> list[DocumentChunk]:
        if self._kb is None:
            return []
        if self.debug:
            print("[debug] Retrieving knowledge")
        return self._kb.retrieve(
            query,
            top_k=self.knowledge_top_k,
            min_score=self.knowledge_min_score,
        )

    def chat(self, message: str, files: Optional[list[str]] = None) -> str:
        """Send a message and get a response, maintaining conversation history across calls."""
        chunks = self.retrieve_knowledge(message)
        system = self.system_prompt
        if chunks:
            context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
            system = f"{system}\n\n<context>\n{context}\n</context>"

        # Built once: attachments are read from disk and base64-encoded here, so
        # building it twice would re-read every file.
        user_message = {"role": "user", "content": _build_user_content(message, files)}

        messages = (
            [{"role": "system", "content": system}]
            + list(self._conversation_messages)
            + [user_message]
        )

        response = execute_tool_loop(
            self._client,
            self.llm_config.model,
            messages,
            self._tools,
            debug=self.debug,
            temperature=self.temperature,
            response_format=self.response_format,
            max_iterations=self.max_iterations,
        )

        self._conversation_messages.append(user_message)
        self._conversation_messages.append({"role": "assistant", "content": response})
        self._trim_history()
        return response

    def _trim_history(self) -> None:
        """Drop the oldest turns once history exceeds ``max_history_messages``."""
        limit = self.max_history_messages
        if limit is None or len(self._conversation_messages) <= limit:
            return
        # Round up to a whole number of turns so history still starts on a user
        # message rather than a dangling assistant reply.
        dropped = len(self._conversation_messages) - limit
        dropped += dropped % 2
        self._conversation_messages = self._conversation_messages[dropped:]
        if self.debug:
            print(f"[debug] trimmed {dropped} message(s) from conversation history")

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
        content = self._client.chat.completions.create(**kwargs).choices[0].message.content
        return content or ""

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
            max_iterations=self.max_iterations,
        )
