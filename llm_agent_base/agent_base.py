from typing import Callable, Optional

from .knowledge_base import DocumentChunk, KnowledgeBase
from .llm_connection_config import LLMConnectionConfig
from .tool_calling import build_tool_schema, execute_tool_loop


class AgentBase:
    def __init__(
        self,
        system_prompt: str,
        llm_config: LLMConnectionConfig,
        knowledge_folder_path: Optional[str] = None,
        knowledge_index_dir: str = ".kb_index",
        knowledge_top_k: int = 5,
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

        if knowledge_folder_path:
            self._kb = KnowledgeBase(
                folder_path=knowledge_folder_path,
                llm_config=llm_config,
                index_dir=knowledge_index_dir,
            )

    def register_tool(self, fn: Callable) -> Callable:
        """Register a Python function as a callable tool. Can be used as a decorator."""
        self._tools[fn.__name__] = (fn, build_tool_schema(fn))
        return fn

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

    def retrieve_knowledge(self, query: str) -> list[DocumentChunk]:
        if self._kb is None:
            return []
        if self.debug:
            print("[debug] Retrieving knowledge")
        return self._kb.retrieve(query, top_k=self.knowledge_top_k)

    def ask(self, prompt: str) -> str:
        """Single LLM call with no knowledge retrieval or tool calling."""
        kwargs: dict = {
            "model": self.llm_config.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.response_format is not None:
            kwargs["response_format"] = self.response_format
        return self._client.chat.completions.create(**kwargs).choices[0].message.content

    def run(self, prompt: str) -> str:
        chunks = self.retrieve_knowledge(prompt)

        system = self.system_prompt
        if chunks:
            context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
            system = f"{system}\n\n<context>\n{context}\n</context>"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
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
