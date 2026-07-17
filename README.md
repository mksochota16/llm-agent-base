# llm-agent-base

A lightweight Python library for building LLM agents with tool calling and retrieval-augmented generation (RAG). Works with any OpenAI-compatible API (OpenRouter, OpenAI, Ollama, etc.).

Available on PyPI: https://pypi.org/project/llm-agent-base/

## Installation

```bash
pip install llm-agent-base
```

## Features

- **Simple LLM calls** — single-call `ask()` with no overhead for straightforward completions
- **Tool calling** — register plain Python functions as LLM-callable tools; schemas are built automatically from type hints and docstrings
- **RAG** — ingest a folder of documents (`.txt`, `.md`, `.json`, `.pdf`) into a FAISS vector index and inject relevant chunks into every prompt
- **Pipelines** — chain multiple agents so each agent's output becomes the next agent's input
- **Temperature control** — set per-agent temperature for precise or creative responses
- **Response format** — enforce JSON output or structured schemas via the OpenAI response format API
- **Debug mode** — optional logging of tool calls and knowledge retrievals

## Quick start

```python
from llm_agent_base import AgentBase, LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

agent = AgentBase(
    system_prompt="You are a helpful assistant.",
    llm_config=config,
)

# Simple one-shot call
print(agent.ask("What is the capital of France?"))

# Full agentic loop (knowledge retrieval + tool calling)
print(agent.run("What is the capital of France?"))
```

The default `base_url` points to [OpenRouter](https://openrouter.ai), which gives access to many models through a single API key. You can swap it for the OpenAI base URL or any other compatible endpoint.

## Usage

### Simple LLM call

`ask()` makes a single completion call with no knowledge retrieval or tool calling — the fastest path when you just need a plain response.

```python
from llm_agent_base import AgentBase, LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

agent = AgentBase(system_prompt="You are a helpful assistant.", llm_config=config)
print(agent.ask("Summarise the water cycle in one sentence."))
```

### Temperature and response format

```python
# Low temperature for deterministic, factual responses
precise = AgentBase(
    system_prompt="You are a helpful assistant.",
    llm_config=config,
    temperature=0.1,
)

# High temperature for creative responses
creative = AgentBase(
    system_prompt="You are a poet.",
    llm_config=config,
    temperature=1.4,
)

# Enforce JSON output
json_agent = AgentBase(
    system_prompt="Always respond with valid JSON.",
    llm_config=config,
    response_format={"type": "json_object"},
)
print(json_agent.ask('Return {"city": "Paris", "country": "France"}'))
```

### Tool calling

Register any Python function as a tool. The function name becomes the tool name, the docstring becomes its description, and the type hints define the parameter schema.

```python
from llm_agent_base import AgentBase, LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

agent = AgentBase(
    system_prompt="You are a helpful assistant. Use the available tools when needed.",
    llm_config=config,
)

@agent.register_tool
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    return f"The weather in {city} is sunny and 22°C."

@agent.register_tool
def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

print(agent.run("What is the weather in Tokyo and what is 10 + 20?"))
```

`register_tool` can also be called directly:

```python
agent.register_tool(get_weather)
```

### RAG (knowledge base)

Place your documents in a folder (organised into subdirectories by topic). Call `ingest_knowledge` once to embed and index them, then `run` as normal — relevant chunks are automatically injected into the system prompt.

```
knowledge/
├── products/
│   ├── faq.md
│   └── pricing.json
└── support/
    └── sla.md
```

```python
from llm_agent_base import AgentBase, LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

agent = AgentBase(
    system_prompt="You are a product assistant. Answer using only the provided context.",
    llm_config=config,
    knowledge_folder_path="knowledge",
    knowledge_index_dir=".kb_index",  # where the FAISS index is saved
    knowledge_top_k=3,                # number of chunks injected per prompt
)

# Build and persist the index (run once, or when documents change)
agent.ingest_knowledge(save=True)

# On subsequent runs, load from disk instead of re-embedding
# agent.load_knowledge()

print(agent.run("Who founded the company and when?"))
```

### Agent pipelines

Chain agents so the output of one becomes the input of the next:

```python
from llm_agent_base import AgentBase, AgentPipelineBase, LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

researcher = AgentBase(
    system_prompt="Extract the key facts from the user's question.",
    llm_config=config,
)
writer = AgentBase(
    system_prompt="Turn the provided facts into a concise, friendly summary.",
    llm_config=config,
)

pipeline = AgentPipelineBase(agents=[researcher, writer])
print(pipeline.run("Tell me about the Acme Corp product lineup."))
```

### Debug mode

Pass `debug=True` to any agent to print tool invocations and knowledge retrievals to stdout:

```python
agent = AgentBase(..., debug=True)
```

```
[debug] Retrieving knowledge
[debug] tool 'get_weather' args={'city': 'Tokyo'} result=The weather in Tokyo is sunny and 22°C.
```

## API reference

### `LLMConnectionConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — | Model identifier (e.g. `"openai/gpt-4o-mini"`) |
| `base_url` | `str` | OpenRouter | API base URL |
| `api_key` | `str \| None` | `None` | API key (falls back to `OPENROUTER_API_KEY` env var) |
| `embedding_model` | `str` | `"openai/text-embedding-3-small"` | Model used for RAG embeddings |

### `AgentBase`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | — | System prompt sent on every call |
| `llm_config` | `LLMConnectionConfig` | — | Connection and model settings |
| `temperature` | `float \| None` | `None` | Sampling temperature (model default when omitted) |
| `response_format` | `dict \| None` | `None` | OpenAI response format (e.g. `{"type": "json_object"}`) |
| `knowledge_folder_path` | `str \| None` | `None` | Folder of documents to index for RAG |
| `knowledge_index_dir` | `str` | `".kb_index"` | Directory where the FAISS index is persisted |
| `knowledge_top_k` | `int` | `5` | Number of chunks injected per prompt |
| `debug` | `bool` | `False` | Print tool calls and retrievals to stdout |

| Method | Description |
|---|---|
| `ask(prompt)` | Single LLM call — no knowledge retrieval or tool calling |
| `run(prompt)` | Full agentic loop — knowledge retrieval + tool-calling until text response |
| `register_tool(fn)` | Register a function as a tool; usable as a decorator |
| `ingest_knowledge(save)` | Parse, embed, and index documents in `knowledge_folder_path` |
| `load_knowledge()` | Restore a previously saved index from `knowledge_index_dir` |

### Other exports

| Class / function | Description |
|---|---|
| `AgentPipelineBase` | Chains multiple `AgentBase` instances in sequence |
| `KnowledgeBase` | Document ingestion, embedding, and FAISS retrieval |
| `DocumentChunk` | Dataclass representing a retrieved text chunk |
| `build_tool_schema` | Builds an OpenAI-compatible tool schema from a function |
| `execute_tool_loop` | Runs the agentic tool-calling loop against any OpenAI-compatible client |
