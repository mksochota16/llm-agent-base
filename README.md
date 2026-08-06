# llm-agent-base

A lightweight Python library for building LLM agents with tool calling and retrieval-augmented generation (RAG). Works with any OpenAI-compatible API (OpenRouter, OpenAI, Ollama, etc.).

Available on PyPI: https://pypi.org/project/llm-agent-base/

## Installation

```bash
pip install llm-agent-base
```

## Features

- **Simple LLM calls** — single-call `ask()` with optional knowledge retrieval for straightforward completions
- **Agentic loop** — `run()` with knowledge retrieval and tool calling until a final text response
- **Conversational chat** — `chat()` maintains conversation history across calls; `reset_conversation()` starts fresh
- **File attachments** — pass images, PDFs, and text files directly to `ask()`, `run()`, or `chat()` via the `files` parameter
- **Tool calling** — register plain Python functions as LLM-callable tools; schemas are built automatically from type hints and docstrings, including per-parameter descriptions and enum values
- **RAG** — ingest a folder of documents (`.txt`, `.md`, `.json`, `.pdf`) into a FAISS vector index and inject relevant chunks into every prompt
- **Incremental re-indexing** — edit a document and only that file is re-embedded; unchanged chunks keep their vectors
- **Knowledge search tool** — when a knowledge base is configured, the agent automatically gains a `search_knowledge` tool (semantic/vector) and a `read_knowledge_files` tool (keyword search returning full file contents); both are optional and independently toggleable
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

`ask()` makes a single completion call with no tool calling. If a knowledge base is configured, relevant chunks are automatically retrieved and injected as context.

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

Schema generation reads more than the bare type. A docstring `Args:` section (Google style) or `:param name:` lines (Sphinx style) become per-parameter descriptions, `Literal` and `Enum` annotations become `enum` constraints, and `list[str]` gets a proper `items` type — which strict endpoints require and which measurably improves how accurately the model fills arguments. `Returns:`/`Raises:` sections are kept out of the tool description.

```python
@agent.register_tool
def search_orders(customer: str, status: Literal["open", "shipped", "cancelled"] = "open") -> str:
    """Look up a customer's orders.

    Args:
        customer: The customer's account ID.
        status: Which order state to filter on.
    """
```

The tool-calling loop handles both the standard OpenAI `tool_calls` field and the fallback case where a model emits a tool call as inline JSON text in the message content — a behaviour seen on some OpenAI-compatible endpoints (e.g. GLM models). Two inline shapes are recognised:

```json
{"name": "get_weather", "arguments": {"city": "Tokyo"}}
{"get_weather": {"city": "Tokyo"}}
```

The second form — the tool name as the single top-level key, arguments as the nested object — is also produced by GLM models. When an inline call is detected and the tool name is registered, the tool is executed, the result is fed back, and the loop continues exactly as with a structural call. Unknown tool names and non-tool JSON are returned unchanged. A safety cap of 5 consecutive text-emitted calls prevents infinite loops.

The loop is bounded by `max_iterations` (default 10). Once the budget is spent, the tools are withheld from the request so the model has to answer in text — the loop always terminates and always returns a string, even against a model that would otherwise keep calling tools forever.

```python
agent = AgentBase(..., max_iterations=25)  # allow longer tool chains
```

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

### Conversational chat

`chat()` works like `run()` — knowledge retrieval and tool calling are both active — but it accumulates the conversation history across calls so the LLM retains context between turns. Call `reset_conversation()` to wipe the history and start a new session.

```python
from llm_agent_base import AgentBase, LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

agent = AgentBase(
    system_prompt="You are a helpful assistant. Keep answers concise.",
    llm_config=config,
)

print(agent.chat("My name is Alice."))
print(agent.chat("What is my name?"))  # agent remembers: Alice

agent.reset_conversation()

print(agent.chat("What is my name?"))  # history cleared, agent no longer knows
```

History is unbounded by default and will eventually exceed the model's context window in a long session. Set `max_history_messages` to cap it — the oldest turns are dropped once the limit is passed:

```python
agent = AgentBase(..., max_history_messages=20)  # keep the last 10 turns
```

### RAG (knowledge base)

Place your documents in a folder (organised into subdirectories by topic). Call `ingest_knowledge` once to embed and index them, then use `run`, `ask`, or `chat` — relevant chunks are automatically retrieved and injected into the system prompt on every call.

When a knowledge base is configured, the agent registers two tools the LLM can call mid-reasoning:

- **`search_knowledge`** — semantic vector search; returns the most relevant chunks for a query. Requires an ingested FAISS index.
- **`read_knowledge_files`** — keyword search by filename or file content; returns complete file text. Works directly on files without a vector index.

Both tools are enabled by default. Use `knowledge_search_tool=False` or `knowledge_file_tool=False` to disable either one.

`read_knowledge_files` returns at most `knowledge_file_max_chars` (default 20 000) per call, so a broad keyword cannot return the entire knowledge base and overflow the context window. When the limit is hit, the result ends with a note telling the model how many files were omitted and to narrow its search.

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

# Load from disk if a saved index exists, otherwise ingest and save automatically
agent.load_or_ingest_knowledge()

print(agent.run("Who founded the company and when?"))
```

Pass `auto_load_or_ingest=True` to do this in the constructor:

```python
agent = AgentBase(
    system_prompt="You are a product assistant. Answer using only the provided context.",
    llm_config=config,
    knowledge_folder_path="knowledge",
    auto_load_or_ingest=True,
)

print(agent.run("Who founded the company and when?"))
```

For more control, call `ingest_knowledge()` and `load_knowledge()` directly:

```python
# Build and persist the index (run once, or when documents change)
agent.ingest_knowledge(save=True)

# On subsequent runs, load from disk instead of re-embedding
agent.load_knowledge()
```

#### Keeping the index current

`load_or_ingest_knowledge()` compares a content hash of every source file against the saved index. If documents were added, edited, or deleted, only those files are re-embedded — everything else keeps its existing vector. Editing one file in a 500-document corpus costs one file's worth of embedding calls, not 500.

Call `sync_knowledge()` to do it explicitly:

```python
summary = agent.sync_knowledge()
# {'added': 1, 'updated': 1, 'removed': 0,
#  'reused_chunks': 128, 'embedded_chunks': 4, 'total_chunks': 132}
```

Embeddings are sent in batches (100 chunks per request by default), so a first-time ingest of 1 000 chunks is 10 requests rather than 1 000.

#### Filtering weak matches

`retrieve` returns chunks in similarity order, each carrying its cosine similarity in `score`. By default it always returns `knowledge_top_k` chunks, even when none are actually relevant. Set `knowledge_min_score` to drop weak matches instead:

```python
agent = AgentBase(
    ...,
    knowledge_top_k=5,
    knowledge_min_score=0.35,  # return fewer than 5 rather than pad with noise
)
```

#### Keyword file search without a vector index

`read_knowledge_files` searches filenames and file contents directly — no embedding or FAISS index needed. Disable `search_knowledge` to use only this tool:

```python
agent = AgentBase(
    system_prompt="You are a product assistant. Answer using only the provided context.",
    llm_config=config,
    knowledge_folder_path="knowledge",
    knowledge_search_tool=False,  # no vector index required
)

print(agent.run("What are the support SLA terms?"))
```

The LLM can search with fine-grained control over where to look and how many keywords must match:

```python
# filename or content, at least 2 of the 3 keywords must match
read_knowledge_files(
    keywords=["pricing plan", "enterprise", "discount"],
    search_in="both",
    match_mode="min",
    min_matches=2,
)
```

| `search_in` | Where keywords are matched |
|---|---|
| `"filename"` | File name only |
| `"content"` | File contents only |
| `"both"` (default) | Filename first, then contents |

| `match_mode` | Files returned |
|---|---|
| `"any"` (default) | At least one keyword matches |
| `"all"` | Every keyword must match |
| `"min"` | At least `min_matches` keywords match |

### File attachments

All three call methods accept an optional `files` parameter — a list of file paths to attach to the prompt. Files are embedded directly in the message sent to the model.

| Type | Extensions |
|---|---|
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp` |
| Documents | `.pdf` |
| Text | `.txt`, `.md`, `.json`, `.csv`, `.xml`, `.html`, `.yaml`, `.yml`, `.py`, `.js`, `.ts` |

Passing a file with any other extension raises a `ValueError`.

```python
# Describe an image (requires a vision-capable model)
print(agent.ask("What's in this diagram?", files=["architecture.png"]))

# Summarise a PDF
print(agent.run("Summarise the key findings.", files=["report.pdf"]))

# Multi-turn with an attached file — the file stays in conversation history
agent.chat("Here's our codebase overview.", files=["overview.md"])
agent.chat("Which module handles authentication?")
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

## Upgrading to 0.3.0

The knowledge index catalog is now stored as `catalog.json` instead of `catalog.pkl`. Loading an index required unpickling it, which meant a saved index could execute arbitrary code — so the pickle path was removed rather than kept as a fallback.

Existing indexes are not readable. Delete the index directory and re-ingest once:

```python
agent.ingest_knowledge(save=True)   # or just delete .kb_index and run as usual
```

`load_knowledge()` raises a `FileNotFoundError` explaining this if it finds a legacy `catalog.pkl`.

## API reference

### `LLMConnectionConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | — | Model identifier (e.g. `"openai/gpt-4o-mini"`) |
| `base_url` | `str` | OpenRouter | API base URL |
| `api_key` | `str \| None` | `None` | API key (falls back to `OPENROUTER_API_KEY` env var) |
| `embedding_model` | `str` | `"openai/text-embedding-3-small"` | Model used for RAG embeddings |
| `max_retries` | `int` | `3` | Retries on 429/5xx/connection errors, with exponential backoff and jitter |
| `timeout` | `float \| None` | `None` | Per-request timeout in seconds (SDK default when omitted) |

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
| `knowledge_min_score` | `float \| None` | `None` | Drop retrieved chunks below this cosine similarity |
| `auto_load_or_ingest` | `bool` | `False` | Load saved index on init, or ingest and save if none exists |
| `knowledge_search_tool` | `bool` | `True` | Register the `search_knowledge` semantic vector search tool |
| `knowledge_file_tool` | `bool` | `True` | Register the `read_knowledge_files` keyword file search tool |
| `knowledge_file_max_chars` | `int` | `20000` | Max characters `read_knowledge_files` may return in one result |
| `max_iterations` | `int` | `10` | Max tool-calling rounds before a final answer is forced |
| `max_history_messages` | `int \| None` | `None` | Cap on stored `chat()` history; oldest turns are dropped (unlimited when `None`) |
| `debug` | `bool` | `False` | Print tool calls and retrievals to stdout |

| Method | Description |
|---|---|
| `ask(prompt, files)` | Single LLM call with optional knowledge retrieval; no tool calling |
| `run(prompt, files)` | Full agentic loop — knowledge retrieval + tool calling until text response |
| `chat(message, files)` | Like `run()` but accumulates conversation history across calls |
| `reset_conversation()` | Clear the stored conversation history |
| `register_tool(fn)` | Register a function as a tool; usable as a decorator |
| `ingest_knowledge(save)` | Parse, embed, and index documents in `knowledge_folder_path` |
| `load_knowledge()` | Restore a previously saved index from `knowledge_index_dir` |
| `load_or_ingest_knowledge()` | Load saved index if one exists (re-syncing changed files), otherwise ingest and save |
| `sync_knowledge(save)` | Re-embed only the source files that changed since the last ingest |
| `retrieve_knowledge(query)` | Manually retrieve the top-k chunks for a query |

**`read_knowledge_files` tool parameters** (called by the LLM, not directly):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keywords` | `list[str]` | — | Phrases to search for; each entry matched as-is |
| `search_in` | `str` | `"both"` | `"filename"`, `"content"`, or `"both"` |
| `match_mode` | `str` | `"any"` | `"any"` (OR), `"all"` (AND), or `"min"` (at least `min_matches`) |
| `min_matches` | `int \| None` | `None` | Minimum number of matching keywords when `match_mode="min"` |

### Other exports

| Class / function | Description |
|---|---|
| `AgentPipelineBase` | Chains multiple `AgentBase` instances in sequence |
| `KnowledgeBase` | Document ingestion, embedding, and FAISS retrieval; `sync()` re-embeds only changed files |
| `DocumentChunk` | Dataclass representing a retrieved text chunk, including its `score` |
| `build_tool_schema` | Builds an OpenAI-compatible tool schema from a function's hints and docstring |
| `execute_tool_loop` | Runs the agentic tool-calling loop against any OpenAI-compatible client; handles both structural `tool_calls` and inline JSON text tool calls |
