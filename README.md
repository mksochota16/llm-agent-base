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
- **Tool calling** — register plain Python functions as LLM-callable tools; schemas are built automatically from type hints and docstrings
- **RAG** — ingest a folder of documents (`.txt`, `.md`, `.json`, `.pdf`) into a FAISS vector index and inject relevant chunks into every prompt
- **Knowledge search tool** — when a knowledge base is configured, the agent automatically gains a `search_knowledge` tool it can call mid-reasoning for targeted lookups
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

### RAG (knowledge base)

Place your documents in a folder (organised into subdirectories by topic). Call `ingest_knowledge` once to embed and index them, then use `run`, `ask`, or `chat` — relevant chunks are automatically retrieved and injected into the system prompt on every call.

When a knowledge base is configured, the agent also registers a `search_knowledge` tool automatically. This lets the LLM perform targeted knowledge lookups mid-reasoning, in addition to the upfront retrieval that seeds each prompt.

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
| `ask(prompt, files)` | Single LLM call with optional knowledge retrieval; no tool calling |
| `run(prompt, files)` | Full agentic loop — knowledge retrieval + tool calling until text response |
| `chat(message, files)` | Like `run()` but accumulates conversation history across calls |
| `reset_conversation()` | Clear the stored conversation history |
| `register_tool(fn)` | Register a function as a tool; usable as a decorator |
| `ingest_knowledge(save)` | Parse, embed, and index documents in `knowledge_folder_path` |
| `load_knowledge()` | Restore a previously saved index from `knowledge_index_dir` |
| `retrieve_knowledge(query)` | Manually retrieve the top-k chunks for a query |

### Other exports

| Class / function | Description |
|---|---|
| `AgentPipelineBase` | Chains multiple `AgentBase` instances in sequence |
| `KnowledgeBase` | Document ingestion, embedding, and FAISS retrieval |
| `DocumentChunk` | Dataclass representing a retrieved text chunk |
| `build_tool_schema` | Builds an OpenAI-compatible tool schema from a function |
| `execute_tool_loop` | Runs the agentic tool-calling loop against any OpenAI-compatible client |
