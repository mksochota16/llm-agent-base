# llm-agent-base

A lightweight Python base for building LLM agents with tool calling and retrieval-augmented generation (RAG). It works with any OpenAI-compatible API (OpenRouter, OpenAI, local models via Ollama, etc.).

## Features

- **Tool calling** — register plain Python functions as LLM-callable tools; schemas are built automatically from type hints and docstrings
- **RAG** — ingest a folder of documents (`.txt`, `.md`, `.json`, `.pdf`) into a FAISS vector index and inject relevant chunks into every prompt
- **Pipelines** — chain multiple agents so each agent's output becomes the next agent's input
- **Debug mode** — optional logging of tool calls and knowledge retrievals

## Project structure

```
agent_base.py           # AgentBase class
agent_pipeline_base.py  # AgentPipelineBase class
tool_calling.py         # Schema building and tool-call execution loop
knowledge_base.py       # Document ingestion, embedding, and FAISS retrieval
llm_connection_config.py# LLM client configuration
knowledge/              # Knowledge files (subdirectory per topic)
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment**

Copy `.env` and fill in your credentials:

```env
OPENROUTER_API_KEY=your-api-key-here
MODEL=openai/gpt-4o-mini
BASE_URL=https://openrouter.ai/api/v1
```

The default `BASE_URL` points to [OpenRouter](https://openrouter.ai), which gives access to many models through a single API key. You can swap it for the OpenAI base URL or any other compatible endpoint.

## Usage

### Basic agent

```python
from agent_base import AgentBase
from llm_connection_config import LLMConnectionConfig

config = LLMConnectionConfig(model="openai/gpt-4o-mini", api_key="...")

agent = AgentBase(
    system_prompt="You are a helpful assistant.",
    llm_config=config,
)

print(agent.run("What is the capital of France?"))
```

### Tool calling

Register any Python function as a tool. The function name becomes the tool name, the docstring becomes its description, and the type hints define the parameter schema.

```python
def get_weather(city: str) -> str:
    """Return the current weather for a given city."""
    return f"The weather in {city} is sunny and 22°C."

def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b

agent = AgentBase(
    system_prompt="You are a helpful assistant. Use the available tools when needed.",
    llm_config=config,
)
agent.register_tool(get_weather)
agent.register_tool(add)

print(agent.run("What is the weather in Tokyo and what is 10 + 20?"))
```

`register_tool` can also be used as a decorator:

```python
@agent.register_tool
def search_orders(order_id: str) -> str:
    """Look up an order by ID."""
    ...
```

### RAG (knowledge base)

Place your documents in a folder (organised into subdirectories by topic). Call `ingest_knowledge` once to embed and index them, then `run` as normal — relevant chunks are automatically injected into the system prompt.

```
knowledge/
├── topic/
│   └── overview.txt
├── products/
│   ├── faq.md
│   └── pricing.json
└── support/
    └── sla.md
```

```python
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
from agent_pipeline_base import AgentPipelineBase

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
