# Cogniforge AI Service

FastAPI-based AI service providing Agent execution, LLM calls, RAG knowledge base, and Memory management.

## Features

- **Agent Execution**: Run AI agents with tools and memory
- **Multi-Provider LLM**: OpenAI GPT, Anthropic Claude
- **RAG Knowledge Base**: Document parsing, embedding, vector search
- **Memory Management**: Conversation history and context
- **Function Calling**: Tool registration and execution

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Cogniforge AI Service                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────┐│
│  │  Agent    │  │   LLM     │  │   RAG     │  │ Memory  ││
│  │ Service   │  │ Service   │  │ Service   │  │ Service ││
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └────┬────┘│
│        │               │               │              │       │
│        └───────────────┼───────────────┴──────────────┘       │
│                        │                                      │
│                        ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Tool Registry                         │ │
│  │  - search_knowledge (RAG)  - get_time                  │ │
│  │  - calculate               - http_request              │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Parsers                               │ │
│  │  PDF  |  DOCX  |  TXT  |  Markdown  |  HTML           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Embedders                             │ │
│  │  OpenAI Embeddings  |  Local (Sentence-Transformers)   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Vector Store                          │ │
│  │               PostgreSQL + pgvector                     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
cogniforge-ai/
├── app/
│   ├── main.py              # FastAPI application
│   └── routers/             # API routers
│       ├── agent.py         # Agent endpoints
│       ├── llm.py           # LLM endpoints
│       ├── memory.py        # Memory endpoints
│       └── rag.py           # RAG endpoints
├── llm/                     # LLM providers
│   ├── base.py              # Base provider interface
│   ├── openai_provider.py   # OpenAI implementation
│   └── anthropic_provider.py # Anthropic implementation
├── agent/                   # Agent execution
│   └── executor.py          # Agent executor
├── tools/                   # Function calling
│   ├── base.py              # Base tool class
│   ├── registry.py          # Tool registry
│   └── builtins.py          # Built-in tools
├── memory/                  # Memory management
│   └── manager.py           # Memory manager
├── services/
│   └── rag/                 # RAG service
│       ├── parsers/         # Document parsers
│       ├── splitters/       # Text chunkers
│       ├── embedding/       # Embedding models
│       └── vector_store/    # Vector storage
├── tests/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── start.sh
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env

# Edit .env with your configuration
```

### 3. Start the Server

```bash
./start.sh
# or
uvicorn app.main:app --host 0.0.0.0 --port 8086 --reload
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `DEFAULT_MODEL` | Default LLM model | `gpt-4o` |
| `EMBEDDER_TYPE` | Embedder type (`openai` or `local`) | `openai` |
| `CHUNK_SIZE` | Text chunk size | `512` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `cogniforge` |

## API Endpoints

### Health Check

```bash
curl http://localhost:8086/health
```

### LLM Endpoints (`/api/llm`)

```bash
# Chat
curl -X POST http://localhost:8086/api/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Agent Endpoints (`/api/agent`)

```bash
# Agent chat with tools
curl -X POST http://localhost:8086/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "system_prompt": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "session_id": "user_123"
  }'
```

### Memory Endpoints (`/api/memory`)

```bash
# Save to memory
curl -X POST http://localhost:8086/api/memory/save \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user_123",
    "messages": [{"role": "user", "content": "My name is John"}]
  }'

# Get memory
curl http://localhost:8086/api/memory/user_123?limit=10

# Clear memory
curl -X DELETE http://localhost:8086/api/memory/user_123
```

### RAG Endpoints (`/api/rag`)

```bash
# Process document
curl -X POST http://localhost:8086/api/rag/process \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/document.pdf",
    "document_id": "doc_001",
    "collection_name": "my_kb"
  }'

# Search knowledge base
curl -X POST http://localhost:8086/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "collection_name": "my_kb",
    "top_k": 5
  }'

# Upload and process
curl -X POST http://localhost:8086/api/rag/upload \
  -F "file=@document.pdf" \
  -F "document_id=doc_001" \
  -F "collection_name=my_kb"
```

## Built-in Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `search_knowledge` | Search RAG knowledge base | query, top_k, collection_name |
| `get_time` | Get current date/time | - |
| `calculate` | Evaluate math expressions | expression |
| `http_request` | Make HTTP requests | method, url, headers, body |

## Integration with Go Backend

```yaml
# Go config.yaml
ai:
  python_service_url: http://localhost:8086
```

## License

MIT
