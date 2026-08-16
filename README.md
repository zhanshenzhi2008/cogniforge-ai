# Cogniforge AI Service

FastAPI-based AI service: Agent、RAG（解析/分块/向量库）、Memory。  
**不直接调 DeepSeek/OpenAI。** 聊天和 embedding 都回调 Go 后端，模型配置只有控制台「模型」页这一套。

## Features

- **Agent Execution**: Run AI agents with tools and memory（LLM 经 Go）
- **LLM via Go hub**: `COGNIFORGE_API_URL` → `/api/v1/chat/completions` / `/api/v1/chat/stream`
- **RAG Knowledge Base**: Document parsing, chunking, pgvector；embedding 经 Go `/api/v1/embeddings`
- **Memory Management**: Conversation history and context
- **Function Calling**: Tool registration and execution

## Architecture

```
浏览器 / 控制台
        │
        ▼
   Go 后端 :8080          ← 唯一拿 Key、调供应商
   ai_providers
        │
        ├─ 聊天：直接调 DeepSeek / OpenAI
        ├─ 知识库 CRUD：HTTP → Python :8086 /api/rag
        └─ embeddings：直接调供应商 /v1/embeddings
                ▲
                │  Python Agent / LLM / RAG embedding
                │  回调 Go，不自己持有 API Key
        Python :8086
        解析 / 分块 / pgvector / Agent 工具
```

不要把 Go 聊天再转到 Python `/api/llm`，否则会绕一圈。

## Project Structure

```
cogniforge-ai/
├── app/
│   ├── main.py              # FastAPI application
│   └── routers/             # API routers
│       ├── agent.py         # Agent endpoints
│       ├── llm.py           # LLM endpoints（转发 Go）
│       ├── memory.py        # Memory endpoints
│       └── rag.py           # RAG endpoints
├── llm/                     # LLM providers
│   ├── base.py              # Base provider interface
│   ├── go_gateway.py        # 回调 Go 聊天接口
│   └── model_config.py      # 本地 cachetools + Redis 模型配置缓存
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
│       ├── embedding/       # Go hub / Local
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
```

本地开发把 `COGNIFORGE_API_URL` 设为 `http://localhost:8080`，并先启动 Go 后端。

### 3. Start the Server

```bash
./start.sh
# or
uvicorn app.main:app --host 0.0.0.0 --port 8086 --reload
```

## Environment Variables

LLM 密钥**不要**写在 Python `.env` 里。只配 Go 地址；Key 在控制台「模型」页。

| Variable | Description | Default |
|----------|-------------|---------|
| `COGNIFORGE_API_URL` | Go 后端地址 | `http://localhost:8080` |
| `REDIS_HOST` | Redis（与 Go 共用 `cogniforge:modelcfg:*`） | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `REDIS_PWD` | Redis password | empty |
| `PGSQL_HOST` | PostgreSQL host（pgvector） | `localhost` |
| `PGSQL_PORT` | PostgreSQL port | `5432` |
| `PGSQL_DB` | Database name | `cogniforge` |
| `PGSQL_USERNAME` | Database user | `postgres` |
| `PGSQL_PASSWORD` | Database password | - |
| `EMBEDDER_TYPE` | `go`（回调 Go）或 `local` | `go` |
| `CHUNK_SIZE` | Text chunk size | `512` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |

已废弃（请从 Python `.env` 删除）：`ENCRYPTION_KEY`、`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`OPENROUTER_API_KEY`、`OPENAI_BASE_URL`、`DEFAULT_MODEL`、`OPENROUTER_HTTP_REFERER`、`OPENROUTER_TITLE`

若当前「模型」页供应商不支持 `/v1/embeddings`（例如只做聊天的 DeepSeek），把 `EMBEDDER_TYPE=local`。

## API Endpoints

### Health Check

```bash
curl http://localhost:8086/health
```

### LLM Endpoints (`/api/llm`)

```bash
# Chat（实际转发到 Go，用模型页当前启用的供应商）
curl -X POST http://localhost:8086/api/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Agent Endpoints (`/api/agent`)

```bash
# Agent chat with tools
curl -X POST http://localhost:8086/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{
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

```
Go knowledge  →  Python /api/rag     （解析、分块、pgvector）
Python LLM    →  Go /api/v1/chat/*   （同一套 ai_providers）
Python embed  →  Go /api/v1/embeddings
```

生产 compose：`COGNIFORGE_API_URL=http://cogniforge:8080`

## License

MIT
