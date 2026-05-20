"""
Cogniforge AI Service
FastAPI application providing Agent, LLM, RAG, and Memory services.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llm import OpenAIProvider, AnthropicProvider
from agent import AgentExecutor
from memory import MemoryManager
from services.rag import DocumentProcessor

from app.routers import rag_router, agent_router, llm_router, memory_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Cogniforge AI Service...")
    
    # Initialize LLM Providers
    app.state.llm_providers = {}
    
    if os.getenv("OPENAI_API_KEY"):
        try:
            app.state.llm_providers["openai"] = OpenAIProvider()
            logger.info("OpenAI provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            app.state.llm_providers["anthropic"] = AnthropicProvider()
            logger.info("Anthropic provider initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Anthropic provider: {e}")
    
    # Initialize Memory Manager
    app.state.memory = MemoryManager()
    logger.info("Memory manager initialized")
    
    # Initialize Agent Executor
    app.state.agent_executor = AgentExecutor(
        providers=app.state.llm_providers,
        memory=app.state.memory
    )
    logger.info("Agent executor initialized")
    
    # Initialize RAG Processor
    embedder_type = os.getenv("EMBEDDER_TYPE", "openai")
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "pgvector")
    
    try:
        app.state.rag_processor = DocumentProcessor(
            embedder_type=embedder_type,
            vector_store_type=vector_store_type,
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50"))
        )
        app.state.rag_processor.connect()
        logger.info("RAG processor initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize RAG processor: {e}")
        app.state.rag_processor = None
    
    # Register routers with their dependencies
    from app.routers.rag import set_processor
    from app.routers.agent import set_executor
    from app.routers.llm import set_providers
    from app.routers.memory import set_memory
    
    set_providers(app.state.llm_providers)
    set_memory(app.state.memory)
    set_executor(app.state.agent_executor)
    if app.state.rag_processor:
        set_processor(app.state.rag_processor)
    
    logger.info("Cogniforge AI Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cogniforge AI Service...")
    if app.state.rag_processor:
        app.state.rag_processor.disconnect()


app = FastAPI(
    title="Cogniforge AI Service",
    description="AI service providing Agent execution, LLM calls, RAG knowledge base, and Memory management",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(llm_router)
app.include_router(agent_router)
app.include_router(memory_router)
app.include_router(rag_router)


@app.get("/health")
async def health_check():
    """Overall health check."""
    return {
        "status": "ok",
        "services": {
            "llm": list(app.state.llm_providers.keys()),
            "memory": app.state.memory is not None,
            "agent": app.state.agent_executor is not None,
            "rag": app.state.rag_processor is not None,
        }
    }


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "name": "Cogniforge AI Service",
        "version": "1.0.0",
        "services": {
            "llm": "/api/llm",
            "agent": "/api/agent",
            "memory": "/api/memory",
            "rag": "/api/rag",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8086,
        reload=True
    )
