"""
Built-in Tools - Default tools provided by the service
"""
import os
import json
import httpx
import logging
from datetime import datetime
from typing import Any

from .base import Tool

logger = logging.getLogger(__name__)


class SearchKnowledgeTool(Tool):
    """Search knowledge base using the RAG service"""
    
    @property
    def name(self) -> str:
        return "search_knowledge"
    
    @property
    def description(self) -> str:
        return "Search the knowledge base for relevant documents. Use this when you need to find information from the user's knowledge base."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                },
                "collection_name": {
                    "type": "string",
                    "description": "The knowledge collection to search"
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, top_k: int = 5, collection_name: str = "default") -> Any:
        rag_url = os.getenv("RAG_SERVICE_URL", "http://localhost:8085")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{rag_url}/api/knowledge/search",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "collection_name": collection_name,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"RAG service error: {e}")
            return {"error": f"Failed to search knowledge base: {str(e)}"}


class GetTimeTool(Tool):
    """Get the current date and time"""
    
    @property
    def name(self) -> str:
        return "get_time"
    
    @property
    def description(self) -> str:
        return "Get the current date and time. Use this when you need to know the current time or date."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {}
        }
    
    async def execute(self) -> Any:
        now = datetime.now()
        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": "UTC+8",
        }


class CalculatorTool(Tool):
    """Simple calculator for mathematical expressions"""
    
    @property
    def name(self) -> str:
        return "calculate"
    
    @property
    def description(self) -> str:
        return "Evaluate a mathematical expression. Use this for calculations."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> Any:
        try:
            import math
            
            safe_dict = {
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "sqrt": math.sqrt, "sin": math.sin,
                "cos": math.cos, "tan": math.tan, "log": math.log,
                "log10": math.log10, "exp": math.exp, "pi": math.pi,
                "e": math.e,
            }
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": f"Calculation error: {str(e)}"}


class HttpRequestTool(Tool):
    """Make HTTP requests"""
    
    @property
    def name(self) -> str:
        return "http_request"
    
    @property
    def description(self) -> str:
        return "Make an HTTP request to an external API."
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    "description": "HTTP method"
                },
                "url": {
                    "type": "string",
                    "description": "The URL to request"
                },
                "headers": {
                    "type": "object",
                    "description": "HTTP headers"
                },
                "body": {
                    "type": "object",
                    "description": "Request body (for POST/PUT/PATCH)"
                }
            },
            "required": ["method", "url"]
        }
    
    async def execute(
        self,
        method: str,
        url: str,
        headers: dict = None,
        body: dict = None,
    ) -> Any:
        try:
            async with httpx.AsyncClient() as client:
                kwargs = {"headers": headers or {}}
                if body and method in ["POST", "PUT", "PATCH"]:
                    kwargs["json"] = body
                
                response = await client.request(method, url, **kwargs, timeout=30.0)
                return {
                    "status": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                }
        except httpx.HTTPError as e:
            return {"error": f"HTTP request failed: {str(e)}"}
