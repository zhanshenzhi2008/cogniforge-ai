"""
Agent Executor - Handles agent execution with tools and memory
"""
import logging
import json
from typing import Any, Dict, List, Optional

from llm.base import LLMProvider
from memory.manager import MemoryManager
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentExecutor:
    """Executes agent tasks with LLM, tools, and memory"""
    
    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        memory: MemoryManager,
        max_iterations: int = 10,
    ):
        self.providers = providers
        self.memory = memory
        self.tool_registry = ToolRegistry()
        self.max_iterations = max_iterations
    
    async def execute(self, request: dict) -> dict:
        """
        Execute an agent task.
        
        Request format:
        {
            "provider": "openai",  # LLM provider
            "model": "gpt-4o",     # Optional, uses default if not specified
            "system_prompt": "...", # Agent's system prompt
            "messages": [...],     # Conversation history
            "tools": [...],        # Available tools
            "session_id": "...",   # For memory management
            "stream": false,       # Whether to stream response
        }
        """
        provider_name = request.get("provider", "openai")
        provider = self.providers.get(provider_name)
        
        if not provider:
            return {"error": f"Provider '{provider_name}' not available"}
        
        model = request.get("model")
        system_prompt = request.get("system_prompt", "You are a helpful AI assistant.")
        messages = request.get("messages", [])
        tools = request.get("tools", [])
        session_id = request.get("session_id", "default")
        stream = request.get("stream", False)
        
        if session_id != "default":
            history = await self.memory.get(session_id)
            if history:
                messages = history + messages
        
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        llm_request = {
            "model": model,
            "messages": messages,
            "temperature": request.get("temperature", 0.7),
            "max_tokens": request.get("max_tokens", 4096),
        }
        
        if tools and provider.supports_functions():
            llm_request["functions"] = tools
        
        try:
            if stream:
                return await self._execute_stream(provider, llm_request, session_id)
            else:
                return await self._execute_normal(provider, llm_request, session_id)
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            return {"error": str(e)}
    
    async def _execute_normal(
        self,
        provider: LLMProvider,
        request: dict,
        session_id: str,
    ) -> dict:
        """Execute a single turn without streaming"""
        response = await provider.chat(request)
        
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls and self.max_iterations > 0:
            content = await self._handle_tool_calls(
                provider, request, tool_calls, session_id
            )
        
        if session_id != "default":
            await self.memory.append(
                session_id,
                request["messages"][1:] if request["messages"] and request["messages"][0]["role"] == "system" else request["messages"]
            )
            await self.memory.append(session_id, [{"role": "assistant", "content": content}])
        
        return {
            "id": response.get("id"),
            "model": response.get("model"),
            "content": content,
            "tool_calls": tool_calls,
            "usage": response.get("usage"),
        }
    
    async def _execute_stream(
        self,
        provider: LLMProvider,
        request: dict,
        session_id: str,
    ):
        """Execute with streaming response"""
        full_content = ""
        tool_calls = []
        
        async for chunk in provider.chat_stream(request):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            
            if delta.get("content"):
                full_content += delta["content"]
            
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    if len(tool_calls) <= tc.get("index", 0):
                        tool_calls.append(tc)
                    else:
                        existing = tool_calls[tc["index"]]
                        if tc.get("function", {}).get("arguments"):
                            existing["function"]["arguments"] = (
                                existing.get("function", {}).get("arguments", "") 
                                + tc["function"]["arguments"]
                            )
            
            yield chunk
        
        if tool_calls and self.max_iterations > 0:
            full_content = await self._handle_tool_calls(
                provider, request, tool_calls, session_id
            )
            yield {"final_content": full_content}
    
    async def _handle_tool_calls(
        self,
        provider: LLMProvider,
        request: dict,
        tool_calls: List[dict],
        session_id: str,
    ) -> str:
        """Handle tool calls from the LLM"""
        results = []
        
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name")
            arguments = func.get("arguments", "{}")
            
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                args = {}
            
            logger.info(f"Executing tool: {name} with args: {args}")
            
            result = await self.tool_registry.execute(name, args)
            results.append({
                "tool_call_id": tc.get("id"),
                "tool_name": name,
                "result": result,
            })
            
            request["messages"].append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tc],
            })
            request["messages"].append({
                "role": "tool",
                "tool_call_id": tc.get("id"),
                "content": json.dumps(result),
            })
        
        response = await provider.chat(request)
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")
