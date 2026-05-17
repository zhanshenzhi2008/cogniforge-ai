"""
Tool Registry - Manages and executes tools
"""
import logging
from typing import Any, Dict, Optional

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and executing tools"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool"""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self) -> list:
        """List all registered tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]
    
    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name with arguments"""
        tool = self._tools.get(name)
        
        if not tool:
            logger.warning(f"Tool not found: {name}")
            return ToolResult(success=False, error=f"Tool '{name}' not found").to_dict()
        
        try:
            result = await tool.execute(**arguments)
            return ToolResult(success=True, data=result).to_dict()
        except Exception as e:
            logger.error(f"Tool execution error: {name} - {e}")
            return ToolResult(success=False, error=str(e)).to_dict()
    
    def get_openai_schema(self) -> list:
        """Get tools in OpenAI function calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self._tools.values()
        ]


_default_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the default tool registry"""
    return _default_registry


def register_default_tools():
    """Register default built-in tools"""
    from .builtins import (
        SearchKnowledgeTool,
        GetTimeTool,
        CalculatorTool,
    )
    
    _default_registry.register(SearchKnowledgeTool())
    _default_registry.register(GetTimeTool())
    _default_registry.register(CalculatorTool())
