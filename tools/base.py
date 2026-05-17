"""
Tool Base Classes
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Tool(ABC):
    """Base class for tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool"""
        pass


class ToolResult:
    """Result from tool execution"""
    
    def __init__(self, success: bool, data: Any = None, error: str = None):
        self.success = success
        self.data = data
        self.error = error
    
    def to_dict(self) -> dict:
        if self.success:
            return {"success": True, "data": self.data}
        else:
            return {"success": False, "error": self.error}
