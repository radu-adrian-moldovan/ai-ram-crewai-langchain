"""
Tool Manager - Core Tool Abstraction and Framework Adapters
Provides unified tool abstraction, framework adapters, and tool registry system.
"""

from typing import Any, Dict, List, Callable, Awaitable, Union, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Core Unified Tool System
# ============================================================================

class UnifiedTool:
    """Unified tool that works across LangChain, CrewAI, and MCP."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        func: Union[Callable[..., Any], Awaitable[Any]]
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func

    async def run(self, **kwargs) -> Any:
        """Execute the tool with given parameters asynchronously."""
        result = self.func(**kwargs)
        if hasattr(result, "__await__"):  # async func
            return await result
        return result
    
    def sync_run(self, **kwargs) -> Any:
        """Execute the tool with given parameters synchronously."""
        result = self.func(**kwargs)
        if hasattr(result, "__await__"):  # async func
            # Run in event loop if it's async
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new event loop in a thread if current loop is running
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, result)
                        return future.result()
                else:
                    return loop.run_until_complete(result)
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(result)
        return result

# ============================================================================
# LangChain Adapter
# ============================================================================

try:
    from langchain_core.tools.base import BaseTool
    from pydantic import Field
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    
    class LangChainToolAdapter(BaseTool):
        """Adapter to use UnifiedTool with LangChain."""
        
        unified_tool: UnifiedTool = Field(...)
        
        class Config:
            arbitrary_types_allowed = True
        
        def _run(
            self, 
            tool_input: str = "", 
            run_manager: Optional[CallbackManagerForToolRun] = None,
            **kwargs: Any
        ) -> str:
            """Synchronous run method for LangChain."""
            # Parse tool_input if it's a string with key=value pairs
            parsed_kwargs = self._parse_tool_input(tool_input)
            parsed_kwargs.update(kwargs)
            
            result = self.unified_tool.sync_run(**parsed_kwargs)
            return str(result)
        
        async def _arun(
            self, 
            tool_input: str = "", 
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
            **kwargs: Any
        ) -> str:
            """Asynchronous run method for LangChain."""
            # Parse tool_input if it's a string with key=value pairs
            parsed_kwargs = self._parse_tool_input(tool_input)
            parsed_kwargs.update(kwargs)
            
            result = await self.unified_tool.run(**parsed_kwargs)
            return str(result)
        
        def _parse_tool_input(self, tool_input: str) -> Dict[str, Any]:
            """Parse tool input string into kwargs."""
            if not tool_input:
                return {}
            
            result = {}
            # Simple parsing for "key=value,key2=value2" format
            if '=' in tool_input:
                parts = tool_input.split(',')
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        result[key.strip()] = value.strip()
            else:
                # If no key=value format, treat as image_path
                result['image_path'] = tool_input
            
            return result

    LANGCHAIN_AVAILABLE = True
    
except ImportError:
    LANGCHAIN_AVAILABLE = False
    
    class LangChainToolAdapter:
        """Dummy adapter when LangChain is not available."""
        def __init__(self, unified_tool: UnifiedTool):
            raise ImportError("LangChain not available")

# ============================================================================
# CrewAI Adapter
# ============================================================================

try:
    def to_crewai_tool(unified_tool: UnifiedTool):
        """Convert UnifiedTool to CrewAI compatible tool."""
        
        def tool_func(tool_input: str = "", **kwargs) -> str:
            """Tool function for CrewAI agents."""
            # Parse tool_input
            parsed_kwargs = {}
            if tool_input:
                if '=' in tool_input:
                    parts = tool_input.split(',')
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            parsed_kwargs[key.strip()] = value.strip()
                else:
                    parsed_kwargs['image_path'] = tool_input
            
            # Merge with direct kwargs
            parsed_kwargs.update(kwargs)
            
            result = unified_tool.sync_run(**parsed_kwargs)
            return str(result)
        
        # Return a dictionary with the tool definition that CrewAI expects
        return {
            "name": unified_tool.name,
            "description": unified_tool.description,
            "func": tool_func,
            "args": unified_tool.input_schema.get("properties", {}),
            "return_type": "str"
        }
    
    CREWAI_AVAILABLE = True
    
except ImportError:
    CREWAI_AVAILABLE = False
    
    def to_crewai_tool(unified_tool: UnifiedTool):
        """Dummy adapter when CrewAI dependencies are not available."""
        raise ImportError("CrewAI dependencies not available")

# ============================================================================
# Tool Registry and Management
# ============================================================================

# Global tool registry
TOOL_REGISTRY: List[UnifiedTool] = []

def register_tool(tool: UnifiedTool) -> UnifiedTool:
    """Register a tool in the global registry."""
    if tool not in TOOL_REGISTRY:
        TOOL_REGISTRY.append(tool)
    return tool

def get_tool(name: str) -> Optional[UnifiedTool]:
    """Get a tool by name from the registry."""
    return next((tool for tool in TOOL_REGISTRY if tool.name == name), None)

def list_tools() -> List[UnifiedTool]:
    """List all registered tools."""
    return TOOL_REGISTRY.copy()

# ============================================================================
# Utility Functions for Framework Integration
# ============================================================================

def create_langchain_tools(unified_tools: List[UnifiedTool]) -> List[Any]:
    """Create LangChain tools from a list of unified tools."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available")
    
    return [LangChainToolAdapter(
        name=tool.name,
        description=tool.description,
        unified_tool=tool
    ) for tool in unified_tools]

def create_crewai_tools(unified_tools: List[UnifiedTool]) -> List[Any]:
    """Create CrewAI tools from a list of unified tools."""
    if not CREWAI_AVAILABLE:
        raise ImportError("CrewAI dependencies not available")
    
    return [to_crewai_tool(tool) for tool in unified_tools]

# ============================================================================
# Convenience Exports
# ============================================================================

# Export main components for easy importing
__all__ = [
    'UnifiedTool',
    'LangChainToolAdapter',
    'to_crewai_tool',
    'register_tool',
    'get_tool',
    'list_tools',
    'create_langchain_tools',
    'create_crewai_tools',
    'LANGCHAIN_AVAILABLE',
    'CREWAI_AVAILABLE',
    'TOOL_REGISTRY'
]
