"""
MCP Manager - Model Context Protocol Service Management
Provides MCP service communication, manager class, and utilities.
"""

from typing import Any, Dict, List
import asyncio
import logging
import requests
from datetime import datetime
from .manager_tool import (
    UnifiedTool, 
    register_tool, 
    get_tool, 
    list_tools,
    LANGCHAIN_AVAILABLE,
    CREWAI_AVAILABLE
)

logger = logging.getLogger(__name__)

# ============================================================================
# MCP Service Implementation
# ============================================================================

def mcp_service_call(method: str = "document/analyze", image_path: str = "", mcp_url: str = "http://localhost:3000", **kwargs: Any) -> Dict[str, Any]:
    """Call MCP service with given parameters."""
    
    if not image_path:
        return {"error": "image_path is required"}
    
    try:
        request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Prepare parameters
        params = {"image_path": image_path}
        params.update(kwargs)
        
        # Prepare request payload
        request_payload = {
            "method": method,
            "params": params,
            "id": request_id
        }
        
        # Send request to MCP server
        response = requests.post(
            mcp_url,
            json=request_payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"MCP {method} request successful")
            return result.get("result", {})
        else:
            logger.error(f"MCP request failed with status {response.status_code}")
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"MCP request failed: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"MCP tool error: {e}")
        return {"error": str(e)}

# ============================================================================
# MCP Tool Definition
# ============================================================================

# Create MCP tool instance
mcp_tool = UnifiedTool(
    name="mcp_client",
    description="Integrates with MCP services for enhanced document processing",
    input_schema={
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "description": "MCP method to call (e.g., 'document/classify')",
                "default": "document/analyze"
            },
            "image_path": {
                "type": "string", 
                "description": "Path to the document image"
            },
            "mcp_url": {
                "type": "string",
                "description": "MCP server URL",
                "default": "http://localhost:3000"
            }
        },
        "required": ["image_path"]
    },
    func=mcp_service_call
)

# Register the MCP tool in the global registry
register_tool(mcp_tool)

# ============================================================================
# MCP Manager Class
# ============================================================================

class MCPManager:
    """
    Central manager for MCP tools and framework adapters.
    Provides a unified interface for managing tools across CrewAI and LangChain.
    """
    
    def __init__(self, mcp_url: str = "http://localhost:3000"):
        self.mcp_url = mcp_url
        self.tools = list_tools()
    
    def get_mcp_tool(self) -> UnifiedTool:
        """Get the main MCP tool."""
        return mcp_tool
    
    def create_langchain_adapter(self, tool_name: str = "mcp_client") -> Any:
        """Create a LangChain adapter for the specified tool."""
        from .manager_tool import LangChainToolAdapter
        
        tool = get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available")
        
        return LangChainToolAdapter(
            name=tool.name,
            description=tool.description,
            unified_tool=tool
        )
    
    def create_crewai_adapter(self, tool_name: str = "mcp_client") -> Any:
        """Create a CrewAI adapter for the specified tool."""
        from .manager_tool import to_crewai_tool
        
        tool = get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI dependencies not available")
        
        return to_crewai_tool(tool)
    
    def list_available_tools(self) -> List[str]:
        """List names of all available tools."""
        return [tool.name for tool in self.tools]
    
    def test_mcp_connection(self) -> Dict[str, Any]:
        """Test connection to MCP server."""
        try:
            response = requests.get(f"{self.mcp_url}/health", timeout=5)
            if response.status_code == 200:
                return {"status": "connected", "url": self.mcp_url}
            else:
                return {"status": "error", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ============================================================================
# Convenience Exports
# ============================================================================

# Export main components for easy importing
__all__ = [
    'MCPManager', 
    'mcp_tool',
    'mcp_service_call',
]

# Default manager instance
default_manager = MCPManager()
