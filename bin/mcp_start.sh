#!/bin/bash

# MCP Server Startup Script
# Starts the MCP server for document processing

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MCP Document Processing Server...${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if requirements are installed
echo -e "${YELLOW}Checking Python requirements...${NC}"
cd "$PROJECT_ROOT"

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to install requirements.${NC}"
        exit 1
    fi
fi

# Check if the MCP server file exists
MCP_SERVER_PATH="$PROJECT_ROOT/src/main/mcp/server.py"
if [ ! -f "$MCP_SERVER_PATH" ]; then
    echo -e "${RED}Error: MCP server not found at $MCP_SERVER_PATH${NC}"
    exit 1
fi

# Check if port 3000 is available
if command -v lsof &> /dev/null; then
    if lsof -i :3000 &> /dev/null; then
        echo -e "${YELLOW}Port 3000 is currently in use. Details:${NC}"
        lsof -i :3000
        echo ""
        
        # Get the PID
        PID=$(lsof -t -i:3000)
        echo -e "${YELLOW}Process ID using port 3000: $PID${NC}"
        
        # Ask if user wants to kill it
        read -p "Do you want to stop this process and continue? (Y/n): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo -e "${RED}Exiting. Please stop the process manually or use a different port.${NC}"
            exit 1
        else
            echo -e "${YELLOW}Stopping process $PID...${NC}"
            kill $PID 2>/dev/null || true
            sleep 2
            
            # Verify it's stopped
            if lsof -i :3000 &> /dev/null; then
                echo -e "${YELLOW}Process still running. Forcing termination...${NC}"
                kill -9 $PID 2>/dev/null || true
                sleep 1
            fi
            
            # Final check
            if lsof -i :3000 &> /dev/null; then
                echo -e "${RED}Failed to free port 3000. Please stop the process manually:${NC}"
                echo -e "${YELLOW}kill \$(lsof -t -i:3000)${NC}"
                exit 1
            else
                echo -e "${GREEN}✓ Port 3000 is now available${NC}"
            fi
        fi
    fi
fi

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Start the MCP server
echo -e "${GREEN}Starting MCP server on localhost:3000...${NC}"
echo -e "${YELLOW}Using MCP server: src/main/mcp/server.py${NC}"
echo -e "${YELLOW}Server logs will be saved to logs/mcp_server.log${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start server with logging
python3 "$MCP_SERVER_PATH" 2>&1 | tee "$PROJECT_ROOT/logs/mcp_server.log"

# Cleanup on exit
trap 'echo -e "\n${YELLOW}Stopping MCP server...${NC}"; exit 0' INT TERM
