#!/bin/bash

# Cogniforge AI Service - Start Script

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default values
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8086}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

echo "======================================"
echo "  Cogniforge AI Service"
echo "======================================"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Log Level: $LOG_LEVEL"
echo "======================================"

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Warning: No API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
fi

# Start the server
echo "Starting Cogniforge AI Service..."
uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level "${LOG_LEVEL,,}"
