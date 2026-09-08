#!/bin/bash
set -e

echo "=== XAI-DocGuard Setup & Run ==="

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting backend server..."
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
