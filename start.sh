#!/usr/bin/env bash
# Start the Unhuman Clips backend server
cd "$(dirname "$0")" || { echo "Project directory not found"; exit 1; }

# Activate the Python virtual environment (deps are installed here)
if [ -f "backend/venv/bin/activate" ]; then
  source backend/venv/bin/activate
  # Add CUDA libraries from nvidia pip packages to library path
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/backend/venv/lib/python3.12/site-packages/nvidia/cublas/lib"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/backend/venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
else
  echo "Warning: venv not found, using system Python" >&2
fi

echo "=== Unhuman Clips Backend ==="
PORT="${PORT:-9000}"
HOST="${HOST:-127.0.0.1}"

echo "Starting server on http://${HOST}:${PORT}"
echo ""

# Run uvicorn server
exec uvicorn backend.main:app --host "${HOST}" --port "${PORT}" --reload
