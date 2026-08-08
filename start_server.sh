#!/bin/bash
# AION Server Start — L40 48GB
# Usage: bash start_server.sh
# Copy to server alongside .env.server

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== AION Server Mode ==="

# Load server env for this session only
if [ -f ".env.server" ]; then
    set -a
    source ".env.server"
    set +a
    echo "Loaded: .env.server"
fi

# Activate venv if present
[ -f ".venv/bin/activate" ] && source ".venv/bin/activate"

# Show resolved model
python -c "
from core.config.production_model import get_production_model, get_resolution_info
m = get_production_model()
i = get_resolution_info()
print(f'  Model  : {m}')
print(f'  Source : {i[\"source\"]}')
print(f'  Device : {i[\"device\"]}')
"

echo ""
echo "Starting AION backend on port 8100..."
python aion_api.py
