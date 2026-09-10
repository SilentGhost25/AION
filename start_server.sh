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

# Optional port argument overrides env
DESIRED_PORT="${1:-${AION_PORT:-8100}}"

# Dynamically resolve port if preferred port is occupied
RESOLVED_PORT=$(python -c "from core.config.server_config import find_available_port; print(find_available_port($DESIRED_PORT))" 2>/dev/null || echo "$DESIRED_PORT")

if [ "$RESOLVED_PORT" != "$DESIRED_PORT" ]; then
    echo "[AIQ] Port $DESIRED_PORT is occupied. Dynamically switched to port $RESOLVED_PORT"
fi

export AION_PORT="$RESOLVED_PORT"
export BACKEND_PORT="$RESOLVED_PORT"

# Record active port in .aiq/ports.env
mkdir -p "$ROOT/.aiq"
cat <<EOF > "$ROOT/.aiq/ports.env"
BACKEND_PORT=$RESOLVED_PORT
AION_PORT=$RESOLVED_PORT
EOF

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
echo "Starting AION backend on port $AION_PORT..."
python aion_api.py
