#!/bin/bash
echo "Starting AION Embeddings V2..."
# Start the background daemon
python -m core.auto_trainer &
# Start the API server
uvicorn api.server:app --host 0.0.0.0 --port 8000 &
# Start the admin dashboard
streamlit run admin/dashboard.py --server.port 8501
