#!/bin/bash
echo "Initializing AION Embeddings V2..."
pip install -r requirements.txt
python -c "from storage.database import init_db; init_db()"
echo "Done. Run scripts/start.sh to launch."
