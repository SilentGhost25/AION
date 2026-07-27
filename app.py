"""
AION Streamlit Web App Entrypoint.
Forwards to aion-embeddings/app.py seamlessly.
"""
import sys
from pathlib import Path

# Add aion-embeddings to sys.path
embeddings_dir = Path(__file__).parent / "aion-embeddings"
sys.path.insert(0, str(embeddings_dir))

# Execute aion-embeddings/app.py
app_path = embeddings_dir / "app.py"
exec(app_path.read_text(encoding="utf-8"), {"__file__": str(app_path), "__name__": "__main__"})
