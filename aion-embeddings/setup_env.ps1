# Setup Environment Script for AION Embeddings

Write-Host "Creating conda environment 'aion' with Python 3.11..."
conda create -y -n aion python=3.11
conda activate aion

Write-Host "Installing dependencies..."
pip install sentence-transformers transformers datasets
pip install faiss-gpu torch torchvision
pip install fastapi uvicorn streamlit
pip install pdfplumber pymupdf marker-pdf
pip install peft accelerate bitsandbytes
pip install watchdog sqlalchemy pyyaml
pip install wandb

Write-Host "Environment setup complete!"
