from setuptools import setup, find_packages

setup(
    name="aion-trainer",
    version="1.0.0",
    description="AION Academic Foundation Model Trainer",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "datasets>=2.12",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "numpy>=1.24",
        "pymupdf>=1.23",
        "python-docx>=1.1",
        "sentence-transformers>=2.5",
        "accelerate>=0.20",
        "wandb>=0.15",
    ],
    entry_points={
        "console_scripts": [
            "aion-trainer=cli:main",
        ],
    },
)
