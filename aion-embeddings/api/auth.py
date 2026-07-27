# api/auth.py
import hashlib
import yaml
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

def get_admin(api_key_header: str = Security(api_key_header)):
    config = load_config()
    # Simple hash check for admin
    if not api_key_header:
        raise HTTPException(status_code=401, detail="Missing API Key")
    
    hashed = hashlib.sha256(api_key_header.encode()).hexdigest()
    if hashed != config["admin"]["password_hash"]:
        raise HTTPException(status_code=403, detail="Invalid Admin Key")
    return True
