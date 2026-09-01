"""
AION Ollama Diagnostic Script
==============================
Run before running emergency_cli.py or aion_api.py to verify queue state & model availability.
"""

import sys
import requests
import json


def diagnose() -> bool:
    print("="*60)
    print("Ollama Diagnostic Check")
    print("="*60)

    base = "http://localhost:11434"

    # Test 1: Connectivity
    print("\n[1] Connectivity test...")
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        if r.status_code == 200:
            print("    [OK] Ollama responding")
        else:
            print(f"    [X] HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"    [X] Cannot connect: {e}")
        print("    -> Run: ollama serve")
        return False

    # Test 2: Check queue state
    print("\n[2] Queue state test...")
    try:
        r = requests.post(
            f"{base}/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": "OK",
                "stream": False,
                "options": {"num_predict": 200},
                "keep_alive": 0,
            },
            timeout=10
        )

        if r.status_code == 503:
            print("    [X] Queue full (HTTP 503)")
            print("    -> Ollama state corrupted")
            print("    -> ACTION: Delete %LOCALAPPDATA%\\Ollama\\*.db")
            print("    -> Restart: ollama serve")
            return False

        if r.status_code == 200:
            print("    [OK] Queue accepting requests")
        else:
            print(f"    [!] Unexpected HTTP {r.status_code}")

    except requests.Timeout:
        print("    [X] Request hung (timeout)")
        print("    -> Model stuck processing")
        print("    -> ACTION: taskkill /F /IM ollama.exe")
        return False

    # Test 3: Model availability
    print("\n[3] Model availability test...")
    try:
        r = requests.get(f"{base}/api/tags", timeout=3)
        models = r.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        if any("qwen2.5:7b" in name for name in model_names):
            print("    [OK] qwen2.5:7b available")
        else:
            print(f"    [X] qwen2.5:7b not found")
            print(f"    Available: {model_names}")
            print("    -> Run: ollama pull qwen2.5:7b")
            return False

    except Exception as e:
        print(f"    [X] Error checking models: {e}")
        return False

    # Test 4: Generation test
    print("\n[4] Generation test...")
    try:
        r = requests.post(
            f"{base}/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": "Say: Test OK",
                "stream": False,
                "options": {"num_predict": 200},
                "keep_alive": 0,
            },
            timeout=15
        )

        if r.status_code != 200:
            print(f"    [X] HTTP {r.status_code}")
            return False

        response_text = r.json().get("response", "").strip()

        if len(response_text) > 0:
            print(f"    [OK] Generated: '{response_text[:50]}'")
        else:
            print("    [X] Empty response")
            return False

    except requests.Timeout:
        print("    [X] Generation timeout")
        return False

    print("\n" + "="*60)
    print("[OK] Ollama is healthy and ready")
    print("="*60)
    return True


if __name__ == "__main__":
    healthy = diagnose()

    if not healthy:
        print("\n[!] Ollama NOT ready")
        print("Fix the issues reported above before running emergency_cli.py")
        sys.exit(1)
    else:
        print("\nReady to run:")
        print("  python emergency_cli.py <your_pdf> 5")
        sys.exit(0)
