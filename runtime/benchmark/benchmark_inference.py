# runtime/benchmark/benchmark_inference.py

import time, json, logging, subprocess
from runtime.profiles import LAPTOP_FAST_PROFILE

LOG = logging.getLogger("aion.benchmark")

TEST_PROMPT = (
    "Generate a 4-mark L3 question about sorting algorithms. "
    "Return JSON: {\"instruction\": \"...\", \"question_text\": \"...\"}"
)

CANDIDATE_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b",
]

QUALITY_PASS_THRESHOLD = 0.70   # JSON valid + bloom verb present


def benchmark_model(model_name: str) -> dict:
    import requests
    start = time.monotonic()
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model" : model_name,
                "prompt": TEST_PROMPT,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 200}
            },
            timeout=60,
        )
        elapsed = time.monotonic() - start
        if not resp.ok:
            return {"model": model_name, "status": "HTTP_ERROR", "elapsed": elapsed}

        text = resp.json().get("response", "")

        try:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            parsed = json.loads(match.group()) if match else {}
            json_valid = bool(parsed.get("instruction") or parsed.get("question_text"))
        except Exception:
            json_valid = False

        bloom_ok = any(
            text.lower().startswith(v)
            for v in {"calculate","apply","demonstrate","solve","derive","explain","analyze"}
        )

        quality = (0.5 if json_valid else 0.0) + (0.5 if bloom_ok else 0.0)

        return {
            "model"     : model_name,
            "status"    : "OK",
            "elapsed_s" : round(elapsed, 2),
            "json_valid": json_valid,
            "bloom_ok"  : bloom_ok,
            "quality"   : quality,
        }
    except Exception as e:
        return {"model": model_name, "status": "ERROR", "error": str(e)}


def select_best_model() -> str:
    LOG.info("[BENCHMARK] Testing laptop models...")
    results = []
    for model in CANDIDATE_MODELS:
        r = benchmark_model(model)
        LOG.info(f"  {model}: {r}")
        if r.get("status") == "OK" and r.get("quality", 0) >= QUALITY_PASS_THRESHOLD:
            results.append(r)

    if not results:
        LOG.warning("No candidate model passed quality threshold, returning default qwen2.5:3b")
        return "qwen2.5:3b"

    # Select fastest among quality-passing models
    best = min(results, key=lambda r: r["elapsed_s"])
    LOG.info(f"[BENCHMARK] Selected: {best['model']} ({best['elapsed_s']}s)")
    return best["model"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = select_best_model()
    print(f"SELECTED_MODEL={model}")
