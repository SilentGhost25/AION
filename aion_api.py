#!/usr/bin/env python3
"""
AION API Server
Flask-based HTTP server bridging the React frontend to the AION pipeline.
Run:    python aion_api.py
URL:    http://localhost:8100
"""

import os
import sys
import json
import uuid
import threading
import traceback
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AION_MODEL", "qwen2.5:3b")

# ── Flask imports ─────────────────────────────────────────────
try:
    from flask import Flask, request, jsonify, Response, stream_with_context
    from flask_cors import CORS
except ImportError:
    print("[ERROR] Flask not installed.")
    print("Run: pip install flask flask-cors")
    sys.exit(1)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")

# ── Storage ───────────────────────────────────────────────────
UPLOAD_DIR = ROOT / "workspace" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

file_registry: dict = {}
job_store:     dict = {}

ALLOWED = {".pdf", ".txt", ".docx", ".pptx", ".md"}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def warmup_model():
    """Pre-load model into memory — runs in background thread."""
    import time
    time.sleep(2)  # Let Flask start first

    model = os.environ.get("AION_MODEL", "qwen2.5:3b")
    print(f"[AION] Warming up '{model}'...", flush=True)

    try:
        payload = json.dumps({
            "model":      model,
            "messages":   [{"role": "user", "content": "hi"}],
            "keep_alive": -1,
            "stream":     False,
            "options":    {"num_predict": 1, "num_ctx": 512}
        }).encode("utf-8")

        import urllib.request
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data    = payload,
            headers = {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            status = "OK [OK]" if r.status == 200 else f"status={r.status}"
            print(f"[AION] Warmup {status}", flush=True)

    except Exception as e:
        print(f"[AION] Warmup skipped ({e})", flush=True)



# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    ollama_ok = False
    model = os.environ.get("AION_MODEL", "qwen2.5:3b")
    try:
        import urllib.request
        with urllib.request.urlopen(
            "http://localhost:11434/api/tags", timeout=3
        ) as r:
            data   = json.loads(r.read())
            models = [m["name"] for m in data.get("models", [])]
            ollama_ok = any(model in m for m in models)
    except Exception:
        pass

    return jsonify({
        "status":  "ok" if ollama_ok else "degraded",
        "ollama":  ollama_ok,
        "model":   model,
        "version": "0.1.0",
        "uploads": len(file_registry),
    })


# ─────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file in request. Use field name 'file'."}), 400

    f = request.files["file"]

    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({
            "error":   f"Unsupported type: {ext}",
            "allowed": sorted(ALLOWED),
        }), 400

    file_id = str(uuid.uuid4())[:12]
    dest    = UPLOAD_DIR / f"{file_id}{ext}"
    f.save(str(dest))

    size = dest.stat().st_size

    record = {
        "id":          file_id,
        "filename":    f.filename,
        "storedPath":  str(dest),
        "subject":     request.form.get("subject",  "unknown"),
        "category":    request.form.get("category", "notes"),
        "uploadedAt":  datetime.now().isoformat(),
        "sizeBytes":   size,
    }
    file_registry[file_id] = record

    print(f"[UPLOAD] OK: {f.filename} -> {dest} ({size:,} bytes)")
    return jsonify(record), 201


@app.route("/api/files", methods=["GET"])
def list_files():
    return jsonify(list(file_registry.values()))


@app.route("/api/files/<file_id>", methods=["DELETE"])
def delete_file(file_id):
    record = file_registry.pop(file_id, None)
    if not record:
        return jsonify({"error": "Not found"}), 404
    try:
        Path(record["storedPath"]).unlink(missing_ok=True)
    except Exception:
        pass
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
# SSE helper
# ─────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────────────────────
# Generate — SSE stream
# ─────────────────────────────────────────────────────────────

@app.route("/api/generate/stream", methods=["POST"])
def generate_stream():
    body = request.get_json(silent=True) or {}

    file_id   = body.get("fileId")
    file_path = body.get("filePath", "")

    if file_id:
        record = file_registry.get(file_id)
        if not record:
            def err_stream():
                yield _sse("error", {
                    "message": f"File ID '{file_id}' not found. Upload first."
                })
            return Response(
                stream_with_context(err_stream()),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control":     "no-cache",
                    "Connection":        "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        file_path = record["storedPath"]

    if not file_path or not Path(file_path).exists():
        def err_stream():
            yield _sse("error", {"message": f"File not found: '{file_path}'"})
        return Response(
            stream_with_context(err_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control":     "no-cache",
                "Connection":        "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    subject      = body.get("subject",      "Unknown")
    exam_type    = body.get("examType",     "see")
    mode         = body.get("mode",         "turbo")
    max_concepts = int(body.get("maxConcepts", 10))

    def stream():
        try:
            # ── 1. Notify frontend: started ───────────────────
            yield _sse("status", {
                "status":  "started",
                "message": f"Processing: {Path(file_path).name}",
            })

            # ── 2. Notify: pipeline running ───────────────────
            yield _sse("log", {
                "message": f"Running pipeline in {mode} mode..."
            })

            # ── 3. Run pipeline (blocking) ────────────────────
            print(f"[STREAM] Starting pipeline for: {file_path}", flush=True)

            try:
                from v0_1.main import run_pipeline
                paper, rejected = run_pipeline(
                    file_path,
                    max_concepts = max_concepts,
                    mode         = mode,
                    exam_type    = exam_type,
                )
                print(f"[STREAM] Pipeline done. Modules: {len(paper) if isinstance(paper, list) else 0}", flush=True)

            except ImportError as e:
                print(f"[STREAM] Import error: {e}", flush=True)
                yield _sse("error", {
                    "message": f"Pipeline import failed: {str(e)}",
                    "trace":   traceback.format_exc(),
                })
                return

            except Exception as e:
                print(f"[STREAM] Pipeline error: {e}", flush=True)
                yield _sse("error", {
                    "message": f"Pipeline failed: {str(e)}",
                    "trace":   traceback.format_exc(),
                })
                return

            # ── 4. Validate pipeline output ───────────────────
            if not paper:
                yield _sse("error", {
                    "message": "Pipeline returned empty result. Check your file content."
                })
                return

            # ── 5. Format result ──────────────────────────────
            try:
                result = _format_paper(paper, subject, exam_type, mode)
                print(f"[STREAM] Formatted paper: {len(result.get('modules', []))} modules", flush=True)
            except Exception as e:
                print(f"[STREAM] Format error: {e}", flush=True)
                yield _sse("error", {
                    "message": f"Result formatting failed: {str(e)}",
                    "trace":   traceback.format_exc(),
                })
                return

            # ── 6. Send result to frontend ────────────────────
            yield _sse("result", result)
            print(f"[STREAM] Result sent to frontend", flush=True)

            # ── 7. Done ───────────────────────────────────────
            yield _sse("done", {"status": "done"})
            print(f"[STREAM] Done event sent", flush=True)

        except GeneratorExit:
            print("[STREAM] Client disconnected", flush=True)

        except Exception as e:
            print(f"[STREAM] Unexpected error: {e}", flush=True)
            try:
                yield _sse("error", {
                    "message": str(e),
                    "trace":   traceback.format_exc(),
                })
            except Exception:
                pass

    return Response(
        stream_with_context(stream()),
        mimetype = "text/event-stream",
        headers  = {
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────
# Generate — async job
# ─────────────────────────────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def generate_async():
    body = request.get_json(silent=True) or {}

    file_id   = body.get("fileId")
    file_path = body.get("filePath", "")

    if file_id:
        record = file_registry.get(file_id)
        if not record:
            return jsonify({"error": "File not found"}), 404
        file_path = record["storedPath"]

    if not file_path or not Path(file_path).exists():
        return jsonify({"error": f"File not found: '{file_path}'"}), 404

    job_id = str(uuid.uuid4())[:8]
    job_store[job_id] = {
        "status":   "queued",
        "progress": 0.0,
        "result":   None,
        "error":    None,
    }

    def _run():
        job = job_store[job_id]
        job["status"] = "running"
        try:
            from v0_1.main import run_pipeline
            paper, _ = run_pipeline(
                file_path,
                max_concepts = int(body.get("maxConcepts", 10)),
                mode         = body.get("mode",     "turbo"),
                exam_type    = body.get("examType", "see"),
            )
            job["result"]   = _format_paper(
                paper,
                body.get("subject",  ""),
                body.get("examType", "see"),
                body.get("mode",     "turbo"),
            )
            job["status"]   = "done"
            job["progress"] = 1.0
        except Exception as e:
            job["status"] = "failed"
            job["error"]  = str(e)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"jobId": job_id, "status": "queued"}), 202


@app.route("/api/generate/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = job_store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "jobId":    job_id,
        "status":   job["status"],
        "progress": job["progress"],
        "result":   job.get("result"),
        "error":    job.get("error"),
    })


# ─────────────────────────────────────────────────────────────
# Format helper
# ─────────────────────────────────────────────────────────────

def _format_paper(paper, subject, exam_type, mode):
    modules     = []
    total_marks = 0

    if isinstance(paper, list):
        for mod in paper:
            questions = []
            for mq in mod.get("questions", []):
                subs = []
                for sq in mq.get("sub_questions", []):
                    subs.append({
                        "letter": sq.get("letter"),
                        "text":   sq.get("text", ""),
                        "marks":  sq.get("marks", 5),
                    })
                    total_marks += sq.get("marks", 0)

                questions.append({
                    "mqIndex":      mq.get("mq_index",    1),
                    "totalMarks":   mq.get("total_marks", 10),
                    "bloomLevel":   mq.get("bloom_level", 2),
                    "bloomName":    mq.get("bloom_name",  "Understand"),
                    "subQuestions": subs,
                })

            modules.append({
                "moduleIndex": mod.get("module_index", 1),
                "moduleTitle": mod.get("module_title", "Module"),
                "questions":   questions,
            })

    return {
        "id":          str(uuid.uuid4())[:8],
        "subject":     subject,
        "examType":    exam_type,
        "mode":        mode,
        "modules":     modules,
        "generatedAt": datetime.now().isoformat(),
        "totalMarks":  total_marks,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("AION_PORT", 8100))

    print("==================================================")
    print("         AION Flask API Server  v0.1              ")
    print("==================================================")
    print(f"  URL   : http://localhost:{port}")
    print(f"  Model : {os.environ.get('AION_MODEL', 'qwen2.5:3b')}")
    print("==================================================")
    print("  GET  /api/health")
    print("  POST /api/upload")
    print("  GET  /api/files")
    print("  POST /api/generate/stream    (SSE)")
    print("  POST /api/generate           (async)")
    print("  GET  /api/generate/status/<id>")
    print("==================================================")
    print()

    # Warmup runs in background — Flask starts immediately
    threading.Thread(target=warmup_model, daemon=True).start()

    app.run(
        host     = "0.0.0.0",
        port     = port,
        debug    = False,
        threaded = True,
    )
