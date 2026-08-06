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
import time
import traceback
from pathlib import Path
from datetime import datetime
import requests

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AION_MODEL", "aion-exam")

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

# ✅ PERMANENT FIX 1: Allow ALL origins, ALL methods, ALL headers
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    }
})

# ✅ PERMANENT FIX 2: Force CORS headers on EVERY response
@app.after_request
def force_cors(response):
    response.headers["Access-Control-Allow-Origin"]          = "*"
    response.headers["Access-Control-Allow-Headers"]         = "*"
    response.headers["Access-Control-Allow-Methods"]         = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    response.headers["Access-Control-Max-Age"]               = "86400"
    response.headers["Vary"]                                 = "Origin"
    return response

# ✅ PERMANENT FIX 3: Handle ALL OPTIONS preflight requests globally
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        response.headers["Access-Control-Allow-Origin"]          = "*"
        response.headers["Access-Control-Allow-Headers"]         = "*"
        response.headers["Access-Control-Allow-Methods"]         = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Access-Control-Max-Age"]               = "86400"
        return response

# ── API v1 Blueprint ──────────────────────────────────────────
try:
    from api.v1 import api_v1_bp
    app.register_blueprint(api_v1_bp)
    print("[API] Successfully registered /api/v1 adapter layer.")
except Exception as e:
    print(f"[API ERROR] Failed to register /api/v1 blueprint: {e}")


@app.route("/api/v1", methods=["GET"])
@app.route("/api/v1/", methods=["GET"])
def api_v1_root():
    return jsonify({
        "name": "AION API Gateway",
        "version": "v1.0",
        "status": "online",
        "documentation": "AION Intelligence & Backend REST API v1",
        "workspaces": {
            "health": "/api/v1/health",
            "diagnostics": "/api/v1/diagnostics",
            "dashboard": "/api/v1/dashboard/summary",
            "uploads": "/api/v1/uploads",
            "training": "/api/v1/training/analyze",
            "question_forge": "/api/v1/questions/generate",
            "paper_forge": "/api/v1/papers/generate",
            "critic": "/api/v1/critic/evaluate",
            "datasets": "/api/v1/datasets",
            "models": "/api/v1/models",
            "knowledge": "/api/v1/knowledge/subjects",
        },
    })

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

    model = os.environ.get("AION_MODEL", "aion-exam")
    print(f"[AION] Warming up '{model}'...", flush=True)

    try:
        payload = json.dumps({
            "model":      model,
            "messages":   [{"role": "user", "content": "hi"}],
            "keep_alive": -1,
            "stream":     False,
            "options":    {
                "num_predict": 1,
                "num_ctx":     2048  # Force context on load
            }
        }).encode("utf-8")

        import urllib.request
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data    = payload,
            headers = {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            status = "OK" if r.status == 200 else f"status={r.status}"
            print(f"[AION] Warmup {status}", flush=True)

    except Exception as e:
        print(f"[AION] Warmup skipped ({e})", flush=True)



# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────

@app.route("/api/tags", methods=["GET"])
def get_tags():
    try:
        r = requests.get(f"{OLLAMA_URL.rstrip('/')}/api/tags", timeout=3)
        return jsonify(r.json() if r.status_code == 200 else {"models": []})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


@app.route("/api/health", methods=["GET"])
def health():
    """
    Proper health check that tests Ollama,
    not just Flask.
    """
    import requests as req

    # Test Ollama
    ollama_ok = False
    ollama_detail = "not checked"
    try:
        r = req.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )
        ollama_ok = r.status_code == 200
        models = [
            m["name"]
            for m in r.json().get("models", [])
        ]
        ollama_detail = f"{len(models)} models loaded"
    except Exception as e:
        ollama_detail = str(e)

    overall = "ok" if ollama_ok else "degraded"

    return jsonify({
        "status":         overall,
        "flask":          "ok",
        "ollama":         "ok" if ollama_ok else "down",
        "ollama_detail":  ollama_detail,
        "model":          app.config.get(
                              "MODEL", "qwen2.5:7b"
                          ),
        "timestamp":      time.time(),
    }), 200 if overall == "ok" else 503


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

    subject        = body.get("subject",        "Unknown")
    exam_type      = body.get("examType",       "see")
    mode           = body.get("mode",           "turbo")
    difficulty     = body.get("difficulty",     "mixed")
    max_concepts   = int(body.get("maxConcepts", 10))
    include_visual = bool(body.get("includeVisual", True))

    print(f"\n{'='*50}")
    print(f"[STREAM] START: {file_id or file_path}")
    print(f"[STREAM] Time: {datetime.now()}")

    def stream():
        import time
        import threading
        start_time = time.time()

        try:
            yield _sse("status", {
                "status":  "started",
                "message": f"Processing: {Path(file_path).name}",
            })
            yield _sse("log", {
                "message": f"Running pipeline in {mode} mode ({difficulty.upper()} difficulty)..."
            })

            pipeline_done = threading.Event()
            result_holder = {"paper": None, "qa_report": None, "error": None, "trace": None}

            def run_worker():
                try:
                    from v0_1.main import run_pipeline
                    paper, qa_report = run_pipeline(
                        file_path,
                        max_concepts   = max_concepts,
                        mode           = mode,
                        exam_type      = exam_type,
                        difficulty     = difficulty,
                        include_visual = include_visual,
                    )
                    result_holder["paper"] = paper
                    result_holder["qa_report"] = qa_report
                except Exception as e:
                    import traceback as tb
                    result_holder["error"] = str(e)
                    result_holder["trace"] = tb.format_exc()
                finally:
                    pipeline_done.set()

            pipeline_thread = threading.Thread(target=run_worker, daemon=True)
            pipeline_thread.start()

            last_keepalive = time.time()
            KEEPALIVE_INTERVAL = 4

            while not pipeline_done.is_set():
                pipeline_done.wait(timeout=0.5)
                now = time.time()
                if now - last_keepalive >= KEEPALIVE_INTERVAL:
                    yield _sse("log", {
                        "message": f"Processing... ({int(now - start_time)}s elapsed)",
                        "type": "keepalive",
                    })
                    last_keepalive = now

            elapsed = time.time() - start_time

            if result_holder["error"]:
                yield _sse("error", {
                    "message": f"Pipeline failed: {result_holder['error']}",
                    "trace":   result_holder["trace"],
                })
                return

            paper = result_holder["paper"]
            qa_report = result_holder["qa_report"]

            if not paper:
                yield _sse("error", {
                    "message": "Pipeline returned empty result. Check your file content."
                })
                return

            try:
                result = _format_paper(paper, subject, exam_type, mode, qa_report=qa_report)
                print(f"[STREAM] Formatted paper in {elapsed:.1f}s: {len(result.get('modules', []))} modules", flush=True)
            except Exception as e:
                print(f"[STREAM] Format error: {e}", flush=True)
                yield _sse("error", {
                    "message": f"Result formatting failed: {str(e)}",
                    "trace":   traceback.format_exc(),
                })
                return

            yield _sse("result", result)
            yield _sse("done", {"status": "done", "elapsed": elapsed})
            print(f"[STREAM] Complete event sent in {elapsed:.1f}s", flush=True)

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
            paper, qa_report = run_pipeline(
                file_path,
                max_concepts   = int(body.get("maxConcepts", 10)),
                mode           = body.get("mode",           "turbo"),
                exam_type      = body.get("examType",       "see"),
                difficulty     = body.get("difficulty",     "mixed"),
                include_visual = bool(body.get("includeVisual", True)),
            )
            job["result"]   = _format_paper(
                paper,
                body.get("subject",  ""),
                body.get("examType", "see"),
                body.get("mode",     "turbo"),
                qa_report = qa_report,
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


def _format_paper(paper, subject, exam_type, mode, qa_report=None):
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
                        "image":  sq.get("image"),
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
        "qaReport":    qa_report or {},
    }


@app.route("/api/preview", methods=["POST"])
def preview_paper():
    from v0_1.paper_formatter import get_preview_html
    data = request.get_json()
    html = get_preview_html(data)
    return Response(html, mimetype="text/html")


@app.route("/api/download/pdf", methods=["POST"])
def download_pdf():
    from v0_1.paper_formatter import export_pdf
    from flask import send_file
    data = request.get_json()
    try:
        out_path = export_pdf(data)
        return send_file(str(out_path), mimetype="application/pdf", as_attachment=True)
    except Exception as e:
        print(f"[ERROR] PDF Generation failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/emergency", methods=["POST"])
def generate_emergency():
    """
    Emergency generation endpoint.
    Ultra-fast local generation bypassing heavy pipelines.
    """
    data = request.get_json() or {}
    file_path = data.get("file_path")
    n_questions = int(data.get("n_questions", 5))

    if not file_path or not Path(file_path).exists():
        return jsonify({"error": "File not found"}), 404

    try:
        from v0_1.minimal_pipeline import emergency_pipeline
        result = emergency_pipeline(
            pdf_path=file_path,
            n_questions=n_questions
        )

        return jsonify({
            "status":    "success",
            "mode":      "emergency",
            "data":      result,
            "timestamp": time.time(),
        }), 200

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc(),
        }), 500


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("AION_PORT", 8100))

    print("==================================================")
    print("         AION Flask API Server  v0.1              ")
    print("==================================================")
    print(f"  URL   : http://localhost:{port}")
    print(f"  Model : {os.environ.get('AION_MODEL', 'qwen2.5:7b')}")
    print("==================================================")
    print("  GET  /api/health")
    print("  GET  /api/tags")
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

