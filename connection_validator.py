"""
AION Connection Validator
Runs at startup and diagnoses all possible connection issues.
Prevents upload errors before they happen.
"""

from __future__ import annotations

import os
import sys
import json
import re
import socket
import urllib.request
import urllib.error
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ConnectionValidator:
    """
    Validates entire AION stack at startup.
    Fixes what it can, reports what it cannot.
    """

    def __init__(
        self,
        api_port:    int = 8100,
        ollama_port: int = 11434,
        upload_dir:  str = "workspace/uploads",
    ):
        self.api_port    = api_port
        self.ollama_port = ollama_port
        self.upload_dir  = Path(upload_dir)
        self.errors      = []
        self.warnings    = []
        self.fixes       = []

    def validate(self) -> bool:
        """
        Run all checks. Returns True if safe to start.
        Prints full diagnostic report.
        """
        print("\n" + "═" * 55)
        print("  AION Startup Validator")
        print("═" * 55)

        checks = [
            self._check_python_version,
            self._check_required_packages,
            self._check_upload_directory,
            self._check_port_available,
            self._check_ollama_running,
            self._check_ollama_model,
            self._check_network_interfaces,
            self._check_env_consistency,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                self.errors.append(f"Check crashed: {check.__name__}: {e}")

        self._print_report()
        return len(self.errors) == 0

    def _check_python_version(self):
        major, minor = sys.version_info[:2]
        if major < 3 or (major == 3 and minor < 9):
            self.errors.append(
                f"Python {major}.{minor} too old. Need 3.9+"
            )
        else:
            self._ok(f"Python {major}.{minor}")

    def _check_required_packages(self):
        required = {
            "flask":       "flask",
            "flask_cors":  "flask-cors",
            "requests":    "requests",
        }
        optional = {
            "fitz":        "pymupdf",
            "docx":        "python-docx",
            "ollama":      "ollama",
            "lxml":        "lxml",
        }

        for imp, pkg in required.items():
            try:
                __import__(imp)
                self._ok(f"Package: {pkg}")
            except ImportError:
                self.errors.append(
                    f"Missing required package: {pkg}\n"
                    f"  Fix: pip install {pkg}"
                )

        for imp, pkg in optional.items():
            try:
                __import__(imp)
            except ImportError:
                self.warnings.append(
                    f"Optional package missing: {pkg} "
                    f"(pip install {pkg})"
                )

    def _check_upload_directory(self):
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        test_file = self.upload_dir / ".write_test"
        try:
            test_file.write_text("ok")
            test_file.unlink()
            self._ok(f"Upload dir writable: {self.upload_dir}")
        except Exception as e:
            self.errors.append(
                f"Upload directory not writable: {self.upload_dir}\n"
                f"  Error: {e}"
            )

    def _check_port_available(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", self.api_port))

        if result == 0:
            self.warnings.append(
                f"Port {self.api_port} already in use. "
                f"Another instance may be running."
            )
        else:
            self._ok(f"Port {self.api_port} available")

    def _check_ollama_running(self):
        try:
            req = urllib.request.Request(
                f"http://localhost:{self.ollama_port}/api/tags"
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    self._ok("Ollama is running")
                    return
        except urllib.error.URLError:
            pass
        except Exception:
            pass

        self.errors.append(
            f"Ollama not reachable on port {self.ollama_port}\n"
            f"  Fix: run 'ollama serve' in a terminal"
        )

    def _check_ollama_model(self):
        model = os.environ.get("AION_MODEL", "aion-qwen")
        base  = "qwen2.5:3b"

        try:
            req = urllib.request.Request(
                f"http://localhost:{self.ollama_port}/api/tags"
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                data   = json.loads(r.read())
                names  = [m["name"] for m in data.get("models", [])]

            if any(model in n for n in names):
                self._ok(f"Model ready: {model}")
                return

            if any(base in n for n in names):
                self.warnings.append(
                    f"Custom model '{model}' not found. "
                    f"Using base '{base}'. "
                    f"Run: ollama create aion-qwen -f AION.Modelfile"
                )
                os.environ["AION_MODEL"] = base
                self.fixes.append(f"Switched model to: {base}")
                return

            self.errors.append(
                f"No suitable model found in Ollama.\n"
                f"  Installed: {names}\n"
                f"  Fix: ollama pull qwen2.5:3b"
            )

        except Exception:
            pass

    def _check_network_interfaces(self):
        hostname = socket.gethostname()
        try:
            ips = socket.getaddrinfo(
                hostname, None,
                socket.AF_INET
            )
            local_ips = list({
                ip[4][0] for ip in ips
                if not ip[4][0].startswith("127.")
            })
        except Exception:
            local_ips = []

        local_ips_str = ", ".join(local_ips) if local_ips else "none"
        self._ok(
            f"Network interfaces: localhost, {local_ips_str}"
        )

        if local_ips:
            self.warnings.append(
                f"If accessing frontend from network IP "
                f"({local_ips[0]}:5173), ensure .env has:\n"
                f"  VITE_AION_API_URL=http://localhost:{self.api_port}\n"
                f"  AND open browser at: http://localhost:5173"
            )

    def _check_env_consistency(self):
        env_paths = [
            Path("frontend/artifacts/qp-maker/.env"),
            Path("frontend/.env"),
        ]

        for env_path in env_paths:
            if env_path.exists():
                content = env_path.read_text(encoding="utf-8")
                self._check_env_content(env_path, content)

    def _check_env_content(self, path: Path, content: str):
        lines    = content.strip().splitlines()
        env_vars = {}

        for line in lines:
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()

        api_url = env_vars.get("VITE_AION_API_URL", "")

        ip_pattern = re.compile(r"http://\d+\.\d+\.\d+\.\d+")
        if ip_pattern.search(api_url):
            self.warnings.append(
                f"Frontend .env at {path} uses IP address:\n"
                f"  {api_url}\n"
                f"  This causes CORS errors in Chrome!\n"
                f"  Auto-fixing to localhost..."
            )
            fixed = ip_pattern.sub(
                f"http://localhost:{self.api_port}",
                content
            )
            try:
                path.write_text(fixed, encoding="utf-8")
                self.fixes.append(
                    f"Fixed {path}: replaced IP with localhost"
                )
            except Exception as e:
                self.warnings.append(
                    f"Could not auto-fix {path}: {e}\n"
                    f"  Manually change to: "
                    f"VITE_AION_API_URL=http://localhost:{self.api_port}"
                )
        elif f"localhost:{self.api_port}" in api_url or "localhost" in api_url:
            self._ok(f"Frontend .env: {path.name} ✓")
        elif api_url:
            self.warnings.append(
                f"Unexpected API URL in {path.name}: {api_url}"
            )

    def _ok(self, msg: str):
        print(f"  ✅ {msg}")

    def _print_report(self):
        if self.fixes:
            print("\n  🔧 Auto-fixes applied:")
            for f in self.fixes:
                print(f"     • {f}")

        if self.warnings:
            print("\n  ⚠️  Warnings:")
            for w in self.warnings:
                for line in w.splitlines():
                    print(f"     {line}")

        if self.errors:
            print("\n  ❌ Errors (must fix before starting):")
            for e in self.errors:
                for line in e.splitlines():
                    print(f"     {line}")
            print()
        else:
            print("\n  ✅ All checks passed — AION is ready!\n")

        print("═" * 55 + "\n")
