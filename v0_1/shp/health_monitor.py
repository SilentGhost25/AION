"""
AION SHP Stage 0 — System Health Monitor
=========================================
Runs before touching any document.
Verifies every dependency is ready.
Attempts auto-repair for recoverable failures.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .error_knowledge import ErrorKnowledgeBase, Severity


@dataclass
class HealthStatus:
    healthy:      bool
    checks:       dict[str, bool]  = field(default_factory=dict)
    repairs_done: list[str]        = field(default_factory=list)
    warnings:     list[str]        = field(default_factory=list)
    fatal:        str              = ""

    def summary(self) -> str:
        total   = len(self.checks)
        passed  = sum(1 for v in self.checks.values() if v)
        status  = "HEALTHY" if self.healthy else "DEGRADED"
        return f"System {status}: {passed}/{total} checks passed"


class SystemHealthMonitor:
    """
    Stage 0: Verify all pipeline dependencies before execution.
    Auto-repairs what it can. Reports what it cannot fix.
    """

    OLLAMA_URL    = "http://127.0.0.1:11434"
    REQUIRED_DIRS = [
        "workspace/uploads",
        "workspace/cache",
        "logs",
        ".aiq",
        "generated_papers",
        "templates",
    ]

    def __init__(self, kb: ErrorKnowledgeBase, root: Path = None):
        self.kb   = kb
        self.root = root or Path(__file__).parent.parent.parent

    def check(self) -> HealthStatus:
        """Run all health checks. Attempt auto-repair where possible."""
        status = HealthStatus(healthy=True)

        for d in self.REQUIRED_DIRS:
            path = self.root / d
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                status.repairs_done.append(f"Created missing directory: {d}")
            status.checks[f"dir:{d}"] = path.exists()

        ollama_ok = self._check_ollama()
        status.checks["ollama:running"] = ollama_ok
        if not ollama_ok:
            rec = self.kb.record("SH-060", "S0_HEALTH",
                                 "Ollama not running", Severity.ERROR)
            repaired = self._start_ollama()
            if repaired:
                status.repairs_done.append("Started ollama serve")
                self.kb.resolve(rec, "ollama serve started")
                status.checks["ollama:running"] = True
            else:
                status.warnings.append(
                    "Ollama not running and could not be started. "
                    "Generation will fail."
                )

        if status.checks.get("ollama:running"):
            model_ok, model_name = self._check_model()
            status.checks[f"model:{model_name}"] = model_ok
            if not model_ok:
                rec = self.kb.record("SH-070", "S0_HEALTH",
                                     f"Model {model_name} not found",
                                     Severity.ERROR)
                fallback = self._find_fallback_model()
                if fallback:
                    status.repairs_done.append(f"Using fallback model: {fallback}")
                    os.environ["AION_MODEL"] = fallback
                    self.kb.resolve(rec, f"Switched to {fallback}")
                    status.checks[f"model:{model_name}"] = True
                else:
                    status.fatal = f"No usable model found in Ollama"
                    status.healthy = False

        try:
            import shutil
            free_gb = shutil.disk_usage(self.root).free / (1024**3)
            ok = free_gb >= 2.0
            status.checks["disk:free_2gb"] = ok
            if not ok:
                status.warnings.append(
                    f"Low disk space: {free_gb:.1f}GB free. "
                    f"PDF generation may fail."
                )
        except Exception:
            status.checks["disk:free_2gb"] = True

        for pkg in ["fitz", "requests", "flask"]:
            try:
                __import__(pkg)
                status.checks[f"dep:{pkg}"] = True
            except ImportError:
                status.checks[f"dep:{pkg}"] = False
                status.warnings.append(f"Missing package: {pkg}")

        try:
            from core.config.production_model import get_production_model
            model = get_production_model()
            status.checks["config:model_authority"] = bool(model)
        except Exception as e:
            status.checks["config:model_authority"] = False
            status.warnings.append(f"Config authority failed: {e}")

        if status.fatal:
            status.healthy = False
        elif any(
            not v for k, v in status.checks.items()
            if k.startswith("model:")
        ):
            status.healthy = False

        print(f"[SHP-S0] {status.summary()}")
        for repair in status.repairs_done:
            print(f"  ✓ Repaired: {repair}")
        for warning in status.warnings:
            print(f"  ⚠ Warning: {warning}")

        return status

    def _check_ollama(self) -> bool:
        try:
            import requests
            r = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def _start_ollama(self) -> bool:
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(8)
            return self._check_ollama()
        except Exception:
            return False

    def _check_model(self) -> tuple[bool, str]:
        from core.config.production_model import get_production_model
        model = get_production_model()
        try:
            import requests
            r = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                names = [m["name"] for m in r.json().get("models", [])]
                found = any(model in n or n.startswith(model) for n in names)
                return found, model
        except Exception:
            pass
        return False, model

    def _find_fallback_model(self) -> Optional[str]:
        try:
            import requests
            r = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=3)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                for preferred in ["qwen2.5:14b", "qwen2.5:7b", "qwen2.5:3b"]:
                    for m in models:
                        if preferred in m:
                            return m
                return models[0] if models else None
        except Exception:
            return None
