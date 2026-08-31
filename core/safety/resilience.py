"""AION Master Resilience Layer"""
import os, sys, json, re, time, socket, logging, hashlib, warnings, threading
from typing import Any, Optional, Dict, List
from contextvars import ContextVar

logger = logging.getLogger("AION.RESILIENCE")
request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


import re as _re

def _sanitize_latex_escapes(raw: str) -> str:
    r"""
    Prevent JSON parser from interpreting LaTeX backslash sequences
    as JSON escape chars.  \b -> backspace, \f -> formfeed, etc.
    Targets: \bowtie, \beta, \begin, \bar, \bullet, \bf, \frac,
             \text, \times, \tau, \sigma, \pi, \rho, \cup, \cap,
             \left, \right, \mathbf, \mathrm, \mathcal, \setminus,
             \le, \ge, \ne, \in, \notin, \subset, \supset,
             \forall, \exists, \nexists, \sum, \prod, \int,
             \infty, \partial, \nabla, \cdot, \ldots, \cdots,
             \alpha, \gamma, \delta, \epsilon, \zeta, \eta,
             \theta, \iota, \kappa, \lambda, \mu, \nu, \xi,
             \omicron, \phi, \chi, \psi, \omega, \Gamma, \Delta,
             \Theta, \Lambda, \Xi, \Pi, \Sigma, \Phi, \Psi, \Omega
    """
    # Double any single backslash that precedes a letter but is NOT
    # already a valid JSON escape (\, \/, \", \b, \f, \n, \r, \t, \uXXXX)
    return _re.sub(
        r'(?<!\\)\\(?![\\/"bfnrtu]|u[0-9a-fA-F]{4})([a-zA-Z])',
        r'\\\\\1',
        raw,
    )

def set_request_context(key: str, value: Any) -> None:
    ctx = dict(request_context.get({}))
    ctx[key] = value
    request_context.set(ctx)

def get_request_context(key: str, default: Any = None) -> Any:
    return request_context.get({}).get(key, default)

def repair_json(raw: str) -> Optional[dict]:
    if not raw or not str(raw).strip(): return None
    text = str(raw).strip()
    try: return json.loads(_sanitize_latex_escapes(text))
    except: pass
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        try: return json.loads(code_block.group(1).strip())
        except: pass
    first_b, last_b = text.find("{"), text.rfind("}")
    if first_b != -1 and last_b > first_b:
        try: return json.loads(re.sub(r",\s*([\}\]])", r"\1", text[first_b:last_b+1]))
        except: pass
    return None

def latex_safe_dumps(obj: Any, **kwargs) -> str:
    return json.dumps(obj, ensure_ascii=False, **kwargs)

def ensure_port_free(port: int = 8100) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
        return True
    except OSError:
        sock.close()
        try:
            import subprocess
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=3)
            time.sleep(0.5)
            return True
        except: return False

def install_all_safety_layers() -> None:
    ensure_port_free(8100)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*fitz.*deprecated.*")
    print("[RESILIENCE] ✅ Safety layers active.")
