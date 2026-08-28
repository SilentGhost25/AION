# core/validation/math_validator.py

import re
import subprocess
import logging
from core.generation.output_schema import QuestionOutput, MathBlock
from core.validation.common import CheckResult, RetryAction

LOG = logging.getLogger(__name__)


class MathRenderFailure(Exception):
    """Raised when KaTeX math rendering fails."""
    pass


class KaTeXAvailabilityGate:
    _initialized = False
    _ready = False

    @classmethod
    def probe(cls) -> bool:
        """Executes a startup probe. Verifies Node, NPX and KaTeX compile capabilities."""
        if cls._initialized:
            return cls._ready
        cls._initialized = True
        try:
            # Probe using npx --no-install katex first
            res = subprocess.run(
                ["npx", "--no-install", "katex"],
                input="x = y",
                capture_output=True,
                text=True,
                shell=True
            )
            if res.returncode == 0 and "katex" in res.stdout.lower():
                cls._ready = True
                LOG.info("KaTeXAvailabilityGate verified: npx --no-install katex is ready.")
            else:
                LOG.warning(f"KaTeX --no-install probe failed (code {res.returncode}): {res.stderr or res.stdout}. Trying fallback standard npx...")
                res_fallback = subprocess.run(
                    ["npx", "katex"],
                    input="x = y",
                    capture_output=True,
                    text=True,
                    shell=True
                )
                if res_fallback.returncode == 0 and "katex" in res_fallback.stdout.lower():
                    cls._ready = True
                    LOG.info("KaTeXAvailabilityGate verified (fallback): npx katex is ready.")
                else:
                    LOG.error(f"KaTeX probe failed completely: exit {res_fallback.returncode}. {res_fallback.stderr}")
        except Exception as e:
            LOG.error(f"KaTeX probe exception: {e}")
        return cls._ready

    @classmethod
    def render(cls, latex: str, display_mode: bool = False) -> str:
        """Compiles LaTeX to HTML via subprocess."""
        if not cls.probe():
            raise MathRenderFailure("MATH_OK")
        
        cmd = ["npx", "--no-install", "katex"]
        if display_mode:
            cmd.append("--display-mode")
            
        try:
            res = subprocess.run(
                cmd,
                input=latex,
                capture_output=True,
                text=True,
                shell=True
            )
            if res.returncode != 0:
                # Fall back to standard npx in case --no-install failed to find local copy
                fallback_cmd = ["npx", "katex"]
                if display_mode:
                    fallback_cmd.append("--display-mode")
                res_fallback = subprocess.run(
                    fallback_cmd,
                    input=latex,
                    capture_output=True,
                    text=True,
                    shell=True
                )
                if res_fallback.returncode != 0:
                    raise MathRenderFailure(res_fallback.stderr or res_fallback.stdout)
                return res_fallback.stdout
            return res.stdout
        except Exception as e:
            raise MathRenderFailure(f"KaTeX subprocess failed: {str(e)}")


def validate_math_consistency(output: QuestionOutput) -> CheckResult:
    """
    H3 — Missing or orphan [MATH:id] references are hard failures.
    Called after schema validation (which also checks this).
    """
    referenced = set(re.findall(r"\[MATH:([^\]]+)\]", output.question_text))
    declared   = {b.block_id for b in output.math_blocks}

    orphan = referenced - declared
    if orphan:
        return CheckResult.fail(
            "MATH_PLACEHOLDER_UNRESOLVED",
            f"[MATH:...] references not in declared blocks: {orphan}. "
            f"Final question would contain broken placeholders.",
            action=RetryAction.REGENERATE
        )

    unused = declared - referenced
    if unused:
        return CheckResult.fail(
            "MATH_BLOCK_UNREFERENCED",
            f"Declared MathBlocks never appear in question text: {unused}. "
            f"Qwen declared math it didn't use.",
            action=RetryAction.REGENERATE
        )

    return CheckResult.pass_()


def validate_math_block_with_render(block: MathBlock) -> CheckResult:
    """Renderer-aware validation. KaTeX is mandatory — no silent skip."""

    if "\ufffd" in block.latex or "\x00" in block.latex:
        return CheckResult.critical(
            "MATH_CORRUPTED",
            f"Block {block.block_id} contains corruption chars"
        )

    if not block.latex.strip():
        return CheckResult.fail("MATH_EMPTY", f"Block {block.block_id} is empty")

    try:
        block.latex.encode("utf-8").decode("utf-8")
    except Exception as e:
        return CheckResult.critical("MATH_ENCODING_FAILURE", str(e))

    # Brace balance
    depth = 0
    for ch in block.latex:
        if ch == "{": depth += 1
        elif ch == "}": depth -= 1
        if depth < 0:
            return CheckResult.fail("MATH_UNBALANCED_BRACES",
                                    f"Block {block.block_id}: unmatched }}")
    if depth != 0:
        return CheckResult.fail("MATH_UNCLOSED_BRACES",
                                f"Block {block.block_id}: {depth} unclosed braces")

    # \\frac structure
    frac_count = block.latex.count(r"\frac")
    frac_ok    = len(re.findall(r"\\frac\s*\{[^}]*\}\s*\{[^}]*\}", block.latex))
    if frac_count > 0 and frac_ok < frac_count:
        return CheckResult.fail("MATH_INCOMPLETE_FRAC",
                                f"Block {block.block_id}: \\frac missing argument")

    # KaTeX render (mandatory — not optional)
    try:
        rendered = KaTeXAvailabilityGate.render(
            block.latex, display_mode=block.display_mode
        )
        if "katex-error" in rendered.lower():
            return CheckResult.fail("MATH_RENDER_ERROR",
                                    f"Block {block.block_id}: KaTeX error class")
        if len(rendered.strip()) < 20:
            return CheckResult.fail("MATH_RENDER_EMPTY",
                                    f"Block {block.block_id}: render too short")
    except MathRenderFailure as e:
        return CheckResult.fail("MATH_RENDER_FAILURE",
                                f"Block {block.block_id}: {e}")

    return CheckResult.pass_()
