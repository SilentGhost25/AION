"""
AION OpenVINO LLM
=================
Runs Qwen2.5-7B directly on Intel Arc iGPU using OpenVINO GenAI.
Bypasses Ollama entirely when selected.

Intel Arc iGPU on Core Ultra 5 125H:
- Supports INT4 quantized models
- Shared VRAM (from system RAM)
- High generation speed vs CPU via Ollama

Usage:
    from v0_1.openvino_llm import OpenVINOLLM
    llm = OpenVINOLLM()
    result = llm.generate("Your prompt here")
"""

import os
import time
import threading
import queue
from pathlib import Path
from typing import Optional


DEFAULT_MODEL_PATH = str(
    Path(__file__).parent.parent / "models" / "qwen2.5-7b-ov"
)

MODEL_PATH = os.environ.get("AION_OV_MODEL", DEFAULT_MODEL_PATH)


class OpenVINOLLM:
    """
    OpenVINO GenAI LLM runner for Intel Arc iGPU.
    Drop-in replacement for Ollama in the AION pipeline on local laptop.
    """

    def __init__(
        self,
        model_path:  str = MODEL_PATH,
        device:      str = "GPU",       # GPU = Intel Arc, CPU = fallback
        max_tokens:  int = 500,
        temperature: float = 0.1,
        timeout_sec: int = 180,
    ):
        self.model_path  = model_path
        self.device      = device
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self._pipe       = None
        self._lock       = threading.Lock()
        self._load_error = None

        print(f"[OV-LLM] Initializing on {device}")
        print(f"[OV-LLM] Model path: {model_path}")

    def _load(self):
        """Load the OpenVINO pipeline. Called once on first use."""
        if self._pipe is not None:
            return True
        if self._load_error:
            return False

        path = Path(self.model_path)
        if not path.exists():
            self._load_error = (
                f"Model not found at {self.model_path}. "
                f"Run the conversion step first."
            )
            print(f"[OV-LLM] ERROR: {self._load_error}")
            return False

        try:
            import openvino_genai as ov_genai

            print(f"[OV-LLM] Loading model on {self.device}...")
            t0 = time.time()

            self._pipe = ov_genai.LLMPipeline(
                self.model_path,
                self.device
            )

            elapsed = round(time.time() - t0, 1)
            print(f"[OV-LLM] Model loaded in {elapsed}s on {self.device}")
            return True

        except Exception as e:
            self._load_error = str(e)
            print(f"[OV-LLM] Load failed on {self.device}: {e}")

            if self.device != "CPU":
                print("[OV-LLM] Retrying on CPU...")
                self.device = "CPU"
                self._load_error = None
                return self._load()

            return False

    def generate(
        self,
        prompt:      str,
        max_tokens:  Optional[int]   = None,
        temperature: Optional[float] = None,
        system:      str = "",
    ) -> Optional[str]:
        """
        Generate text from a prompt.
        Returns the generated string or None on failure.
        """
        with self._lock:
            if not self._load():
                return None

        tokens  = max_tokens  or self.max_tokens
        temp    = temperature or self.temperature

        if system:
            full_prompt = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            full_prompt = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        result_queue: queue.Queue = queue.Queue()

        def _worker():
            try:
                import openvino_genai as ov_genai

                config = ov_genai.GenerationConfig()
                config.max_new_tokens = tokens
                config.temperature    = temp
                config.do_sample      = temp > 0

                result = self._pipe.generate(full_prompt, config)
                result_queue.put(str(result).strip())

            except Exception as e:
                print(f"[OV-LLM] Generation error: {e}")
                result_queue.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout_sec)

        if thread.is_alive():
            print(f"[OV-LLM] Generation timed out after {self.timeout_sec}s")
            return None

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None

    def call(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """
        Alias matching RobustLLMCaller interface.
        Drop-in replacement for llm.call()
        """
        return self.generate(prompt, max_tokens=max_tokens)

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None


_instance: Optional[OpenVINOLLM] = None


def get_ov_llm(device: str = "GPU") -> OpenVINOLLM:
    global _instance
    if _instance is None:
        _instance = OpenVINOLLM(device=device)
    return _instance
