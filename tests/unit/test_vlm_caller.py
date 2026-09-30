"""
Unit tests for the VLM caller.

Uses unittest.mock.patch to control requests.Session.post.
Image encoding is tested with a real temp file.
"""

import base64
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from core.api.vlm_caller import VLMCaller, APIUnavailable


# -----------------------------------------------------------------------------
# Test image helpers
# -----------------------------------------------------------------------------


def _make_png(path: Path, size=(100, 80), color=(255, 0, 0)) -> None:
    """Write a small PNG to `path` using Pillow if available, else a 1x1 fallback."""
    try:
        from PIL import Image
        img = Image.new("RGB", size, color)
        img.save(path, format="PNG")
    except ImportError:
        # Minimal valid 1x1 PNG
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
            "1f15c4890000000d49444154789c6360000002000100ffff0300000600"
            "05570b2e0000000049454e44ae426082"
        )
        path.write_bytes(png_bytes)


@pytest.fixture
def png_file(tmp_path):
    p = tmp_path / "test.png"
    _make_png(p)
    return str(p)


@pytest.fixture
def large_png_file(tmp_path):
    p = tmp_path / "large.png"
    _make_png(p, size=(2000, 1500))
    return str(p)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def caller():
    return VLMCaller(
        base_url="http://localhost:8001/v1",
        model="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        timeout_seconds=5.0,
        max_attempts=2,
    )


def ok_response(content: str, usage=None) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 500, "completion_tokens": 100},
    }
    r.text = content
    return r


def error_response(status: int, body: str = "error") -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.text = body
    r.json.side_effect = ValueError("not json")
    return r


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


def test_call_success_returns_parsed_json(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"question_text": "..."}')
        result = caller.call("analyze this figure", image_path=png_file)
    assert isinstance(result, dict)
    assert result["question_text"] == "..."


def test_call_success_returns_string_for_non_json(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response("plain text response")
        result = caller.call("analyze", image_path=png_file)
    assert isinstance(result, str)
    assert result == "plain text response"


def test_call_records_usage(caller, png_file):
    usage = {"prompt_tokens": 1000, "completion_tokens": 200}
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}', usage=usage)
        caller.call("analyze", image_path=png_file)
    assert caller.last_usage == usage


# -----------------------------------------------------------------------------
# Request shape
# -----------------------------------------------------------------------------


def test_call_sends_multimodal_message(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("analyze the orbit", image_path=png_file)
    body = mock_post.call_args_list[0][1]["json"]
    content = body["messages"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "analyze the orbit"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/")


def test_call_encodes_image_as_base64(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("analyze", image_path=png_file)
    body = mock_post.call_args_list[0][1]["json"]
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    # Format: data:image/png;base64,<data>
    assert ";base64," in url
    b64_data = url.split(";base64,", 1)[1]
    # Decodable
    decoded = base64.b64decode(b64_data)
    assert len(decoded) > 0


def test_call_resizes_large_image(caller, large_png_file):
    """Large images are resized to max_image_dimension."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        pytest.skip("Pillow not installed; resize path untested")

    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("analyze", image_path=large_png_file)
    body = mock_post.call_args_list[0][1]["json"]
    url = body["messages"][0]["content"][1]["image_url"]["url"]
    b64_data = url.split(";base64,", 1)[1]
    decoded = base64.b64decode(b64_data)

    from PIL import Image
    img = Image.open(io.BytesIO(decoded))
    long_edge = max(img.size)
    assert long_edge <= caller._max_image_dimension


def test_call_includes_seed_and_temperature(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('{"ok": true}')
        caller.call("analyze", image_path=png_file)
    body = mock_post.call_args_list[0][1]["json"]
    assert body["seed"] == 42
    assert body["temperature"] == 0.1


# -----------------------------------------------------------------------------
# Retry / circuit
# -----------------------------------------------------------------------------


def test_call_retries_on_500(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            error_response(500),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("analyze", image_path=png_file)
    assert result == {"ok": True}


def test_call_retries_on_timeout(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [
            requests.Timeout(),
            ok_response('{"ok": true}'),
        ]
        result = caller.call("analyze", image_path=png_file)
    assert result == {"ok": True}


def test_call_raises_after_max_attempts(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [error_response(500), error_response(500)]
        with pytest.raises(APIUnavailable):
            caller.call("analyze", image_path=png_file)


def test_call_no_retry_on_404(caller, png_file):
    """Config errors (404) fail fast — no retry."""
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = error_response(404, "model not found")
        with pytest.raises(APIUnavailable):
            caller.call("analyze", image_path=png_file)
    # Only one POST — no retry
    assert mock_post.call_count == 1


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------


def test_missing_base_url_rejected():
    with pytest.raises(ValueError):
        VLMCaller(base_url="", model="x")


def test_missing_model_rejected():
    with pytest.raises(ValueError):
        VLMCaller(base_url="http://x", model="")


def test_invalid_timeout_rejected():
    with pytest.raises(ValueError):
        VLMCaller(base_url="http://x", model="y", timeout_seconds=0)


def test_invalid_max_attempts_rejected():
    with pytest.raises(ValueError):
        VLMCaller(base_url="http://x", model="y", max_attempts=0)


def test_invalid_image_dimension_rejected():
    with pytest.raises(ValueError):
        VLMCaller(base_url="http://x", model="y", max_image_dimension=10)


def test_missing_image_file_rejected(caller):
    with pytest.raises(ValueError, match="does not exist"):
        caller.call("analyze", image_path="/nonexistent/path.png")


def test_empty_prompt_rejected(caller, png_file):
    with pytest.raises(ValueError):
        caller.call("", image_path=png_file)


# -----------------------------------------------------------------------------
# Response parsing edge cases
# -----------------------------------------------------------------------------


def test_empty_choices_raises(caller, png_file):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"choices": []}
    r.text = ""
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.side_effect = [r, r]
        with pytest.raises(APIUnavailable):
            caller.call("analyze", image_path=png_file)


def test_code_fenced_json_parsed(caller, png_file):
    with patch("core.api.vlm_caller.requests.Session.post") as mock_post:
        mock_post.return_value = ok_response('```json\n{"k": "v"}\n```')
        result = caller.call("analyze", image_path=png_file)
    assert result == {"k": "v"}
