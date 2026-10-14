# tests/unit/test_json_extractor.py
import pytest
from core.generation.robust_llm_caller import _parse_json_response


def test_parse_json_pure():
    raw = '{"question": "What is IoT?", "marks": 5}'
    parsed = _parse_json_response(raw)
    assert parsed["question"] == "What is IoT?"
    assert parsed["marks"] == 5


def test_parse_json_markdown_fenced():
    raw = """```json
{
  "question": "Explain sensor networks",
  "marks": 6
}
```"""
    parsed = _parse_json_response(raw)
    assert parsed["question"] == "Explain sensor networks"
    assert parsed["marks"] == 6


def test_parse_json_with_leading_and_trailing_prose():
    raw = """Certainly! Here is the JSON output requested according to your specification:
```json
{
  "question": "Compare IPv4 and IPv6",
  "marks": 8
}
```
I hope this meets your standards! Let me know if you need any adjustments."""
    parsed = _parse_json_response(raw)
    assert parsed["question"] == "Compare IPv4 and IPv6"
    assert parsed["marks"] == 8


def test_parse_json_unfenced_with_prose():
    raw = """Sure, here you go: {"status": "SUCCESS", "count": 10}. Have a nice day!"""
    parsed = _parse_json_response(raw)
    assert parsed["status"] == "SUCCESS"
    assert parsed["count"] == 10


def test_parse_json_invalid_raises():
    with pytest.raises(ValueError):
        _parse_json_response("This is just prose without any brackets at all.")
