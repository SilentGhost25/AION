# tests/unit/test_serialization.py
import pytest
from dataclasses import dataclass
from pathlib import Path
from core.serialization import jsonable


@dataclass
class SampleDifficulty:
    level: str
    score: float


class UnknownCustomType:
    def __init__(self, value):
        self._secret = value

    def __repr__(self):
        return f"<UnknownCustomType {self._secret}>"


def test_jsonable_primitives_and_dataclass():
    data = {
        "text": "sample",
        "num": 42,
        "flag": True,
        "path": Path("/tmp/test.txt"),
        "nested": SampleDifficulty(level="HARD", score=0.85),
        "tags": {"a", "b"},
    }
    result = jsonable(data, _strict=True)
    assert result["text"] == "sample"
    assert result["num"] == 42
    assert result["flag"] is True
    assert result["path"] == str(Path("/tmp/test.txt"))
    assert result["nested"] == {"level": "HARD", "score": 0.85}
    assert set(result["tags"]) == {"a", "b"}


def test_jsonable_strict_raises_on_unhandled():
    # An object without __dict__ or custom to_dict (like a custom socket or object instance with slots)
    class SlottedUnknown:
        __slots__ = ("val",)
        def __init__(self, val):
            self.val = val

    unhandled = SlottedUnknown("mystery")

    with pytest.raises(TypeError) as exc:
        jsonable(unhandled, _strict=True)
    assert "unhandled type SlottedUnknown" in str(exc.value)

    # In non-strict mode, falls back to repr
    rep = jsonable(unhandled, _strict=False)
    assert "SlottedUnknown" in rep
