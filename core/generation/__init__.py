"""AION Generation Package — Single Composer Interface (V2 only, V1 deprecated)"""

from .composer_v2 import ComposerV2
from .question_composer import ComposedQuestion  # Keep type for compatibility

__all__ = ["ComposerV2", "ComposedQuestion"]
