"""
AION Core Artifacts — Package Initializer
=========================================
Exports ArtifactStore, DocumentManifest, SourceArtifact, DerivedArtifact,
MIME Detector, GenerationRequestResolver, and diagnostic assertions.
"""

from .manifest import DerivedArtifact, DocumentManifest, SourceArtifact
from .mime_detector import detect_mime_from_header
from .store import (
    ArtifactStore, DocumentNotFoundError, SourceFileMissingError, SourceIntegrityError
)
from .resolver import ExtractionSource, ExtractionSourceMissingError, GenerationRequestResolver
from .assertions import assert_gateway_receives_original, assert_not_txt_source
from .cache_manager import DerivedCacheManager

__all__ = [
    "SourceArtifact",
    "DerivedArtifact",
    "DocumentManifest",
    "detect_mime_from_header",
    "ArtifactStore",
    "DocumentNotFoundError",
    "SourceFileMissingError",
    "SourceIntegrityError",
    "ExtractionSource",
    "ExtractionSourceMissingError",
    "GenerationRequestResolver",
    "assert_not_txt_source",
    "assert_gateway_receives_original",
    "DerivedCacheManager",
]
