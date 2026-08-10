"""
AION Artifact Store — Generation Request Resolver
===================================================
Resolves incoming GenerationRequests to an authoritative ExtractionSource.
Guarantees that ExtractionGateway ALWAYS receives the original uploaded document,
never a derived plain text file.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from .manifest import DocumentManifest
from .store import ArtifactStore, DocumentNotFoundError, SourceFileMissingError

logger = logging.getLogger("AION.RequestResolver")


class ExtractionSourceMissingError(Exception):
    """Raised when the authoritative source file cannot be resolved."""
    pass


@dataclass
class ExtractionSource:
    """Target source descriptor provided to ExtractionGateway."""
    path        : str
    mime_type   : str
    document_id : str
    manifest    : DocumentManifest


class GenerationRequestResolver:
    """Resolves requests to an authoritative ExtractionSource."""

    @classmethod
    def resolve(cls, request_input: Union[str, Dict[str, Any]], store: Optional[ArtifactStore] = None) -> ExtractionSource:
        store = store or ArtifactStore()

        # Extract document_id or file_path from request_input
        document_id = ""
        file_path = ""

        if isinstance(request_input, str):
            if os.path.exists(request_input):
                file_path = request_input
            else:
                document_id = request_input
        elif isinstance(request_input, dict):
            document_id = request_input.get("file_id") or request_input.get("document_id") or ""
            file_path = request_input.get("file_path") or ""

        manifest: Optional[DocumentManifest] = None

        # STEP 1: LOAD MANIFEST
        if document_id:
            try:
                manifest = store.get(document_id)
            except DocumentNotFoundError:
                manifest = None

        # If document_id was not in manifest store, try resolving from file_path
        if not manifest and file_path and os.path.exists(file_path):
            manifest = store.store_from_temp(file_path, os.path.basename(file_path))

        if not manifest:
            raise ExtractionSourceMissingError(
                f"Could not resolve document manifest for id='{document_id}' path='{file_path}'"
            )

        # STEP 2: VERIFY SOURCE EXISTS
        if not os.path.exists(manifest.source.path):
            raise ExtractionSourceMissingError(
                f"Authoritative source file missing at {manifest.source.path} for document {manifest.document_id}"
            )

        # STEP 3: LOG EXTRACTION MODE
        if manifest.is_pdf():
            logger.info(f"[RESOLVE] Document {manifest.document_id} -> MULTIMODAL mode (PDF source: {manifest.source.path})")
        elif manifest.is_docx():
            logger.info(f"[RESOLVE] Document {manifest.document_id} -> STRUCTURED mode (DOCX source: {manifest.source.path})")
        else:
            logger.info(f"[RESOLVE] Document {manifest.document_id} -> TEXT_ONLY mode (Source: {manifest.source.path})")

        # STEP 4: CHECK DERIVED CACHE
        derived_text = manifest.get_derived_text()
        if derived_text and os.path.exists(derived_text):
            logger.info(f"[RESOLVE] Derived text cache exists at {derived_text} (for reference only)")

        # STEP 5: RETURN EXTRACTION SOURCE (ALWAYS ORIGINAL FILE)
        return ExtractionSource(
            path=manifest.source.path,
            mime_type=manifest.source.mime_type,
            document_id=manifest.document_id,
            manifest=manifest,
        )
