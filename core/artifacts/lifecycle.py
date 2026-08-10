"""
AION Core Artifact Lifecycle — State Machine & Generation Guard
===============================================================
Defines ArtifactStatus, ArtifactStatusTransition, and GenerationGuard
as specified in Part II of the Production Hardening Specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Set, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .manifest import DocumentManifest
    from .store import ArtifactStore

logger = logging.getLogger("AION.ArtifactLifecycle")


class ArtifactStatus(str, Enum):
    UPLOADED           = "UPLOADED"           # bytes stored, nothing else done
    VALIDATING         = "VALIDATING"         # MIME + integrity check running
    EXTRACTING         = "EXTRACTING"         # ExtractionGateway running
    EVIDENCE_VALIDATED = "EVIDENCE_VALIDATED" # chunks passed quality gate
    READY              = "READY"              # generation eligible
    FAILED_EXTRACTION  = "FAILED_EXTRACTION"  # extraction could not produce evidence
    QUARANTINED        = "QUARANTINED"        # safety violation in content
    STALE              = "STALE"              # cache invalidated, needs re-extraction


GENERATION_ELIGIBLE_STATUSES: Set[ArtifactStatus] = {ArtifactStatus.READY}

NON_ELIGIBLE_STATUSES: Set[ArtifactStatus] = {
    ArtifactStatus.UPLOADED,
    ArtifactStatus.VALIDATING,
    ArtifactStatus.EXTRACTING,
    ArtifactStatus.FAILED_EXTRACTION,
    ArtifactStatus.QUARANTINED,
    ArtifactStatus.STALE,
}


VALID_TRANSITIONS: Dict[ArtifactStatus, Set[ArtifactStatus]] = {
    ArtifactStatus.UPLOADED:           {ArtifactStatus.VALIDATING, ArtifactStatus.FAILED_EXTRACTION},
    ArtifactStatus.VALIDATING:         {ArtifactStatus.EXTRACTING, ArtifactStatus.FAILED_EXTRACTION, ArtifactStatus.QUARANTINED},
    ArtifactStatus.EXTRACTING:         {ArtifactStatus.EVIDENCE_VALIDATED, ArtifactStatus.FAILED_EXTRACTION, ArtifactStatus.QUARANTINED},
    ArtifactStatus.EVIDENCE_VALIDATED: {ArtifactStatus.READY, ArtifactStatus.FAILED_EXTRACTION},
    ArtifactStatus.READY:              {ArtifactStatus.STALE, ArtifactStatus.QUARANTINED},
    ArtifactStatus.STALE:              {ArtifactStatus.EXTRACTING},
    ArtifactStatus.FAILED_EXTRACTION:  {ArtifactStatus.EXTRACTING},   # re-extract allowed
    ArtifactStatus.QUARANTINED:        set(),                         # terminal — manual review only
}


class IllegalStatusTransitionError(Exception):
    """Raised when an illegal status transition is requested."""

    def __init__(self, document_id: str, from_status: ArtifactStatus, to_status: ArtifactStatus, message: str):
        super().__init__(message)
        self.document_id = document_id
        self.from_status = from_status
        self.to_status = to_status


class ArtifactStatusTransition:
    """Manages legal transitions between ArtifactStatus states."""

    @classmethod
    def transition(cls, manifest: DocumentManifest, new_status: ArtifactStatus, store: Optional[ArtifactStore] = None) -> DocumentManifest:
        current = manifest.status
        if current != new_status and new_status not in VALID_TRANSITIONS.get(current, set()):
            raise IllegalStatusTransitionError(
                document_id=manifest.document_id,
                from_status=current,
                to_status=new_status,
                message=f"Illegal status transition: {current.value} -> {new_status.value} for document {manifest.document_id}"
            )

        manifest.status = new_status
        manifest.status_changed_at = datetime.now(timezone.utc).isoformat()
        if store:
            store.save_manifest(manifest)
        logger.info(f"[LIFECYCLE] {manifest.document_id}: {current.value} -> {new_status.value}")
        return manifest


@dataclass
class GuardDecision:
    allowed : bool
    code    : str = "OK"
    message : str = "Generation permitted."
    status  : Optional[ArtifactStatus] = None


class GenerationGuard:
    """Guards generation execution to ensure document is in READY state."""

    @classmethod
    def check(cls, document_id: str, store: Optional[Any] = None) -> GuardDecision:
        if store is None:
            from .store import ArtifactStore
            store_inst = ArtifactStore()
        else:
            store_inst = store

        try:
            manifest = store_inst.get(document_id)
        except Exception as e:
            return GuardDecision(
                allowed=False,
                code="MANIFEST_NOT_FOUND",
                message=f"Document manifest not found for id '{document_id}': {e}"
            )

        if not manifest.source.authoritative:
            return GuardDecision(
                allowed=False,
                code="SOURCE_AUTHORITY_INVALID",
                message="Generation requires an original authoritative document.",
                status=manifest.status
            )

        if manifest.status not in GENERATION_ELIGIBLE_STATUSES:
            return GuardDecision(
                allowed=False,
                code="ARTIFACT_NOT_READY",
                message=f"Document status is '{manifest.status.value}'. Only READY documents may generate papers.",
                status=manifest.status
            )

        return GuardDecision(allowed=True, status=manifest.status)
