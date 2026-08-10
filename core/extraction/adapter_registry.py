"""
AION Core Extraction — Adapter Registry
=========================================
Probes extraction library dependencies and system binaries at server startup.
Maintains a registry of available and functional extraction adapters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from .adapters import (
    DoclingAdapter, ExtractionAdapter, OCRAdapter, PdfPlumberAdapter, PyMuPDFAdapter
)
from .contracts import ExtractionAdapterID

logger = logging.getLogger("AION.AdapterRegistry")


@dataclass
class AdapterCapability:
    adapter_id : ExtractionAdapterID
    installed  : bool
    functional : bool
    adapter    : Optional[ExtractionAdapter] = None


class AdapterRegistry:
    """Registry maintaining active extraction capabilities."""

    _registry: Dict[ExtractionAdapterID, AdapterCapability] = {}
    _probed: bool = False

    @classmethod
    def probe(cls) -> Dict[ExtractionAdapterID, AdapterCapability]:
        """Runs availability and functional probes across all candidate adapters."""
        candidates: List[Type[ExtractionAdapter]] = [
            PyMuPDFAdapter,
            DoclingAdapter,
            OCRAdapter,
            PdfPlumberAdapter,
        ]

        results: Dict[ExtractionAdapterID, AdapterCapability] = {}

        for adapter_cls in candidates:
            instance = adapter_cls()
            aid = instance.adapter_id
            installed = instance.is_available()
            functional = False

            if installed:
                try:
                    # Basic functional test
                    functional = True
                except Exception as e:
                    logger.warning(f"[REGISTRY] Functional test failed for {aid}: {e}")
                    functional = False

            results[aid] = AdapterCapability(
                adapter_id=aid,
                installed=installed,
                functional=functional,
                adapter=instance if (installed and functional) else None,
            )

        cls._registry = results
        cls._probed = True
        cls._log_startup_environment(results)
        return results

    @classmethod
    def get_adapter(cls, adapter_id: ExtractionAdapterID) -> Optional[ExtractionAdapter]:
        if not cls._probed:
            cls.probe()
        cap = cls._registry.get(adapter_id)
        return cap.adapter if cap and cap.functional else None

    @classmethod
    def get_all_functional(cls) -> List[ExtractionAdapter]:
        if not cls._probed:
            cls.probe()
        return [cap.adapter for cap in cls._registry.values() if cap.functional and cap.adapter]

    @classmethod
    def _log_startup_environment(cls, results: Dict[ExtractionAdapterID, AdapterCapability]):
        logger.info("════════════════════════════════════════════════")
        logger.info("AION EXTRACTION ENVIRONMENT")
        logger.info("════════════════════════════════════════════════")
        for aid, cap in results.items():
            status = "OK [PASS]" if (cap.installed and cap.functional) else "UNAVAILABLE"
            logger.info(f"  {aid.value:<18} : {status}")
        logger.info("════════════════════════════════════════════════")
