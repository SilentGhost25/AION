# core/extraction/figure_invariant.py

import logging

LOG = logging.getLogger("aion.figures")


class FigurePropagationFailure(RuntimeError):
    def __init__(self, extracted: int, mapped: int, message: str):
        super().__init__(message)
        self.extracted = extracted
        self.mapped = mapped


def assert_figures_propagated(
    extraction_figure_count : int,
    mapper_figure_count     : int,
    visual_rag_enabled      : bool,
) -> None:
    """
    Asserts figures are not silently dropped between extraction and mapper.
    """
    if not visual_rag_enabled:
        LOG.info("[FIGURES] Visual RAG disabled — figure propagation not required.")
        return

    if extraction_figure_count > 0 and mapper_figure_count == 0:
        raise FigurePropagationFailure(
            extracted  = extraction_figure_count,
            mapped     = mapper_figure_count,
            message    = (
                f"Extraction found {extraction_figure_count} figures "
                f"but mapper received 0. "
                f"Figure data was lost in the pipeline. "
                f"Trace: ExtractionGateway -> EvidenceArtifact -> ModuleSegment -> TextChunk -> ChunkImageMapper."
            )
        )

    LOG.info(
        f"[FIGURES] extraction={extraction_figure_count} "
        f"mapped={mapper_figure_count} — OK"
    )
