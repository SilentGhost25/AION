"""
AION v2 Artifact Fusion Layer
=============================
Transforms intermediate extraction elements into canonical DOM artifacts.
Preserves rich academic semantics (Definitions, Examples, Procedures, Algorithms)
without flattening into text chunks.
Saves binary assets to the ArtifactRegistry with content-addressable hashes.
Enforces 100% page accounting into DocumentDOM.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    TextArtifact,
    HeadingArtifact,
    EquationArtifact,
    FigureArtifact,
    TableArtifact,
    AlgorithmArtifact,
    ExampleArtifact,
    DefinitionArtifact,
    ProcedureArtifact,
    CodeArtifact,
    ListArtifact,
)
from aion.core.dom.artifact_registry import ArtifactRegistry
from aion.core.dom.document_dom import (
    DocumentDOM,
    ModuleNode,
    SectionNode,
    PageAuditRecord,
    PageState,
)
from aion.core.dom.relationships import (
    DeterministicReferenceLinker,
    RelationshipGraph,
)
from aion.core.extraction.contracts import (
    PageExtractionResult,
    RawTextBlock,
    RawHeading,
    RawEquation,
    RawFigure,
    RawTable,
)


class ArtifactFusionEngine:
    """
    Assembles extracted page streams into a populated DocumentDOM.
    """

    DEF_PATTERN = re.compile(r'(?i)^\s*(?:definition\s*(\d+(?:\.\d+)?|\b)?\s*[:\-–]\s*)(.*)', re.DOTALL)
    EX_PATTERN = re.compile(r'(?i)^\s*(?:(?:worked\s+)?example\s*(\d+(?:\.\d+)?|\b)?\s*[:\-–]\s*)(.*)', re.DOTALL)
    PROC_PATTERN = re.compile(r'(?i)^\s*(?:procedure\s*(\d+(?:\.\d+)?|\b)?\s*[:\-–]\s*)(.*)', re.DOTALL)
    ALG_PATTERN = re.compile(r'(?i)^\s*(?:algorithm\s*(\d+(?:\.\d+)?|\b)?\s*[:\-–]\s*)(.*)', re.DOTALL)

    def __init__(self, dom: DocumentDOM):
        self.dom = dom
        self.registry = dom.registry
        self.graph = dom.relationships
        self.linker = DeterministicReferenceLinker()

        # Tracking state
        self._current_module_idx: int = 1
        self._current_section_id: str = "SEC-M1-INIT"
        self._section_counter: int = 0
        self._artifact_counter: int = 0
        self._all_fused_artifacts: List[BaseArtifact] = []

    def fuse_page(
        self,
        page_res: PageExtractionResult,
        module_mapping: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> None:
        """
        Process a single page result:
        1. Determine module index based on mapping or page span.
        2. Fuse headings, text blocks, equations, figures, tables.
        3. Record strict page accounting in dom.page_audit.
        """
        pno = page_res.page_number

        # 1. Update module context if module_mapping provided
        if module_mapping:
            for mod_idx, (start_p, end_p) in module_mapping.items():
                if start_p <= pno <= end_p:
                    self._current_module_idx = mod_idx
                    break

        # Ensure ModuleNode exists in DOM
        if self._current_module_idx not in self.dom.modules:
            self.dom.add_module(ModuleNode(
                module_index=self._current_module_idx,
                title=f"Module {self._current_module_idx}",
                start_page=pno,
                end_page=pno,
            ))
        else:
            self.dom.modules[self._current_module_idx].end_page = max(
                self.dom.modules[self._current_module_idx].end_page, pno
            )

        page_artifact_ids: List[str] = []

        # 2. Process Headings
        for rh in page_res.headings:
            h_art = self._fuse_heading(rh, pno)
            self.dom.attach_artifact(h_art, self._current_module_idx, self._current_section_id)
            page_artifact_ids.append(h_art.artifact_id)
            self._all_fused_artifacts.append(h_art)

        # 3. Process Equations
        for req in page_res.equations:
            eq_art = self._fuse_equation(req, pno)
            self.dom.attach_artifact(eq_art, self._current_module_idx, self._current_section_id)
            page_artifact_ids.append(eq_art.artifact_id)
            self._all_fused_artifacts.append(eq_art)

        # 4. Process Figures (Binary images stored in registry vault)
        for rfig in page_res.figures:
            fig_art = self._fuse_figure(rfig, pno)
            self.dom.attach_artifact(fig_art, self._current_module_idx, self._current_section_id)
            page_artifact_ids.append(fig_art.artifact_id)
            self._all_fused_artifacts.append(fig_art)

        # 5. Process Tables
        for rtbl in page_res.tables:
            tbl_art = self._fuse_table(rtbl, pno)
            self.dom.attach_artifact(tbl_art, self._current_module_idx, self._current_section_id)
            page_artifact_ids.append(tbl_art.artifact_id)
            self._all_fused_artifacts.append(tbl_art)

        # 6. Process Text Blocks (with semantic classification)
        for rtxt in page_res.text_blocks:
            classified_art = self._fuse_text_block(rtxt, pno)
            self.dom.attach_artifact(classified_art, self._current_module_idx, self._current_section_id)
            page_artifact_ids.append(classified_art.artifact_id)
            self._all_fused_artifacts.append(classified_art)

        # 7. Record strict page accounting
        audit_record = PageAuditRecord(
            page_number=pno,
            state=page_res.status.to_page_state(),
            char_count=page_res.char_count,
            artifact_count=len(page_artifact_ids),
            artifact_ids=page_artifact_ids,
            error_message=page_res.error_message,
            ocr_applied=page_res.ocr_applied,
        )
        self.dom.record_page_audit(audit_record)

    def finalize(self) -> None:
        """
        Execute final multi-strategy relationship linking after all pages are ingested.
        """
        self.linker.link(self._all_fused_artifacts, self.graph)

    # -----------------------------------------------------------------------
    # Internal Fusion Helpers
    # -----------------------------------------------------------------------

    def _next_id(self, prefix: str, page_number: int) -> str:
        self._artifact_counter += 1
        return f"{prefix}-M{self._current_module_idx}-P{page_number:02d}-{self._artifact_counter:04d}"

    def _fuse_heading(self, rh: RawHeading, page_number: int) -> HeadingArtifact:
        self._section_counter += 1
        sec_id = f"SEC-M{self._current_module_idx}-{self._section_counter:03d}"
        self._current_section_id = sec_id

        # Register section node in DOM
        self.dom.add_section(SectionNode(
            section_id=sec_id,
            module_index=self._current_module_idx,
            title=rh.title,
            level=rh.level,
            numbering=rh.numbering,
            start_page=page_number,
            end_page=page_number,
        ))

        return HeadingArtifact(
            artifact_id=self._next_id("HDG", page_number),
            module_index=self._current_module_idx,
            section_id=sec_id,
            page_number=page_number,
            source_bbox=rh.bbox,
            title=rh.title,
            level=rh.level,
            numbering=rh.numbering,
        )

    def _fuse_equation(self, req: RawEquation, page_number: int) -> EquationArtifact:
        # Extract basic variables from LaTeX if not provided
        vars_found = req.variables if req.variables else list(set(re.findall(r'\\?[a-zA-Z](?:_[a-zA-Z0-9]+)?', req.latex)))
        clean_vars = [v for v in vars_found if not v.startswith("\\") and len(v) <= 4][:6]

        return EquationArtifact(
            artifact_id=self._next_id("EQ", page_number),
            module_index=self._current_module_idx,
            section_id=self._current_section_id,
            page_number=page_number,
            source_bbox=req.bbox,
            latex=req.latex,
            variables=clean_vars,
            is_inline=req.is_inline,
            equation_label=req.label,
            verification_status=req.verification_status,
        )

    def _fuse_figure(self, rfig: RawFigure, page_number: int) -> FigureArtifact:
        aid = self._next_id("FIG", page_number)
        asset_key = ""
        sha = ""

        if rfig.image_bytes:
            asset_key, sha, _ = self.registry.store_asset_bytes(
                rfig.image_bytes,
                relative_subpath=f"figures/m{self._current_module_idx}",
                filename_hint=f"{aid}.png"
            )

        return FigureArtifact(
            artifact_id=aid,
            module_index=self._current_module_idx,
            section_id=self._current_section_id,
            page_number=page_number,
            source_bbox=rfig.bbox,
            asset_key=asset_key,
            image_hash=sha,
            caption=rfig.caption_hint or "",
            width=rfig.width,
            height=rfig.height,
        )

    def _fuse_table(self, rtbl: RawTable, page_number: int) -> TableArtifact:
        return TableArtifact(
            artifact_id=self._next_id("TBL", page_number),
            module_index=self._current_module_idx,
            section_id=self._current_section_id,
            page_number=page_number,
            source_bbox=rtbl.bbox,
            headers=rtbl.headers,
            rows=rtbl.rows,
            markdown_repr=rtbl.markdown,
            caption=rtbl.caption_hint or "",
        )

    def _fuse_text_block(self, rtxt: RawTextBlock, page_number: int) -> BaseArtifact:
        text = rtxt.text.strip()
        norm_text = re.sub(r'\s+', ' ', text)

        # 1. Check for formal Definition
        def_m = self.DEF_PATTERN.match(text)
        if def_m:
            term_part, body = def_m.groups()
            return DefinitionArtifact(
                artifact_id=self._next_id("DEF", page_number),
                module_index=self._current_module_idx,
                section_id=self._current_section_id,
                page_number=page_number,
                source_bbox=rtxt.bbox,
                term=term_part.strip() if term_part else "Definition",
                definition=body.strip(),
            )

        # 2. Check for Worked Example
        ex_m = self.EX_PATTERN.match(text)
        if ex_m:
            ex_title, body = ex_m.groups()
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            sub_title = lines[0] if (lines and not re.match(r'^(?:\d+\.|\-|\*)\s*', lines[0])) else ""
            full_title = f"Example {ex_title.strip()}" if ex_title else "Example"
            if sub_title and len(sub_title) < 80:
                full_title = f"{full_title}: {sub_title}"
            return ExampleArtifact(
                artifact_id=self._next_id("EX", page_number),
                module_index=self._current_module_idx,
                section_id=self._current_section_id,
                page_number=page_number,
                source_bbox=rtxt.bbox,
                title=full_title,
                problem_statement=body.strip(),
            )

        # 3. Check for Procedure
        proc_m = self.PROC_PATTERN.match(text)
        if proc_m:
            proc_title, body = proc_m.groups()
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            sub_title = lines[0] if (lines and not re.match(r'^(?:\d+\.|\-|\*)\s*', lines[0])) else ""
            remaining = "\n".join(lines[1:]) if sub_title else body
            steps = [s.strip() for s in re.split(r'\n(?:\d+\.|\-|\*)\s*', remaining) if s.strip()]
            full_title = f"Procedure {proc_title.strip()}" if proc_title else "Procedure"
            if sub_title and len(sub_title) < 80:
                full_title = f"{full_title}: {sub_title}"
            return ProcedureArtifact(
                artifact_id=self._next_id("PROC", page_number),
                module_index=self._current_module_idx,
                section_id=self._current_section_id,
                page_number=page_number,
                source_bbox=rtxt.bbox,
                title=full_title,
                steps=steps if steps else [body.strip()],
            )

        # 4. Check for Algorithm
        alg_m = self.ALG_PATTERN.match(text)
        if alg_m:
            alg_title, body = alg_m.groups()
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
            sub_title = lines[0] if (lines and not re.match(r'^(?:\d+\.|\-|\*)\s*', lines[0])) else ""
            remaining = "\n".join(lines[1:]) if sub_title else body
            steps = [s.strip() for s in re.split(r'\n(?:\d+\.|\-|\*)\s*', remaining) if s.strip()]
            full_title = f"Algorithm {alg_title.strip()}" if alg_title else "Algorithm"
            if sub_title and len(sub_title) < 80:
                full_title = f"{full_title}: {sub_title}"
            return AlgorithmArtifact(
                artifact_id=self._next_id("ALG", page_number),
                module_index=self._current_module_idx,
                section_id=self._current_section_id,
                page_number=page_number,
                source_bbox=rtxt.bbox,
                title=full_title,
                steps=steps if steps else [body.strip()],
                pseudocode=body.strip(),
            )

        # 5. Default TextArtifact
        return TextArtifact(
            artifact_id=self._next_id("TXT", page_number),
            module_index=self._current_module_idx,
            section_id=self._current_section_id,
            page_number=page_number,
            source_bbox=rtxt.bbox,
            raw_text=text,
            normalized_text=norm_text,
            reading_order=rtxt.reading_order,
        )
