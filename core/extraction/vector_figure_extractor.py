"""
AION Vector Figure & Diagram Extractor
======================================
Solves the vector drawing problem in scientific/engineering PDFs.
1. Uses PyMuPDF page.get_drawings() to detect vector path clusters (lines, curves, rects).
2. Renders pages at 300 DPI high resolution.
3. Crops bounding boxes of vector diagrams and encodes them to Base64 Data URIs.
4. Zero external APIs, 100% offline & local.
"""

from __future__ import annotations
import base64
import os
from io import BytesIO
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
    from PIL import Image
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


class VectorFigureExtractor:
    """Extracts vector draw-path diagrams & raster images from PDF pages."""

    @classmethod
    def extract_all_figures(
        cls, pdf_path: str, min_area: float = 2500, max_area_ratio: float = 0.85
    ) -> List[Dict[str, Any]]:
        if not HAS_FITZ or not pdf_path or not os.path.exists(pdf_path):
            return []

        figures = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height

                # Step 1: Detect vector drawing bounding boxes
                drawings = page.get_drawings()
                clusters = []

                if drawings:
                    for draw in drawings:
                        r = draw.get("rect")
                        if not r:
                            continue

                        # Exclude full-page borders and tiny lines
                        if r.width > page_rect.width * 0.92 or r.height > page_rect.height * 0.92:
                            continue
                        if r.width * r.height < 300:
                            continue

                        # Cluster nearby vector drawings together
                        merged = False
                        for cluster in clusters:
                            if cluster.intersects(r) or cluster.distance_to(r) < 35:
                                union_rect = fitz.Rect(cluster).include_rect(r)
                                if union_rect.width * union_rect.height < page_area * max_area_ratio:
                                    cluster.include_rect(r)
                                    merged = True
                                    break
                        if not merged:
                            clusters.append(fitz.Rect(r))

                # Step 2: Render page at 300 DPI for high-res cropping
                pix = page.get_pixmap(dpi=300)
                scale = 300.0 / 72.0
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Step 3: Crop vector clusters
                for idx, c_rect in enumerate(clusters):
                    area = c_rect.width * c_rect.height
                    if area < min_area or area > page_area * max_area_ratio:
                        continue

                    crop_box = (
                        int(max(0, c_rect.x0 - 8) * scale),
                        int(max(0, c_rect.y0 - 8) * scale),
                        int(min(page_rect.width, c_rect.x1 + 8) * scale),
                        int(min(page_rect.height, c_rect.y1 + 8) * scale),
                    )

                    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                        continue

                    try:
                        cropped_img = page_img.crop(crop_box)
                        buf = BytesIO()
                        cropped_img.save(buf, format="PNG")
                        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                        data_uri = f"data:image/png;base64,{b64_str}"

                        figures.append({
                            "page": page_num + 1,
                            "figure_id": f"vec_fig_p{page_num+1}_{idx+1}",
                            "data_uri": data_uri,
                            "url": data_uri,
                            "type": "base64",
                            "caption": f"Extracted Diagram (Page {page_num+1}, Figure {idx+1})"
                        })
                    except Exception as ce:
                        print(f"[VECTOR EXTRACTOR] Crop error page {page_num+1}: {ce}")

            print(f"[VECTOR EXTRACTOR] Successfully extracted {len(figures)} vector figures across {len(doc)} pages.")
            doc.close()
        except Exception as e:
            print(f"[VECTOR EXTRACTOR] Error processing PDF: {e}")

        return figures
