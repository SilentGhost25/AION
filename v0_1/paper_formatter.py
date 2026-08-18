"""
AION Paper Formatter
Renders generated JSON -> HTML -> PDF (VTU standard)
"""
from __future__ import annotations
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
OUTPUT_DIR   = Path("generated_papers")
OUTPUT_DIR.mkdir(exist_ok=True)

env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))

def render_paper_html(paper_data: dict) -> str:
    """Render paper data to HTML string."""
    template = env.get_template("vtu_paper.html")
    # Clean visual field mismatch keys
    return template.render(**paper_data)

def export_pdf(paper_data: dict, filename: str = None) -> Path:
    """Export paper to PDF using WeasyPrint."""
    from weasyprint import HTML
    html_content = render_paper_html(paper_data)
    
    if not filename:
        filename = f"{paper_data.get('subject', 'VTU')}_{paper_data.get('id', 'paper')}.pdf"
    
    out_path = OUTPUT_DIR / filename
    HTML(string=html_content, base_url=str(TEMPLATE_DIR)).write_pdf(str(out_path))
    
    print(f"[FORMAT] PDF exported: {out_path}")
    return out_path

def get_preview_html(paper_data: dict) -> str:
    """Return HTML for frontend live preview."""
    return render_paper_html(paper_data)
