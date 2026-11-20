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

def _export_pdf_reportlab(paper_data: dict, out_path: Path):
    """Fallback PDF generator using ReportLab when native WeasyPrint libraries are absent."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VTUTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        alignment=1,  # Center
    )
    meta_style = ParagraphStyle(
        "VTUMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
    )
    q_style = ParagraphStyle(
        "VTUQ",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
    )
    bold_style = ParagraphStyle(
        "VTUBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
    )

    story = []
    story.append(Paragraph("<b>VISVESVARAYA TECHNOLOGICAL UNIVERSITY, BELAGAVI</b>", title_style))
    story.append(Spacer(1, 4))
    dept = paper_data.get("department") or "Department of Computer Science & Engineering"
    exam = (paper_data.get("exam_type") or "IA").upper()
    story.append(Paragraph(f"<b>{dept} — {exam} EXAMINATION</b>", ParagraphStyle("SubTitle", parent=title_style, fontSize=10, leading=13)))
    story.append(Spacer(1, 8))

    subj = paper_data.get("subject") or paper_data.get("subject_name") or "Course Examination"
    code = paper_data.get("subject_code") or "VTU2026"
    max_m = paper_data.get("max_marks", 50)
    dur = paper_data.get("duration", "90 Minutes")

    table_data = [
        [Paragraph(f"<b>Course:</b> {subj}", meta_style), Paragraph(f"<b>Course Code:</b> {code}", meta_style)],
        [Paragraph(f"<b>Max Marks:</b> {max_m}", meta_style), Paragraph(f"<b>Duration:</b> {dur}", meta_style)],
    ]
    meta_table = Table(table_data, colWidths=[340, 200])
    meta_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    q_rows = [[
        Paragraph("<b>Q.No</b>", bold_style),
        Paragraph("<b>Question Description</b>", bold_style),
        Paragraph("<b>Marks</b>", bold_style),
        Paragraph("<b>CO</b>", bold_style),
        Paragraph("<b>RBT</b>", bold_style),
    ]]

    modules = paper_data.get("modules") or []
    for mod in modules:
        m_idx = mod.get("module_index", 1)
        m_title = mod.get("title") or f"Module {m_idx}"
        q_rows.append([
            Paragraph(f"<b>M{m_idx}</b>", bold_style),
            Paragraph(f"<b>{m_title}</b>", bold_style),
            Paragraph("", bold_style),
            Paragraph("", bold_style),
            Paragraph("", bold_style),
        ])
        for q in mod.get("questions", []):
            q_num = q.get("question_number") or q.get("mq_index") or 1
            subs = q.get("sub_questions") or q.get("subquestions") or []
            if subs:
                for sq in subs:
                    let = sq.get("letter", "a")
                    txt = sq.get("text", "")
                    marks = sq.get("marks", 0)
                    co = sq.get("co", "CO1")
                    bloom = sq.get("bloom", "L2")
                    q_rows.append([
                        Paragraph(f"{q_num}.{let}", q_style),
                        Paragraph(txt, q_style),
                        Paragraph(str(marks), q_style),
                        Paragraph(str(co), q_style),
                        Paragraph(str(bloom), q_style),
                    ])
            else:
                txt = q.get("text", "")
                marks = q.get("marks", 10)
                co = q.get("co", "CO1")
                bloom = q.get("bloom", "L2")
                q_rows.append([
                    Paragraph(str(q_num), q_style),
                    Paragraph(txt, q_style),
                    Paragraph(str(marks), q_style),
                    Paragraph(str(co), q_style),
                    Paragraph(str(bloom), q_style),
                ])

    main_table = Table(q_rows, colWidths=[40, 360, 45, 45, 50])
    main_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#0F172A")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F1F5F9")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(main_table)
    doc.build(story)


def export_pdf(paper_data: dict, filename: str = None) -> Path:
    """Export paper to PDF using WeasyPrint with ReportLab fallback."""
    if not filename:
        subj = paper_data.get('subject') or paper_data.get('subject_name') or 'VTU'
        pid = paper_data.get('id') or paper_data.get('paper_id') or 'paper'
        filename = f"{subj}_{pid}.pdf"
    
    out_path = OUTPUT_DIR / filename
    
    try:
        from weasyprint import HTML
        html_content = render_paper_html(paper_data)
        HTML(string=html_content, base_url=str(TEMPLATE_DIR)).write_pdf(str(out_path))
        print(f"[FORMAT] PDF exported via WeasyPrint: {out_path}")
    except (ImportError, OSError, Exception) as weasy_err:
        print(f"[FORMAT] WeasyPrint unavailable ({weasy_err}), falling back to ReportLab: {out_path}")
        _export_pdf_reportlab(paper_data, out_path)
        print(f"[FORMAT] PDF exported via ReportLab fallback: {out_path}")
    
    return out_path

def get_preview_html(paper_data: dict) -> str:
    """Return HTML for frontend live preview."""
    return render_paper_html(paper_data)
