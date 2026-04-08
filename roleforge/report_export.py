from io import BytesIO
from typing import List, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


def _styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="RFTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#1f3c88"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RFSub",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#555555"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RFHead",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#1f3c88"),
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RFBody",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="RFBullet",
            parent=styles["Normal"],
            fontSize=10.5,
            leading=15,
            leftIndent=12,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    return styles


def _kv_table(rows: List[Tuple[str, str]]):
    table = Table(rows, colWidths=[55 * mm, 110 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d0d7de")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def _bullet_paragraphs(items: List[str], style):
    out = []
    for item in items:
        out.append(Paragraph(item, style, bulletText="-"))
    return out


def build_roleforge_report_pdf(
    target_role: str,
    user_skills: List[str],
    readiness_score: float,
    reality_verdict: str,
    fastest_role: str,
    confidence: str,
    estimated_months_to_ready: int,
    matched_skills: List[str],
    bottlenecks: List[Tuple[str, float]],
    roadmap: List[Tuple[str, List[str]]],
) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="RoleForge Report",
        author="RoleForge",
    )
    styles = _styles()
    story = []

    story.append(Paragraph("RoleForge Career Report", styles["RFTitle"]))
    story.append(
        Paragraph(
            "Constraint-aware readiness snapshot with skill gaps, role recommendation, and roadmap.",
            styles["RFSub"],
        )
    )
    story.append(Spacer(1, 4))

    summary_rows = [
        ("Target role", target_role),
        ("Readiness", f"{readiness_score:.1f}%"),
        ("Reality verdict", reality_verdict),
        ("Fastest role", fastest_role),
        ("Confidence", confidence),
        ("Estimated months to ready", str(estimated_months_to_ready)),
    ]
    story.append(_kv_table(summary_rows))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Current Skills", styles["RFHead"]))
    if user_skills:
        story.extend(_bullet_paragraphs(user_skills, styles["RFBullet"]))
    else:
        story.append(Paragraph("No skills provided.", styles["RFBody"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Matched Skills", styles["RFHead"]))
    if matched_skills:
        story.extend(_bullet_paragraphs(matched_skills, styles["RFBullet"]))
    else:
        story.append(Paragraph("No matched skills found.", styles["RFBody"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Top Skill Gaps", styles["RFHead"]))
    if bottlenecks:
        gap_rows = [("Skill", "Weight")]
        for skill, weight in bottlenecks[:8]:
            gap_rows.append((skill, f"{weight:.2f}"))
        gap_table = Table(gap_rows, colWidths=[120 * mm, 35 * mm], hAlign="LEFT")
        gap_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f0fe")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f3c88")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d0d7de")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e7eb")),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("LEADING", (0, 0), (-1, -1), 12),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(gap_table)
    else:
        story.append(Paragraph("No major skill gaps detected.", styles["RFBody"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Roadmap", styles["RFHead"]))
    if roadmap:
        for interval, tasks in roadmap:
            story.append(Paragraph(f"<b>{interval}</b>", styles["RFBody"]))
            for task in tasks:
                story.append(Paragraph(task, styles["RFBullet"], bulletText="-"))
            story.append(Spacer(1, 4))
    else:
        story.append(Paragraph("No roadmap available.", styles["RFBody"]))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes