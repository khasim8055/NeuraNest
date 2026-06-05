# app/core/pdf_exporter.py
# ================================================================
# NeuraCare — PDF Export Engine
# ================================================================
# Generates a properly formatted Arztbrief / discharge letter PDF.
# Uses ReportLab for layout — same library as NeuraNest.py MVP.
#
# Features:
#   - Professional header with clinic name and patient ID
#   - Patient information table
#   - Full letter text body
#   - Risk score callout box
#   - GDPR footer (data stored locally, nothing transmitted)
#   - Saves to desktop or user-specified path
#   - Opens file automatically after saving
# ================================================================

import os
import subprocess
import platform
from datetime import datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib.pagesizes   import A4
from reportlab.lib.units       import cm
from reportlab.lib.styles      import getSampleStyleSheet
from reportlab.lib             import colors as rl_colors
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable,
)
from reportlab.lib.styles      import ParagraphStyle
from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT

from app.core.risk import compute_risk, get_risk_label_de


# ================================================================
# COLOUR PALETTE — matches app UI
# ================================================================
C_DARK_BLUE  = rl_colors.HexColor("#1A3A5C")
C_MID_BLUE   = rl_colors.HexColor("#4A90D9")
C_MUTED      = rl_colors.HexColor("#8B90A8")
C_TEXT       = rl_colors.HexColor("#1C2B3A")
C_ROW_A      = rl_colors.HexColor("#F7F9FB")
C_ROW_B      = rl_colors.HexColor("#FFFFFF")
C_BORDER     = rl_colors.HexColor("#D1DCE8")
C_GREEN      = rl_colors.HexColor("#2D6A4F")
C_AMBER      = rl_colors.HexColor("#B7791F")
C_RED        = rl_colors.HexColor("#C53030")
C_GREEN_BG   = rl_colors.HexColor("#D4EDDA")
C_AMBER_BG   = rl_colors.HexColor("#FFF3CD")
C_RED_BG     = rl_colors.HexColor("#F8D7DA")


# ================================================================
# STYLE DEFINITIONS
# ================================================================



def get_clinic_settings() -> dict:
    """Read clinic letterhead from app_config."""
    defaults = {
        "clinic_name": "NeuraCare Klinik",
        "clinic_doctor": "",
        "clinic_street": "",
        "clinic_city": "",
        "clinic_phone": "",
        "clinic_email": "",
        "clinic_bsnr": "",
        "clinic_lanr": "",
    }
    try:
        import sqlite3, os
        from pathlib import Path as _Path
        base = _Path(os.environ.get("NEURACARE_BASE_DIR", ""))
        if not base or not base.exists():
            base = _Path(__file__).parent.parent.parent
        db = base / "app" / "data" / "neuranest.db"
        if not db.exists():
            return defaults
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT key,value FROM app_config WHERE key LIKE 'clinic_%'"
        ).fetchall()
        conn.close()
        for key, val in rows:
            if key in defaults and val:
                defaults[key] = val
    except Exception:
        pass
    return defaults

def _make_styles() -> dict:
    """Build all paragraph styles used in the PDF."""
    return {
        "clinic_name": ParagraphStyle(
            "clinic_name",
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=C_DARK_BLUE,
            spaceAfter=6,
        ),
        "clinic_sub": ParagraphStyle(
            "clinic_sub",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_MUTED,
            spaceAfter=8,
        ),
        "doc_title": ParagraphStyle(
            "doc_title",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=C_DARK_BLUE,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "meta",
            fontName="Helvetica",
            fontSize=9,
            textColor=C_MUTED,
            spaceAfter=8,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=C_DARK_BLUE,
            spaceBefore=12,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            leading=15,
            textColor=C_TEXT,
            spaceAfter=6,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=15,
            textColor=C_TEXT,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=7,
            textColor=C_MUTED,
            alignment=TA_CENTER,
            spaceBefore=8,
        ),
        "risk_text": ParagraphStyle(
            "risk_text",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=C_TEXT,
            spaceAfter=2,
        ),
    }


# ================================================================
# PATIENT INFO TABLE
# ================================================================

def _make_patient_table(patient: dict, lang: str) -> Table:
    """Build the patient information header table."""
    def fmt_date(d):
        if not d:
            return "—"
        try:
            dt = datetime.strptime(str(d)[:10], "%Y-%m-%d")
            return dt.strftime("%d.%m.%Y") if lang == "Deutsch" \
                   else dt.strftime("%d %b %Y")
        except ValueError:
            return str(d)[:10]

    pid    = f"PT-{patient.get('id', 0):04d}"
    name   = patient.get("name", "—")
    age    = str(patient.get("age", "—"))
    diag   = patient.get("diagnosis", "—")
    icd10  = patient.get("icd10_code", "") or ""
    adm    = fmt_date(patient.get("admission_date"))
    dis    = fmt_date(patient.get("discharge_date"))
    los    = patient.get("length_of_stay", 0) or 0
    phy    = patient.get("physician_name", "") or "—"

    if lang == "Deutsch":
        data = [
            ["Patient",         name,          "Patienten-ID",  pid],
            ["Alter",           f"{age} J.",   "Diagnose",      f"{diag} {icd10}".strip()],
            ["Aufnahme",        adm,           "Entlassung",    dis],
            ["Verweildauer",    f"{los} Tage", "Behandl. Arzt", phy],
        ]
    else:
        data = [
            ["Patient",         name,          "Patient ID",    pid],
            ["Age",             f"{age} yrs",  "Diagnosis",     f"{diag} {icd10}".strip()],
            ["Admission",       adm,           "Discharge",     dis],
            ["Length of stay",  f"{los} days", "Physician",     phy],
        ]

    col_widths = [3.2*cm, 6.5*cm, 3.2*cm, 4.6*cm]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",      (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,0), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TEXTCOLOR",     (0,0), (0,-1),  C_MUTED),
        ("TEXTCOLOR",     (2,0), (2,-1),  C_MUTED),
        ("TEXTCOLOR",     (1,0), (-1,-1), C_TEXT),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [C_ROW_A, C_ROW_B]),
        ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.25, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    return tbl


# ================================================================
# RISK CALLOUT BOX
# ================================================================

def _make_risk_box(patient: dict, lang: str, styles: dict) -> list:
    """Build a coloured risk score callout."""
    risk, score = compute_risk(patient)
    from app.core.risk import get_risk_explanation
    reasons = get_risk_explanation(patient, lang)

    risk_colors = {
        "High":   (C_RED,   C_RED_BG),
        "Medium": (C_AMBER, C_AMBER_BG),
        "Low":    (C_GREEN, C_GREEN_BG),
    }
    text_color, bg_color = risk_colors.get(risk, (C_MUTED, C_ROW_A))

    if lang == "Deutsch":
        risk_label = get_risk_label_de(risk)
        header_txt = f"Wiederaufnahmerisiko: {risk_label} (Score {score}/6)"
    else:
        header_txt = f"Readmission Risk: {risk} (Score {score}/6)"

    reason_lines = "\n".join(f"  • {r}" for r in reasons[:3])

    risk_style = ParagraphStyle(
        "risk_hdr",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=text_color,
        spaceAfter=3,
    )
    reason_style = ParagraphStyle(
        "risk_reason",
        fontName="Helvetica",
        fontSize=9,
        textColor=C_TEXT,
        leading=13,
    )

    data = [[
        Paragraph(header_txt, risk_style),
    ]]
    for r in reasons[:3]:
        data.append([Paragraph(f"  • {r}", reason_style)])

    tbl = Table(data, colWidths=[17.1*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bg_color),
        ("BOX",           (0,0), (-1,-1), 0.5, text_color),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("RIGHTPADDING",  (0,0), (-1,-1), 10),
    ]))
    return [tbl, Spacer(1, 0.3*cm)]


# ================================================================
# LETTER TEXT BODY
# ================================================================

def _make_letter_body(letter_text: str, styles: dict) -> list:
    """Parse and format the letter text into PDF flowables."""
    elements = []
    body_style = styles["body"]
    header_style = styles["section_header"]

    for line in letter_text.splitlines():
        stripped = line.strip()
        if not stripped:
            elements.append(Spacer(1, 0.15*cm))
            continue

        # Section headers — all caps lines or lines ending with ─
        if stripped.startswith("─") or stripped.startswith("═"):
            elements.append(HRFlowable(
                width="100%", thickness=0.5,
                color=C_BORDER, spaceAfter=4, spaceBefore=4
            ))
            continue

        # Detect section titles (all caps, short)
        if (stripped.isupper() and len(stripped) < 50
                and not stripped.startswith("•")):
            elements.append(Paragraph(stripped, header_style))
            continue

        # Bullet points
        if stripped.startswith("•") or stripped.startswith("·"):
            safe = stripped.replace("•", "·").replace("&", "&amp;").replace("<", "&lt;")
            elements.append(Paragraph(safe, body_style))
            continue

        # Regular lines — skip divider lines already handled
        if all(c in "─═ " for c in stripped):
            continue

        safe = stripped.replace("&", "&amp;").replace("<", "&lt;")
        elements.append(Paragraph(safe, body_style))

    return elements


# ================================================================
# MAIN PDF GENERATION FUNCTION
# ================================================================

def generate_pdf(
    patient: dict,
    letter_text: str,
    lang: str = "Deutsch",
    clinic_name: str = "NeuraCare Klinik",
    level: int = 1,
) -> tuple[bool, bytes, str]:
    """
    Generate a PDF discharge letter.

    Args:
        patient:     patient dict from get_patient()
        letter_text: pre-generated letter text from letters.py
        lang:        "Deutsch" or "English"
        clinic_name: clinic name for header
        level:       AL level (0/1/2) for footer

    Returns:
        (True,  pdf_bytes, "")       on success
        (False, b"",       error)    on failure
    """
    if not patient:
        return False, b"", "No patient data."
    if not letter_text:
        return False, b"", "No letter text to export."

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm,
        )

        styles = _make_styles()
        story  = []

        # ── Header ───────────────────────────────────────────────
        # Load clinic settings from database
        cs = get_clinic_settings()
        if clinic_name == "NeuraCare Klinik":
            clinic_name = cs["clinic_name"]

        story.append(Paragraph(clinic_name, styles["clinic_name"]))

        # Build address line from settings
        addr_parts = []
        if cs.get("clinic_doctor"): addr_parts.append(cs["clinic_doctor"])
        if cs.get("clinic_street"): addr_parts.append(cs["clinic_street"])
        if cs.get("clinic_city"):   addr_parts.append(cs["clinic_city"])
        if addr_parts:
            story.append(Paragraph("  ·  ".join(addr_parts), styles["clinic_sub"]))

        contact_parts = []
        if cs.get("clinic_phone"): contact_parts.append("Tel: " + cs["clinic_phone"])
        if cs.get("clinic_email"): contact_parts.append(cs["clinic_email"])
        if cs.get("clinic_bsnr"):  contact_parts.append("BSNR: " + cs["clinic_bsnr"])
        if contact_parts:
            story.append(Paragraph("  ·  ".join(contact_parts), styles["clinic_sub"]))

        if lang == "Deutsch":
            sub = "Vertraulich · Arztbrief · Datenschutzkonform"
        else:
            sub = "Confidential · Discharge Letter · GDPR Compliant"
        story.append(Paragraph(sub, styles["clinic_sub"]))

        story.append(HRFlowable(
            width="100%", thickness=1.5,
            color=C_MID_BLUE, spaceBefore=4, spaceAfter=8
        ))

        # ── Document title ────────────────────────────────────────
        if lang == "Deutsch":
            doc_title = f"ENTLASSUNGSBRIEF — {patient.get('name', '—')}"
        else:
            doc_title = f"DISCHARGE LETTER — {patient.get('name', '—')}"

        story.append(Paragraph(doc_title, styles["doc_title"]))

        now = datetime.now()
        if lang == "Deutsch":
            meta = (f"Erstellt: {now.strftime('%d.%m.%Y %H:%M')}  ·  "
                    f"AL{level}  ·  NeuraCare v1.0")
        else:
            meta = (f"Generated: {now.strftime('%d %b %Y %H:%M')}  ·  "
                    f"AL{level}  ·  NeuraCare v1.0")
        story.append(Paragraph(meta, styles["meta"]))

        story.append(Spacer(1, 0.3*cm))

        # ── Patient info table ────────────────────────────────────
        story.append(_make_patient_table(patient, lang))
        story.append(Spacer(1, 0.4*cm))

        # ── Risk callout box ──────────────────────────────────────
        story.extend(_make_risk_box(patient, lang, styles))

        # ── Letter body ───────────────────────────────────────────
        story.extend(_make_letter_body(letter_text, styles))

        # ── GDPR footer ───────────────────────────────────────────
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=C_BORDER, spaceAfter=4
        ))

        if lang == "Deutsch":
            footer = (
                "Dieses Dokument wurde von NeuraCare generiert. "
                "Alle Patientendaten werden ausschließlich lokal gespeichert und verschlüsselt. "
                "Es wurden keine Patientendaten extern übertragen. "
                f"Patienten-ID: PT-{patient.get('id', 0):04d}"
            )
        else:
            footer = (
                "This document was generated by NeuraCare. "
                "All patient data is stored locally and encrypted. "
                "No patient data was transmitted externally. "
                f"Patient ID: PT-{patient.get('id', 0):04d}"
            )
        story.append(Paragraph(footer, styles["footer"]))

        # ── Build PDF ─────────────────────────────────────────────
        doc.build(story)
        return True, buffer.getvalue(), ""

    except Exception as e:
        return False, b"", f"PDF generation failed: {str(e)}"


# ================================================================
# SAVE TO DISK AND OPEN
# ================================================================

def save_pdf(
    pdf_bytes: bytes,
    patient_name: str,
    save_dir: str | None = None,
) -> tuple[bool, str, str]:
    """
    Save PDF bytes to disk.

    Args:
        pdf_bytes:    raw PDF bytes from generate_pdf()
        patient_name: used for filename
        save_dir:     directory to save in (default: user desktop)

    Returns:
        (True,  file_path, "")       on success
        (False, "",        error)    on failure
    """
    if not pdf_bytes:
        return False, "", "No PDF data to save."

    # Default save location — desktop
    if save_dir is None:
        desktop = Path.home() / "Desktop"
        if not desktop.exists():
            desktop = Path.home()
        save_dir = str(desktop)

    # Build filename
    safe_name  = "".join(c for c in patient_name if c.isalnum() or c in " _-")
    safe_name  = safe_name.strip().replace(" ", "_")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    filename   = f"NeuraCare_{safe_name}_{timestamp}.pdf"
    file_path  = Path(save_dir) / filename

    try:
        with open(file_path, "wb") as f:
            f.write(pdf_bytes)
        return True, str(file_path), ""
    except Exception as e:
        return False, "", f"Could not save PDF: {str(e)}"


def open_pdf(file_path: str) -> bool:
    """
    Open a PDF file using the system default viewer.
    Works on Windows, macOS, Linux.
    Returns True if opened successfully.
    """
    try:
        system = platform.system()
        if system == "Windows":
            os.startfile(file_path)
        elif system == "Darwin":
            subprocess.run(["open", file_path], check=True)
        else:
            subprocess.run(["xdg-open", file_path], check=True)
        return True
    except Exception:
        return False


def export_and_open(
    patient: dict,
    letter_text: str,
    lang: str = "Deutsch",
    level: int = 1,
    clinic_name: str = "NeuraCare Klinik",
) -> tuple[bool, str, str]:
    """
    Full pipeline: generate → save → open.
    Single function call from the UI.

    Returns:
        (True,  file_path, "")       on success
        (False, "",        error)    on failure
    """
    # Generate
    ok, pdf_bytes, err = generate_pdf(
        patient, letter_text, lang=lang,
        clinic_name=clinic_name, level=level,
    )
    if not ok:
        return False, "", err

    # Save
    patient_name = patient.get("name", "Patient")
    ok, file_path, err = save_pdf(pdf_bytes, patient_name)
    if not ok:
        return False, "", err

    # Open
    open_pdf(file_path)

    return True, file_path, ""
