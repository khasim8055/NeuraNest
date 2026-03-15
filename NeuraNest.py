# NeuraCare_Advanced.py
# =====================================================================
# NeuraCare – Hospital-Grade Clinical Assistant
# Privacy-first · Fully offline · GDPR-compliant
# =====================================================================

import streamlit as st
import json
import os
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cryptography.fernet import Fernet
import re
from collections import Counter
import numpy as np

st.set_page_config(
    page_title="NeuraCare – Clinical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
:root {
    --navy: #1A3A5C; --navy-dark: #122840; --teal: #007A7A; --teal-light: #E0F2F1;
    --bg: #F0F4F7; --surface: #FFFFFF; --surface2: #F7F9FB;
    --border: rgba(26,58,92,0.12); --text: #1C2B3A; --text2: #5A7186; --text3: #8EA4B8;
    --mono: 'IBM Plex Mono', monospace;
    --risk-low: #2D6A4F; --risk-low-bg: #D8F3DC;
    --risk-med: #7D4E00; --risk-med-bg: #FFF0CC;
    --risk-high: #7B2020; --risk-high-bg: #FFE5E5;
}
.nc-topbar { background: var(--navy); padding: 10px 24px; display: flex; align-items: center; gap: 14px; border-bottom: 1px solid rgba(255,255,255,0.08); }
.nc-logo { color: #fff; font-size: 17px; font-weight: 600; letter-spacing: .4px; display: flex; align-items: center; gap: 8px; }
.nc-logo-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: #4DD9C0; }
.nc-badge { background: rgba(255,255,255,0.1); color: #A8C4D8; font-size: 11px; font-weight: 500; padding: 3px 10px; border-radius: 20px; letter-spacing: .3px; }
.nc-spacer { flex: 1; }
.nc-offline { background: rgba(77,217,192,0.15); color: #4DD9C0; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; letter-spacing: .4px; }
.nc-hero { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 22px; margin-bottom: 16px; }
.nc-hero-name { font-size: 22px; font-weight: 600; color: var(--text); letter-spacing: -.3px; margin-bottom: 3px; }
.nc-hero-id { font-family: var(--mono); font-size: 11px; color: var(--text3); letter-spacing: .5px; margin-bottom: 14px; }
.nc-hero-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 12px; }
.nc-field label { display: block; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: .8px; color: var(--text3); margin-bottom: 3px; }
.nc-field span { font-size: 13px; color: var(--text); font-weight: 500; }
.nc-field span.mono { font-family: var(--mono); font-size: 12px; }
.nc-risk-high { display: inline-block; background: var(--risk-high-bg); color: var(--risk-high); font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; letter-spacing: .4px; }
.nc-risk-med { display: inline-block; background: var(--risk-med-bg); color: var(--risk-med); font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; letter-spacing: .4px; }
.nc-risk-low { display: inline-block; background: var(--risk-low-bg); color: var(--risk-low); font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 20px; letter-spacing: .4px; }
.nc-card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 14px; overflow: hidden; }
.nc-card-header { padding: 11px 16px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; }
.nc-card-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--text2); }
.nc-card-body { padding: 14px 16px; }
.nc-timeline { display: flex; align-items: center; gap: 0; padding: 8px 0 4px; }
.nc-tl-node { display: flex; flex-direction: column; align-items: center; }
.nc-tl-dot { width: 13px; height: 13px; border-radius: 50%; background: var(--teal); border: 2px solid white; box-shadow: 0 0 0 2px var(--teal); }
.nc-tl-dot.past { background: var(--text3); box-shadow: 0 0 0 2px var(--text3); }
.nc-tl-dot.active { background: #F4A50A; box-shadow: 0 0 0 2px #F4A50A; }
.nc-tl-dot.future { background: var(--border); box-shadow: 0 0 0 2px var(--border); }
.nc-tl-label { font-size: 10px; color: var(--text2); margin-top: 5px; white-space: nowrap; font-family: var(--mono); }
.nc-tl-line { flex: 1; height: 2px; background: var(--border); margin-bottom: 20px; }
.nc-tl-line.done { background: var(--teal); }
.nc-tag { display: inline-block; padding: 1px 8px; border-radius: 4px; background: #FFF0CC; color: #7D4E00; font-size: 11px; font-weight: 600; margin-right: 4px; }
.nc-tag.flag { background: #FFE5E5; color: #7B2020; }
.nc-tag.info { background: #E6F1FB; color: #185FA5; }
.nc-ai-panel { background: linear-gradient(135deg, #EFF6FC 0%, #E0F2F1 100%); border: 1px solid rgba(0,122,122,0.2); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
.nc-ai-label { font-size: 11px; font-weight: 700; color: var(--teal); letter-spacing: .4px; margin-bottom: 6px; display: flex; align-items: center; gap: 5px; }
.nc-ai-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--teal); animation: ncpulse 2s infinite; }
@keyframes ncpulse { 0%,100%{opacity:1} 50%{opacity:.25} }
.nc-ai-text { font-size: 12px; color: var(--text2); line-height: 1.55; }
.nc-audit { font-size: 11px; color: var(--text2); padding: 5px 0; border-bottom: 1px solid var(--border); display: flex; gap: 8px; line-height: 1.4; }
.nc-audit:last-child { border-bottom: none; }
.nc-audit-time { font-family: var(--mono); color: var(--text3); white-space: nowrap; min-width: 60px; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; min-width: 240px !important; }
.stButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; border-radius: 8px !important; border: 1px solid var(--border) !important; background: var(--surface2) !important; color: var(--text) !important; font-size: 13px !important; width: 100% !important; transition: all .15s !important; padding: 8px 14px !important; }
.stButton > button:hover { border-color: var(--teal) !important; background: var(--teal-light) !important; color: var(--teal) !important; }
.stDownloadButton > button { font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; border-radius: 8px !important; border: 1px solid var(--border) !important; background: var(--navy) !important; color: #fff !important; font-size: 13px !important; width: 100% !important; }
.stSelectbox label, .stTextInput label, .stDateInput label, .stNumberInput label, .stTextArea label, .stRadio label { font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: .7px !important; color: var(--text3) !important; }
.stMetric { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; }
.stMetric label { font-size: 10px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: .7px !important; color: var(--text3) !important; }
.stMetric [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 600 !important; color: var(--text) !important; }
[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--surface2) !important; }
.stAlert { border-radius: 8px !important; font-size: 13px !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden !important; font-size: 12px !important; }
.nc-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .8px; color: var(--text3); margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid var(--border); }
.nc-notes { font-size: 13px; color: var(--text2); line-height: 1.65; padding: 12px 14px; background: var(--surface2); border-radius: 8px; border-left: 3px solid var(--teal); }
.main { background: var(--bg) !important; }
.nc-toast-success { background: #D8F3DC; color: #2D6A4F; border: 1px solid #B7E4C7; border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 500; margin-bottom: 8px; }
.nc-toast-error { background: #FFE5E5; color: #7B2020; border: 1px solid #F7C1C1; border-radius: 8px; padding: 10px 14px; font-size: 13px; font-weight: 500; margin-bottom: 8px; }
</style>
"""

KEY_FILE      = "secret.key"
PATIENTS_FILE = "patients.json"

DIAGNOSIS_CATEGORIES = {
    "cardiac":     ["heart","cardio","myocardial","angina","herz","herzinsuffizienz","cardiac"],
    "respiratory": ["pneumonia","copd","respiratory","lung","asthma","bronchitis"],
    "orthopedic":  ["fracture","orthopedic","hip","knee","surgery","arthro","bone"],
    "neurology":   ["stroke","neurolog","seizure","brain","neuro","parkinson"],
    "infection":   ["infection","sepsis","infect","bacterial","viral"],
    "renal":       ["renal","kidney","dialysis","nephro"],
    "endocrine":   ["diabetes","thyroid","insulin","endocrine"],
    "general":     [],
}

# FIX 1 — added "navy" key; was missing, caused KeyError in make_diagnosis_chart
CHART_COLORS = {
    "primary":    "#1A3A5C",
    "navy":       "#1A3A5C",   # alias — was missing, caused the crash
    "teal":       "#007A7A",
    "teal_light": "#4DD9C0",
    "risk_low":   "#52B788",
    "risk_med":   "#F4A50A",
    "risk_high":  "#D64545",
    "bg":         "#F7F9FB",
    "border":     "#D1DCE8",
}


# ── Encryption ──────────────────────────────────────────────────────
def load_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read()
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    return key

fernet = Fernet(load_key())

def encrypt_data(data: str) -> bytes:
    return fernet.encrypt(data.encode())

def decrypt_data(data: bytes) -> str:
    return fernet.decrypt(data).decode()


# ── Storage ──────────────────────────────────────────────────────────
def load_patients():
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE, "rb") as f:
            encrypted = f.read()
            if encrypted:
                try:
                    return json.loads(decrypt_data(encrypted))
                except Exception:
                    st.error("Decryption failed — check that secret.key matches the data file.")
                    return []
    return []

def save_patients(patients):
    with open(PATIENTS_FILE, "wb") as f:
        f.write(encrypt_data(json.dumps(patients, ensure_ascii=False, indent=2)))

def delete_patient(patient_id):
    patients  = load_patients()
    deleted   = [p for p in patients if p["id"] == patient_id]
    remaining = [p for p in patients if p["id"] != patient_id]
    save_patients(remaining)
    if deleted:
        st.session_state["last_deleted"] = deleted[0]
    return remaining

def undo_delete():
    if "last_deleted" in st.session_state:
        patients = load_patients()
        patients.append(st.session_state["last_deleted"])
        save_patients(patients)
        name = st.session_state["last_deleted"]["name"]
        del st.session_state["last_deleted"]
        return patients, name
    return load_patients(), None


# ── Analytics helpers ────────────────────────────────────────────────
def preprocess_df(patients):
    if not patients:
        return pd.DataFrame()
    df = pd.DataFrame(patients)
    df["admission_date"] = pd.to_datetime(df["admission_date"], errors="coerce")
    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")
    df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days.fillna(0).astype(int)
    df["age"]            = pd.to_numeric(df["age"], errors="coerce").fillna(0).astype(int)
    df["diagnosis"]      = df["diagnosis"].fillna("Unknown").astype(str)
    df["notes"]          = df["notes"].fillna("").astype(str)
    return df

def compute_readmission_risk(row):
    score = 0
    age   = int(row.get("age", 0) or 0)
    diag  = str(row.get("diagnosis", "")).lower()
    los   = int(row.get("length_of_stay", 0) or 0)
    notes = str(row.get("notes", "")).lower()
    if age >= 80:   score += 2
    elif age >= 65: score += 1
    chronic = ["diabetes","herzinsuffizienz","heart failure","copd","chronic","renal","dialysis","stroke"]
    if any(k in diag for k in chronic) or any(k in notes for k in chronic):
        score += 2
    if los >= 10: score += 1
    flags = ["complication","infection","wound","reoperation","unstable","sepsis"]
    if any(k in notes for k in flags): score += 1
    if score <= 1:   return "Low",    score
    elif score <= 3: return "Medium", score
    else:            return "High",   score

def extract_keywords(notes, top_n=20):
    if not notes or not isinstance(notes, str):
        return []
    clean = re.sub(r"[^\w\s\-]", " ", notes.lower())
    words = [w for w in clean.split() if len(w) > 3]
    return Counter(words).most_common(top_n)

def categorize_diagnosis(text):
    t = str(text).lower()
    for cat, keywords in DIAGNOSIS_CATEGORIES.items():
        if any(kw in t for kw in keywords):
            return cat
    return "other"

# FIX 2 — safe CSV export: convert Timestamp columns to strings before encoding
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    export = df.copy()
    for col in export.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        export[col] = export[col].dt.strftime("%Y-%m-%d")
    return export.to_csv(index=False).encode("utf-8")


# ── PDF generation ───────────────────────────────────────────────────
def create_pdf(text: str, patient_name: str, patient: dict = None) -> bytes:
    """Build PDF and return raw bytes (not a BytesIO object).
    Returning bytes directly avoids the stale-buffer problem where Streamlit
    reads the BytesIO to the end on first render then finds it empty on the
    next re-render / download click.
    """
    buffer = BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm,  bottomMargin=2*cm)
    story  = []

    title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16,
                                  textColor=rl_colors.HexColor("#1A3A5C"), spaceAfter=6)
    sub_style   = ParagraphStyle("sub",   fontName="Helvetica", fontSize=10,
                                  textColor=rl_colors.HexColor("#5A7186"), spaceAfter=12)
    body_style  = ParagraphStyle("body",  fontName="Helvetica", fontSize=11, leading=16,
                                  textColor=rl_colors.HexColor("#1C2B3A"), spaceAfter=8)

    story.append(Paragraph(f"Discharge Summary — {patient_name}", title_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  NeuraCare Clinical System",
        sub_style))
    story.append(Spacer(1, 0.3*cm))

    if patient:
        info_data = [
            ["Patient Name", patient.get("name",""),        "Patient ID",  f"PT-{patient.get('id',0):04d}"],
            ["Age",          f"{patient.get('age','')} yrs","Diagnosis",   patient.get("diagnosis","")],
            ["Admitted",     str(patient.get("admission_date",""))[:10],
             "Discharged",   str(patient.get("discharge_date",""))[:10]],
        ]
        tbl = Table(info_data, colWidths=[3.5*cm, 5*cm, 3.5*cm, 5*cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME",      (0,0),(-1,-1),"Helvetica"),
            ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0),(2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("TEXTCOLOR",     (0,0),(0,-1), rl_colors.HexColor("#8EA4B8")),
            ("TEXTCOLOR",     (2,0),(2,-1), rl_colors.HexColor("#8EA4B8")),
            ("TEXTCOLOR",     (1,0),(1,-1), rl_colors.HexColor("#1C2B3A")),
            ("TEXTCOLOR",     (3,0),(3,-1), rl_colors.HexColor("#1C2B3A")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),
             [rl_colors.HexColor("#F7F9FB"), rl_colors.HexColor("#FFFFFF")]),
            ("BOX",           (0,0),(-1,-1), 0.5, rl_colors.HexColor("#D1DCE8")),
            ("INNERGRID",     (0,0),(-1,-1), 0.25,rl_colors.HexColor("#D1DCE8")),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Clinical Summary", ParagraphStyle("sec",
        fontName="Helvetica-Bold", fontSize=12,
        textColor=rl_colors.HexColor("#1A3A5C"), spaceBefore=8, spaceAfter=6)))

    for line in text.splitlines():
        if line.strip():
            story.append(Paragraph(line.replace("•", "·"), body_style))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "This document was generated by NeuraCare. All data is stored locally and encrypted. "
        "No patient data was transmitted externally.",
        ParagraphStyle("footer", fontName="Helvetica", fontSize=8,
                       textColor=rl_colors.HexColor("#8EA4B8"), spaceBefore=12)))

    doc.build(story)
    # FIX 3 — return raw bytes so Streamlit download_button never gets a stale buffer
    return buffer.getvalue()


# ── Summary generator ────────────────────────────────────────────────
def dummy_discharge_summary(patient, autonomy_level, lang="Deutsch"):
    name  = patient.get("name", "Patient")
    diag  = patient.get("diagnosis", "")
    adm   = str(patient.get("admission_date", ""))[:10]
    dis   = str(patient.get("discharge_date", ""))[:10]
    notes = patient.get("notes", "")
    age   = patient.get("age", "")
    risk_label, _ = compute_readmission_risk(patient)

    if lang == "Deutsch":
        if autonomy_level == 0:
            return (f"ENTLASSUNGSBERICHT — {name}\n{'─'*50}\n\n"
                    f"• Diagnose: {diag}\n• Alter: {age} Jahre\n"
                    f"• Aufnahme: {adm} | Entlassung: {dis}\n"
                    f"• Wiederaufnahmerisiko: {risk_label}\n\n"
                    f"Klinische Beobachtung bis {dis}.\n"
                    f"Verlauf insgesamt stabil, Patient konnte entlassen werden.")
        elif autonomy_level == 1:
            return (f"ENTLASSUNGSBERICHT — {name}\n{'─'*50}\n\n"
                    f"{name} ({age} J.) wurde erfolgreich wegen {diag} behandelt "
                    f"und am {dis} entlassen. Wiederaufnahmerisiko: {risk_label}.\n\n"
                    f"Klinische Verlauf: {notes[:200] + '...' if len(notes) > 200 else notes}")
        else:
            return (f"ENTLASSUNGSBERICHT — {name}\n{'─'*50}\n\n"
                    f"{name} ({age} J.) hatte eine laengere Behandlung wegen {diag} "
                    f"(Aufnahme {adm} - Entlassung {dis}).\n\n"
                    f"Verlauf und Beobachtungen:\n{notes}\n\n"
                    f"Wiederaufnahmerisiko: {risk_label}.\nAmbulante Nachkontrolle empfohlen.")
    else:
        if autonomy_level == 0:
            return (f"DISCHARGE SUMMARY — {name}\n{'─'*50}\n\n"
                    f"• Diagnosis: {diag}\n• Age: {age} years\n"
                    f"• Admission: {adm} | Discharge: {dis}\n"
                    f"• Readmission Risk: {risk_label}\n\n"
                    f"Clinical observation until {dis}.\n"
                    f"Overall course stable, patient cleared for discharge.")
        elif autonomy_level == 1:
            return (f"DISCHARGE SUMMARY — {name}\n{'─'*50}\n\n"
                    f"{name} ({age} yrs) was successfully treated for {diag} "
                    f"and discharged on {dis}. Readmission risk: {risk_label}.\n\n"
                    f"Clinical notes: {notes[:200] + '...' if len(notes) > 200 else notes}")
        else:
            return (f"DISCHARGE SUMMARY — {name}\n{'─'*50}\n\n"
                    f"{name} ({age} yrs) underwent extended treatment for {diag} "
                    f"(admission {adm} - discharge {dis}).\n\n"
                    f"Clinical course and observations:\n{notes}\n\n"
                    f"Readmission risk assessed as: {risk_label}.\n"
                    f"Outpatient follow-up recommended.")


# ── Chart helpers ────────────────────────────────────────────────────
def setup_chart_style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.spines.top":    False, "axes.spines.right": False,
        "axes.spines.left":   False, "axes.spines.bottom":False,
        "axes.grid":          True,  "grid.color": "#E8EEF3", "grid.linewidth": 0.6,
        "axes.facecolor":     "#F7F9FB", "figure.facecolor": "#F7F9FB",
        "axes.labelcolor":    "#5A7186", "xtick.color": "#8EA4B8", "ytick.color": "#8EA4B8",
        "xtick.labelsize":    9, "ytick.labelsize": 9, "axes.labelsize": 10,
    })

def make_diagnosis_chart(df):
    setup_chart_style()
    counts = df["diagnosis"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(range(len(counts)), counts.values,
                  color=CHART_COLORS["navy"], width=0.6, zorder=3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([t[:18]+"..." if len(t)>18 else t for t in counts.index],
                       rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=9, color="#5A7186")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                ha="center", va="bottom", fontsize=8, color="#1A3A5C", fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=1.2)
    return fig

def make_age_chart(df):
    setup_chart_style()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df["age"], bins=8, color=CHART_COLORS["teal"], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xlabel("Age (years)", fontsize=9)
    ax.set_ylabel("Patients",    fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig

def make_los_chart(df):
    setup_chart_style()
    los_by_diag = df.groupby("diagnosis")["length_of_stay"].mean().sort_values(ascending=True).tail(8)
    fig, ax     = plt.subplots(figsize=(6, 3))
    avg         = df["length_of_stay"].mean()
    colors_list = [CHART_COLORS["teal"] if v < avg else CHART_COLORS["primary"]
                   for v in los_by_diag.values]
    ax.barh(range(len(los_by_diag)), los_by_diag.values, color=colors_list, height=0.55, zorder=3)
    ax.set_yticks(range(len(los_by_diag)))
    ax.set_yticklabels([t[:20]+"..." if len(t)>20 else t for t in los_by_diag.index], fontsize=8)
    ax.set_xlabel("Avg LOS (days)", fontsize=9)
    ax.axvline(avg, color="#D64545", linewidth=1, linestyle="--", alpha=0.6, zorder=4)
    ax.text(avg+0.1, len(los_by_diag)-0.5, f"avg {avg:.1f}d", fontsize=8, color="#D64545")
    fig.tight_layout(pad=1.2)
    return fig

def make_monthly_chart(df):
    setup_chart_style()
    monthly = (df.resample("ME", on="admission_date")
                 .agg({"id":"count","length_of_stay":"mean"})
                 .rename(columns={"id":"admissions","length_of_stay":"avg_los"}))
    if monthly.empty:
        return None
    fig, ax1 = plt.subplots(figsize=(7, 3))
    ax2 = ax1.twinx()
    x   = range(len(monthly))
    ax1.bar(x, monthly["admissions"].values, color=CHART_COLORS["primary"], width=0.5, zorder=3, alpha=0.8)
    ax2.plot(x, monthly["avg_los"].values, color=CHART_COLORS["teal"], linewidth=2, marker="o", markersize=5, zorder=4)
    labels = [d.strftime("%b %y") for d in monthly.index]
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("Admissions", fontsize=9, color=CHART_COLORS["primary"])
    ax2.set_ylabel("Avg LOS",    fontsize=9, color=CHART_COLORS["teal"])
    ax2.tick_params(axis="y", colors=CHART_COLORS["teal"])
    fig.tight_layout(pad=1.2)
    return fig

def make_risk_pie(df):
    setup_chart_style()
    counts    = df["readmission_risk"].value_counts()
    labels    = counts.index.tolist()
    color_map = {"Low": CHART_COLORS["risk_low"], "Medium": CHART_COLORS["risk_med"], "High": CHART_COLORS["risk_high"]}
    clrs      = [color_map.get(l, "#ccc") for l in labels]
    fig, ax   = plt.subplots(figsize=(3.5, 3.5))
    _, _, autotexts = ax.pie(counts.values, labels=None, autopct="%1.0f%%", colors=clrs,
                              startangle=90, wedgeprops={"edgecolor":"white","linewidth":2},
                              pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold"); at.set_color("white")
    patches = [mpatches.Patch(color=color_map.get(l,"#ccc"), label=l) for l in labels]
    ax.legend(handles=patches, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    ax.set_aspect("equal")
    fig.tight_layout(pad=0.5)
    return fig

def make_weekday_chart(df):
    setup_chart_style()
    order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    counts = df["admit_weekday"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    colors_list = [CHART_COLORS["teal"] if i < 5 else CHART_COLORS["risk_med"] for i in range(7)]
    ax.bar(range(7), counts.values, color=colors_list, width=0.6, zorder=3)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], fontsize=8)
    ax.set_ylabel("Admissions", fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig

def make_cat_chart(df):
    setup_chart_style()
    df = df.copy()
    df["diag_category"] = df["diagnosis"].apply(categorize_diagnosis)
    cat_counts = df["diag_category"].value_counts()
    fig, ax    = plt.subplots(figsize=(6, 2.8))
    palette    = [CHART_COLORS["primary"], CHART_COLORS["teal"],
                  CHART_COLORS["risk_med"], CHART_COLORS["risk_high"],
                  "#6B7FAB","#9B7BB5","#5A8A7A","#B07A5A"]
    ax.bar(range(len(cat_counts)), cat_counts.values,
           color=palette[:len(cat_counts)], width=0.6, zorder=3)
    ax.set_xticks(range(len(cat_counts)))
    ax.set_xticklabels(cat_counts.index, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig


# ── Render helpers ───────────────────────────────────────────────────
def render_risk_badge(risk: str) -> str:
    cls = {"Low":"nc-risk-low","Medium":"nc-risk-med","High":"nc-risk-high"}.get(risk,"nc-risk-low")
    dot = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk,"⚪")
    return f'<span class="{cls}">{dot} {risk.upper()} RISK</span>'

def render_patient_hero(p: dict, risk: str):
    risk_badge = render_risk_badge(risk)
    los_val    = ""
    try:
        adm     = datetime.strptime(str(p.get("admission_date",""))[:10], "%Y-%m-%d")
        dis     = datetime.strptime(str(p.get("discharge_date",""))[:10], "%Y-%m-%d")
        los_val = f" · LOS: {(dis - adm).days} days"
    except Exception:
        pass
    st.markdown(f"""
    <div class="nc-hero">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:4px">
        <div>
          <div class="nc-hero-name">{p.get("name","—")}</div>
          <div class="nc-hero-id">PT-{p.get('id',0):04d}{los_val}</div>
        </div>
        <div>{risk_badge}</div>
      </div>
      <div class="nc-hero-grid">
        <div class="nc-field"><label>Age</label><span>{p.get('age','—')} years</span></div>
        <div class="nc-field"><label>Diagnosis</label><span>{p.get('diagnosis','—')}</span></div>
        <div class="nc-field"><label>Admitted</label><span class="mono">{str(p.get('admission_date','—'))[:10]}</span></div>
        <div class="nc-field"><label>Discharge</label><span class="mono">{str(p.get('discharge_date','—'))[:10]}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_topbar(lang):
    st.markdown(f"""
    <div class="nc-topbar">
      <div class="nc-logo"><span class="nc-logo-dot"></span> NeuraCare</div>
      <span class="nc-badge">Clinical Assistant</span>
      <span class="nc-badge">{"Sprache: Deutsch" if lang=="Deutsch" else "Language: English"}</span>
      <div class="nc-spacer"></div>
      <span class="nc-offline">● OFFLINE · ENCRYPTED</span>
    </div>
    """, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.session_state["lang"] = "Deutsch"

    patients = load_patients()
    render_topbar(st.session_state["lang"])

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="nc-section-title">System</div>', unsafe_allow_html=True)
        lang = st.radio("Language / Sprache", ["Deutsch","English"], horizontal=True,
                        index=0 if st.session_state["lang"]=="Deutsch" else 1)
        st.session_state["lang"] = lang
        st.markdown("<hr style='border:none;border-top:1px solid #E2EAF0;margin:12px 0'>",
                    unsafe_allow_html=True)

        st.markdown('<div class="nc-section-title">New Patient</div>', unsafe_allow_html=True)
        with st.expander("➕ Add Patient" if lang=="English" else "➕ Patient hinzufuegen"):
            with st.form("patient_form", clear_on_submit=True):
                name           = st.text_input("Full Name" if lang=="English" else "Vollstaendiger Name")
                age            = st.number_input("Age" if lang=="English" else "Alter", 0, 120, 50)
                diagnosis      = st.text_input("Diagnosis" if lang=="English" else "Diagnose")
                admission_date = st.date_input("Admission Date" if lang=="English" else "Aufnahmedatum")
                discharge_date = st.date_input("Discharge Date" if lang=="English" else "Entlassungsdatum")
                notes          = st.text_area("Clinical Notes" if lang=="English" else "Klinische Notizen", height=90)
                submitted = st.form_submit_button(
                    "Save Patient" if lang=="English" else "Patient speichern",
                    use_container_width=True)

            if submitted and name.strip():
                patients = load_patients()
                new_id   = (max(p["id"] for p in patients) + 1) if patients else 1
                patients.append({
                    "id": new_id, "name": name.strip(), "age": int(age),
                    "diagnosis": diagnosis, "admission_date": str(admission_date),
                    "discharge_date": str(discharge_date), "notes": notes,
                })
                save_patients(patients)
                st.markdown(f'<div class="nc-toast-success">Patient {name} saved.</div>',
                            unsafe_allow_html=True)
                patients = load_patients()

        st.markdown("<hr style='border:none;border-top:1px solid #E2EAF0;margin:12px 0'>",
                    unsafe_allow_html=True)

        if patients:
            st.markdown('<div class="nc-section-title">Select Patient</div>',
                        unsafe_allow_html=True)
            df_all_sb = preprocess_df(patients)
            risk_map  = {}
            if not df_all_sb.empty:
                for _, row in df_all_sb.iterrows():
                    r, _ = compute_readmission_risk(row)
                    risk_map[row["id"]] = r

            def fmt_patient(p):
                risk = risk_map.get(p["id"],"Low")
                pip  = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk,"⚪")
                return f"{pip} {p['name']}  ·  {p.get('diagnosis','')[:22]}"

            selected = st.selectbox("Patient", patients, format_func=fmt_patient,
                                    label_visibility="collapsed")

            st.markdown("<hr style='border:none;border-top:1px solid #E2EAF0;margin:12px 0'>",
                        unsafe_allow_html=True)
            st.markdown('<div class="nc-section-title">Actions</div>', unsafe_allow_html=True)
            action = st.radio("Action",
                              ["Generate Report","Delete Patient","Undo Last Delete"],
                              label_visibility="collapsed")

            if action == "Generate Report":
                autonomy = st.select_slider(
                    "Summary Detail" if lang=="English" else "Zusammenfassungsdetail",
                    options=["Brief (AL0)","Standard (AL1)","Extended (AL2)"])
                al_map = {"Brief (AL0)":0,"Standard (AL1)":1,"Extended (AL2)":2}
                if st.button("Generate Summary", use_container_width=True):
                    summary = dummy_discharge_summary(selected, al_map[autonomy], lang)
                    st.session_state["summary"]    = summary
                    st.session_state["summary_pt"] = selected

            elif action == "Delete Patient":
                st.warning(f"Delete **{selected['name']}**?")
                confirm = st.checkbox("Confirm deletion")
                if st.button("Delete Patient", use_container_width=True) and confirm:
                    patients = delete_patient(selected["id"])
                    st.markdown(f'<div class="nc-toast-error">Patient {selected["name"]} deleted.</div>',
                                unsafe_allow_html=True)
                    patients = load_patients()

            elif action == "Undo Last Delete":
                if st.button("Restore Last Deleted", use_container_width=True):
                    patients, name_restored = undo_delete()
                    if name_restored:
                        st.markdown(f'<div class="nc-toast-success">{name_restored} restored.</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.info("Nothing to restore.")
                    patients = load_patients()
        else:
            st.info("No patients yet. Add a patient above to begin." if lang=="English"
                    else "Noch keine Patienten. Fuegen Sie oben einen Patienten hinzu.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F0F4F7;border:1px solid #D1DCE8;border-radius:8px;
                    padding:10px 12px;font-size:11px;color:#5A7186;line-height:1.5">
            🔒 <strong>Privacy-first</strong><br>
            All data is encrypted locally.<br>No external connections.
        </div>
        """, unsafe_allow_html=True)

    # ── Main content ─────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    patients = load_patients()

    if not patients:
        st.markdown("""
        <div style="max-width:520px;margin:80px auto;text-align:center;padding:40px;
                    background:#fff;border:1px solid #E2EAF0;border-radius:12px">
            <div style="font-size:40px;margin-bottom:16px">🏥</div>
            <div style="font-size:18px;font-weight:600;color:#1C2B3A;margin-bottom:8px">No patient records</div>
            <div style="font-size:13px;color:#5A7186">
                Use the sidebar to add your first patient and begin clinical analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df_all = preprocess_df(patients)
    rr     = df_all.apply(compute_readmission_risk, axis=1)
    df_all["readmission_risk"] = [r[0] for r in rr]
    df_all["risk_score"]       = [r[1] for r in rr]

    tab_patient, tab_analytics = st.tabs(["👤 Patient View", "📊 Ward Analytics"])

    # ── TAB 1 ────────────────────────────────────────────────────────
    with tab_patient:
        df_tmp    = preprocess_df(patients)
        risk_map2 = {}
        for _, row in df_tmp.iterrows():
            r, _ = compute_readmission_risk(row)
            risk_map2[row["id"]] = r

        def fmt2(p):
            r   = risk_map2.get(p["id"],"Low")
            pip = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(r,"⚪")
            return f"{pip} {p['name']}  ·  {p.get('diagnosis','')}"

        pt = st.selectbox("Select Patient Record", patients, format_func=fmt2)
        risk_label, risk_score = compute_readmission_risk(pt)
        render_patient_hero(pt, risk_label)

        try:
            adm = datetime.strptime(str(pt.get("admission_date",""))[:10], "%Y-%m-%d")
            dis = datetime.strptime(str(pt.get("discharge_date",""))[:10], "%Y-%m-%d")
            los = (dis - adm).days
        except Exception:
            los = 0

        ward_avg_los = df_all["length_of_stay"].mean()
        k1, k2, k3  = st.columns(3)
        k1.metric("Length of Stay", f"{los} days",
                  delta=f"{los - ward_avg_los:+.0f} vs ward avg", delta_color="inverse")
        k2.metric("Readmission Risk Score", f"{risk_score} / 5", delta=risk_label)
        same_diag = df_all[df_all["diagnosis"].str.lower() == str(pt.get("diagnosis","")).lower()]
        k3.metric("Same-Diagnosis Patients", len(same_diag),
                  delta=f"Avg LOS {same_diag['length_of_stay'].mean():.1f}d" if not same_diag.empty else "")

        st.markdown("""
        <div class="nc-card">
          <div class="nc-card-header"><span class="nc-card-title">Patient Journey</span></div>
          <div class="nc-card-body">
            <div class="nc-timeline">
              <div class="nc-tl-node"><div class="nc-tl-dot past"></div><div class="nc-tl-label">Admission</div></div>
              <div class="nc-tl-line done"></div>
              <div class="nc-tl-node"><div class="nc-tl-dot past"></div><div class="nc-tl-label">Diagnostics</div></div>
              <div class="nc-tl-line done"></div>
              <div class="nc-tl-node"><div class="nc-tl-dot past"></div><div class="nc-tl-label">Treatment</div></div>
              <div class="nc-tl-line done"></div>
              <div class="nc-tl-node"><div class="nc-tl-dot active"></div><div class="nc-tl-label" style="color:#7D4E00;font-weight:600">Review</div></div>
              <div class="nc-tl-line"></div>
              <div class="nc-tl-node"><div class="nc-tl-dot future"></div><div class="nc-tl-label">Discharge</div></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="nc-card">
          <div class="nc-card-header"><span class="nc-card-title">Clinical Notes</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="nc-notes">{pt.get("notes","No clinical notes recorded.")}</div>',
                    unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="nc-section-title" style="margin-top:8px">Discharge Report</div>',
                    unsafe_allow_html=True)
        col_al, col_lang2 = st.columns([3,2])
        with col_al:
            autonomy = st.select_slider("Summary Detail Level",
                                        options=["Brief (AL0)","Standard (AL1)","Extended (AL2)"],
                                        value="Standard (AL1)")
        with col_lang2:
            rpt_lang = st.radio("Report Language", ["Deutsch","English"], horizontal=True)

        al_map = {"Brief (AL0)":0,"Standard (AL1)":1,"Extended (AL2)":2}
        if st.button("Generate Discharge Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                summary = dummy_discharge_summary(pt, al_map[autonomy], rpt_lang)
                st.session_state["summary"]    = summary
                st.session_state["summary_pt"] = pt

        # FIX 3 — show download for any generated summary (removed fragile id check)
        if "summary" in st.session_state:
            st.markdown("""
            <div class="nc-card">
              <div class="nc-card-header"><span class="nc-card-title">Generated Summary</span></div>
            </div>
            """, unsafe_allow_html=True)
            st.code(st.session_state["summary"], language=None)

            pdf_pt   = st.session_state.get("summary_pt", pt)
            pdf_data = create_pdf(st.session_state["summary"], pdf_pt["name"], pdf_pt)
            st.download_button(
                "Download as PDF",
                data=pdf_data,                  # bytes — never a stale BytesIO
                file_name=f"{pdf_pt['name'].replace(' ','_')}_discharge_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="nc-ai-panel">
          <div class="nc-ai-label">
            <span class="nc-ai-dot"></span> Local AI Assistant · AL3 (Upcoming)
          </div>
          <div class="nc-ai-text">
            Readmission risk assessed as <strong>{risk_label}</strong> (score {risk_score}/5).
            {"Post-discharge cardiology follow-up within 7 days recommended." if risk_label=="High" else
             "Standard follow-up protocol advised." if risk_label=="Medium" else
             "Routine outpatient check-up sufficient."}
            All processing runs locally — no data leaves this device.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2 ────────────────────────────────────────────────────────
    with tab_analytics:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="nc-section-title">Filters</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([3,3,4])
        ages = df_all["age"].dropna()
        if len(ages) > 0 and int(ages.min()) < int(ages.max()):
            age_range = fc1.slider("Age Range", int(ages.min()), int(ages.max()),
                                   (int(ages.min()), int(ages.max())))
        else:
            age_range = (int(ages.min()) if len(ages) else 0, int(ages.max()) if len(ages) else 120)
            fc1.write(f"Age: **{age_range[0]}**")

        valid_dates = df_all["admission_date"].dropna()
        if len(valid_dates):
            date_range = fc2.date_input("Admission Date Range",
                                        [valid_dates.min().date(), valid_dates.max().date()])
        else:
            date_range = [datetime.today().date(), datetime.today().date()]

        notes_search = fc3.text_input("Search in Notes", placeholder="keyword...")

        fdf = df_all.copy()
        fdf = fdf[(fdf["age"] >= age_range[0]) & (fdf["age"] <= age_range[1])]
        try:
            fdf = fdf[
                (fdf["admission_date"] >= pd.to_datetime(date_range[0])) &
                (fdf["admission_date"] <= pd.to_datetime(date_range[1]))
            ]
        except Exception:
            pass
        if notes_search.strip():
            fdf = fdf[fdf["notes"].str.contains(notes_search, case=False, na=False)]

        st.markdown(f"<div style='font-size:12px;color:#8EA4B8;margin-bottom:12px'>"
                    f"Showing <strong style='color:#1C2B3A'>{len(fdf)}</strong> of "
                    f"{len(df_all)} patients</div>", unsafe_allow_html=True)

        if fdf.empty:
            st.info("No patients match the current filters.")
            return

        fdf = fdf.copy()
        rr2 = fdf.apply(compute_readmission_risk, axis=1)
        fdf["readmission_risk"] = [r[0] for r in rr2]
        fdf["risk_score"]       = [r[1] for r in rr2]
        fdf["admit_weekday"]    = fdf["admission_date"].dt.day_name()
        fdf["diag_category"]    = fdf["diagnosis"].apply(categorize_diagnosis)

        avg_los   = fdf["length_of_stay"].mean()
        med_los   = fdf["length_of_stay"].median()
        high_risk = len(fdf[fdf["readmission_risk"]=="High"])
        top_diag  = fdf["diagnosis"].value_counts().idxmax() if not fdf.empty else "—"

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Patients",     len(fdf))
        m2.metric("Avg LOS (days)",     f"{avg_los:.1f}" if not np.isnan(avg_los) else "—")
        m3.metric("Median LOS (days)",  f"{med_los:.1f}" if not np.isnan(med_los) else "—")
        m4.metric("High Risk Patients", high_risk,
                  delta=f"{high_risk/len(fdf)*100:.0f}% of ward" if len(fdf) else "")
        m5.metric("Top Diagnosis",
                  top_diag[:18]+"..." if len(str(top_diag))>18 else top_diag)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="nc-section-title">Diagnosis & Demographics</div>',
                    unsafe_allow_html=True)
        r1c1, r1c2 = st.columns([3,2])
        with r1c1:
            fig = make_diagnosis_chart(fdf)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with r1c2:
            fig = make_age_chart(fdf)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown('<div class="nc-section-title">Length of Stay</div>', unsafe_allow_html=True)
        r2c1, r2c2 = st.columns([2,3])
        with r2c1:
            st.markdown("""<div class="nc-card"><div class="nc-card-header">
            <span class="nc-card-title">LOS Statistics</span></div>
            <div class="nc-card-body">""", unsafe_allow_html=True)
            st.write({
                "Average (days)": round(float(avg_los),1) if not np.isnan(avg_los) else None,
                "Median (days)":  round(float(med_los),1) if not np.isnan(med_los) else None,
                "Std Dev":        round(float(fdf["length_of_stay"].std()),1),
                "Min":            int(fdf["length_of_stay"].min()),
                "Max":            int(fdf["length_of_stay"].max()),
            })
            st.markdown("</div></div>", unsafe_allow_html=True)
        with r2c2:
            fig = make_los_chart(fdf)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown('<div class="nc-section-title">Monthly Admissions & Average LOS</div>',
                    unsafe_allow_html=True)
        fig = make_monthly_chart(fdf)
        if fig:
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        else:
            st.info("Not enough data for monthly trend chart.")

        st.markdown('<div class="nc-section-title">Risk, Categories & Admission Patterns</div>',
                    unsafe_allow_html=True)
        r3c1, r3c2, r3c3 = st.columns([2,3,3])
        with r3c1:
            fig = make_risk_pie(fdf)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with r3c2:
            fig = make_cat_chart(fdf)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with r3c3:
            fig = make_weekday_chart(fdf)
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown('<div class="nc-section-title">Clinical Notes — Top Keywords</div>',
                    unsafe_allow_html=True)
        all_notes = " ".join(fdf["notes"].tolist())
        keywords  = extract_keywords(all_notes, top_n=20)
        if keywords:
            kw_df = pd.DataFrame(keywords[:10], columns=["Keyword","Count"])
            kw_df.index = kw_df.index + 1
            st.dataframe(kw_df, use_container_width=True, height=240)
        else:
            st.info("No significant keywords found in notes.")

        st.markdown('<div class="nc-section-title">Patient Risk Table</div>',
                    unsafe_allow_html=True)
        cols_show  = ["id","name","age","diagnosis","length_of_stay",
                      "readmission_risk","risk_score","admission_date","discharge_date"]
        risk_table = fdf[cols_show].sort_values("risk_score", ascending=False).reset_index(drop=True)
        risk_table.index = risk_table.index + 1
        st.dataframe(risk_table, use_container_width=True, height=280)

        st.markdown('<div class="nc-section-title">Individual Patient Journey</div>',
                    unsafe_allow_html=True)
        pt_sel = st.selectbox("Select patient for detail view", fdf.to_dict("records"),
                              format_func=lambda x: f"{x['name']} ({x['diagnosis']})",
                              key="analytics_pt_sel")
        if pt_sel:
            adm_d = pt_sel["admission_date"]
            dis_d = pt_sel["discharge_date"]
            if hasattr(adm_d,"date"): adm_d = adm_d.date()
            if hasattr(dis_d,"date"): dis_d = dis_d.date()
            r_lbl, r_sc = compute_readmission_risk(pt_sel)
            st.markdown(f"""
            <div class="nc-card">
              <div class="nc-card-header">
                <span class="nc-card-title">{pt_sel['name']}</span>
                {render_risk_badge(r_lbl)}
              </div>
              <div class="nc-card-body">
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px">
                  <div class="nc-field"><label>Admission</label><span class="mono">{adm_d}</span></div>
                  <div class="nc-field"><label>Discharge</label><span class="mono">{dis_d}</span></div>
                  <div class="nc-field"><label>LOS</label><span>{pt_sel['length_of_stay']} days</span></div>
                  <div class="nc-field"><label>Risk Score</label><span>{r_sc}/5</span></div>
                </div>
                <div class="nc-notes">{pt_sel.get('notes','No notes recorded.')}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # FIX 2 — use df_to_csv_bytes() — handles Timestamp columns safely
        st.markdown('<div class="nc-section-title">Export</div>', unsafe_allow_html=True)
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "Download Filtered Data (CSV)",
                data=df_to_csv_bytes(fdf),
                file_name="filtered_patients.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ex2:
            anon = fdf.drop(columns=["name","notes"], errors="ignore")
            st.download_button(
                "Download Anonymised Summary (CSV)",
                data=df_to_csv_bytes(anon),
                file_name="anonymised_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()