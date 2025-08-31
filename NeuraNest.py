# NeuraNest.py - Fully Updated Closed Beta Version with German Translation
import streamlit as st
import json
from datetime import datetime
import os
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import requests

# ---------------------------
# Storage
# ---------------------------
PATIENTS_FILE = "patients.json"
AUDIT_FILE = "audit_log.json"

def load_patients():
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_patients(patients):
    with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(patients, f, ensure_ascii=False, indent=2)

def load_audit_logs():
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f.readlines()]
    return []

# ---------------------------
# PDF Generator
# ---------------------------
def create_pdf(text, patient_name):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text_object = c.beginText(50, height - 50)
    text_object.setFont("Helvetica", 11)
    for line in text.splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------
# Dummy Generator
# ---------------------------
def dummy_discharge_summary(patient, autonomy_level, lang="Deutsch"):
    if autonomy_level == 0:
        return f"• {patient['diagnosis']} erfolgreich behandelt\n• Beobachtung bis {patient['discharge_date']}" \
            if lang == "Deutsch" else f"• {patient['diagnosis']} successfully treated\n• Observation until {patient['discharge_date']}"
    elif autonomy_level == 1:
        return f"{patient['name']} wurde erfolgreich behandelt." if lang == "Deutsch" else f"{patient['name']} was successfully treated."
    else:
        return f"{patient['name']} hatte eine längere Behandlung mit {patient['diagnosis']}." if lang == "Deutsch" else f"{patient['name']} had extended treatment for {patient['diagnosis']}."

# ---------------------------
# AI Generators
# ---------------------------
def generate_openai_summary(patient, autonomy_level, lang, api_key, model_choice):
    client = OpenAI(api_key=api_key)
    prompt = f"""
    Patient: {patient['name']}, {patient['age']} Jahre
    Diagnose: {patient['diagnosis']}
    Aufenthalt: {patient['admission_date']} bis {patient['discharge_date']}
    Notizen: {patient['notes']}
    """
    if autonomy_level == 0:
        prompt += "\nErstelle kurze Stichpunkte." if lang == "Deutsch" else "\nCreate short bullet points."
    elif autonomy_level == 1:
        prompt += "\nErstelle einfache Zusammenfassung." if lang == "Deutsch" else "\nCreate a simple summary."
    else:
        prompt += "\nErstelle detaillierte Zusammenfassung mit Empfehlungen." if lang == "Deutsch" else "\nCreate a detailed summary with recommendations."
    response = client.chat.completions.create(
        model=model_choice,
        messages=[
            {"role": "system", "content": "Du bist ein deutscher medizinischer Assistent." if lang == "Deutsch" else "You are a medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def generate_togetherai_summary(patient, autonomy_level, lang, api_key, model_choice):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"""
    Patient: {patient['name']}, {patient['age']} Jahre
    Diagnose: {patient['diagnosis']}
    Aufenthalt: {patient['admission_date']} bis {patient['discharge_date']}
    Notizen: {patient['notes']}
    """
    if autonomy_level == 0:
        prompt += "\nErstelle kurze Stichpunkte." if lang == "Deutsch" else "\nCreate short bullet points."
    elif autonomy_level == 1:
        prompt += "\nErstelle einfache Zusammenfassung." if lang == "Deutsch" else "\nCreate a simple summary."
    else:
        prompt += "\nErstelle detaillierte Zusammenfassung mit Empfehlungen." if lang == "Deutsch" else "\nCreate a detailed summary with recommendations."
    data = {
        "model": model_choice,
        "messages": [
            {"role": "system", "content": "Du bist ein deutscher medizinischer Assistent." if lang == "Deutsch" else "You are a medical assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ---------------------------
# Hybrid Generator
# ---------------------------
def generate_summary(patient, autonomy_level, lang, mode, api_key, model_choice):
    if mode == "Dummy Mode":
        return dummy_discharge_summary(patient, autonomy_level, lang)
    elif mode == "OpenAI":
        return generate_openai_summary(patient, autonomy_level, lang, api_key, model_choice)
    elif mode == "TogetherAI":
        return generate_togetherai_summary(patient, autonomy_level, lang, api_key, model_choice)
    else:
        return "⚠️ Unknown mode selected."

# ---------------------------
# Delete / Undo Functions
# ---------------------------
def delete_patient(patient_id):
    patients = load_patients()
    for i, patient in enumerate(patients):
        if patient["id"] == patient_id:
            st.session_state["last_deleted_patient"] = patient
            del patients[i]
            save_patients(patients)
            return patients
    return patients

def undo_delete():
    if "last_deleted_patient" in st.session_state:
        patients = load_patients()
        patients.append(st.session_state["last_deleted_patient"])
        save_patients(patients)
        del st.session_state["last_deleted_patient"]
        return patients
    return load_patients()

# ---------------------------
# Step 1: Language selection first
# ---------------------------
lang = st.radio("🌍 Language / Sprache:", ["Deutsch", "English"], horizontal=True)

# ---------------------------
# Step 2: Helper function for translation
# ---------------------------
def _(en_text, de_text):
    return de_text if lang == "Deutsch" else en_text

# ---------------------------
# Streamlit UI
# ---------------------------
st.title(_("🏥 NeuraNest MVP – Hybrid AI Assistant", "🏥 NeuraNest MVP – Hybrid KI-Assistent"))

# Mode selection
mode = st.radio(_("⚙️ Select Mode:", "⚙️ Modus wählen:"), ["Dummy Mode", "OpenAI", "TogetherAI"], horizontal=True)
api_key = None
model_choice = None
if mode in ["OpenAI", "TogetherAI"]:
    api_key = st.text_input(_("🔑 Enter your API Key:", "🔑 API-Schlüssel eingeben:"), type="password")
    if mode == "OpenAI":
        model_choice = st.selectbox(_("🤖 Choose Model:", "🤖 Modell wählen:"), ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"])
    else:
        model_choice = st.text_input(_("🤖 Together.ai Model:", "🤖 Together.ai Modell:"), value="togethercomputer/llama-2-70b-chat")

# Load patients
patients = load_patients()

# ---------------------------
# Add New Patient
# ---------------------------
st.subheader(_("📋 Add New Patient", "📋 Neuen Patienten hinzufügen"))
with st.form("patient_form"):
    name = st.text_input(_("Name", "Name"))
    age = st.number_input(_("Age", "Alter"), min_value=0, max_value=120, step=1)
    diagnosis = st.text_input(_("Diagnosis", "Diagnose"))
    admission_date = st.date_input(_("Admission Date", "Aufnahmedatum"))
    discharge_date = st.date_input(_("Discharge Date", "Entlassungsdatum"))
    notes = st.text_area(_("Notes", "Notizen"))
    submitted = st.form_submit_button(_("Save Patient", "Patient speichern"))
if submitted:
    new_id = len(patients) + 1
    patient = {
        "id": new_id,
        "name": name,
        "age": age,
        "diagnosis": diagnosis,
        "admission_date": str(admission_date),
        "discharge_date": str(discharge_date),
        "notes": notes
    }
    patients.append(patient)
    save_patients(patients)
    st.success(_(f"✅ Patient {name} saved!", f"✅ Patient {name} gespeichert!"))

# ---------------------------
# Select Patient and Actions
# ---------------------------
if patients:
    selected_patient = st.selectbox(
        _("👤 Select Patient:", "👤 Patienten auswählen:"),
        patients,
        format_func=lambda x: f"{x['name']} ({x['diagnosis']})",
        key="select_patient"
    )

    # ---------------------------
    # Autonomy Level Guide Panel (Visual)
    # ---------------------------
    st.subheader(_("🤖 Autonomy Level Guide", "🤖 Leitfaden zur Autonomie"))
    al_styles = {
        0: {"color": "#A0C4FF", "icon": "🟦", "title": _("AL0 – Bullet Points", "AL0 – Stichpunkte")},
        1: {"color": "#B5E48C", "icon": "🟩", "title": _("AL1 – Simple Summary", "AL1 – Einfache Zusammenfassung")},
        2: {"color": "#FFADAD", "icon": "🟥", "title": _("AL2 – Detailed Summary", "AL2 – Detaillierte Zusammenfassung")}
    }
    al_examples = {
        0: _("• Diagnosis: Pneumonia\n• Treatment: Antibiotics\n• Observation until discharge",
             "• Diagnose: Pneumonie\n• Behandlung: Antibiotika\n• Beobachtung bis Entlassung"),
        1: _("John Doe was treated successfully for pneumonia from 01/08/2025 to 07/08/2025. The treatment was effective and he is ready for discharge.",
             "John Doe wurde erfolgreich wegen Pneumonie vom 01.08.2025 bis 07.08.2025 behandelt. Die Behandlung war erfolgreich, Entlassung möglich."),
        2: _("John Doe was admitted for pneumonia from 01/08/2025 to 07/08/2025. The patient responded well to antibiotics. Vital signs remained stable. Recommendations: follow-up X-ray in two weeks, hydration, and monitoring for recurring symptoms.",
             "John Doe wurde vom 01.08.2025 bis 07.08.2025 wegen Pneumonie aufgenommen. Patient reagierte gut auf Antibiotika. Vitalzeichen stabil. Empfehlungen: Kontrollröntgen in zwei Wochen, Hydration, Überwachung auf Rückfälle.")
    }
    for level in range(3):
        st.markdown(
            f"""
            <div style='background-color: {al_styles[level]['color']}; padding: 10px; border-radius: 10px; margin-bottom:10px'>
            <h4>{al_styles[level]['icon']} {al_styles[level]['title']}</h4>
            <pre style='font-size:12px'>{al_examples[level]}</pre>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Autonomy selection
    autonomy = st.radio(
        _("Autonomy Level:", "Autonomie-Level:"),
        [0, 1, 2],
        format_func=lambda x: ["AL0", "AL1", "AL2"][x],
        horizontal=True,
        key="autonomy_radio"
    )

    # ---------------------------
    # Generate Report
    # ---------------------------
    if st.button(_("Generate Report", "Bericht generieren"), key="generate_report"):
        if mode != "Dummy Mode" and not api_key:
            st.error(_("⚠️ Please enter API Key first.", "⚠️ Bitte zuerst API-Schlüssel eingeben."))
        else:
            summary = generate_summary(selected_patient, autonomy, lang, mode, api_key, model_choice)
            st.subheader(_("📄 Generated Report", "📄 Generierter Bericht"))
            st.write(summary)
            pdf_buffer = create_pdf(summary, selected_patient['name'])
            st.download_button(
                _("📥 Download as PDF", "📥 Als PDF herunterladen"),
                data=pdf_buffer,
                file_name=f"{selected_patient['name']}_summary.pdf",
                mime="application/pdf",
                key="download_pdf"
            )

    # ---------------------------
    # Delete / Undo Patient
    # ---------------------------
    col1, col2 = st.columns(2)
    if col1.button(_("🗑️ Delete Patient", "🗑️ Patient löschen"), key="delete_patient"):
        patients = delete_patient(selected_patient["id"])
        st.success(_(f"✅ Patient {selected_patient['name']} deleted!", f"✅ Patient {selected_patient['name']} gelöscht!"))
    if col2.button(_("↩️ Undo Delete", "↩️ Löschen rückgängig"), key="undo_delete"):
        patients = undo_delete()
        st.success(_("✅ Undo successful!", "✅ Rückgängig erfolgreich!"))

# ---------------------------
# Enhanced Analytics & Filtering
# ---------------------------
if patients:
    st.subheader(_("📈 Patient Analytics & Filters", "📈 Patientenanalyse & Filter"))
    df = pd.DataFrame(patients)
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days

    col_a, col_b, col_c = st.columns(3)
    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    age_range = col_a.slider(_("Age Range", "Altersspanne"), min_value=min_age, max_value=max_age, value=(min_age, max_age))
    date_range = col_b.date_input(_("Admission Date Range", "Aufnahmezeitraum"), [df["admission_date"].min(), df["discharge_date"].max()])
    notes_search = col_c.text_input(_("Search in Notes", "Suche in Notizen"))

    filtered_df = df[
        (df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) &
        (df["admission_date"] >= pd.to_datetime(date_range[0])) & (df["discharge_date"] <= pd.to_datetime(date_range[1]))
    ]
    if notes_search:
        filtered_df = filtered_df[filtered_df["notes"].str.contains(notes_search, case=False, na=False)]

    st.write(_(f"Filtered Patients: {len(filtered_df)}", f"Gefilterte Patienten: {len(filtered_df)}"))

    if not filtered_df.empty:
        st.subheader(_("Most Common Diagnoses", "Häufigste Diagnosen"))
        diag_counts = filtered_df["diagnosis"].value_counts()
        fig1, ax1 = plt.subplots()
        diag_counts.plot(kind="bar", ax=ax1, color="skyblue")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig1)

        st.subheader(_("Age Distribution", "Altersverteilung"))
        fig2, ax2 = plt.subplots()
        filtered_df["age"].plot(kind="hist", bins=10, ax=ax2, color="lightgreen")
        st.pyplot(fig2)

        st.subheader(_("Length of Stay", "Aufenthaltsdauer"))
        fig3, ax3 = plt.subplots()
        filtered_df.plot(x="name", y="length_of_stay", kind="bar", ax=ax3, color="salmon")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig3)
