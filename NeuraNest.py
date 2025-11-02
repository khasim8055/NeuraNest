# NeuraNest_Local_Advanced.py
import streamlit as st
import json
import os
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import pandas as pd
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
import re
from collections import Counter
import numpy as np

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="NeuraNest Insights", layout="wide")

KEY_FILE = "secret.key"
PATIENTS_FILE = "patients.json"

# ---------------------------
# Encryption Setup
# ---------------------------
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

# ---------------------------
# Storage helpers
# ---------------------------
def load_patients():
    if os.path.exists(PATIENTS_FILE):
        with open(PATIENTS_FILE, "rb") as f:
            encrypted = f.read()
            if encrypted:
                try:
                    return json.loads(decrypt_data(encrypted))
                except Exception as e:
                    st.error("Failed to decrypt patient file. Is the secret.key matching the file?")
                    return []
    return []

def save_patients(patients):
    with open(PATIENTS_FILE, "wb") as f:
        f.write(encrypt_data(json.dumps(patients, ensure_ascii=False, indent=2)))

# ---------------------------
# Basic Utilities
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
# Core CRUD: delete & undo
# ---------------------------
def delete_patient(patient_id):
    patients = load_patients()
    deleted = [p for p in patients if p["id"] == patient_id]
    patients = [p for p in patients if p["id"] != patient_id]
    save_patients(patients)
    if deleted:
        st.session_state["last_deleted"] = deleted[0]
    return patients

def undo_delete():
    if "last_deleted" in st.session_state:
        patients = load_patients()
        patients.append(st.session_state["last_deleted"])
        save_patients(patients)
        st.success(f" Patient {st.session_state['last_deleted']['name']} restored.")
        del st.session_state["last_deleted"]
        return patients
    else:
        st.warning(" No patient to restore.")
        return load_patients()

# ---------------------------
# Simple Local Summary Generator (Dummy)
# ---------------------------
def dummy_discharge_summary(patient, autonomy_level, lang="Deutsch"):
    if autonomy_level == 0:
        return (f"• {patient['diagnosis']} erfolgreich behandelt\n• Beobachtung bis {patient['discharge_date']}"
                if lang == "Deutsch" else
                f"• {patient['diagnosis']} successfully treated\n• Observation until {patient['discharge_date']}")
    elif autonomy_level == 1:
        return (f"{patient['name']} wurde erfolgreich behandelt." if lang == "Deutsch"
                else f"{patient['name']} was successfully treated.")
    else:
        return (f"{patient['name']} hatte eine längere Behandlung mit {patient['diagnosis']}." if lang == "Deutsch"
                else f"{patient['name']} had extended treatment for {patient['diagnosis']}.")

def generate_summary(patient, autonomy_level, lang, mode="Dummy"):
    return dummy_discharge_summary(patient, autonomy_level, lang)

# ---------------------------
# Advanced Analytics Helpers
# ---------------------------
def preprocess_df(patients):
    df = pd.DataFrame(patients)
    if df.empty:
        return df
    # Ensure dates are datetimes
    df["admission_date"] = pd.to_datetime(df["admission_date"])
    df["discharge_date"] = pd.to_datetime(df["discharge_date"])
    df["length_of_stay"] = (df["discharge_date"] - df["admission_date"]).dt.days
    # Basic normalization for diagnosis text
    df["diagnosis"] = df["diagnosis"].fillna("Unknown").astype(str)
    df["notes"] = df["notes"].fillna("").astype(str)
    return df

# Rule-based readmission risk (simple, transparent)
def compute_readmission_risk(row):
    score = 0
    age = int(row.get("age", 0) or 0)
    diag = row.get("diagnosis", "").lower()
    los = int(row.get("length_of_stay", 0) or 0)
    notes = row.get("notes", "").lower()

    # Age-based
    if age >= 80:
        score += 2
    elif age >= 65:
        score += 1

    # Diagnosis heuristics (chronic disease keywords)
    chronic_keywords = ["diabetes", "herzinsuffizienz", "heart failure", "copd", "chronic", "renal", "dialysis", "stroke"]
    if any(k in diag for k in chronic_keywords) or any(k in notes for k in chronic_keywords):
        score += 2

    # Long stay
    if los >= 10:
        score += 1

    # Notes flags
    flags = ["complication", "infection", "wound", "reoperation", "unstable"]
    if any(k in notes for k in flags):
        score += 1

    # Simple mapping
    if score <= 1:
        return "Low", score
    elif score <= 3:
        return "Medium", score
    else:
        return "High", score

# Simple keyword extractor from notes
def extract_keywords(notes, top_n=10):
    if not notes or not isinstance(notes, str):
        return []
    # lowercase, remove punctuation except intra-word hyphens
    clean = re.sub(r"[^\w\s\-]", " ", notes.lower())
    words = [w for w in clean.split() if len(w) > 3]  # filter short tokens
    c = Counter(words)
    return c.most_common(top_n)

# Diagnosis categorization map (extendable)
DIAGNOSIS_CATEGORIES = {
    "cardiac": ["heart", "cardio", "myocardial", "angina", "herz", "herzinsuffizienz"],
    "respiratory": ["pneumonia", "copd", "respiratory", "lungs", "lung", "asthma"],
    "orthopedic": ["fracture", "orthopedic", "hip", "knee", "surgery", "arthro"],
    "neurology": ["stroke", "neurolog", "seizure", "brain", "neuro"],
    "infection": ["infection", "sepsis", "infect"],
    "renal": ["renal", "kidney", "dialysis"],
    "general": []
}

def categorize_diagnosis(text):
    t = text.lower()
    for cat, keywords in DIAGNOSIS_CATEGORIES.items():
        for kw in keywords:
            if kw in t:
                return cat
    return "other"

# ---------------------------
# UI: Header & Load Data
# ---------------------------
st.title(" NeuraNest – GDPR Local Analytics (Advanced)")

lang = st.radio(" Language / Sprache:", ["Deutsch", "English"], horizontal=True)
patients = load_patients()

st.markdown("**Data storage:** Local encrypted JSON. All processing is local to this instance.")

# ---------------------------
# Left: Patient Management Panel
# ---------------------------
with st.sidebar:
    st.header(" Patient Management")
    with st.expander("➕ Add New Patient"):
        with st.form("patient_form"):
            name = st.text_input("Name", key="add_name")
            age = st.number_input("Age" if lang == "English" else "Alter", 0, 120, 1, key="add_age")
            diagnosis = st.text_input("Diagnosis" if lang == "English" else "Diagnose", key="add_diagnosis")
            admission_date = st.date_input("Admission Date" if lang == "English" else "Aufnahmedatum", key="add_admission")
            discharge_date = st.date_input("Discharge Date" if lang == "English" else "Entlassungsdatum", key="add_discharge")
            notes = st.text_area("Notes" if lang == "English" else "Notizen", key="add_notes")
            submitted = st.form_submit_button("Save Patient" if lang == "English" else "Patient speichern")

        if submitted:
            patients = load_patients()
            new_id = (max([p["id"] for p in patients]) + 1) if patients else 1
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
            st.success(f" Patient {name} saved!")
            # reload
            patients = load_patients()

    st.write("---")
    if patients:
        selected_patient = st.selectbox("👤 Select Patient:", patients, format_func=lambda x: f"{x['name']} ({x['diagnosis']})", key="patient_select")
        action = st.radio("Choose Action:", ["Generate Report", "Delete Patient", "Undo Last Delete"], horizontal=False)

        if action == "Generate Report":
            autonomy = st.radio("Autonomy Level:", [0,1,2], format_func=lambda x: ["AL0","AL1","AL2"][x], horizontal=False)
            if st.button(" Generate Report"):
                summary = generate_summary(selected_patient, autonomy, lang)
                st.subheader(" Generated Report")
                st.write(summary)
                pdf_buffer = create_pdf(summary, selected_patient['name'])
                st.download_button(" Download as PDF", data=pdf_buffer, file_name=f"{selected_patient['name']}_summary.pdf", mime="application/pdf")

        elif action == "Delete Patient":
            confirm_delete = st.checkbox(f"Confirm deletion of {selected_patient['name']}?")
            if st.button("🗑 Delete Patient") and confirm_delete:
                patients = delete_patient(selected_patient["id"])
                st.success(f" Patient {selected_patient['name']} deleted.")
                patients = load_patients()

        elif action == "Undo Last Delete":
            if st.button("↩ Restore Last Deleted Patient"):
                patients = undo_delete()
                patients = load_patients()

    else:
        st.info("No patients yet. Add a patient to begin analytics.")

st.write("---")

# ---------------------------
# Analytics Main Area
# ---------------------------
df = preprocess_df(patients)

if df.empty:
    st.warning("No patient data available. Add patients from the left panel to view analytics.")
else:
    # Filters
    st.subheader(" Filters")
    col_a, col_b, col_c = st.columns([3,3,4])
    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    if min_age == max_age:
        age_range = (min_age, max_age)
        col_a.write(f"Only one age value: **{min_age}**")
    else:
        age_range = col_a.slider(
            "Alter / Age Range",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age)
        )

    date_range = col_b.date_input("Aufnahmezeitraum / Admission Date Range", [df["admission_date"].min().date(), df["admission_date"].max().date()])
    notes_search = col_c.text_input("Suche in Notizen / Search in Notes")

    # apply filters
    filtered_df = df[
        (df["age"] >= age_range[0]) & (df["age"] <= age_range[1]) &
        (df["admission_date"] >= pd.to_datetime(date_range[0])) & (df["admission_date"] <= pd.to_datetime(date_range[1]))
    ]
    if notes_search:
        filtered_df = filtered_df[filtered_df["notes"].str.contains(notes_search, case=False, na=False)]

    st.write(f"Gefilterte Patienten / Filtered Patients: **{len(filtered_df)}**")

    # Compute KPIs
    total_patients = len(filtered_df)
    avg_los = filtered_df["length_of_stay"].mean()
    median_los = filtered_df["length_of_stay"].median()
    std_los = filtered_df["length_of_stay"].std()
    most_common_diag = filtered_df["diagnosis"].value_counts().idxmax() if total_patients else "N/A"

    # Compute readmission risk
    risk_results = filtered_df.apply(lambda r: compute_readmission_risk(r), axis=1)
    filtered_df["readmission_risk"] = [res[0] for res in risk_results]
    filtered_df["risk_score"] = [res[1] for res in risk_results]

    high_risk_count = len(filtered_df[filtered_df["readmission_risk"] == "High"])
    medium_risk_count = len(filtered_df[filtered_df["readmission_risk"] == "Medium"])
    low_risk_count = len(filtered_df[filtered_df["readmission_risk"] == "Low"])

    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Patients", total_patients)
    k2.metric("Avg LOS (days)", f"{avg_los:.1f}" if not np.isnan(avg_los) else "N/A")
    k3.metric("Median LOS (days)", f"{median_los:.1f}" if not np.isnan(median_los) else "N/A")
    k4.metric("High Risk Patients", high_risk_count)
    k5.metric("Most Common Diagnosis", most_common_diag)

    st.write("---")

    # ---------------------------
    # Top Row Charts: Diagnosis counts & Age distribution
    # ---------------------------
    col1, col2 = st.columns((2,1))
    with col1:
        st.subheader(" Häufigste Diagnosen / Most Common Diagnoses")
        diag_counts = filtered_df["diagnosis"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(8,4))
        diag_counts.plot(kind="bar", ax=ax1)
        ax1.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig1)

    with col2:
        st.subheader(" Altersverteilung / Age Distribution")
        fig2, ax2 = plt.subplots(figsize=(4,4))
        filtered_df["age"].plot(kind="hist", bins=10, ax=ax2)
        ax2.set_xlabel("Age")
        st.pyplot(fig2)

    st.write("---")

    # ---------------------------
    # LOS analytics and trends
    # ---------------------------
    st.subheader(" Length of Stay Analysis & Trends")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.markdown("**Summary statistics**")
        st.write({
            "Average LOS (days)": float(avg_los) if not np.isnan(avg_los) else None,
            "Median LOS (days)": float(median_los) if not np.isnan(median_los) else None,
            "Std Dev (days)": float(std_los) if not np.isnan(std_los) else None,
            "Min LOS": int(filtered_df["length_of_stay"].min()),
            "Max LOS": int(filtered_df["length_of_stay"].max())
        })

    with col_l2:
        st.markdown("**LOS by Diagnosis (avg)**")
        los_by_diag = filtered_df.groupby("diagnosis")["length_of_stay"].mean().sort_values(ascending=False)
        fig_l2, ax_l2 = plt.subplots(figsize=(6,3))
        los_by_diag.plot(kind="bar", ax=ax_l2)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_l2)

    # Trend by month
    st.markdown("**Admissions & Average LOS by Month**")
    monthly = filtered_df.resample("M", on="admission_date").agg({"id":"count", "length_of_stay":"mean"})
    monthly = monthly.rename(columns={"id":"admissions", "length_of_stay":"avg_los"})
    fig_m, ax_m = plt.subplots(figsize=(10,3))
    ax_m2 = ax_m.twinx()
    monthly["admissions"].plot(ax=ax_m, kind="bar", width=0.6, position=1)
    monthly["avg_los"].plot(ax=ax_m2, marker='o', linewidth=2, color='orange')
    ax_m.set_ylabel("Admissions")
    ax_m2.set_ylabel("Avg LOS")
    st.pyplot(fig_m)

    st.write("---")

    # ---------------------------
    # Diagnosis categories mapping
    # ---------------------------
    st.subheader(" Diagnosis Categories")
    filtered_df["diag_category"] = filtered_df["diagnosis"].apply(categorize_diagnosis)
    cat_counts = filtered_df["diag_category"].value_counts()
    fig_cat, ax_cat = plt.subplots(figsize=(8,3))
    cat_counts.plot(kind="bar", ax=ax_cat)
    plt.xticks(rotation=30)
    ax_cat.set_ylabel("Count")
    st.pyplot(fig_cat)

    st.write("---")

    # ---------------------------
    # Admission load & weekday patterns
    # ---------------------------
    st.subheader(" Admission Load & Weekday Patterns")
    filtered_df["admit_weekday"] = filtered_df["admission_date"].dt.day_name()
    weekday_counts = filtered_df["admit_weekday"].value_counts().reindex([
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    ]).fillna(0)
    fig_w, ax_w = plt.subplots(figsize=(8,2))
    weekday_counts.plot(kind="bar", ax=ax_w)
    ax_w.set_ylabel("Admissions")
    st.pyplot(fig_w)

    st.write("---")

    # ---------------------------
    # Notes keyword insights
    # ---------------------------
    st.subheader(" Notes Keyword Insights")
    all_notes = " ".join(filtered_df["notes"].tolist())
    keywords = extract_keywords(all_notes, top_n=20)
    if keywords:
        kw_df = pd.DataFrame(keywords, columns=["keyword","count"])
        st.table(kw_df.head(10))
    else:
        st.info("No significant keywords found in notes.")

    st.write("---")

    # ---------------------------
    # High-risk patients table and patient timeline
    # ---------------------------
    st.subheader(" High / Medium Risk Patients")
    risk_table = filtered_df.sort_values(by="risk_score", ascending=False)[["id","name","age","diagnosis","length_of_stay","readmission_risk","risk_score","admission_date","discharge_date"]]
    st.dataframe(risk_table.reset_index(drop=True), height=250)

    st.write("---")
    st.subheader(" Patient Journey Timeline")
    patient_selection = st.selectbox("Select patient for timeline:", filtered_df.to_dict("records"), format_func=lambda x: f"{x['name']} ({x['diagnosis']})")
    # Show timeline
    st.markdown(f"**{patient_selection['name']}** — Admission: {patient_selection['admission_date'].date()} — Discharge: {patient_selection['discharge_date'].date()} — LOS: {patient_selection['length_of_stay']} days")
    st.write("Notes:")
    st.write(patient_selection["notes"])

    # ---------------------------
    # Export filtered data
    # ---------------------------
    st.write("---")
    st.subheader(" Export / Download")
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(" Download Filtered Data (CSV)", data=csv_bytes, file_name="filtered_patients.csv", mime="text/csv")
    # Also allow exporting anonymized summary
    anon_df = filtered_df.copy()
    anon_df = anon_df.drop(columns=["name","notes"])
    st.download_button(" Download Anonymized Summary (CSV)", data=anon_df.to_csv(index=False).encode("utf-8"), file_name="anonymized_summary.csv", mime="text/csv")

# End of file
