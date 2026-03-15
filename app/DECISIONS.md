# DECISIONS.md
# Plain English explanation of every file and function
# Update this every time a new function is added
# This file saves you hours of debugging at midnight

---

## Project Structure

```
neuranest_desktop/
├── core/
│   ├── database.py       — SQLite connection, encryption, schema init
│   ├── patients.py       — all patient CRUD operations (add/edit/delete/list)
│   ├── auth.py           — user login, bcrypt password check, session
│   ├── audit.py          — write every action to audit_log table
│   ├── risk.py           — readmission risk scoring (rule-based)
│   └── pdf.py            — discharge letter PDF generation (ReportLab)
├── ui/
│   ├── main_window.py    — PyQt6 main 3-panel window
│   ├── login_dialog.py   — login screen shown before main window
│   ├── patient_form.py   — add/edit patient form dialog
│   ├── patient_list.py   — left panel: scrollable patient list
│   ├── patient_detail.py — centre panel: hero card, notes, timeline
│   ├── summary_panel.py  — right panel: generate summary, download PDF
│   └── analytics_tab.py  — Practice Insights tab with 5 charts
├── assets/
│   └── logo.png          — clinic logo placeholder for PDF header
├── tests/
│   ├── test_patients.py  — tests for patient CRUD
│   ├── test_risk.py      — tests for risk scoring
│   ├── test_pdf.py       — tests for PDF generation
│   └── test_auth.py      — tests for login / password hashing
├── data/
│   └── neuranest.db.enc  — encrypted SQLite database (auto-created)
├── schema.sql            — database table definitions
├── requirements.txt      — Python dependencies
├── generate_key.py       — one-time key generation (delete after use)
├── secret.key            — encryption key (NEVER commit to GitHub)
├── DECISIONS.md          — this file
└── NEURANEST_CONTEXT.md  — product context for Claude sessions
```

---

## core/database.py

| Function / Class | What it does |
|---|---|
| `load_key()` | Reads secret.key file, returns Fernet object. Raises clear error if key is missing. |
| `encrypt_db()` | Takes the plain SQLite .db file, encrypts it to .db.enc, deletes the plain file. Called when closing the app. |
| `decrypt_db()` | Takes the .db.enc file, decrypts it to plain .db file. Called when opening the app. |
| `Database` class | Context manager — decrypts on enter, re-encrypts on exit. Use with `with Database() as db:` |
| `Database.fetchall()` | Run a SELECT query, return list of dicts (column name as key) |
| `Database.fetchone()` | Run a SELECT query, return one dict or None |

---

## schema.sql — 4 tables

| Table | Purpose |
|---|---|
| `users` | Login accounts. Stores bcrypt password hash, role (admin/doctor). |
| `patients` | Full patient record. 7 original fields + 4 new clinical fields. LOS is auto-computed. |
| `audit_log` | Every action logged with user, timestamp, patient. GDPR Article 30. Never deleted. |
| `app_config` | Key-value settings (clinic name, language, first run flag). |

---

## patients table — all fields

| Field | Type | Required | Notes |
|---|---|---|---|
| id | INTEGER | auto | Primary key |
| name | TEXT | yes | Patient full name |
| age | INTEGER | yes | 0–130, validated |
| date_of_birth | TEXT | no | Optional, YYYY-MM-DD |
| diagnosis | TEXT | yes | Free text |
| icd10_code | TEXT | no | e.g. I50.0 |
| admission_date | TEXT | yes | YYYY-MM-DD |
| discharge_date | TEXT | yes | YYYY-MM-DD, must be >= admission_date |
| notes | TEXT | no | Clinical notes |
| medication | TEXT | no | Medications at discharge — NEW |
| physician_name | TEXT | no | Responsible physician — NEW |
| followup_date | TEXT | no | Follow-up appointment date — NEW |
| length_of_stay | INTEGER | auto | Computed: discharge - admission days |
| created_at | TEXT | auto | Timestamp of record creation |
| updated_at | TEXT | auto | Timestamp of last edit |
| created_by | INTEGER | no | Foreign key to users.id |
| is_deleted | INTEGER | auto | 0=active, 1=soft deleted (enables undo) |

---

## Key rules (do not break these)

1. Never open DB_FILE directly — always use `with Database() as db:`
2. Never store plain text passwords — always bcrypt hash
3. Every patient action must write a row to audit_log
4. Discharge date must always be >= admission date — enforced at DB level AND form level
5. secret.key never goes to GitHub — it is in .gitignore
