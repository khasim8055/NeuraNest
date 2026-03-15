# NEURANEST_CONTEXT.md
# Paste this file at the start of every Claude session with "NeuraCare session"
# Last updated: 75-day build plan locked

---

## PRODUCT
**Name:** NeuraCare
**Type:** Offline desktop clinical documentation assistant
**Tagline:** Privacy-first discharge letter generation for European private clinics

**Core value proposition:**
A doctor opens the app, adds a patient, generates a professional discharge letter
(Arztbrief) in 30 seconds, downloads as PDF. 100% offline. Zero data leaves the device.

---

## TARGET USERS
- Primary: Private GP and specialist practices, 1–10 doctors (DACH region)
- Secondary: Private clinics in Switzerland, UAE/GCC
- NOT: Public hospitals, NHS, US market (for now)

---

## CURRENT STACK (MVP — Streamlit prototype)
- Language: Python 3.x
- UI: Streamlit (to be replaced with PyQt6)
- Storage: JSON encrypted with Fernet (to be replaced with SQLite)
- PDF: ReportLab (keep)
- Encryption: Fernet / cryptography (keep)
- Charts: Matplotlib (keep, 5 charts only)
- LLM: None yet (to add: Ollama + Mistral 7B)

---

## CURRENT STATE
- File: neuranest.py (955 lines, 25 functions)
- Build progress: 38% of total product / 72% of core clinical logic
- Status: Working Streamlit prototype, not yet a shippable desktop product

---

## BUGS ALREADY FIXED (do not reintroduce)
1. CHART_COLORS["navy"] KeyError — added "navy" alias to dict
2. PDF download broken — create_pdf() now returns bytes not BytesIO
3. CSV download broken — df_to_csv_bytes() converts Timestamps before encoding
4. AL1 missing admission date — still needs one-line fix in dummy_discharge_summary
5. AL0 hardcoded "stable" — still needs conditional check against notes flags

---

## FEATURES LOCKED IN (build these, nothing else)

### CORE — Patient Management
- [x] Add / edit / delete patient record
- [x] Undo last delete
- [x] Patient list with risk colour indicators
- [x] Fernet encryption on database file
- [ ] SQLite backend (replace JSON)
- [ ] User login with bcrypt (Admin + Doctor roles)
- [ ] Real audit log to DB (every action: user, action, patient_id, timestamp)
- [ ] Input validation: discharge date cannot be before admission date
- [ ] Key backup warning on first launch

### PATIENT DATA MODEL — new fields to add
- [x] Name, age, diagnosis, admission date, discharge date, notes
- [ ] ICD-10 code
- [ ] Medication at discharge
- [ ] Responsible physician name
- [ ] Follow-up date

### DOCUMENT GENERATION
- [x] AL0 — bullet summary (fix hardcoded "stable" text)
- [x] AL1 — standard paragraph (fix missing admission date)
- [x] AL2 — extended with full notes
- [ ] AL3 — real LLM generation via Ollama + Mistral 7B
- [ ] Arztbrief PDF template (DIN 5008 format)
- [ ] New fields in PDF output (medication, physician, ICD-10, follow-up)

### ANALYTICS — Practice Insights (secondary tab)
- [x] Diagnosis frequency chart (keep)
- [x] Age distribution chart (keep)
- [x] LOS by diagnosis chart (keep)
- [x] Monthly admissions + avg LOS trend (keep — add 60-day data guard)
- [x] Readmission risk pie chart (keep)
- [x] Risk breakdown panel with progress bars (new — replaces removed charts)
- [ ] Minimum data guard on monthly chart (only render if 60+ days of data)

### EXPORT
- [x] PDF discharge summary download
- [x] Filtered data CSV export
- [x] Anonymised summary CSV export

### DESKTOP PRODUCT
- [ ] PyQt6 main window (3-panel layout)
- [ ] PyInstaller .exe packaging for Windows 10/11
- [ ] Inno Setup installer (NeuraCare_Setup.exe)
- [ ] Demo database with 5 test patients

---

## FEATURES PERMANENTLY REMOVED (do not rebuild)
- ~~Weekday admission pattern chart~~ — hospital ops metric, not clinical
- ~~Notes keyword frequency table~~ — no NLP context = noise, not insight
- ~~Diagnosis category chart~~ — duplicate of diagnosis frequency chart
- ~~Counter / re imports~~ — removed with the features above

---

## FEATURES DEFERRED TO V2 (not in 75-day build)
- Voice input / ambient scribing
- HL7 / FHIR export
- Multi-clinic cloud sync
- Mobile app
- Arabic language support
- Polish / Czech localisation
- EHR integration

---

## 75-DAY BUILD SCHEDULE

| Days    | Milestone                                      | Status  |
|---------|------------------------------------------------|---------|
| 1–5     | PyCharm + GitHub setup, SQLite schema          | TODO    |
| 6–15    | User auth (bcrypt), real audit log, key backup | TODO    |
| 16–28   | PyQt6 desktop UI (3 panels, patient form)      | TODO    |
| 29–42   | Document generation + Ollama AL3 + Arztbrief   | TODO    |
| 43–53   | Analytics dashboard (5 charts) + export        | TODO    |
| 54–63   | PyInstaller packaging + Windows .exe           | TODO    |
| 64–75   | Buffer, polish, demo database, clinic outreach | TODO    |

---

## TOOL STACK FOR BUILDING
- Editor: PyCharm Community (free)
- AI co-founder: Claude (this conversation)
- Version control: GitHub Desktop (free, no command line)
- LLM engine: Ollama + Mistral 7B (free, offline)
- NOT using: Cursor, ChatGPT, Copilot, Electron, Flutter

---

## MARKET STRATEGY
- Beachhead: DACH private clinics (Germany, Austria, Switzerland)
- Price: €99/month per practice (1–3 doctors), €199/month (4–8 doctors)
- First goal: 3 free pilots → 3 paying customers → fundable
- Funding path: EXIST Gründerstipendium (non-dilutive, up to €150k) → pre-seed
- Regulatory path: DiGA provisional listing after 3 paying clinics + CE marking
- Secondary markets: UAE/GCC (parallel), India (v2), CEE (v2)
- Avoid: USA (HIPAA + competition), public hospital tenders (too slow)

---

## COMPETITIVE POSITION
- Nuance DAX: Cloud-only, €600–800/month, launched DACH Oct 2025 — your biggest threat
- Abridge: Cloud-only, English-only, no German discharge letter
- Suki: Cloud-only, English-only
- NeuraCare advantage: offline + GDPR-native + German Arztbrief + €99 price + risk scoring

---

## HOW TO USE THIS FILE
Start every Claude session with:
"NeuraCare session — [what you are building today]"
Then paste this file.
Claude will have full context in 10 seconds.
Update the Status column in the schedule as milestones complete.
