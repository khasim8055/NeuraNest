# NeuraCare

A clinical documentation tool for German private practices. Offline. Simple. Built because I got frustrated reading about how much time doctors waste on paperwork.

**[github.com/khasim8055/NeuraNest](https://github.com/khasim8055/NeuraNest)**

---

## The problem that started this

I was reading a Marburger Bund survey for a university project. One number stopped me: German senior physicians spend only 2.5 hours per day on actual patient care. The other 5+ hours? Documentation, bureaucracy, writing letters that follow the same structure every single time.

I looked at what software existed to fix this. Nuance DAX: €600/month, requires internet, sends patient data to Microsoft's servers in the US. Suki, Abridge: same story. Under GDPR, many German clinics cannot legally use any of them. And even if they could, €600/month is not a small practice budget.

I am a healthcare AI student, not a doctor. But this felt like a problem I could actually do something about. So I started building.

---

## What NeuraCare does

It generates a complete Arztbrief (discharge letter) in about 30 seconds. The doctor opens the app, selects a patient, picks a detail level, and gets a formatted clinical letter — in German or English — ready to download as a PDF.

Everything runs locally. No internet required. No data leaves the machine. The patient database is encrypted on the clinic's own computer. I did not make this offline because it sounded good in a pitch. I made it offline because it was the only way a German private practice could legally use it.

The app has four autonomy levels:
- **AL0** — structured bullet summary, fast handover
- **AL1** — standard paragraph letter, most common use
- **AL2** — extended letter with full notes and risk assessment
- **AL3** — AI-generated via Ollama + Mistral, runs entirely on the clinic's machine

---

## What I actually built

I started this with no desktop development experience. I had never used PyQt6. I had never packaged a Python app into a .exe. I had never built anything that needed to work on someone else's computer without Python installed.

Thirteen sessions later, here is what exists:

| File | What it does |
|---|---|
| `app/core/database.py` | SQLite connection with Fernet encryption |
| `app/core/auth.py` | bcrypt login, role-based sessions (admin / doctor) |
| `app/core/audit.py` | Every action logged — GDPR Article 30 compliance |
| `app/core/patients.py` | Patient CRUD with soft delete and restore |
| `app/core/risk.py` | Rule-based readmission risk scoring, fully explainable |
| `app/core/letters.py` | Letter generation engine — AL0 through AL2 |
| `app/core/ollama_client.py` | Ollama API client for AL3 local AI generation |
| `app/core/pdf_exporter.py` | ReportLab PDF with patient table, risk callout, GDPR footer |
| `app/core/analytics.py` | Data layer for five dashboard charts |
| `app/core/csv_export.py` | Anonymised export and audit log CSV for compliance |
| `app/ui/main_window.py` | PyQt6 three-panel layout — the whole app shell |
| `app/ui/patient_form.py` | Add and edit patients with inline validation |
| `app/ui/letter_panel.py` | Letter generation UI with level selector and preview |
| `app/ui/analytics_panel.py` | Matplotlib charts embedded in PyQt6 |
| `app/ui/settings_panel.py` | Change password, create users, manage accounts |

By the end: 14 modules, around 6,500 lines, 100 passing pytest tests, and a Windows .exe that opens on a machine with no Python installed.

---

## The honest account of building with Claude

I used Claude in claude.ai — not an API, not Cursor, just the chat interface. Every session I uploaded a zip of the project and started fresh. Here is what I actually learned, including the parts that did not work.

**What worked well**

The most important thing I did was maintain a `NEURANEST_CONTEXT.md` file. A running document tracking every decision, every bug fixed, every feature permanently removed and why. At the start of each session I uploaded the full project zip and Claude read that file. It knew why we chose SQLite over PostgreSQL, why we removed the weekday chart, why AL3 was deferred. Without that file the sessions would have been chaotic.

One module per session, tested before committing. This sounds obvious but it matters more when working with an AI. If I committed broken code, the next session started with Claude trying to understand a broken codebase. Every commit in this repo is a green test run.

Pasting exact terminal output rather than describing errors. "It doesn't work" tells Claude nothing. The full stack trace tells it everything. This alone saved me hours of back and forth.

Writing the test file before the implementation. For `auth.py` and `patients.py` I asked Claude to write what the tests should check first. Then we wrote the code to pass them. It caught three real bugs I would not have found otherwise.

**What did not work**

Trying to build two connected modules in one session always produced wiring bugs. I tried this twice, both times ended up with import errors and mismatched function signatures. One module per session became a rule after that.

Long conversations where I did not upload a fresh project state. After about 50 messages Claude's suggestions started drifting — small inconsistencies in naming, forgetting that a function signature had changed three sessions ago. The zip-upload workflow fixed this completely.

**The hardware constraint that shaped everything**

My laptop has 8GB of soldered RAM. Not upgradeable. This is not a footnote — it changed every technical decision. Mistral 3B instead of 7B. VS Code instead of PyCharm. SQLite instead of PostgreSQL. PyInstaller instead of Electron. Telling Claude this constraint upfront on Day 1 meant every suggestion it made was hardware-aware. AL3 runs but is slow on this machine. On a 16GB clinic machine it would be noticeably faster.

---

## Five things I would tell someone starting a similar project

**Keep a CONTEXT.md and update it every session.** It is your AI's memory between conversations. Without it you will repeat decisions and relitigate choices you already made.

**End every session with a zip and a git commit.** Not because of version control hygiene — because it gives you a clean starting point next time. I called this the zip-upload workflow and it was the single biggest productivity improvement I made.

**Schema first, everything else second.** I wrote `schema.sql` on Day 1. Every module that came after it — auth, audit, patients — referenced that schema. Claude never had to guess the data model. When the schema was wrong we fixed it in one place.

**Hardware constraints are product constraints.** If you tell your AI tools what you are actually working with, the advice gets dramatically better. Vague questions get generic answers. Specific constraints get specific solutions.

**PyInstaller will fail the first four times.** This is not a skill problem. It is a dependency discovery problem. Each failure gives you one missing hidden import. Patch the spec file, rebuild. By the fifth build it works. Budget time for this.

---

## What I would do differently

I would add the help documentation earlier. The app works but a doctor handed it alone would not know where to start. I built the settings panel on Day 13. It should have been Day 7.

AL3 is wired correctly but slow on 8GB RAM. The architecture is right — Ollama runs locally, the prompt is clinically structured, the output is good German. But on this machine it sometimes hangs. On a 16GB machine it runs fine. I shipped it anyway because the code is correct even if the hardware is the bottleneck.

---

## What inspires me

Linear. I use it daily. The thing I tried to copy from it into NeuraCare is that removing a feature is a product decision as important as adding one. I cut the weekday chart, the full CSV export, the keyword frequency table. Each one felt wrong to cut at the time. Each one was the right call. A product that does three things well is more useful than one that does ten things adequately.

---

## Stack

Python 3.13 · PyQt6 · SQLite · Fernet encryption · bcrypt · ReportLab · Matplotlib · Ollama + Mistral · PyInstaller · pytest

---

## Running it

```bash
# Clone and install
git clone https://github.com/khasim8055/NeuraNest.git
cd NeuraNest
pip install -r requirements.txt

# First run — creates database and default admin
python main.py
# Username: admin  |  Password: NeuraCare2024!

# Seed demo patients (optional)
python demo_seed.py

# Build Windows .exe
.\build.bat
```

---

## License

MIT — see [LICENSE](LICENSE)

---

*Khasim Bin Saleh · Healthcare AI · SRH Berlin*
