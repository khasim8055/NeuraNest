# demo_seed.py
# ================================================================
# NeuraCare — Demo Database Seeder
# ================================================================
# Run this ONCE to populate a fresh database with realistic
# German clinical data for demos and pilots.
#
# Usage:
#   python demo_seed.py
#
# WARNING: This adds patients to the existing database.
#          Run only on a fresh install or demo machine.
# ================================================================

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.core.database import init_db
from app.core.auth     import setup_first_admin, hash_password

# ── Demo patient data — realistic German clinical cases ──────────
DEMO_PATIENTS = [
    {
        "name":           "Hans Müller",
        "age":            72,
        "date_of_birth":  "1952-03-14",
        "diagnosis":      "Herzinsuffizienz",
        "icd10_code":     "I50.0",
        "admission_date": "2025-11-03",
        "discharge_date": "2025-11-17",
        "notes":          (
            "Patient wurde mit zunehmender Dyspnoe und peripheren Ödemen aufgenommen. "
            "Echokardiographie zeigte EF 35%. Unter medikamentöser Einstellung mit "
            "Furosemid und Bisoprolol deutliche Besserung. "
            "Bei Entlassung kompensiert, keine Ruhedyspnoe."
        ),
        "medication":     "Furosemid 40mg 1-0-0, Bisoprolol 5mg 1-0-0, Ramipril 5mg 1-0-0",
        "physician_name": "Dr. Schmidt",
        "followup_date":  "2025-12-01",
    },
    {
        "name":           "Ingrid Weber",
        "age":            68,
        "date_of_birth":  "1956-07-22",
        "diagnosis":      "COPD Exazerbation",
        "icd10_code":     "J44.1",
        "admission_date": "2025-12-01",
        "discharge_date": "2025-12-08",
        "notes":          (
            "Patientin mit bekannter COPD GOLD III, aufgenommen wegen akuter Exazerbation "
            "mit Zunahme der Dyspnoe und produktivem Husten. "
            "Prednisolon-Stoßtherapie und Antibiotikum (Amoxicillin) angesetzt. "
            "Spirometrie bei Entlassung FEV1 42% Soll."
        ),
        "medication":     "Tiotropium 18mcg 0-0-1, Salbutamol bei Bedarf, Prednisolon ausgeschlichen",
        "physician_name": "Dr. Schmidt",
        "followup_date":  "2025-12-22",
    },
    {
        "name":           "Klaus Bauer",
        "age":            81,
        "date_of_birth":  "1943-11-05",
        "diagnosis":      "Ischämischer Schlaganfall",
        "icd10_code":     "I63.9",
        "admission_date": "2026-01-15",
        "discharge_date": "2026-02-01",
        "notes":          (
            "Patient mit akutem Mediainfarkt links, stationäre Aufnahme über Stroke Unit. "
            "Lyse nicht möglich da Zeitfenster überschritten. "
            "Rehabilitative Maßnahmen eingeleitet, Logopädie und Physiotherapie. "
            "Bei Entlassung leichte Wortfindungsstörungen, Mobilisation mit Hilfe möglich."
        ),
        "medication":     "ASS 100mg 1-0-0, Atorvastatin 40mg 0-0-1, Ramipril 2.5mg 1-0-0",
        "physician_name": "Dr. Hoffmann",
        "followup_date":  "2026-02-15",
    },
    {
        "name":           "Maria Schmidt",
        "age":            55,
        "date_of_birth":  "1969-04-18",
        "diagnosis":      "Diabetes mellitus Typ 2, entgleist",
        "icd10_code":     "E11.9",
        "admission_date": "2026-01-20",
        "discharge_date": "2026-01-25",
        "notes":          (
            "Patientin mit entgleistem Diabetes mellitus Typ 2, BZ bei Aufnahme 420 mg/dl. "
            "Insulineinstellung unter stationären Bedingungen optimiert. "
            "HbA1c 10.2% bei Aufnahme. Diabetesschulung durchgeführt. "
            "Bei Entlassung BZ-Werte im Zielbereich 80-140 mg/dl."
        ),
        "medication":     "Insulin glargin 20 IE zur Nacht, Metformin 1000mg 1-0-1",
        "physician_name": "Dr. Hoffmann",
        "followup_date":  "2026-02-10",
    },
    {
        "name":           "Franz Wagner",
        "age":            76,
        "date_of_birth":  "1948-09-30",
        "diagnosis":      "Akuter Myokardinfarkt (NSTEMI)",
        "icd10_code":     "I21.4",
        "admission_date": "2026-02-03",
        "discharge_date": "2026-02-14",
        "notes":          (
            "Patient mit NSTEMI, Troponin-Anstieg auf 8.4 ng/ml. "
            "Koronarangiographie: 3-Gefäßerkrankung, RIVA-Stenose 80%. "
            "PCI mit Stentimplantation in RIVA erfolgreich. "
            "Kardiale Rehabilitation empfohlen. EF post-interventionell 50%."
        ),
        "medication":     "ASS 100mg 1-0-0, Clopidogrel 75mg 1-0-0, Bisoprolol 2.5mg 1-0-0, Atorvastatin 80mg 0-0-1",
        "physician_name": "Dr. Schmidt",
        "followup_date":  "2026-03-03",
    },
    {
        "name":           "Anna Hoffmann",
        "age":            63,
        "date_of_birth":  "1961-12-08",
        "diagnosis":      "Pneumonie",
        "icd10_code":     "J18.9",
        "admission_date": "2026-02-10",
        "discharge_date": "2026-02-17",
        "notes":          (
            "Patientin mit ambulant erworbener Pneumonie rechter Unterlappen. "
            "CRP 220 mg/l, Leukozytose 18.000/µl bei Aufnahme. "
            "Antibiotische Therapie mit Amoxicillin/Clavulansäure i.v., "
            "nach 3 Tagen auf orale Therapie umgestellt. "
            "Fieber sistiert Tag 2, radiologische Besserung bei Entlassung."
        ),
        "medication":     "Amoxicillin/Clavulansäure 875/125mg 1-0-1 für 7 Tage",
        "physician_name": "Dr. Meier",
        "followup_date":  "2026-03-01",
    },
    {
        "name":           "Peter Schulz",
        "age":            45,
        "date_of_birth":  "1979-06-15",
        "diagnosis":      "Appendizitis",
        "icd10_code":     "K37",
        "admission_date": "2026-03-01",
        "discharge_date": "2026-03-04",
        "notes":          (
            "Patient mit akuter Appendizitis, laparoskopische Appendektomie komplikationslos. "
            "OP-Dauer 35 Minuten, kein intraabdomineller Befund. "
            "Postoperativer Verlauf unauffällig, Kostaufbau problemlos. "
            "Wundheilung primär, Faden entfernt."
        ),
        "medication":     "Ibuprofen 400mg bei Bedarf, Metamizol 500mg bei starken Schmerzen",
        "physician_name": "Dr. Meier",
        "followup_date":  "2026-03-18",
    },
    {
        "name":           "Sabine Klein",
        "age":            70,
        "date_of_birth":  "1954-02-28",
        "diagnosis":      "Chronische Niereninsuffizienz Stadium 3",
        "icd10_code":     "N18.3",
        "admission_date": "2026-03-10",
        "discharge_date": "2026-03-22",
        "notes":          (
            "Patientin mit progredienter Niereninsuffizienz, Kreatinin 2.8 mg/dl bei Aufnahme. "
            "Ursache: hypertensive Nephropathie. Blutdruckeinstellung optimiert. "
            "Nephrologisches Konsil: Dialysepflichtigkeit noch nicht absehbar. "
            "Ernährungsberatung durchgeführt, eiweiß- und kaliumarme Kost empfohlen."
        ),
        "medication":     "Ramipril 5mg 1-0-0, Amlodipin 5mg 1-0-0, Darbepoetin alfa s.c. wöchentlich",
        "physician_name": "Dr. Hoffmann",
        "followup_date":  "2026-04-05",
    },
    {
        "name":           "Werner Braun",
        "age":            58,
        "date_of_birth":  "1966-08-12",
        "diagnosis":      "Hüft-TEP links",
        "icd10_code":     "Z96.6",
        "admission_date": "2026-03-05",
        "discharge_date": "2026-03-14",
        "notes":          (
            "Patient mit Koxarthrose links, elektive Implantation einer zementfreien Hüft-TEP. "
            "OP komplikationslos, intraoperativer Blutverlust 320ml. "
            "Physiotherapie ab Tag 1 postoperativ, Vollbelastung ab Tag 2. "
            "Entlassung in Anschlussheilbehandlung (AHB)."
        ),
        "medication":     "Rivaroxaban 10mg 1-0-0 für 35 Tage, Ibuprofen 600mg 1-1-1",
        "physician_name": "Dr. Meier",
        "followup_date":  "2026-04-05",
    },
    {
        "name":           "Elisabeth Vogel",
        "age":            84,
        "date_of_birth":  "1940-01-19",
        "diagnosis":      "Schenkelhalsfraktur rechts",
        "icd10_code":     "S72.0",
        "admission_date": "2026-02-20",
        "discharge_date": "2026-03-08",
        "notes":          (
            "Patientin nach Sturz mit medialer Schenkelhalsfraktur rechts. "
            "Hemi-Endoprothese implantiert. Vorerkrankungen: Osteoporose, Hypertonie, VHF. "
            "Postoperativ geriatrisches Assessment: Barthel-Index 45/100. "
            "Entlassung in geriatrische Rehabilitationseinrichtung. Sturzprophylaxe eingeleitet."
        ),
        "medication":     "Rivaroxaban 10mg, Bisoprolol 2.5mg, Alendronsäure 70mg wöchentlich",
        "physician_name": "Dr. Schmidt",
        "followup_date":  "2026-04-01",
    },
    {
        "name":           "Thomas Fischer",
        "age":            49,
        "date_of_birth":  "1975-10-03",
        "diagnosis":      "Bandscheibenvorfall L4/L5",
        "icd10_code":     "M51.1",
        "admission_date": "2026-01-08",
        "discharge_date": "2026-01-13",
        "notes":          (
            "Patient mit lumbaler Radikulopathie L5 rechts bei BSV L4/L5. "
            "Konservative Therapie: Physiotherapie, Schmerzmedikation, Wärmebehandlung. "
            "MRT bestätigte BSV ohne Myelonkompression. "
            "Rückenschule empfohlen, OP-Indikation derzeit nicht gegeben."
        ),
        "medication":     "Ibuprofen 800mg 1-1-1, Diazepam 5mg zur Nacht (max. 5 Tage)",
        "physician_name": "Dr. Meier",
        "followup_date":  "2026-02-08",
    },
    {
        "name":           "Gisela Neumann",
        "age":            66,
        "date_of_birth":  "1958-05-25",
        "diagnosis":      "Vorhofflimmern, tachykard",
        "icd10_code":     "I48.0",
        "admission_date": "2025-12-14",
        "discharge_date": "2025-12-19",
        "notes":          (
            "Patientin mit neu aufgetretenem tachykardem Vorhofflimmern, HF 140/min bei Aufnahme. "
            "Kardioversion nach 48h erfolgreich, Sinusrhythmus wiederhergestellt. "
            "TEE vor Kardioversion: kein Thrombus. "
            "Langzeit-EKG bei Entlassung geplant, orale Antikoagulation eingeleitet."
        ),
        "medication":     "Apixaban 5mg 1-0-1, Bisoprolol 5mg 1-0-0, Ramipril 5mg 1-0-0",
        "physician_name": "Dr. Hoffmann",
        "followup_date":  "2026-01-14",
    },
]


def seed_demo_database():
    """Seed the database with demo patients."""
    print("=" * 60)
    print("  NeuraCare — Demo Database Seeder")
    print("=" * 60)
    print()

    # Ensure DB exists
    data_dir = Path(__file__).parent / "app" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    init_db()

    db_path = data_dir / "neuranest.db"
    conn = sqlite3.connect(str(db_path))

    # Create admin if not exists
    existing_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    if existing_users == 0:
        ok, msg = setup_first_admin(
            username="admin",
            password="NeuraCare2024!",
            full_name="Administrator",
        )
        if ok:
            print("✓ Admin account created")
            print("  Username: admin")
            print("  Password: NeuraCare2024!")
        else:
            # Create directly if setup_first_admin fails
            hashed = hash_password("NeuraCare2024!")
            conn.execute(
                "INSERT OR IGNORE INTO users "
                "(username, password_hash, full_name, role) "
                "VALUES (?, ?, ?, ?)",
                ("admin", hashed, "Administrator", "admin")
            )
            conn.commit()
            print("✓ Admin account created (direct)")
    else:
        print("✓ Admin account already exists")

    print()

    # Check existing patients
    existing = conn.execute(
        "SELECT COUNT(*) FROM patients WHERE is_deleted=0"
    ).fetchone()[0]

    if existing > 0:
        print(f"ℹ  Database already has {existing} patient(s).")
        ans = input("   Add demo patients anyway? (y/n): ").strip().lower()
        if ans != "y":
            print("   Skipped. No changes made.")
            conn.close()
            return

    print("Adding 12 demo patients...")
    print()

    # Get admin user id
    admin = conn.execute(
        "SELECT id FROM users WHERE username='admin'"
    ).fetchone()
    admin_id = admin[0] if admin else None

    added = 0
    for p in DEMO_PATIENTS:
        try:
            conn.execute(
                """INSERT INTO patients
                   (name, age, date_of_birth, diagnosis, icd10_code,
                    admission_date, discharge_date, notes,
                    medication, physician_name, followup_date, created_by)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p["name"], p["age"],
                    p.get("date_of_birth"),
                    p["diagnosis"], p.get("icd10_code"),
                    p["admission_date"], p["discharge_date"],
                    p.get("notes", ""),
                    p.get("medication", ""),
                    p.get("physician_name", ""),
                    p.get("followup_date"),
                    admin_id,
                )
            )
            added += 1
            print(f"  ✓ {p['name']} — {p['diagnosis']}")
        except sqlite3.Error as e:
            print(f"  ✗ {p['name']}: {e}")

    conn.commit()
    conn.close()

    print()
    print("=" * 60)
    print(f"  Done. {added} patients added to demo database.")
    print()
    print("  Login credentials:")
    print("  Username: admin")
    print("  Password: NeuraCare2024!")
    print("=" * 60)


if __name__ == "__main__":
    seed_demo_database()
