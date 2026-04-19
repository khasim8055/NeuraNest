# app/core/ollama_client.py
# ================================================================
# NeuraCare — Ollama Local AI Client
# ================================================================
# Communicates with the local Ollama server via HTTP API.
# Ollama must be running: ollama serve
# Model must be pulled:   ollama pull mistral
#
# No data leaves the device — 100% offline inference.
# ================================================================

import json
import urllib.request
import urllib.error
from datetime import datetime

# ── Ollama settings ───────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "phi3:mini"
OLLAMA_TIMEOUT  = 120  # seconds — Mistral 3B takes ~15-25s on 8GB RAM


# ================================================================
# CONNECTION CHECK
# ================================================================

def is_ollama_running() -> bool:
    """
    Check if Ollama server is running and accessible.
    Returns True if running, False if not.
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_model_available(model: str = OLLAMA_MODEL) -> bool:
    """
    Check if the specified model is pulled and available.
    Returns True if available.
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(model in m for m in models)
    except Exception:
        return False


# ================================================================
# CORE GENERATION FUNCTION
# ================================================================

def generate(
    prompt: str,
    model: str = OLLAMA_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> tuple[bool, str, str]:
    """
    Send a prompt to Ollama and return the response.

    Args:
        prompt:      The full prompt text
        model:       Model name (default: mistral)
        temperature: 0.0=deterministic, 1.0=creative (0.3 for clinical)
        max_tokens:  Maximum response length

    Returns:
        (True,  response_text, "")     on success
        (False, "",            error)  on failure
    """
    if not is_ollama_running():
        return False, "", (
            "Ollama is not running. "
            "Please start it with: ollama serve"
        )

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":   temperature,
            "num_predict":   max_tokens,
            "stop":          ["</letter>", "---END---"],
        },
    }

    try:
        data     = json.dumps(payload).encode("utf-8")
        req      = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            result   = json.loads(resp.read())
            response = result.get("response", "").strip()

            if not response:
                return False, "", "Empty response from model."

            return True, response, ""

    except urllib.error.URLError as e:
        return False, "", f"Cannot connect to Ollama: {str(e)}"
    except TimeoutError:
        return False, "", (
            f"Ollama timed out after {OLLAMA_TIMEOUT}s. "
            "Try closing other applications to free RAM."
        )
    except Exception as e:
        return False, "", f"Ollama error: {str(e)}"


# ================================================================
# AL3 PROMPT BUILDERS
# ================================================================

def build_arztbrief_prompt(patient: dict, lang: str = "Deutsch") -> str:
    """
    Build a clinical prompt for AL3 Arztbrief generation.
    The prompt instructs Mistral to write a professional German
    discharge letter following DIN 5008 conventions.
    """
    name       = patient.get("name", "—")
    age        = patient.get("age", "—")
    diag       = patient.get("diagnosis", "—")
    icd10      = patient.get("icd10_code", "") or ""
    adm        = patient.get("admission_date", "—")
    dis        = patient.get("discharge_date", "—")
    los        = patient.get("length_of_stay", 0) or 0
    notes      = patient.get("notes", "") or ""
    medication = patient.get("medication", "") or ""
    physician  = patient.get("physician_name", "") or ""
    followup   = patient.get("followup_date", "") or ""

    from app.core.risk import compute_risk, get_risk_explanation
    risk, score = compute_risk(patient)
    risk_reasons = get_risk_explanation(patient)

    today = datetime.now().strftime("%d.%m.%Y")

    if lang == "Deutsch":
        prompt = f"""Du bist ein erfahrener Klinikarzt und schreibst einen professionellen Arztbrief auf Deutsch.
Schreibe einen vollständigen, medizinisch korrekten Arztbrief im DIN-5008-Format.

PATIENTENDATEN:
- Name: {name}
- Alter: {age} Jahre
- Hauptdiagnose: {diag}{f" ({icd10})" if icd10 else ""}
- Aufnahmedatum: {adm}
- Entlassungsdatum: {dis}
- Verweildauer: {los} Tage
- Behandelnder Arzt: {physician if physician else "nicht angegeben"}
- Wiederaufnahmerisiko: {risk} (Score {score}/6)

KLINISCHE INFORMATIONEN:
{notes if notes else "Keine detaillierten Notizen vorhanden."}

MEDIKATION BEI ENTLASSUNG:
{medication if medication else "Keine Medikation dokumentiert."}

NACHSORGETERMIN: {followup if followup else "Ambulante Kontrolle empfohlen"}

ANWEISUNGEN:
- Schreibe einen professionellen Arztbrief auf Deutsch
- Verwende medizinische Fachterminologie
- Struktur: Anrede, Anamnese, Verlauf, Medikation, Empfehlung, Grußformel
- Maximale Länge: 400 Wörter
- Kein Markdown, keine Formatierungszeichen
- Datum: {today}

Beginne direkt mit dem Arztbrief:"""

    else:
        prompt = f"""You are an experienced hospital physician writing a professional discharge letter.
Write a complete, medically accurate discharge letter.

PATIENT DATA:
- Name: {name}
- Age: {age} years
- Primary diagnosis: {diag}{f" ({icd10})" if icd10 else ""}
- Admission: {adm}
- Discharge: {dis}
- Length of stay: {los} days
- Physician: {physician if physician else "not specified"}
- Readmission risk: {risk} (score {score}/6)

CLINICAL INFORMATION:
{notes if notes else "No detailed notes available."}

MEDICATION AT DISCHARGE:
{medication if medication else "No medication documented."}

FOLLOW-UP: {followup if followup else "Outpatient follow-up recommended"}

INSTRUCTIONS:
- Write a professional discharge letter in English
- Use appropriate medical terminology
- Structure: salutation, history, course, medication, recommendations, closing
- Maximum 400 words
- No markdown or formatting symbols
- Date: {today}

Begin the letter directly:"""

    return prompt


# ================================================================
# MAIN AL3 GENERATION FUNCTION
# ================================================================

def generate_al3(
    patient: dict,
    lang: str = "Deutsch",
) -> tuple[bool, str, str]:
    """
    Generate an AL3 AI-powered discharge letter using Ollama + Mistral.

    Args:
        patient: patient dict from get_patient()
        lang:    "Deutsch" or "English"

    Returns:
        (True,  letter_text, "")     on success
        (False, "",          error)  on failure
    """
    # Check Ollama is available
    if not is_ollama_running():
        return False, "", (
            "Ollama ist nicht gestartet.\n\n"
            "Lösung:\n"
            "1. Öffnen Sie ein Terminal\n"
            "2. Führen Sie aus: ollama serve\n"
            "3. Versuchen Sie es erneut\n\n"
            if lang == "Deutsch" else
            "Ollama is not running.\n\n"
            "Solution:\n"
            "1. Open a terminal\n"
            "2. Run: ollama serve\n"
            "3. Try again\n\n"
        )

    if not is_model_available(OLLAMA_MODEL):
        return False, "", (
            f"Modell '{OLLAMA_MODEL}' nicht gefunden.\n\n"
            f"Lösung: ollama pull {OLLAMA_MODEL}"
            if lang == "Deutsch" else
            f"Model '{OLLAMA_MODEL}' not found.\n\n"
            f"Solution: ollama pull {OLLAMA_MODEL}"
        )

    # Build prompt and generate
    prompt = build_arztbrief_prompt(patient, lang)
    ok, text, err = generate(prompt, temperature=0.3, max_tokens=1500)

    if not ok:
        return False, "", err

    # Clean up the response
    text = text.strip()

    # Remove any accidental prompt echoing
    if "PATIENTENDATEN:" in text:
        idx = text.find("PATIENTENDATEN:")
        text = text[idx:].split("\n\n", 1)[-1].strip()

    return True, text, ""
