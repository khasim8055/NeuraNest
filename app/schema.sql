-- ================================================================
-- NeuraCare Database Schema v1.0
-- SQLite · Encrypted at file level via Fernet
-- ================================================================
-- Tables:
--   1. users          — login accounts (Admin / Doctor roles)
--   2. patients       — full clinical record per patient
--   3. audit_log      — every action logged (GDPR Article 30)
--   4. app_config     — key-value store for app settings
-- ================================================================


-- ── 1. USERS ────────────────────────────────────────────────────
-- Stores clinic staff accounts.
-- Passwords are bcrypt-hashed — plain text never stored.
-- Two roles: admin (full access) and doctor (no delete).

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    full_name     TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'doctor'
                          CHECK(role IN ('admin', 'doctor')),
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login    TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1
                          CHECK(is_active IN (0, 1))
);

-- Default admin account (password: changeme123)
-- MUST be changed on first login
INSERT OR IGNORE INTO users
    (username, password_hash, full_name, role)
VALUES
    ('admin',
     '$2b$12$PLACEHOLDER_CHANGE_ON_FIRST_RUN',
     'Administrator',
     'admin');


-- ── 2. PATIENTS ──────────────────────────────────────────────────
-- Core clinical record.
-- All 7 existing fields kept + 4 new clinical fields added.
-- created_by links to users.id for audit trail.

CREATE TABLE IF NOT EXISTS patients (
    -- Identity
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    age               INTEGER NOT NULL CHECK(age >= 0 AND age <= 130),
    date_of_birth     TEXT,                          -- optional, ISO format YYYY-MM-DD

    -- Clinical
    diagnosis         TEXT    NOT NULL,
    icd10_code        TEXT,                          -- NEW: e.g. "I50.0"
    admission_date    TEXT    NOT NULL,              -- ISO format YYYY-MM-DD
    discharge_date    TEXT    NOT NULL,              -- ISO format YYYY-MM-DD
    notes             TEXT    DEFAULT '',

    -- NEW clinical fields (required for Arztbrief)
    medication        TEXT    DEFAULT '',            -- medications at discharge
    physician_name    TEXT    DEFAULT '',            -- responsible physician
    followup_date     TEXT,                          -- ISO format YYYY-MM-DD

    -- Computed / cached
    length_of_stay    INTEGER                        -- computed on insert/update trigger
                      GENERATED ALWAYS AS (
                          CAST(
                              (julianday(discharge_date) - julianday(admission_date))
                          AS INTEGER)
                      ) VIRTUAL,

    -- Record management
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    created_by        INTEGER REFERENCES users(id),
    is_deleted        INTEGER NOT NULL DEFAULT 0
                              CHECK(is_deleted IN (0, 1)),   -- soft delete for undo

    -- Constraints
    CHECK(discharge_date >= admission_date)          -- prevents negative LOS
);

-- Index for fast patient list queries
CREATE INDEX IF NOT EXISTS idx_patients_name
    ON patients(name) WHERE is_deleted = 0;

CREATE INDEX IF NOT EXISTS idx_patients_admission
    ON patients(admission_date) WHERE is_deleted = 0;


-- ── 3. AUDIT LOG ─────────────────────────────────────────────────
-- Every action on patient data is logged here.
-- Required for GDPR Article 30 compliance.
-- Never deleted — append only.

CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL DEFAULT (datetime('now')),
    user_id       INTEGER REFERENCES users(id),
    username      TEXT    NOT NULL,                  -- denormalised for readability
    action        TEXT    NOT NULL
                          CHECK(action IN (
                              'login',
                              'logout',
                              'patient_create',
                              'patient_view',
                              'patient_edit',
                              'patient_delete',
                              'patient_restore',
                              'summary_generate',
                              'pdf_export',
                              'csv_export'
                          )),
    patient_id    INTEGER,                           -- NULL for login/logout actions
    patient_name  TEXT,                              -- denormalised snapshot
    detail        TEXT                               -- extra context if needed
);

-- Index for audit report queries (by date, by user)
CREATE INDEX IF NOT EXISTS idx_audit_timestamp
    ON audit_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_audit_user
    ON audit_log(user_id);


-- ── 4. APP CONFIG ────────────────────────────────────────────────
-- Key-value store for application settings.
-- Avoids hardcoding clinic name, language preference etc.

CREATE TABLE IF NOT EXISTS app_config (
    key           TEXT PRIMARY KEY,
    value         TEXT NOT NULL,
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Default settings
INSERT OR IGNORE INTO app_config (key, value) VALUES
    ('clinic_name',        'My Practice'),
    ('default_language',   'Deutsch'),
    ('app_version',        '1.0.0'),
    ('first_run',          '1'),
    ('key_backup_done',    '0');           -- triggers backup warning on first launch
