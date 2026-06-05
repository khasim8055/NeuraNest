# app/ui/privacy_panel.py
# ================================================================
# NeuraCare — Data & Privacy Panel
# ================================================================
# GDPR compliance tools:
#   - Export anonymised patient data (Article 15)
#   - Export audit log (Article 30)
#   - Export practice summary
#   - Permanent deletion of all data (Article 17)
# ================================================================

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QMessageBox,
)
from PyQt6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

COLORS = {
    "bg_dark":      "#1A1D2E",
    "bg_panel":     "#22253A",
    "bg_card":      "#2A2D42",
    "accent_blue":  "#4A90D9",
    "accent_green": "#52B788",
    "accent_amber": "#F4A50A",
    "accent_red":   "#D64545",
    "text_primary": "#E8EAF6",
    "text_muted":   "#8B90A8",
    "border":       "#35384F",
}


def _divider():
    f = QFrame()
    f.setStyleSheet(f"background-color: {COLORS['border']};")
    f.setFixedHeight(1)
    return f


def _btn(text, color, danger=False):
    bg = COLORS["accent_red"] if danger else color
    btn = QPushButton(text)
    btn.setFixedHeight(38)
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: {bg};
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            padding: 0 16px;
        }}
        QPushButton:hover {{ background-color: {bg}CC; }}
    """)
    return btn


class PrivacyPanel(QWidget):
    """Data and Privacy panel — GDPR compliance tools."""

    def __init__(self, on_close=None):
        super().__init__()
        self.on_close = on_close
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header.setFixedHeight(56)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)

        title = QLabel("Data & Privacy")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")

        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(32)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_blue']}; }}
        """)
        close_btn.clicked.connect(self._on_close)

        hl.addWidget(title, stretch=1)
        hl.addWidget(close_btn)

        # Scrollable body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {COLORS['bg_dark']}; }}"
        )

        body = QWidget()
        body.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(32, 24, 32, 32)
        layout.setSpacing(20)

        # Status message
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet(
            f"color: {COLORS['accent_green']}; font-size: 12px;"
        )

        # Section 1 — Export
        layout.addWidget(self._section_title("Export Patient Data"))
        layout.addWidget(_divider())

        exp_info = QLabel(
            "Export data to your Desktop for compliance reporting or "
            "patient access requests (GDPR Article 15)."
        )
        exp_info.setWordWrap(True)
        exp_info.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        layout.addWidget(exp_info)

        row1 = QHBoxLayout()
        anon_btn = _btn("Export Anonymised Data", COLORS["accent_blue"])
        anon_btn.clicked.connect(self._on_export_anon)
        row1.addWidget(anon_btn)

        audit_btn = _btn("Export Audit Log", COLORS["accent_amber"])
        audit_btn.clicked.connect(self._on_export_audit)
        row1.addWidget(audit_btn)

        summary_btn = _btn("Practice Summary", COLORS["accent_green"])
        summary_btn.clicked.connect(self._on_export_summary)
        row1.addWidget(summary_btn)

        layout.addLayout(row1)
        layout.addWidget(self.msg)

        # Section 2 — Danger Zone
        layout.addSpacing(16)
        layout.addWidget(self._section_title("Danger Zone", danger=True))
        layout.addWidget(_divider())

        danger_info = QLabel(
            "Permanently delete all patient records and audit log from "
            "this device. This cannot be undone. Use only when offboarding "
            "a clinic or responding to a GDPR Article 17 erasure request."
        )
        danger_info.setWordWrap(True)
        danger_info.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        layout.addWidget(danger_info)

        delete_btn = _btn(
            "Permanently Delete ALL Patient Data",
            COLORS["accent_red"],
            danger=True
        )
        delete_btn.clicked.connect(self._on_delete_all)
        layout.addWidget(delete_btn)

        layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(header)
        outer.addWidget(scroll, stretch=1)

    def _section_title(self, text, danger=False):
        label = QLabel(text)
        color = COLORS["accent_red"] if danger else COLORS["accent_blue"]
        label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
        )
        return label

    def _set_msg(self, text, success=True):
        color = COLORS["accent_green"] if success else COLORS["accent_red"]
        self.msg.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.msg.setText(text)

    def _on_export_anon(self):
        from app.core.csv_export import export_anonymised_patients
        ok, path, err = export_anonymised_patients()
        if ok:
            self._set_msg(f"Saved to Desktop: {Path(path).name}")
        else:
            self._set_msg(err, success=False)

    def _on_export_audit(self):
        from app.core.csv_export import export_audit_log
        ok, path, err = export_audit_log()
        if ok:
            self._set_msg(f"Audit log saved: {Path(path).name}")
        else:
            self._set_msg(err, success=False)

    def _on_export_summary(self):
        from app.core.csv_export import export_practice_summary
        ok, path, err = export_practice_summary()
        if ok:
            self._set_msg(f"Summary saved: {Path(path).name}")
        else:
            self._set_msg(err, success=False)

    def _on_delete_all(self):
        reply = QMessageBox.warning(
            self,
            "Delete All Patient Data",
            "This permanently deletes ALL patient records and the "
            "audit log from this device.\n\nThis cannot be undone.\n\n"
            "Are you absolutely sure?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from app.core.csv_export import delete_all_patient_data
        ok, msg = delete_all_patient_data(confirm=True)
        self._set_msg(msg, success=ok)

    def _on_close(self):
        if self.on_close:
            self.on_close()
