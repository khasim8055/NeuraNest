# app/ui/letter_panel.py
# ================================================================
# NeuraCare — Discharge Letter Generation Panel
# ================================================================
# Shows inside the center panel when "Generate Discharge Letter"
# is clicked for a selected patient.
#
# Features:
#   - AL level selector (AL0 / AL1 / AL2)
#   - Language toggle (Deutsch / English)
#   - Live letter preview in scrollable text area
#   - Copy to clipboard button
#   - Generate button → triggers PDF export (Day 8)
# ================================================================

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QFrame, QButtonGroup,
    QRadioButton, QScrollArea, QApplication,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui  import QFont, QTextCursor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.letters import generate_letter, LEVEL_INFO
from app.core.audit   import log_summary_generate
from app.core.auth    import Session

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


class LetterPanel(QWidget):
    """
    Discharge letter generation panel.

    Usage:
        panel = LetterPanel(on_pdf=callback, on_close=callback)
        panel.load_patient(patient_dict)
    """

    def __init__(self, on_pdf=None, on_close=None):
        super().__init__()
        self.on_pdf   = on_pdf
        self.on_close = on_close
        self._patient = None
        self._lang    = "Deutsch"
        self._level   = 1
        self._letter_text = ""
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header bar ───────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        header.setFixedHeight(56)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)

        self.title_label = QLabel("Discharge Letter")
        self.title_label.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        close_btn = QPushButton("← Back")
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
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self._on_close)

        self.pdf_btn = QPushButton("Export PDF")
        self.pdf_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 20px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #5BA3E8; }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.pdf_btn.setFixedHeight(32)
        self.pdf_btn.setEnabled(False)
        self.pdf_btn.clicked.connect(self._on_export_pdf)

        header_layout.addWidget(close_btn)
        header_layout.addSpacing(12)
        header_layout.addWidget(self.title_label, stretch=1)
        header_layout.addWidget(self.pdf_btn)

        # ── Controls bar ─────────────────────────────────────────
        controls = QWidget()
        controls.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        controls.setFixedHeight(60)
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(24, 0, 24, 0)
        ctrl_layout.setSpacing(16)

        # AL Level selector
        level_label = QLabel("Level: AL")
        level_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )

        self.level_group = QButtonGroup()
        self._level_btns = {}

        level_names = {0: "0", 1: "1", 2: "2", 3: "3 (AI)"}
        for lvl, name in level_names.items():
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(lvl == 1)
            btn.setFixedSize(60 if lvl == 3 else 48, 30)
            btn.setStyleSheet(self._level_btn_style(lvl == 1, is_ai=(lvl == 3)))
            btn.clicked.connect(lambda checked, l=lvl: self._on_level_changed(l))
            self.level_group.addButton(btn)
            self._level_btns[lvl] = btn

        ctrl_layout.addWidget(level_label)
        for lvl in [0, 1, 2, 3]:
            ctrl_layout.addWidget(self._level_btns[lvl])

        # Language toggle
        divider = QFrame()
        divider.setStyleSheet(f"background-color: {COLORS['border']};")
        divider.setFixedSize(1, 30)
        ctrl_layout.addWidget(divider)

        lang_label = QLabel("Sprache:")
        lang_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )

        self.lang_de_btn = QPushButton("Deutsch")
        self.lang_en_btn = QPushButton("English")

        for btn, lang in [(self.lang_de_btn, "Deutsch"),
                          (self.lang_en_btn, "English")]:
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet(self._lang_btn_style(lang == "Deutsch"))
            btn.clicked.connect(
                lambda checked, l=lang: self._on_lang_changed(l)
            )

        self.lang_de_btn.setChecked(True)

        ctrl_layout.addWidget(lang_label)
        ctrl_layout.addWidget(self.lang_de_btn)
        ctrl_layout.addWidget(self.lang_en_btn)

        # Generate button
        ctrl_layout.addStretch()
        self.generate_btn = QPushButton("Generate Letter")
        self.generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_green']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 20px;
                font-size: 12px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #63C899; }}
        """)
        self.generate_btn.setFixedHeight(32)
        self.generate_btn.setFixedWidth(120)
        self.generate_btn.clicked.connect(self._on_generate)
        ctrl_layout.addWidget(self.generate_btn)

        # Copy button
        self.copy_btn = QPushButton("Copy Text")
        self.copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 12px;
            }}
            QPushButton:hover {{ border-color: {COLORS['accent_blue']}; }}
            QPushButton:disabled {{
                color: {COLORS['text_muted']};
            }}
        """)
        self.copy_btn.setFixedHeight(32)
        self.copy_btn.setFixedWidth(90)
        self.copy_btn.setEnabled(False)
        self.copy_btn.clicked.connect(self._on_copy)
        ctrl_layout.addWidget(self.copy_btn)

        # ── Level description bar ─────────────────────────────────
        self.desc_bar = QWidget()
        self.desc_bar.setStyleSheet(
            f"background-color: {COLORS['bg_dark']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        self.desc_bar.setFixedHeight(36)
        desc_layout = QHBoxLayout(self.desc_bar)
        desc_layout.setContentsMargins(24, 0, 24, 0)

        self.desc_label = QLabel(
            LEVEL_INFO[1]["description"]
        )
        self.desc_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )
        desc_layout.addWidget(self.desc_label)

        # ── Letter preview area ───────────────────────────────────
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
                border: none;
                padding: 24px 32px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.6;
            }}
        """)
        self.preview.setPlaceholderText(
            "Select a level and click Generate Letter to preview the discharge letter here."
        )

        outer.addWidget(header)
        outer.addWidget(controls)
        outer.addWidget(self.desc_bar)
        outer.addWidget(self.preview, stretch=1)

    # ================================================================
    # STYLE HELPERS
    # ================================================================

    def _level_btn_style(self, active: bool, is_ai: bool = False) -> str:
        active_color = COLORS['accent_amber'] if is_ai else COLORS['accent_blue']
        if active:
            return f"""
                QPushButton {{
                    background-color: {active_color};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 10px;
                    font-weight: bold;
                }}
            """
        return f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                font-size: 10px;
            }}
            QPushButton:hover {{
                border-color: {active_color};
                color: {COLORS['text_primary']};
            }}
        """

    def _lang_btn_style(self, active: bool) -> str:
        if active:
            return f"""
                QPushButton {{
                    background-color: {COLORS['accent_blue']};
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-size: 11px;
                    font-weight: bold;
                    padding: 4px 12px;
                }}
            """
        return f"""
            QPushButton {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                font-size: 11px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['accent_blue']};
                color: {COLORS['text_primary']};
            }}
        """

    # ================================================================
    # PUBLIC METHODS
    # ================================================================

    def load_patient(self, patient: dict):
        """Load a patient and show the panel."""
        self._patient = patient
        name = patient.get("name", "Unknown")
        self.title_label.setText(f"Discharge Letter — {name}")
        self.preview.clear()
        self.preview.setPlaceholderText(
            f"Click Generate Letter to create a discharge letter for {name}."
        )
        self._letter_text = ""
        self.pdf_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)

    # ================================================================
    # PRIVATE METHODS
    # ================================================================

    def _on_level_changed(self, level: int):
        self._level = level
        # Update button styles
        for lvl, btn in self._level_btns.items():
            btn.setStyleSheet(self._level_btn_style(lvl == level, is_ai=(lvl == 3)))
        # Update description
        info = LEVEL_INFO.get(level, {})
        key = "description" if self._lang == "Deutsch" else "description_en"
        self.desc_label.setText(info.get(key, ""))

    def _on_lang_changed(self, lang: str):
        self._lang = lang
        self.lang_de_btn.setChecked(lang == "Deutsch")
        self.lang_en_btn.setChecked(lang == "English")
        self.lang_de_btn.setStyleSheet(
            self._lang_btn_style(lang == "Deutsch")
        )
        self.lang_en_btn.setStyleSheet(
            self._lang_btn_style(lang == "English")
        )
        # Update description text language
        info = LEVEL_INFO.get(self._level, {})
        key = "description" if lang == "Deutsch" else "description_en"
        self.desc_label.setText(info.get(key, ""))

        # Regenerate if already generated
        if self._letter_text:
            self._on_generate()

    def _on_generate(self):
        """Generate and display the letter."""
        if not self._patient:
            return

        self.generate_btn.setEnabled(False)
        if self._level == 3:
            self.generate_btn.setText("AI generating… (15-30s)")
            self.preview.setPlainText(
                "Ollama + Mistral is generating your letter…\n\n"
                "This takes 15-30 seconds on first run.\n"
                "Please wait."
            )
        else:
            self.generate_btn.setText("Generating…")

        ok, text, error = generate_letter(
            self._patient,
            level=self._level,
            lang=self._lang,
        )

        if ok:
            self._letter_text = text
            self.preview.setPlainText(text)
            # Scroll to top
            cursor = self.preview.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.preview.setTextCursor(cursor)

            self.pdf_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)

            # Log the action
            user = Session.get_user() or {}
            log_summary_generate(
                Session.get_username(),
                patient_id=self._patient.get("id"),
                patient_name=self._patient.get("name", ""),
                autonomy_level=f"AL{self._level}",
                user_id=user.get("id"),
            )
        else:
            self.preview.setPlainText(f"Error: {error}")
            self.pdf_btn.setEnabled(False)
            self.copy_btn.setEnabled(False)

        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Letter")

    def _on_copy(self):
        """Copy letter text to clipboard."""
        if self._letter_text:
            QApplication.clipboard().setText(self._letter_text)
            self.copy_btn.setText("Copied!")
            # Reset button text after delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, lambda: self.copy_btn.setText("Copy Text"))

    def _on_export_pdf(self):
        """Export discharge letter as PDF and open it."""
        if not self._patient or not self._letter_text:
            return

        from app.core.pdf_exporter import export_and_open
        self.pdf_btn.setEnabled(False)
        self.pdf_btn.setText("Exporting…")

        ok, file_path, err = export_and_open(
            self._patient,
            self._letter_text,
            lang=self._lang,
            level=self._level,
        )

        if ok:
            # Notify main window
            if self.on_pdf:
                self.on_pdf(self._patient, file_path)
            self.pdf_btn.setText("PDF Saved ✓")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(3000, lambda: self.pdf_btn.setText("Export PDF"))
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "PDF Export Failed",
                f"Could not export PDF:\n{err}"
            )
            self.pdf_btn.setText("Export PDF")

        self.pdf_btn.setEnabled(True)

    def _on_close(self):
        """Go back to patient detail view."""
        if self.on_close:
            self.on_close()
