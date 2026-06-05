# update_editable_letter.py
# Makes the letter preview editable before PDF export
import ast

with open('app/ui/letter_panel.py', encoding='utf-8') as f:
    content = f.read()

# 1. Make preview editable
old1 = "        self.preview.setReadOnly(True)"
new1 = "        self.preview.setReadOnly(False)"
content = content.replace(old1, new1)
print("1. Preview now editable:", old1 not in content)

# 2. Add edit indicator label after preview
old2 = "        outer.addWidget(self.preview, stretch=1)"
new2 = """        # Edit status bar
        self.edit_bar = QLabel("  ✏  You can edit the letter directly above before exporting.")
        self.edit_bar.setStyleSheet(
            f"background-color: {COLORS['bg_panel']}; "
            f"color: {COLORS['text_muted']}; "
            f"font-size: 11px; padding: 6px 16px; "
            f"border-top: 1px solid {COLORS['border']};"
        )
        self.edit_bar.hide()

        outer.addWidget(self.preview, stretch=1)
        outer.addWidget(self.edit_bar)"""
content = content.replace(old2, new2)
print("2. Edit bar added:", "edit_bar" in content)

# 3. Show edit bar after generation
old3 = """            self.pdf_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)"""
new3 = """            self.pdf_btn.setEnabled(True)
            self.copy_btn.setEnabled(True)
            self.edit_bar.show()"""
content = content.replace(old3, new3)
print("3. Edit bar shows after generation:", content.count("self.edit_bar.show()") > 0)

# 4. Update _on_copy to use current text (edited version)
old4 = """    def _on_copy(self):
        \"\"\"Copy letter text to clipboard.\"\"\"
        if self._letter_text:
            QApplication.clipboard().setText(self._letter_text)"""
new4 = """    def _on_copy(self):
        \"\"\"Copy current letter text to clipboard (including edits).\"\"\"
        current_text = self.preview.toPlainText()
        if current_text:
            QApplication.clipboard().setText(current_text)"""
content = content.replace(old4, new4)
print("4. Copy uses edited text:", "toPlainText" in content)

# 5. Update _on_export_pdf to use edited version
old5 = """    def _on_export_pdf(self):
        \"\"\"Export discharge letter as PDF and open it.\"\"\"
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
        )"""
new5 = """    def _on_export_pdf(self):
        \"\"\"Export discharge letter as PDF — uses edited version if modified.\"\"\"
        if not self._patient:
            return

        # Always use current text from preview (may be edited by doctor)
        current_text = self.preview.toPlainText().strip()
        if not current_text:
            return

        # Check if doctor edited the letter
        was_edited = current_text != self._letter_text.strip()
        if was_edited:
            # Log that letter was edited before export
            from app.core.audit import log
            from app.core.auth  import Session
            log(
                "pdf_export",
                Session.get_username(),
                patient_id=self._patient.get("id"),
                patient_name=self._patient.get("name", ""),
                detail=f"AL{self._level} — edited before export",
            )

        from app.core.pdf_exporter import export_and_open
        self.pdf_btn.setEnabled(False)
        self.pdf_btn.setText("Exporting…")

        ok, file_path, err = export_and_open(
            self._patient,
            current_text,
            lang=self._lang,
            level=self._level,
        )"""
content = content.replace(old5, new5)
print("5. PDF export uses edited text:", "current_text = self.preview.toPlainText" in content)

# 6. Hide edit bar when loading a new patient
old6 = """        self._letter_text = ""
        self.pdf_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)"""
new6 = """        self._letter_text = ""
        self.pdf_btn.setEnabled(False)
        self.copy_btn.setEnabled(False)
        if hasattr(self, 'edit_bar'):
            self.edit_bar.hide()"""
content = content.replace(old6, new6)
print("6. Edit bar hides on new patient:", content.count("self.edit_bar.hide()") > 0)

# Save
with open('app/ui/letter_panel.py', 'w', encoding='utf-8') as f:
    f.write(content)

# Verify syntax
with open('app/ui/letter_panel.py', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print(f"\n✓ letter_panel.py syntax OK ({src.count(chr(10))} lines)")
except SyntaxError as e:
    print(f"\n✗ SYNTAX ERROR line {e.lineno}: {e.msg}")

print("\nDone. Run: python main.py")
print("Generate a letter → edit the text directly → Export PDF")
print("The PDF will contain your edited version.")