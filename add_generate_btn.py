# add_generate_btn.py
import ast

with open('app/ui/main_window.py', encoding='utf-8-sig') as f:
    content = f.read()

old = '''        self.edit_btn = QPushButton("Edit Patient")
        self.edit_btn.setObjectName("secondary")
        self.edit_btn.setFixedHeight(38)
        self.edit_btn.clicked.connect(
            lambda: self.on_edit(self._current_patient)
        )'''

new = '''        self.generate_btn = QPushButton("Generate Discharge Letter")
        self.generate_btn.setFixedHeight(38)
        self.generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }}
            QPushButton:hover {{ background-color: #5BA3E8; }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.generate_btn.clicked.connect(
            lambda: self.on_generate(self._current_patient)
        )

        self.edit_btn = QPushButton("Edit Patient")
        self.edit_btn.setObjectName("secondary")
        self.edit_btn.setFixedHeight(38)
        self.edit_btn.clicked.connect(
            lambda: self.on_edit(self._current_patient)
        )'''

if old in content:
    content = content.replace(old, new)
    print("Generate button added back")
else:
    print("Pattern not found")

# Add generate_btn to layout before edit_btn
old2 = '''        layout.addWidget(self.edit_btn)
        layout.addWidget(self.delete_btn)'''

new2 = '''        layout.addWidget(self.generate_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.delete_btn)'''

if old2 in content:
    content = content.replace(old2, new2)
    print("Generate button added to layout")
else:
    print("Layout pattern not found")

# Fix _set_actions_enabled to include generate_btn
old3 = '''        for btn in [self.edit_btn, self.delete_btn]:'''
new3 = '''        for btn in [self.generate_btn, self.edit_btn, self.delete_btn]:'''
if old3 in content:
    content = content.replace(old3, new3)
    print("Button enable/disable fixed")

with open('app/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(content)

try:
    ast.parse(content)
    print("Syntax OK — run: python main.py")
except SyntaxError as e:
    print(f"Syntax error line {e.lineno}: {e.msg}")