# rebuild_letter_ui.py
# Completely rebuilds the controls bar — removes levels, clean German UI
import ast, re

with open('app/ui/letter_panel.py', encoding='utf-8') as f:
    content = f.read()

# Fix header title from "Discharge Letter" to "Arztbrief"
content = content.replace(
    'self.title_label.setText(f"Discharge Letter — {name}")',
    'self.title_label.setText(f"Arztbrief — {name}")'
)
content = content.replace(
    'self.title_label.setText("Discharge Letter")',
    'self.title_label.setText("Arztbrief")'
)
print("1. Header title fixed")

# Fix generate button width — make it wider
content = content.replace(
    'self.generate_btn.setFixedHeight(32)',
    'self.generate_btn.setFixedHeight(32)\n        self.generate_btn.setMinimumWidth(180)'
)
print("2. Generate button width fixed")

# Remove entire level selector block from _build_ui
# Find and replace the level selector section
old_level_block = '''        # AL Level selector
        level_label = QLabel("Level: AL")
        level_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 11px;"
        )

        self.level_group = QButtonGroup()
        self._level_btns = {}

        level_names = {0: "0", 1: "1", 2: "2"}
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
        for lvl in [0, 1, 2]:
            ctrl_layout.addWidget(self._level_btns[lvl])'''

if old_level_block in content:
    content = content.replace(old_level_block, '')
    print("3. Level selector removed from UI")
else:
    print("3. Level selector pattern not found — removing via line scan")
    lines = content.split('\n')
    new_lines = []
    skip = False
    for line in lines:
        if '# AL Level selector' in line:
            skip = True
        if skip and 'ctrl_layout.addWidget(level_label)' in line:
            skip = True
            continue
        if skip and 'for lvl in [0, 1, 2]:' in line:
            continue
        if skip and 'ctrl_layout.addWidget(self._level_btns[lvl])' in line:
            skip = False
            continue
        if not skip:
            new_lines.append(line)
    content = '\n'.join(new_lines)
    print("3. Level selector removed via line scan")

# Fix controls bar height — smaller now without level buttons
content = content.replace(
    'controls.setFixedHeight(52)',
    'controls.setFixedHeight(48)'
)
print("4. Controls bar height adjusted")

# Fix spacing
content = content.replace(
    'ctrl_layout.setSpacing(4)',
    'ctrl_layout.setSpacing(8)'
)
print("5. Spacing adjusted")

with open('app/ui/letter_panel.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('app/ui/letter_panel.py', encoding='utf-8') as f:
    src = f.read()
try:
    ast.parse(src)
    print(f"\nSyntax OK ({src.count(chr(10))} lines)")
    print("Run: python main.py")
    print("Expected UI: Sprache [Deutsch][English]  [Arztbrief generieren] [Kopieren]")
except SyntaxError as e:
    print(f"\nSyntax error line {e.lineno}: {e.msg}")