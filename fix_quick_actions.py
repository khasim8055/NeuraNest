# remove_pdf_btn.py
import ast

with open('app/ui/main_window.py', encoding='utf-8-sig') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Skip pdf_btn definition block
    if 'self.pdf_btn = QPushButton("Export PDF")' in line:
        while i < len(lines) and 'self.edit_btn' not in lines[i]:
            i += 1
        continue
    # Skip pdf_btn in layout
    if 'layout.addWidget(self.pdf_btn)' in line:
        i += 1
        continue
    new_lines.append(line)
    i += 1

content = ''.join(new_lines)

# Fix disabled buttons list
content = content.replace(
    'self.generate_btn, self.pdf_btn,\n                    self.edit_btn, self.delete_btn]',
    'self.generate_btn,\n                    self.edit_btn, self.delete_btn]'
)
content = content.replace(
    'self.generate_btn, self.pdf_btn, self.edit_btn, self.delete_btn]',
    'self.generate_btn, self.edit_btn, self.delete_btn]'
)

with open('app/ui/main_window.py', 'w', encoding='utf-8') as f:
    f.write(content)

try:
    ast.parse(content)
    print("Syntax OK")
    print("Export PDF removed from Quick Actions")
    print("Generate Discharge Letter kept")
except SyntaxError as e:
    print(f"Syntax error line {e.lineno}: {e.msg}")