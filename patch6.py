import os

file_path = r'D:\AI\AI_file_db_reader - Copy\mssql_app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip_mode = False
for line in lines:
    if line.startswith('from file_reader import read_project'):
        continue
    if line.startswith('from search_engine import search_files'):
        continue
        
    # Remove toggle and force Database mode
    if 'is_file_reader = st.toggle' in line:
        continue
    if 'mode = "File Reader AI Assistant" if is_file_reader else "Database AI Assistant"' in line:
        new_lines.append('    mode = "Database AI Assistant"\n')
        continue
        
    new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
    
print("Imports and toggle removed")
