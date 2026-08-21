import re

file_path = r'D:\AI\AI_file_db_reader - Copy\mssql_app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Restore Mode Toggle
content = content.replace(
    '# Mode Toggle\n    mode = "Database AI Assistant"',
    '# Mode Toggle\n    mode = st.sidebar.radio("Mode", ["Database AI Assistant", "File Reader AI Assistant"], label_visibility="collapsed")'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Mode toggle restored.")
