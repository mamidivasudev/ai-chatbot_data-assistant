import os

file_path = r'D:\AI\AI_file_db_reader - Copy\mssql_app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

old_code = '''                        db_ident = f"{db_type}_{conn_params.get('database') or conn_params.get('db_path') or conn_params.get('host') or 'default'}"'''
new_code = '''                        env = st.session_state.get('db_environment', 'dev')
                        db_ident = f"{env}_{db_type}_{conn_params.get('database') or conn_params.get('db_path') or conn_params.get('host') or 'default'}"'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("SUCCESS")
else:
    print("FAILED TO FIND OLD CODE")
