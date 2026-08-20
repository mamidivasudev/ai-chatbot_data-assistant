import os

file_path = r'D:\AI\AI_file_db_reader - Copy\mssql_app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add key to model selectbox
content = content.replace(
    'selected_model = st.selectbox("Model", options=available_models, label_visibility="collapsed")',
    'selected_model = st.selectbox("Model", options=available_models, key="db_model", label_visibility="collapsed")'
)

# 2. Add key to Server and Database
content = content.replace(
    'server   = c1.text_input("Server",   placeholder="Host\\\\Instance", label_visibility="collapsed")',
    'server   = c1.text_input("Server",   placeholder="Host\\\\Instance", key="db_server", label_visibility="collapsed")'
)
content = content.replace(
    'database = c2.text_input("Database", placeholder="Database",       label_visibility="collapsed")',
    'database = c2.text_input("Database", placeholder="Database",       key="db_database", label_visibility="collapsed")'
)

# 3. Add key to Auth mode
content = content.replace(
    'auth_mode = st.selectbox("Auth", ["Windows Authentication", "SQL Server Authentication"], label_visibility="collapsed")',
    'auth_mode = st.selectbox("Auth", ["Windows Authentication", "SQL Server Authentication"], key="db_auth_mode", label_visibility="collapsed")'
)

# 4. Add key to Username and Password
content = content.replace(
    'username = c3.text_input("Username", placeholder="Username", label_visibility="collapsed")',
    'username = c3.text_input("Username", placeholder="Username", key="db_username", label_visibility="collapsed")'
)
content = content.replace(
    'password = c4.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")',
    'password = c4.text_input("Password", type="password", placeholder="Password", key="db_password", label_visibility="collapsed")'
)

# 5. Add button after Reset App
reset_app_code = '''        st.session_state.clear()
        st.rerun()'''

default_btn_code = '''        st.session_state.clear()
        st.rerun()

    if st.button("Load Default", use_container_width=True, help="Load preset connection details"):
        st.session_state["db_server"] = "192.168.1.18"
        st.session_state["db_database"] = "Hims_Zrams"
        st.session_state["db_auth_mode"] = "SQL Server Authentication"
        st.session_state["db_username"] = "Sa"
        st.session_state["db_password"] = "Satra@123"
        st.session_state["db_model"] = "llama3:latest"
        st.rerun()'''

if reset_app_code in content:
    content = content.replace(reset_app_code, default_btn_code, 1)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Modifications applied successfully.")
