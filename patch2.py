import os
import re
import json

fastapi_path = r'D:\AI\AI_file_db_reader - Copy\fastapi_app.py'
mssql_path = r'D:\AI\AI_file_db_reader - Copy\mssql_app.py'
config_path = r'D:\AI\AI_file_db_reader - Copy\admin_db_config.json'

if os.path.exists(config_path):
    os.remove(config_path)

with open(fastapi_path, 'r', encoding='utf-8') as f:
    fc = f.read()

# Update Pydantic models
fc = fc.replace('class ConnectRequest(BaseModel):', 'class ConnectRequest(BaseModel):\n    environment: str = "dev"')
fc = fc.replace('class SchemaRequest(BaseModel):', 'class SchemaRequest(BaseModel):\n    environment: str = "dev"')
fc = fc.replace('class AskRequest(BaseModel):', 'class AskRequest(BaseModel):\n    environment: str = "dev"')
fc = fc.replace('class AdminDbConfigRequest(BaseModel):', 'class AdminDbConfigRequest(BaseModel):\n    environment: str = "dev"')
fc = fc.replace('class GlobalQuestionRequest(BaseModel):', 'class GlobalQuestionRequest(BaseModel):\n    environment: str = "dev"')

# Connect Endpoint JSON save logic
connect_old = '''        try:
            import json
            with open(ADMIN_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(initial_config, f, indent=4)'''
connect_new = '''        try:
            import json
            existing_config = {}
            if os.path.exists(ADMIN_CONFIG_FILE):
                with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
                    try: existing_config = json.load(f)
                    except: pass
            existing_config[req.environment] = initial_config
            with open(ADMIN_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_config, f, indent=4)'''
fc = fc.replace(connect_old, connect_new)

# Save DB Config logic
save_old = '''    config_data["tables"] = parsed_tables

    try:
        with open(ADMIN_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)'''
save_new = '''    if req.environment not in config_data:
        config_data[req.environment] = {}
    config_data[req.environment]["tables"] = parsed_tables

    try:
        with open(ADMIN_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)'''
fc = fc.replace(save_old, save_new)

# Get DB config logic
get_old = '''def get_admin_db_config():
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"status": "not_configured", "config": None}
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
            config["password"] = "********"
            return {"status": "configured", "config": config}'''
get_new = '''def get_admin_db_config(environment: str = "dev"):
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"status": "not_configured", "config": None}
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            full_config = json.load(f)
            config = full_config.get(environment, None)
            if not config:
                return {"status": "not_configured", "config": None}
            config["password"] = "********"
            return {"status": "configured", "config": config}'''
fc = fc.replace(get_old, get_new)

# Check DB status logic
chk_old = '''@app.get("/admin/check-db-status")
def check_db_status():
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"is_configured": False, "is_connected": False, "message": "No database configuration found."}
        
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)'''
chk_new = '''@app.get("/admin/check-db-status")
def check_db_status(environment: str = "dev"):
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"is_configured": False, "is_connected": False, "message": "No database configuration found."}
        
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            full_config = json.load(f)
            config = full_config.get(environment, None)
            if not config:
                return {"is_configured": False, "is_connected": False, "message": f"No config for {environment}."}'''
fc = fc.replace(chk_old, chk_new)

# Fetch answer logic
fetch_old = '''@app.post("/fetch-answer")
def ask_global_db_query(request: GlobalQuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if not os.path.exists(ADMIN_CONFIG_FILE):
        raise HTTPException(status_code=400, detail="Database is not configured. Admin must save config first.")
        
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read admin config: {e}")'''
fetch_new = '''@app.post("/fetch-answer")
def ask_global_db_query(request: GlobalQuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if not os.path.exists(ADMIN_CONFIG_FILE):
        raise HTTPException(status_code=400, detail="Database is not configured. Admin must save config first.")
        
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            full_config = json.load(f)
            config = full_config.get(request.environment, None)
            if not config:
                raise HTTPException(status_code=400, detail=f"Database config not found for {request.environment}.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read admin config: {e}")'''
fc = fc.replace(fetch_old, fetch_new)

with open(fastapi_path, 'w', encoding='utf-8') as f:
    f.write(fc)

# Now edit mssql_app.py
with open(mssql_path, 'r', encoding='utf-8') as f:
    mc = f.read()

# Add Environment Dropdown and passing logic
env_dropdown = '''    # Model selection
    st.markdown('<p class="sidebar-section-label">Model</p>', unsafe_allow_html=True)'''
env_dropdown_new = '''    # Environment selection
    st.markdown('<p class="sidebar-section-label">Environment</p>', unsafe_allow_html=True)
    env = st.selectbox("Environment", ["dev", "qa", "prod"], key="db_environment", label_visibility="collapsed")
    
    # Model selection
    st.markdown('<p class="sidebar-section-label">Model</p>', unsafe_allow_html=True)'''
mc = mc.replace(env_dropdown, env_dropdown_new)

payload_old1 = '''payload = {
                    "server": server,
                    "database": database,
                    "auth_mode": auth_mode,
                    "username": username,
                    "password": password,
                    "driver": driver
                }'''
payload_new1 = '''payload = {
                    "server": server,
                    "database": database,
                    "auth_mode": auth_mode,
                    "username": username,
                    "password": password,
                    "driver": driver,
                    "environment": st.session_state.get("db_environment", "dev")
                }'''
mc = mc.replace(payload_old1, payload_new1)

payload_old2 = '''            payload = {
                "tables": selected_tables
            }'''
payload_new2 = '''            payload = {
                "tables": selected_tables,
                "environment": st.session_state.get("db_environment", "dev")
            }'''
mc = mc.replace(payload_old2, payload_new2)

req_old1 = '''r = httpx.get(f"{API_BASE_URL}/admin/get-db-config", timeout=10.0)'''
req_new1 = '''r = httpx.get(f"{API_BASE_URL}/admin/get-db-config?environment={st.session_state.get('db_environment', 'dev')}", timeout=10.0)'''
mc = mc.replace(req_old1, req_new1)

req_old2 = '''status_resp = httpx.get(f"{API_BASE_URL}/admin/check-db-status", timeout=5.0)'''
req_new2 = '''status_resp = httpx.get(f"{API_BASE_URL}/admin/check-db-status?environment={st.session_state.get('db_environment', 'dev')}", timeout=5.0)'''
mc = mc.replace(req_old2, req_new2)

payload_old3 = '''                    payload = {
                        "question": question
                    }'''
payload_new3 = '''                    payload = {
                        "question": question,
                        "environment": st.session_state.get("db_environment", "dev")
                    }'''
mc = mc.replace(payload_old3, payload_new3)

with open(mssql_path, 'w', encoding='utf-8') as f:
    f.write(mc)

print("Patch complete")
