import sys
import os
import asyncio

# --- FIX FOR OLLAMA CONNECTION ---
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
# ---------------------------------

if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
import pandas as pd
import streamlit as st

from mssql_connector import connect_mssql, get_available_drivers
from ollama_client import list_ollama_models, ask_ollama
from mssql_schema_reader import (
    get_all_tables,
    get_table_columns,
    get_primary_keys,
    get_foreign_keys,
    get_selected_schema_text,
)
from mssql_sql_generator import generate_tsql, generate_answer_summary, generate_rule_from_sql
from mssql_executor import validate_tsql, execute_tsql
from history_manager import init_db, save_chat, get_business_skills, save_business_skills, filter_rules_by_keywords, save_user_suggestion, get_user_suggestions, get_chat_history, clear_chat_history

# Import unified adapters
from db_adapters import (
    DATABASES,
    connect_db,
    get_db_tables,
    get_db_table_metadata,
    get_db_schema_text,
    generate_db_query,
    validate_db_query,
    execute_db_query,
    generate_db_answer_summary,
    generate_suggested_questions,
)

# ─────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title="Database AI Assistant",
    layout="wide",
    page_icon="🗄️",
)

# ─────────────────────────────────────────────
# Custom CSS — clean, professional design
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root tokens ── */
    :root {
        --bg:          #f5f6fa;
        --surface:     #ffffff;
        --surface-2:   #f0f1f7;
        --border:      #e2e4ef;
        --accent:      #4f6ef7;
        --accent-soft: rgba(79,110,247,0.08);
        --green:       #16a34a;
        --green-soft:  rgba(22,163,74,0.08);
        --red:         #dc2626;
        --red-soft:    rgba(220,38,38,0.07);
        --amber:       #d97706;
        --text-1:      #111827;
        --text-2:      #4b5563;
        --text-3:      #9ca3af;
        --radius:      10px;
        --radius-sm:   6px;
    }

    /* ── Global resets ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text-1) !important;
    }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, header, footer,
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    div[data-testid="InputInstructions"] { display: none !important; }

    /* ── Main container ── */
    .stMainBlockContainer {
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1200px !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        padding: 0 !important;
        box-shadow: 1px 0 0 var(--border) !important;
        min-width: 340px !important;
        max-width: 340px !important;
        width: 340px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        min-width: 340px !important;
        width: 340px !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding: 1rem 1rem 1rem !important;
        width: 100% !important;
    }

    /* Hide scrollbars in sidebar */
    [data-testid="stSidebar"]::-webkit-scrollbar,
    [data-testid="stSidebar"] > div::-webkit-scrollbar,
    [data-testid="stSidebarUserContent"]::-webkit-scrollbar {
        display: none !important;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarUserContent"] {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }

    /* Fix selectbox text clipping */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        min-height: 44px !important;
        padding-left: 12px !important;
        padding-right: 40px !important;
        overflow: visible !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] div[class*="singleValue"] {
        overflow: visible !important;
        text-overflow: unset !important;
        white-space: nowrap !important;
        max-width: none !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
        font-size: 0.82rem !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] div[class*="valueContainer"] {
        padding-left: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.4rem !important; }

    /* Sidebar text & labels */
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label {
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        color: var(--text-2) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        margin-bottom: 0.2rem !important;
        margin-top: 0rem !important;
        padding-top: 0rem !important;
        line-height: 1.2 !important;
        overflow: visible !important;
    }

    /* Sidebar inputs */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-1) !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.6rem !important;
    }
    [data-testid="stSidebar"] input:focus,
    [data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-soft) !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-2) !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.7rem !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stSidebar"] button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }
    [data-testid="stSidebar"] button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #fff !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        opacity: 0.88 !important;
        color: #fff !important;
    }

    /* Sidebar scrollbar hidden */
    [data-testid="stSidebar"]::-webkit-scrollbar,
    [data-testid="stSidebarUserContent"]::-webkit-scrollbar { display: none !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
        -ms-overflow-style: none !important;
        scrollbar-width: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 0.4rem !important; }

    /* ── Toggle ── */
    [data-testid="stSidebar"] [data-testid="stToggle"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.4rem 0.6rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stToggle"] p {
        font-size: 0.78rem !important;
        color: var(--text-1) !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] button[role="tab"] {
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        color: var(--text-2) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1rem !important;
        margin-bottom: -1px !important;
        border-radius: 0 !important;
        transition: color 0.15s, border-color 0.15s !important;
    }
    [data-testid="stTabs"] button[role="tab"]:hover { color: var(--text-1) !important; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stExpander"] summary {
        background: var(--surface) !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--text-2) !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--text-1) !important; }
    [data-testid="stExpander"] > div > div { padding: 0.7rem 0.9rem !important; }

    /* ── Code blocks ── */
    [data-testid="stCode"],
    .stCode, code {
        background: #f8f9fc !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        color: #1e293b !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stDataFrame"] thead th {
        background: var(--surface-2) !important;
        color: var(--text-2) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    [data-testid="stDataFrame"] tbody td {
        font-size: 0.8rem !important;
        color: var(--text-1) !important;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 0.9rem 1rem !important;
        margin-bottom: 0.6rem !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: var(--accent-soft) !important;
        border-color: rgba(79,110,247,0.18) !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stChatInput"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-soft) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-1) !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stChatInput"] button {
        background: var(--accent) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Alert / Info / Warning / Error ── */
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-size: 0.82rem !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"] {
        background: var(--surface-2) !important;
    }
    div[data-testid="stAlert"] [data-baseweb="notification"] {
        background: transparent !important;
    }
    /* Info */
    .stAlert .st-ae { background: var(--accent-soft) !important; border-left: 3px solid var(--accent) !important; }
    /* Success */
    .stAlert .st-af { background: var(--green-soft) !important; border-left: 3px solid var(--green) !important; }
    /* Error */
    .stAlert .st-ag { background: var(--red-soft) !important; border-left: 3px solid var(--red) !important; }

    /* ── Success / Warning / Error / Info boxes direct ── */
    div[class*="stSuccess"] > div,
    div[class*="stInfo"] > div,
    div[class*="stWarning"] > div,
    div[class*="stError"] > div {
        border-radius: var(--radius) !important;
        font-size: 0.82rem !important;
    }

    /* ── Buttons (main area) ── */
    .stButton button {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-2) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.35rem 0.85rem !important;
        transition: all 0.15s !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    .stButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
        background: var(--accent-soft) !important;
    }

    /* ── Multiselect chips ── */
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {
        background: var(--accent-soft) !important;
        border-radius: 4px !important;
        color: var(--accent) !important;
        font-size: 0.72rem !important;
    }
    [data-testid="stMultiSelect"] > div {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ── Text area ── */
    textarea {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-1) !important;
        font-size: 0.82rem !important;
    }
    textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px var(--accent-soft) !important;
    }

    /* ── Number input ── */
    [data-testid="stNumberInput"] input {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-1) !important;
    }

    /* ── Popover ── */
    [data-testid="stPopover"] button {
        font-size: 0.78rem !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

    /* ── Caption / small text ── */
    [data-testid="stCaptionContainer"] p,
    .stCaption { font-size: 0.72rem !important; color: var(--text-3) !important; }

    /* ── Page title styling ── */
    h1 { font-size: 1.5rem !important; font-weight: 600 !important; color: var(--text-1) !important; margin-bottom: 0.25rem !important; }
    h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: var(--text-1) !important; }
    h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: var(--text-2) !important; }

    /* ── Sidebar section labels ── */
    .sidebar-section-label {
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        color: var(--text-3) !important;
        text-transform: uppercase !important;
        padding: 0.5rem 0 0.2rem !important;
    }

    /* ── Connection status badge ── */
    .conn-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--green-soft);
        border: 1px solid rgba(22,163,74,0.25);
        border-radius: 20px;
        padding: 4px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--green);
        margin: 0.4rem 0;
    }
    .conn-dot { width:7px; height:7px; background:var(--green); border-radius:50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* ── Welcome card ── */
    .welcome-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .welcome-card h3 { color: var(--text-2) !important; font-size: 0.75rem !important; text-transform: uppercase !important; letter-spacing: 0.08em !important; margin-bottom: 1rem !important; }
    .step-row { display:flex; align-items:flex-start; gap:0.75rem; margin-bottom:0.85rem; }
    .step-num { background:var(--accent-soft); color:var(--accent); width:24px; height:24px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.72rem; font-weight:700; flex-shrink:0; }
    .step-text { font-size:0.85rem; color:var(--text-2); padding-top:2px; }

    /* ── Suggested question buttons ── */
    .stButton button[data-testid*="sq_btn"],
    .stButton button[data-testid*="saved_btn"],
    .stButton button[data-testid*="ai_sq_btn"] {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        text-align: left !important;
        width: 100% !important;
        justify-content: flex-start !important;
        color: var(--text-2) !important;
        font-size: 0.78rem !important;
    }
    .stButton button[data-testid*="sq_btn"]:hover,
    .stButton button[data-testid*="saved_btn"]:hover,
    .stButton button[data-testid*="ai_sq_btn"]:hover {
        color: var(--accent) !important;
        border-color: var(--accent) !important;
    }

    /* ── Disconnect button ── */
    button[data-testid*="Disconnect"] {
        border-color: var(--red) !important;
        color: var(--red) !important;
    }
    button[data-testid*="Disconnect"]:hover {
        background: var(--red-soft) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────
for key, default in {
    "mode": "Database AI Assistant",
    "db_type": "MS SQL",
    "mssql_conn": None,
    "all_tables": [],
    "selected_tables": [],
    "chat_history": [],
    "table_multiselect": [],
    "suggested_questions": [],
    "last_schema_for_questions": "",
    "pending_question": None,
    "db_identifier": "",
    "business_rules": "",
    "project_files": [],
    "project_answer": "",
    "project_chunks": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default



# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────
with st.sidebar:
    # App wordmark
    st.markdown("""
    <div style="padding:0.25rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:0.75rem;">
        <span style="font-size:1.05rem; font-weight:700; color:var(--text-1); letter-spacing:-0.01em;">⬡ AI&nbsp;<span style="color:var(--accent)">Assistant</span></span>
    </div>
    """, unsafe_allow_html=True)

    # Mode Toggle
    mode = st.radio("Mode", ["Database AI Assistant", "File Reader AI Assistant"], key="mode", label_visibility="collapsed")
    st.divider()

    if mode == "Database AI Assistant":
        # Database selection
        st.markdown('<p class="sidebar-section-label">Database</p>', unsafe_allow_html=True)
        db_type = st.selectbox(
            "Database Type",
            options=DATABASES,
            index=DATABASES.index(st.session_state["db_type"]) if st.session_state["db_type"] in DATABASES else 0,
            label_visibility="collapsed",
        )

        if st.session_state["db_type"] != db_type:
            if st.session_state["mssql_conn"] is not None:
                try:
                    st.session_state["mssql_conn"].close()
                except Exception:
                    pass
            st.session_state["mssql_conn"] = None
            st.session_state["all_tables"] = []
            st.session_state["selected_tables"] = []
            st.session_state.pop("table_multiselect", None)
            st.session_state["chat_history"] = []
            st.session_state["db_type"] = db_type
            st.rerun()

        # Environment selection
        st.markdown('<p class="sidebar-section-label">Environment</p>', unsafe_allow_html=True)
        env = st.selectbox("Environment", ["dev", "qa", "prod"], key="db_environment", label_visibility="collapsed")
        
        # Model selection
        st.markdown('<p class="sidebar-section-label">Model</p>', unsafe_allow_html=True)
        available_models = list_ollama_models()
        if available_models:
            selected_model = st.selectbox("Model", options=available_models, key="db_model", label_visibility="collapsed")
        else:
            st.warning("No Ollama models found. Run `ollama pull <model>` first.")
            selected_model = st.text_input("Model name", value="qwen2.5-coder:7b", label_visibility="collapsed")

        # ── Database Mode connection form ──
        st.markdown('<p class="sidebar-section-label">Connection</p>', unsafe_allow_html=True)
        conn_params = {}
        connect_disabled = False

        if db_type == "MS SQL":
            available_drivers = get_available_drivers()
            if available_drivers:
                driver = st.selectbox("ODBC Driver", options=available_drivers, index=len(available_drivers) - 1, label_visibility="collapsed")
            else:
                st.error("No SQL Server ODBC driver found. Install ODBC Driver 17 or 18.")
                driver = None
                connect_disabled = True

            c1, c2 = st.columns(2)
            server   = c1.text_input("Server",   placeholder="Host\\Instance", key="db_server", label_visibility="collapsed")
            database = c2.text_input("Database", placeholder="Database",       key="db_database", label_visibility="collapsed")

            auth_mode = st.selectbox("Auth", ["Windows Authentication", "SQL Server Authentication"], key="db_auth_mode", label_visibility="collapsed")
            username = password = None
            if auth_mode == "SQL Server Authentication":
                c3, c4 = st.columns(2)
                username = c3.text_input("Username", placeholder="Username", key="db_username", label_visibility="collapsed")
                password = c4.text_input("Password", type="password", placeholder="Password", key="db_password", label_visibility="collapsed")

            conn_params = {"server": server, "database": database, "auth_mode": auth_mode, "username": username, "password": password, "driver": driver}

        elif db_type == "MySQL":
            c1, c2 = st.columns(2)
            host = c1.text_input("Host", value="localhost", placeholder="localhost", label_visibility="collapsed")
            port = c2.number_input("Port", value=3306, label_visibility="collapsed")
            c3, c4 = st.columns(2)
            database = c3.text_input("Database", placeholder="Database", label_visibility="collapsed")
            username = c4.text_input("Username", value="root", placeholder="root", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            conn_params = {"host": host, "port": port, "database": database, "username": username, "password": password}

        elif db_type == "PostgreSQL":
            c1, c2 = st.columns(2)
            host = c1.text_input("Host", value="localhost", placeholder="localhost", label_visibility="collapsed")
            port = c2.number_input("Port", value=5432, label_visibility="collapsed")
            c3, c4 = st.columns(2)
            database = c3.text_input("Database", placeholder="Database", label_visibility="collapsed")
            username = c4.text_input("Username", value="postgres", placeholder="postgres", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            conn_params = {"host": host, "port": port, "database": database, "username": username, "password": password}

        elif db_type == "Oracle Database":
            c1, c2 = st.columns(2)
            host = c1.text_input("Host", value="localhost", placeholder="localhost", label_visibility="collapsed")
            port = c2.number_input("Port", value=1521, label_visibility="collapsed")
            c3, c4 = st.columns(2)
            service_name = c3.text_input("Service Name", value="ORCL", placeholder="ORCL", label_visibility="collapsed")
            username = c4.text_input("Username", placeholder="Username", label_visibility="collapsed")
            password = st.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            conn_params = {"host": host, "port": port, "service_name": service_name, "username": username, "password": password}

        elif db_type == "SQLite":
            db_path = st.text_input("Database file path", value="local.db", placeholder="path/to/database.db", label_visibility="collapsed")
            conn_params = {"db_path": db_path}

        elif db_type == "MongoDB":
            uri = st.text_input("Connection URI", value="mongodb://localhost:27017/", label_visibility="collapsed")
            database = st.text_input("Database name", value="test", label_visibility="collapsed")
            conn_params = {"uri": uri, "database": database}

        elif db_type == "Redis":
            c1, c2 = st.columns(2)
            host = c1.text_input("Host", value="localhost", placeholder="localhost", label_visibility="collapsed")
            port = c2.number_input("Port", value=6379, label_visibility="collapsed")
            c3, c4 = st.columns(2)
            password = c3.text_input("Password", type="password", placeholder="Password", label_visibility="collapsed")
            db_index = c4.number_input("DB Index", value=0, min_value=0, label_visibility="collapsed")
            conn_params = {"host": host, "port": port, "password": password, "db_index": db_index}

        connect_clicked = st.button("Connect", use_container_width=True, type="primary", disabled=connect_disabled)

        if connect_clicked:
            if db_type == "MS SQL" and (not conn_params.get("server") or not conn_params.get("database")):
                st.error("Please fill in Server and Database.")
            elif db_type in ["MySQL", "PostgreSQL", "Oracle Database"] and (not conn_params.get("database") and db_type != "Oracle Database" or not conn_params.get("host")):
                st.error("Please fill in Host and Database/Service Name.")
            elif db_type == "SQLite" and not conn_params.get("db_path"):
                st.error("Please fill in Database File Path.")
            elif db_type == "MongoDB" and (not conn_params.get("uri") or not conn_params.get("database")):
                st.error("Please fill in URI and Database Name.")
            else:
                with st.spinner("Connecting…"):
                    try:
                        conn = connect_db(db_type, conn_params)
                        st.session_state["mssql_conn"] = conn
                        st.session_state["all_tables"] = get_db_tables(conn, db_type)
                        st.session_state["selected_tables"] = []
                        st.session_state.pop("table_multiselect", None)
                        st.session_state["chat_history"] = []
                        env = st.session_state.get('db_environment', 'dev')
                        db_ident = f"{env}_{db_type}_{conn_params.get('database') or conn_params.get('db_path') or conn_params.get('host') or 'default'}"
                        st.session_state["db_identifier"] = db_ident
                        st.session_state["business_rules"] = get_business_skills(db_ident)
                    except Exception as exc:
                        st.error(f"Connection failed: {exc}")

        # ── Connected state ──
        if st.session_state["mssql_conn"] is not None:
            st.markdown(f'<div class="conn-badge"><div class="conn-dot"></div>Connected · {db_type}</div>', unsafe_allow_html=True)

            label_name = "Collections" if db_type == "MongoDB" else ("Key Patterns" if db_type == "Redis" else "Tables")
            all_tables = st.session_state["all_tables"]

            if db_type in ["MS SQL", "PostgreSQL", "Oracle Database"]:
                table_labels = [f"{s}.{t}" for s, t in all_tables]
            else:
                table_labels = [t for s, t in all_tables]

            if not table_labels:
                st.warning(f"No {label_name.lower()} found.")
            else:
                st.markdown(f'<p class="sidebar-section-label">{label_name}</p>', unsafe_allow_html=True)
                st.multiselect(
                    label_name,
                    options=table_labels,
                    key="table_multiselect",
                    label_visibility="collapsed",
                    placeholder=f"Select {label_name.lower()}…",
                )

                if db_type in ["MS SQL", "PostgreSQL", "Oracle Database"]:
                    st.session_state["selected_tables"] = [
                        (lbl.split(".", 1)[0], lbl.split(".", 1)[1])
                        for lbl in st.session_state["table_multiselect"]
                    ]
                else:
                    st.session_state["selected_tables"] = [
                        ("Default", lbl) for lbl in st.session_state["table_multiselect"]
                    ]

            if st.button("Disconnect", use_container_width=True):
                try:
                    st.session_state["mssql_conn"].close()
                except Exception:
                    pass
                st.session_state["mssql_conn"] = None
                st.session_state["all_tables"] = []
                st.session_state["selected_tables"] = []
                st.session_state.pop("table_multiselect", None)
                st.session_state["chat_history"] = []
                st.session_state["db_identifier"] = ""
                st.session_state["business_rules"] = ""
                st.rerun()

    elif mode == "File Reader AI Assistant":
        db_type = st.session_state["db_type"]
        st.markdown('<p class="sidebar-section-label">Upload Document</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload .docx or .txt file", type=["docx", "txt"], label_visibility="collapsed")
        
        # Model selection
        st.markdown('<p class="sidebar-section-label">Model</p>', unsafe_allow_html=True)
        available_models = list_ollama_models()
        if available_models:
            selected_model = st.selectbox("Model", options=available_models, key="file_model", label_visibility="collapsed")
        else:
            st.warning("No Ollama models found.")
            selected_model = st.text_input("Model name", value="llama3:latest", label_visibility="collapsed")

        if st.button("Process File", use_container_width=True):
            if not uploaded_file:
                st.warning("Please upload a file first.")
            else:
                with st.spinner("Processing document..."):
                    import os
                    import v2_rag_engine
                    
                    os.makedirs(v2_rag_engine.V2_UPLOAD_DIR, exist_ok=True)
                    dest_path = os.path.join(v2_rag_engine.V2_UPLOAD_DIR, uploaded_file.name)
                    
                    # save file
                    with open(dest_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        
                    try:
                        res = v2_rag_engine.ingest_file_v2(dest_path, uploaded_file.name)
                        st.session_state["project_files"] = [uploaded_file.name]
                        st.session_state["project_answer"] = ""
                        st.session_state["project_chunks"] = []
                        st.success(f"File processed successfully! ({res.get('chunks_added', 0)} chunks loaded)")
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

    st.divider()
    if st.button("↺  Reset App", use_container_width=True, help="Reset all inputs and settings"):
        if "mssql_conn" in st.session_state and st.session_state["mssql_conn"] is not None:
            try:
                st.session_state["mssql_conn"].close()
            except Exception:
                pass
        st.session_state.clear()
        st.rerun()

# ─────────────────────────────────────────────
# Main — File Reader Mode
# ─────────────────────────────────────────────
if mode == "File Reader AI Assistant":
    st.markdown("## 📄 Document AI Assistant")
    st.caption("Ask questions about your uploaded document")

    if not st.session_state.get("project_files"):
        st.markdown("""
        <div class="welcome-card">
            <h3>Getting started</h3>
            <div class="step-row"><div class="step-num">1</div><div class="step-text">Upload a .docx or .txt file in the sidebar</div></div>
            <div class="step-row"><div class="step-num">2</div><div class="step-text">Click <strong>Process File</strong> to analyze it</div></div>
            <div class="step-row"><div class="step-num">3</div><div class="step-text">Ask questions about your document in plain English</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    project_question = st.chat_input("Ask about the document...")

    if project_question:
        if not project_question.strip():
            st.warning("Please enter a question.")
            st.stop()

        import v2_rag_engine
        from ollama_client import ask_ollama

        with st.spinner("Analyzing document..."):
            # Retrieve relevant chunks
            chunks = v2_rag_engine.retrieve_and_rerank(project_question)
            
            if not chunks:
                answer = "This detail is currently not available in the uploaded document."
            else:
                # Build prompt
                prompt = "You must answer the user's question using ONLY the provided document context.\n\n"
                prompt += "DOCUMENT CONTEXT:\n" + "\n\n---\n\n".join(chunks) + "\n\n"
                prompt += f"USER QUESTION: {project_question}\n\n"
                prompt += "Answer directly without 'According to the document'. If the answer is not in the context, say 'This detail is currently not available.'"

                answer = ask_ollama(prompt, model=st.session_state.get("file_model", "llama3:latest"))

        st.session_state["project_answer"] = answer
        st.session_state["project_chunks"] = chunks

    if st.session_state.get("project_answer"):
        st.markdown("**Answer**")
        st.info(st.session_state["project_answer"])

        if st.session_state.get("project_chunks"):
            with st.expander("📎 View Reference Context"):
                for i, chunk in enumerate(st.session_state["project_chunks"]):
                    st.markdown(f"**Chunk {i+1}**\n{chunk}")
                    st.divider()

    st.stop()

# ─────────────────────────────────────────────
# Main — Database Mode landing
# ─────────────────────────────────────────────
if st.session_state["mssql_conn"] is None:
    desc_query_lang = "T-SQL" if db_type == "MS SQL" else ("NoSQL queries" if db_type in ["MongoDB", "Redis"] else "SQL")

    st.markdown(f"## 🗄️ {db_type} Assistant")
    st.markdown(f"""
    <div class="welcome-card">
        <h3>How it works</h3>
        <div class="step-row"><div class="step-num">1</div><div class="step-text">Fill in your connection details in the sidebar and click <strong>Connect</strong></div></div>
        <div class="step-row"><div class="step-num">2</div><div class="step-text">Select the tables (or collections) relevant to your question</div></div>
        <div class="step-row"><div class="step-num">3</div><div class="step-text">Ask a question in plain English — the AI writes {desc_query_lang}, runs it, and explains the result</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────
# Main — Database connected
# ─────────────────────────────────────────────
conn = st.session_state["mssql_conn"]
selected_tables = st.session_state["selected_tables"]
tab_schema, tab_query, tab_rules = st.tabs(["📐  Schema", "💬  Query", "✨ Generate Rule"])


# ── Schema Tab ──
with tab_schema:
    all_tables = st.session_state["all_tables"]

    if not all_tables:
        st.warning("No items found.")
    else:
        browse_target = selected_tables if selected_tables else all_tables
        label_name = "collection(s)" if db_type == "MongoDB" else ("key pattern(s)" if db_type == "Redis" else "table(s)")

        col_info, col_search = st.columns([3, 2])
        with col_info:
            if selected_tables:
                st.caption(f"Showing **{len(selected_tables)}** selected {label_name}")
            else:
                st.caption(f"All **{len(all_tables)}** {label_name} · select in sidebar to filter")
        with col_search:
            search = st.text_input("Filter", placeholder="Search by name…", label_visibility="collapsed")

        for schema_name, table_name in browse_target:
            label = f"{schema_name}.{table_name}" if db_type in ["MS SQL", "PostgreSQL", "Oracle Database"] else table_name
            if search and search.lower() not in label.lower():
                continue

            with st.expander(f"**{label}**", expanded=False):
                try:
                    metadata = get_db_table_metadata(conn, db_type, schema_name, table_name)
                    columns = metadata["columns"]
                    pks = set(metadata["pks"])
                    fks_list = metadata["fks"]
                    fk_map = {fk["column"]: fk for fk in fks_list}

                    rows_data = []
                    for col in columns:
                        type_str = col["type"].upper()
                        if col.get("max_length"):
                            type_str += f"({col['max_length']})"
                        badges = []
                        if col["name"] in pks:
                            badges.append("🔑 PK")
                        if col["name"] in fk_map:
                            fk = fk_map[col["name"]]
                            badges.append(f"🔗 FK → {fk['ref_schema']}.{fk['ref_table']}.{fk['ref_column']}")
                        rows_data.append({
                            "Column": col["name"],
                            "Type": type_str,
                            "Nullable": col.get("nullable", "YES"),
                            "Default": col.get("default") or "",
                            "Keys": "  ".join(badges),
                        })

                    st.dataframe(pd.DataFrame(rows_data), use_container_width=True, hide_index=True)

                    if fks_list:
                        st.markdown("**Relationships**")
                        for fk in fks_list:
                            st.markdown(f"- `{fk['column']}` → `{fk['ref_schema']}.{fk['ref_table']}.{fk['ref_column']}`")

                except Exception as exc:
                    st.error(f"Could not load {label}: {exc}")


# ── Query Tab ──
with tab_query:
    label_name = "collections" if db_type == "MongoDB" else ("key patterns" if db_type == "Redis" else "tables")
    if not selected_tables:
        st.warning(f"Select at least one {label_name.rstrip('s')} from the sidebar so the AI knows your schema.")
        st.stop()

    # Collapsible context panels
    with st.expander("AI Skills (Dynamic Business Rules)", expanded=False):
        st.caption("Add categories, trigger keywords, and rules. Separate keywords with commas. Empty keywords apply globally.")
        skills = st.session_state.get("business_rules", [])
        
        # Convert to a format data_editor can use
        df_data = []
        for s in skills:
            df_data.append({
                "category": s.get("category", ""),
                "keywords": ", ".join(s.get("keywords", [])) if isinstance(s.get("keywords"), list) else s.get("keywords", ""),
                "rule_text": s.get("rule_text", "")
            })
        if not df_data:
            df_data.append({"category": "", "keywords": "", "rule_text": ""})
            
        import pandas as pd
        df = pd.DataFrame(df_data)
        
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="skills_editor", hide_index=True)
        
        if st.button("?? Save Skills"):
            new_skills = []
            for _, row in edited_df.iterrows():
                cat = str(row.get("category", "")).strip()
                r_text = str(row.get("rule_text", "")).strip()
                if r_text:
                    kws_raw = str(row.get("keywords", ""))
                    kws = [k.strip() for k in kws_raw.split(",") if k.strip()]
                    new_skills.append({"category": cat, "keywords": kws, "rule_text": r_text})
            st.session_state["business_rules"] = new_skills
            save_business_skills(st.session_state.get("db_identifier", ""), new_skills)
            st.success("AI Skills saved successfully!")

    with st.expander("💡 Suggested questions", expanded=False):
        try:
            current_schema_text = get_db_schema_text(conn, db_type, selected_tables)
        except Exception:
            current_schema_text = ""

        if st.session_state["last_schema_for_questions"] != current_schema_text:
            st.session_state["suggested_questions"] = []
            st.session_state["last_schema_for_questions"] = current_schema_text

        with st.container(height=350):
            saved_suggs = get_user_suggestions(db_type, selected_tables)
            if saved_suggs:
                st.markdown('<p style="font-size:0.72rem;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem">📌 Saved</p>', unsafe_allow_html=True)
                for i, sq in enumerate(saved_suggs):
                    if st.button(sq, key=f"saved_btn_{i}", use_container_width=True):
                        st.session_state["pending_question"] = sq

            st.markdown('<p style="font-size:0.72rem;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:0.08em;margin:0.6rem 0 0.4rem">✨ AI generated</p>', unsafe_allow_html=True)
            if not st.session_state["suggested_questions"]:
                if st.button("Generate suggestions", use_container_width=True):
                    with st.spinner("Analysing schema…"):
                        try:
                            suggs = generate_suggested_questions(current_schema_text, model=selected_model)
                            st.session_state["suggested_questions"] = suggs
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed: {exc}")

            for i, sq in enumerate(st.session_state["suggested_questions"]):
                if st.button(sq, key=f"ai_sq_btn_{i}", use_container_width=True):
                    st.session_state["pending_question"] = sq

    st.divider()

    # Query language labels
    query_lang_name = "T-SQL" if db_type == "MS SQL" else ("JSON Command" if db_type in ["MongoDB", "Redis"] else "SQL")
    query_code_lang = "json" if db_type in ["MongoDB", "Redis"] else "sql"

    # Toolbar row
    col_write, col_hist, col_clear, col_spacer = st.columns([2, 2, 2, 3])
    with col_write:
        with st.popover("✍️ Write query"):
            raw_sql_input = st.text_area("Raw SQL", height=130, key="raw_sql_text", label_visibility="collapsed", placeholder="Write your query here…")
            if st.button("▶ Run", use_container_width=True):
                st.session_state["raw_query_pending"] = raw_sql_input
                st.rerun()
    with col_hist:
        with st.popover("🕒 History"):
            hist_items = get_chat_history()
            
            if hist_items:
                with st.expander("🗑 Clear DB History", expanded=False):
                    st.warning("Permanently delete all history?")
                    if st.button("Yes, Clear History", key="confirm_clear_history_btn", use_container_width=True):
                        clear_chat_history()
                        st.session_state["chat_history"] = []
                        st.rerun()
                st.divider()

            if not hist_items:
                st.info("No past questions found.")
            else:
                for i, row in enumerate(hist_items[:15]):
                    q = row[0]
                    if st.button(f"{q}", key=f"hist_btn_{i}", use_container_width=True):
                        st.session_state["pending_question"] = q
                        st.rerun()
    with col_clear:
        if st.button("🗑 Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    # Chat history
    for i, entry in enumerate(st.session_state["chat_history"]):
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(f"**{query_lang_name}**")
            st.code(entry["sql"], language=query_code_lang)
            if entry.get("error"):
                st.error(entry["error"])
            else:
                if entry.get("df") is not None:
                    st.dataframe(entry["df"], use_container_width=True)
                    st.caption(f"{entry['row_count']} row(s)")
                if entry.get("summary"):
                    st.markdown("**Answer**")
                    st.info(entry["summary"])

                if st.button("💾 Save as suggestion", key=f"save_sug_{i}"):
                    save_user_suggestion(db_type, selected_tables, entry["question"])
                    st.toast("Saved as a suggestion!")
                    st.rerun()

    # Resolve pending question
    question_clicked = st.session_state.get("pending_question")
    if question_clicked:
        st.session_state["pending_question"] = None

    question_input = st.chat_input("Ask about your data…", key="db_chat_input")
    question = question_input or question_clicked
    raw_query = st.session_state.get("raw_query_pending")
    if raw_query:
        st.session_state["raw_query_pending"] = None

    if question or raw_query:
        display_text = raw_query if raw_query else question
        with st.chat_message("user"):
            st.markdown(f"```sql\n{display_text}\n```" if raw_query else display_text)

        with st.chat_message("assistant"):
            if raw_query:
                sql = raw_query
            else:
                with st.spinner(f"Generating {query_lang_name}…"):
                    try:
                        schema_text = get_db_schema_text(conn, db_type, selected_tables)
                        active_rules = filter_rules_by_keywords(question, st.session_state.get("db_identifier", ""))
                        sql = generate_db_query(
                            db_type, question, schema_text,
                            business_rules=active_rules,
                            model=selected_model
                        )
                    except Exception as exc:
                        st.error(f"Query generation failed: {exc}")
                        st.stop()

            st.markdown(f"**{'Executing' if raw_query else 'Generated'} {query_lang_name}**")
            st.code(sql, language=query_code_lang)

            is_safe, reason = validate_db_query(db_type, sql)
            if not is_safe:
                error_msg = f"Unsafe query blocked: {reason}"
                st.error(error_msg)
                st.session_state["chat_history"].append({
                    "question": f"```sql\n{raw_query}\n```" if raw_query else question,
                    "sql": sql, "error": error_msg, "df": None, "row_count": 0, "summary": None,
                })
                st.stop()

            with st.spinner("Running query…"):
                try:
                    columns, rows = execute_db_query(conn, db_type, sql)
                except Exception as exc:
                    exc_str = str(exc)
                    if any(err in exc_str for err in ["Invalid column name", "Invalid object name", "Incorrect syntax", "could not be bound"]):
                        error_msg = f"⚠️ **Missing tables or invalid query** — make sure all required tables are selected in the sidebar.\n\n*({exc_str})*"
                    else:
                        error_msg = f"Query error: {exc_str}"
                    st.error(error_msg)
                    st.session_state["chat_history"].append({
                        "question": f"```sql\n{raw_query}\n```" if raw_query else question,
                        "sql": sql, "error": error_msg, "df": None, "row_count": 0, "summary": None,
                    })
                    st.stop()

            df = None
            if rows:
                df = pd.DataFrame(rows, columns=columns)
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(rows)} row(s) returned")
            else:
                st.warning("No records returned.")

            with st.spinner("Summarising…"):
                try:
                    summary = generate_db_answer_summary(db_type, question, sql, columns, rows, model=selected_model)
                except Exception as exc:
                    summary = f"Summary failed: {exc}"

            st.markdown("**Answer**")
            st.info(summary)

            question_to_save = raw_query if raw_query else question
            save_chat(question_to_save, sql, summary)
            st.session_state["chat_history"].append({
                "question": f"```sql\n{raw_query}\n```" if raw_query else question,
                "sql": sql, "error": None, "df": df, "row_count": len(rows), "summary": summary,
            })
            st.rerun()

# ── Generate Rules Tab ──
with tab_rules:
    st.markdown("### Generate Rule from Example")
    st.markdown("Provide a question and the correct SQL. The AI will extract the business rule for you to copy.")
    gen_q = st.text_input("Example Question", key="tab_gen_q", placeholder="e.g. Find busiest roads")
    gen_sql = st.text_area("Correct SQL Query", key="tab_gen_sql", placeholder="SELECT * FROM Roads WHERE AADT > 10000", height=100)
    
    if st.button("Generate Rule", use_container_width=True):
        if not gen_q or not gen_sql:
            st.error("Please provide both a question and a SQL query.")
        else:
            with st.spinner("Analyzing..."):
                schema_text = get_db_schema_text(conn, db_type, selected_tables)
                new_rule = generate_rule_from_sql(gen_q, gen_sql, schema_text, selected_model)
                st.success("Rule Generated!")
                st.code(new_rule, language="text")
                st.info("Hover over the rule above and click the Copy icon on the right. Then paste it into your Business Rules.")