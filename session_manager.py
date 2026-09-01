import uuid
import time
import logging

logger = logging.getLogger(__name__)

SESSION_TTL = 86400  # 24 hours in seconds

active_sessions = {}
file_sessions = {}
db_chat_sessions = {}

def cleanup_sessions():
    now = time.time()
    expired_db = [sid for sid, data in active_sessions.items() if now - data["timestamp"] > SESSION_TTL]
    for sid in expired_db:
        active_sessions.pop(sid, None)
        
    expired_file = [sid for sid, data in file_sessions.items() if now - data["timestamp"] > SESSION_TTL]
    for sid in expired_file:
        file_sessions.pop(sid, None)

    expired_chat = [sid for sid, data in db_chat_sessions.items() if now - data["timestamp"] > SESSION_TTL]
    for sid in expired_chat:
        db_chat_sessions.pop(sid, None)

def create_session(connection_info):
    cleanup_sessions()
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {"info": connection_info, "timestamp": time.time()}
    return session_id


def get_session(session_id):
    cleanup_sessions()
    session_data = active_sessions.get(session_id)
    if session_data:
        session_data["timestamp"] = time.time()
        return session_data["info"]
    return None


def remove_session(session_id):
    active_sessions.pop(session_id, None)


def get_file_session_history(session_id: str) -> list:
    cleanup_sessions()
    if not session_id:
        return []
    session_data = file_sessions.get(session_id)
    if session_data:
        session_data["timestamp"] = time.time()
        return session_data["history"]
    return []


# ---------------------------------------------------------------------------
# Database chat sessions
#
# /fetch-answer used to be stateless, so a follow-up such as "name of it" had
# no antecedent and produced SCHEMA_INSUFFICIENT. These keep the recent turns.
# ---------------------------------------------------------------------------
MAX_DB_HISTORY = 6


def get_db_chat_history(session_id: str) -> list:
    cleanup_sessions()
    if not session_id:
        return []
    entry = db_chat_sessions.get(session_id)
    if not entry:
        return []
    entry["timestamp"] = time.time()
    return entry["history"]


MAX_RESULT_ROWS_KEPT = 5
MAX_CELL_CHARS = 80


def add_db_chat_turn(session_id: str, question: str, sql: str, answer: str,
                     columns=None, rows=None, max_history: int = MAX_DB_HISTORY):
    """
    Record one turn, including a preview of the rows it returned.

    The rows matter: a follow-up like "chainages of this road" needs the key
    value from the previous result (RoadCode = 'N0001'). Keeping only the prose
    answer left the model with no key, and it invented one.
    """
    cleanup_sessions()
    if not session_id:
        return

    preview_rows = []
    for row in (rows or [])[:MAX_RESULT_ROWS_KEPT]:
        preview_rows.append([
            (str(v)[:MAX_CELL_CHARS] if v is not None else None) for v in row
        ])

    entry = db_chat_sessions.setdefault(
        session_id, {"history": [], "timestamp": time.time()}
    )
    entry["history"].append({
        "question": question,
        "sql": sql,
        "answer": answer,
        "columns": list(columns or []),
        "rows": preview_rows,
    })
    entry["timestamp"] = time.time()
    if len(entry["history"]) > max_history:
        entry["history"] = entry["history"][-max_history:]


def clear_db_chat_history(session_id: str):
    db_chat_sessions.pop(session_id, None)


def add_file_session_history(session_id: str, question: str, answer: str, max_history: int = 5):
    cleanup_sessions()
    if not session_id:
        return
    if session_id not in file_sessions:
        file_sessions[session_id] = {"history": [], "timestamp": time.time()}
    
    file_sessions[session_id]["history"].append({"question": question, "answer": answer})
    file_sessions[session_id]["timestamp"] = time.time()
    
    if len(file_sessions[session_id]["history"]) > max_history:
        file_sessions[session_id]["history"] = file_sessions[session_id]["history"][-max_history:]

