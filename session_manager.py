import uuid
import time
import logging

logger = logging.getLogger(__name__)

SESSION_TTL = 86400  # 24 hours in seconds

active_sessions = {}
file_sessions = {}

def cleanup_sessions():
    now = time.time()
    expired_db = [sid for sid, data in active_sessions.items() if now - data["timestamp"] > SESSION_TTL]
    for sid in expired_db:
        active_sessions.pop(sid, None)
        
    expired_file = [sid for sid, data in file_sessions.items() if now - data["timestamp"] > SESSION_TTL]
    for sid in expired_file:
        file_sessions.pop(sid, None)

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
