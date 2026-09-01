"""
FastAPI wrapper — MSSQL AI Assistant
Exposes the core logic as REST endpoints consumable by .NET / Java apps.

Endpoints:
  POST /connect          — test DB connection and return available tables
  POST /schema           — get schema text for selected tables
  POST /ask              — natural-language question → SQL → results → AI answer
  GET  /models           — list available Ollama models
  GET  /health           — liveness check

Auth: Bearer JWT (HS256). Set SECRET_KEY env var.
      Pass Authorization: Bearer <token> header on every request.

Run:
  pip install fastapi uvicorn python-jose[cryptography] pydantic
  uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
"""

import os
import uuid
import logging
from datetime import datetime, timezone
import tempfile
import shutil
from typing import Any, Optional

# --- FIX FOR OLLAMA CONNECTION ---
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
# ---------------------------------


from fastapi import Depends, FastAPI, HTTPException, status, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from mssql_connector import connect_mssql
from mssql_schema_reader import get_all_tables, get_selected_schema_text, get_table_metadata
from schema_profiler import profile_and_render, profile_and_render_cached
from mssql_sql_generator import generate_tsql, generate_answer_summary
from mssql_executor import validate_tsql, execute_tsql
import skills as skills_registry
from skills import describe_active_skills
from db_privileges import enforce_read_only, describe as describe_privileges
from query_library import (
    add_examples,
    parse_upload,
    load_library,
    save_library,
    library_stats,
    retrieve,
    load_examples_for_prompt,
)
from ollama_client import list_ollama_models, ask_ollama, ask_ollama_stream
from audit_logger import log_query   # see audit_logger.py
from session_manager import (
    create_session,
    get_session,
    remove_session,
    get_file_session_history,
    add_file_session_history,
    get_db_chat_history,
    add_db_chat_turn,
    clear_db_chat_history,
)
from cryptography.fernet import Fernet

# v2_rag_engine pulls in chromadb, sentence-transformers and torch, and builds a
# cross-encoder on import. Loading that eagerly costs a slow startup and makes
# the database endpoints unavailable on a host without the RAG stack installed,
# so it is imported on first use instead.
_v2_rag_engine = None


def rag_engine():
    """Import v2_rag_engine on demand; 503 if the RAG dependencies are absent."""
    global _v2_rag_engine
    if _v2_rag_engine is None:
        try:
            import v2_rag_engine as _module
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Document/RAG features are unavailable on this host: "
                    f"{exc}. Install chromadb, sentence-transformers, python-docx "
                    "and pypdf to enable them."
                ),
            )
        _v2_rag_engine = _module
    return _v2_rag_engine
# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"

# Operator-only switch for /fetch-answer diagnostics (server, database, tables,
# raw rows, active skills). Leave unset in production: the answer payload should
# never carry infrastructure details to a caller.
FETCH_ANSWER_DEBUG = os.environ.get("FETCH_ANSWER_DEBUG", "").strip().lower() in ("1", "true", "yes")

ENCRYPTION_KEY_FILE = "encryption_secret.key"

def get_cipher():
    if not os.path.exists(ENCRYPTION_KEY_FILE):
        key = Fernet.generate_key()
        with open(ENCRYPTION_KEY_FILE, "wb") as key_file:
            key_file.write(key)
    with open(ENCRYPTION_KEY_FILE, "rb") as key_file:
        key = key_file.read()
    return Fernet(key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mssql_api")

MULTILINGUAL_PROMPT_TEMPLATE = (
    "You are the official AI Assistant for this application.\n\n"
    "CRITICAL RULES:\n"
    "1. DIRECT & NATURAL ANSWERS ONLY: Begin your answer directly with facts and data. NEVER start answers with 'According to...', 'Based on...', 'According to the SYSTEM_DATABASE_RECORDS...', 'According to the provided information...', 'According to the data...', 'The conversation context...', 'After reviewing...', 'After analyzing...' or any intro/preamble phrases. State the fact immediately as the very first word of your answer.\n"
    "2. STRICT NUMERICAL & DATA FACT ACCURACY: NEVER guess, estimate, or hallucinate numbers or statistics. Read exact numerical figures strictly from the records. NEVER perform mathematical calculations, additions, or combinations of totals unless explicitly provided.\n"
    "3. NO EXTERNAL OR GENERAL KNOWLEDGE: ONLY answer using records present in the system context. NEVER use your own training knowledge, general world knowledge, or external information. Even if asked about famous people, capital cities, or basic facts (e.g. Prime Minister of India, capital of France), you MUST return the fallback message.\n"
    "4. NO INTERNAL REASONING: NEVER narrate your reasoning process. Do NOT output phrases like 'To answer this, I will look for...' or 'Since none are found...' Just output the final answer.\n"
    "5. HIDE FILE & META REFERENCES: NEVER mention or use words like 'document', 'file', 'PDF', 'page', 'manual', 'section', 'chapter', 'appendix', 'text', 'provided information', 'SYSTEM_DATABASE_RECORDS'. Present all information natively.\n"
    "6. FORMATTING & LISTS: Use clear line breaks and Markdown formatting for multi-step processes or lists.\n"
    "7. MISSING INFORMATION FALLBACK: If requested details are missing, cannot be answered, or fall outside the provided context, you MUST output EXACTLY ONE SENTENCE: 'This detail is currently not available in our system.' Do NOT add any preamble, do NOT add 'I am sorry', and do NOT explain your reasoning.\n"
    "8. STRICT TRANSLATION: You MUST translate your final answer (including the fallback message) into the EXACT SAME LANGUAGE as the user's question.\n"
    "9. OUT-OF-SCOPE QUERIES: If the question is unrelated to road/bridge/PWD data (e.g. simple arithmetic, jokes, unrelated topics), output exactly: 'This detail is currently not available in our system.'\n"
)

app = FastAPI(
    title="MSSQL AI Assistant API",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(.*\.)?satragroup\.in",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# JWT Auth
# ─────────────────────────────────────────────
bearer_scheme = HTTPBearer()


# def verify_token(
#     credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
# ) -> dict:
#     token = credentials.credentials
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         return payload
#     except JWTError as exc:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail=f"Invalid or expired token: {exc}",
#             headers={"WWW-Authenticate": "Bearer"},
#         )

# TEMPORARY - Disable JWT Authentication for Testing
def verify_token():
    return {
        "sub": "test-user"
    }
# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────
class ConnectRequest(BaseModel):
    environment: str = "default"
    server: str
    database: str
    auth_mode: str                    # "Windows Authentication" | "SQL Server Authentication"
    username: Optional[str] = None
    password: Optional[str] = None
    driver: Optional[str] = None


class SchemaRequest(BaseModel):
    session_id: str
    tables: list[str]

class AskRequest(BaseModel):
    session_id: str
    tables: list[str]
    question: str
    model: Optional[str] = None

class AskFilesRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    model: Optional[str] = None
    filename: Optional[str] = None

class ConnectResponse(BaseModel):
    session_id: str
    status: str
    database: str
    table_count: int
    tables: list[str]

class SchemaResponse(BaseModel):
    schema_text: str


class AskResponse(BaseModel):
    question: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    answer: str

class AskFilesResponse(BaseModel):
    session_id: Optional[str] = None
    question: str
    answer: str
    time_taken: float = 0.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _parse_table(label: str) -> tuple[str, str]:
    """'dbo.Orders' → ('dbo', 'Orders')"""
    parts = label.split(".", 1)
    if len(parts) != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Table '{label}' must be in 'schema.table' format.",
        )
    return parts[0], parts[1]


def _get_conn(req):
    try:
        return connect_mssql(
            server=req.server,
            database=req.database,
            auth_mode=req.auth_mode,
            username=req.username,
            password=req.password,
            driver=req.driver,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB connection failed: {exc}")


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/models")
def get_models(payload: dict = Depends(verify_token)):
    return {"models": list_ollama_models()}


@app.post("/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest, payload: dict = Depends(verify_token)):

    conn = _get_conn(req)

    try:

        tables = get_all_tables(conn)

        table_labels = [
            f"{s}.{t}"
            for s, t in tables
        ]

        session_id = create_session({
            "server": req.server,
            "database": req.database,
            "auth_mode": req.auth_mode,
            "username": req.username,
            "password": req.password,
            "driver": req.driver
        })
        
        # Save initial admin config with empty tables
        cipher = get_cipher()
        encrypted_password = cipher.encrypt(req.password.encode("utf-8")).decode("utf-8") if req.password else ""
        
        initial_config = {
            "server": req.server,
            "database": req.database,
            "username": req.username,
            "password": encrypted_password,
            "auth_mode": req.auth_mode,
            "driver": req.driver,
            "tables": []
        }
        
        try:
            import json
            existing_config = {}
            if os.path.exists(ADMIN_CONFIG_FILE):
                try:
                    with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as _f:
                        existing_config = json.load(_f)
                except Exception:
                    pass
            existing_config[req.environment] = initial_config
            with open(ADMIN_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save initial admin config: {e}")

        return ConnectResponse(
            session_id=session_id,
            status="connected",
            database=req.database,
            table_count=len(tables),
            tables=table_labels,
        )

    finally:


        conn.close()

@app.post("/schema", response_model=SchemaResponse)
def schema(req: SchemaRequest,
           payload: dict = Depends(verify_token)):

    session = get_session(req.session_id)

    if not session:

        raise HTTPException(
            status_code=404,
            detail="Invalid session."
        )

    conn = connect_mssql(
        server=session["server"],
        database=session["database"],
        auth_mode=session["auth_mode"],
        username=session["username"],
        password=session["password"],
        driver=session["driver"]
    )

    try:
        
        if not req.tables:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one table.")
        selected = [
            _parse_table(t)
            for t in req.tables
        ]
    

        schema_text = get_selected_schema_text(
            conn,
            selected
        )

        return SchemaResponse(
            schema_text=schema_text
        )

    finally:

        conn.close()
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, payload: dict = Depends(verify_token)):

    if not req.tables:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one table."
        )

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    session = get_session(req.session_id)

    if not session:

        raise HTTPException(
            status_code=404,
            detail="Invalid session."
        )

    conn = connect_mssql(
        server=session["server"],
        database=session["database"],
        auth_mode=session["auth_mode"],
        username=session["username"],
        password=session["password"],
        driver=session["driver"]
    )

    user_id = payload.get("sub", "unknown")

    try:

        selected = [
            _parse_table(t)
            for t in req.tables
        ]

        schema_text = get_selected_schema_text(
            conn,
            selected
        )

        sql = generate_tsql(
            req.question,
            schema_text,
            model=req.model
        )

        is_safe, reason = validate_tsql(sql)

        if not is_safe:

            logger.warning(
                "Blocked query from user=%s: %s",
                user_id,
                reason
            )

            raise HTTPException(
                status_code=400,
                detail=f"Unsafe query blocked: {reason}"
            )

        columns, rows = execute_tsql(
            conn,
            sql
        )

        answer = generate_answer_summary(
            req.question,
            sql,
            columns,
            rows,
            model=req.model
        )

        log_query(
            user_id=user_id,
            question=req.question,
            sql=sql,
            row_count=len(rows),
            tables=req.tables,
            database=session["database"],
        )

        return AskResponse(
            question=req.question,
            sql=sql,
            columns=columns,
            rows=[list(r) for r in rows],
            row_count=len(rows),
            answer=answer,
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.error(
            "Error processing ask request: %s",
            exc
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )

    finally:

        conn.close()

class DisconnectRequest(BaseModel):
    session_id: str
    
@app.post("/disconnect")
def disconnect(
        req: DisconnectRequest,
        payload: dict = Depends(verify_token)
):

    remove_session(
        req.session_id
    )

    return {
        "status": "disconnected"
    }


@app.get("/rag-tester")
async def serve_rag_tester():
    from fastapi.responses import HTMLResponse
    import os
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_tester.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/db-tester")
async def serve_db_tester():
    from fastapi.responses import HTMLResponse
    import os
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_tester.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload-file")
async def v2_upload_file_endpoint(file: UploadFile = File(...)):
    import shutil
    import os
    try:
        v2_rag_engine = rag_engine()
        os.makedirs(v2_rag_engine.V2_UPLOAD_DIR, exist_ok=True)
        
        # Clear existing files in the directory so we only keep the latest
        for existing_file in os.listdir(v2_rag_engine.V2_UPLOAD_DIR):
            file_path = os.path.join(v2_rag_engine.V2_UPLOAD_DIR, existing_file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning("Could not remove old file %s: %s", file_path, e)
                
        # Save file to V2 directory
        dest_path = os.path.join(v2_rag_engine.V2_UPLOAD_DIR, file.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingest into ChromaDB
        result = v2_rag_engine.ingest_file_v2(dest_path, file.filename)
        
        return {
            "message": "File uploaded and vectorized successfully (V2)", 
            "filename": file.filename,
            "chunks_added": result.get("chunks_added", 0)
        }
    except Exception as e:
        logger.error("V2 Upload Error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

import re
_LEAK_PATTERNS = re.compile(
    r"SYSTEM_DATABASE_RECORDS",
    re.IGNORECASE
)

def sanitize_answer(answer: str) -> str:
    if _LEAK_PATTERNS.search(answer):
        logger.warning("Sanitized a leaked internal reference in answer: %r", answer)
        return "I am the official AI Assistant. All information I provide is sourced natively from our secure internal system database."
    return answer

AGGREGATE_KEYWORDS = ["total length", "combined length", "sum of", "total number of links", "total number of roads", "total number of bridges", "overall length"]

def is_aggregate_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in AGGREGATE_KEYWORDS)

@app.post("/ask-your-query")
async def v2_ask_your_query(
    question: str = Form(...),
    model: Optional[str] = Form("llama3:latest"),
    session_id: Optional[str] = Form(None)
):
    import uuid
    import time
    import httpx
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Identity & Source Protection (Intercept Conversational Questions)
        question_lower = question.lower()
        identity_triggers = ["who are you", "what are you", "where do you get", "source of", "source for", "your source", "how do you know", "where are you getting", "how u getting", "how are you getting", "getting information", "from which", "from where", "which document", "which file", "which manual", "what file", "what document", "what manual", "how are you generating", "your data source"]
        if any(trigger in question_lower for trigger in identity_triggers):
            relevant_chunks = ["I am the official AI Assistant for the Rajasthan Public Works Department (PWD). All information I provide is sourced natively from our secure internal system database."]
            detected_lang = ""
        else:
            if is_aggregate_question(question):
                return AskFilesResponse(
                    session_id=session_id or str(uuid.uuid4()),
                    question=question,
                    answer="This detail is currently not available in our system.",
                    time_taken=0.0
                )
        
            # 2. Advanced Retrieval + Reranking
            search_query = question
            v2_rag_engine = rag_engine()
            detected_lang = v2_rag_engine.needs_translation(question)
            if detected_lang:
                search_query = await v2_rag_engine.translate_to_english(question, model=model)
            relevant_chunks = v2_rag_engine.retrieve_and_rerank(search_query)
            
        # 3. Construct Grounded Prompt
        system_prompt = MULTILINGUAL_PROMPT_TEMPLATE
        context_text = "\n\n---\n\n".join(relevant_chunks)
        system_prompt += f"SYSTEM_DATABASE_RECORDS:\n{context_text}\n\n"
        system_prompt += "You must answer the user's question using ONLY the SYSTEM_DATABASE_RECORDS above.\n\n"
        system_prompt += "CRITICAL OUTPUT CONSTRAINTS (YOU MUST OBEY THESE OR FAIL):\n"
        system_prompt += "- If the exact answer or the raw data needed to answer is not in the SYSTEM_DATABASE_RECORDS, you must output exactly this string and nothing else: \"This detail is currently not available in our system.\"\n"
        system_prompt += "- NEVER perform mathematical calculations or combinations.\n"
        system_prompt += "- Never use introductory phrases like \"According to the records\". Start directly with the answer.\n"
        system_prompt += "- Provide a complete and comprehensive answer using all relevant details from the records (especially if it is a process with steps). Do not leave out important steps."
        
        if detected_lang:
            system_prompt += f"\n\nCRITICAL MANDATORY OVERRIDE: The USER QUESTION is in {detected_lang}. You MUST write your entire response natively in {detected_lang}. Do NOT reply in English. If you reply in English, you will fail."

        import time
        start_time = time.time()
        
        # Call Ollama
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "stream": False,
            "options": {"temperature": 0.0}
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:11434/api/chat", json=payload, timeout=180.0)
            response.raise_for_status()
            response_data = response.json()
            answer = response_data.get("message", {}).get("content", "")
            answer = sanitize_answer(answer)
            # If question was Telugu/Hindi, translate the English answer back using the model
            if detected_lang:
                answer = await v2_rag_engine.translate_from_english(answer, detected_lang, model=model)

        return AskFilesResponse(
            session_id=session_id or str(uuid.uuid4()),
            question=question,
            answer=answer.strip(),
            time_taken=round(time.time() - start_time, 2)
        )
        
    except Exception as e:
        logger.error("V2 Ask Query Error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-your-query-stream")
async def v2_ask_your_query_stream(
    request: Request,
    question: str = Form(...),
    model: Optional[str] = Form("llama3:latest"),
    session_id: Optional[str] = Form(None)
):
    import uuid
    import time
    import httpx
    import json
    from fastapi.responses import StreamingResponse
    
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        # 1. Identity & Source Protection (Intercept Conversational Questions)
        question_lower = question.lower()
        identity_triggers = ["who are you", "what are you", "where do you get", "source of", "source for", "your source", "how do you know", "where are you getting", "how u getting", "how are you getting", "getting information", "from which", "from where", "which document", "which file", "which manual", "what file", "what document", "what manual", "how are you generating", "your data source"]
        if any(trigger in question_lower for trigger in identity_triggers):
            relevant_chunks = ["I am the official AI Assistant for the Rajasthan Public Works Department (PWD). All information I provide is sourced natively from our secure internal system database."]
            detected_lang = ""  # identity shortcut — no translation needed
        else:
            if is_aggregate_question(question):
                async def aggregate_fallback_generator():
                    session = session_id or str(uuid.uuid4())
                    yield f"event: session\ndata: {json.dumps({'session_id': session})}\n\n"
                    yield f"event: delta\ndata: {json.dumps({'text': 'This detail is currently not available in our system.'})}\n\n"
                    yield f"event: done\ndata: {json.dumps({'session_id': session})}\n\n"
                return StreamingResponse(aggregate_fallback_generator(), media_type="text/event-stream")
        
            # 2. Advanced Retrieval + Reranking
            search_query = question
            v2_rag_engine = rag_engine()
            detected_lang = v2_rag_engine.needs_translation(question)
            if detected_lang:
                search_query = await v2_rag_engine.translate_to_english(question, model=model)
            relevant_chunks = v2_rag_engine.retrieve_and_rerank(search_query)
            
        # 4. Build Optimized Prompt
        prompt = MULTILINGUAL_PROMPT_TEMPLATE
        for chunk in relevant_chunks:
            prompt += f"SYSTEM_DATABASE_RECORDS:\n{chunk}\n\n"
            
        prompt += f"USER QUESTION: {question}\n"
        prompt += "You must answer the user's question using ONLY the SYSTEM_DATABASE_RECORDS above.\n\n"
        prompt += "CRITICAL OUTPUT CONSTRAINTS (YOU MUST OBEY THESE OR FAIL):\n"
        prompt += "- If the exact answer or the raw data needed to answer is not in the SYSTEM_DATABASE_RECORDS, you must output exactly this string and nothing else: \"This detail is currently not available in our system.\"\n"
        prompt += "- NEVER perform mathematical calculations or combinations.\n"
        prompt += "- Provide a complete and comprehensive answer using all relevant details from the records (especially if it is a process with steps). Do not leave out important steps."
        
        if detected_lang:
            prompt += f"\n\nCRITICAL MANDATORY OVERRIDE: The USER QUESTION is in {detected_lang}. You MUST write your entire response natively in {detected_lang}. Do NOT reply in English. If you reply in English, you will fail."

        # 5. Stream from Ollama via httpx
        async def event_generator():
            session = session_id or str(uuid.uuid4())
            yield f"event: session\ndata: {json.dumps({'session_id': session})}\n\n"
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "options": {"temperature": 0.0}
            }
            
            buffer = ""
            TARGET_KEYWORD = "SYSTEM_DATABASE_RECORDS"
            target_len = len(TARGET_KEYWORD)
            leaked = False
            
            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", "http://localhost:11434/api/chat", json=payload, timeout=180.0) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    chunk_text = data.get("message", {}).get("content", "")
                                    if chunk_text:
                                        buffer += chunk_text
                                        
                                        if TARGET_KEYWORD in buffer or TARGET_KEYWORD.lower() in buffer.lower():
                                            leaked = True
                                            break
                                            
                                        if len(buffer) > target_len:
                                            safe_to_yield = buffer[:-target_len]
                                            buffer = buffer[-target_len:]
                                            yield f"event: delta\ndata: {json.dumps({'text': safe_to_yield})}\n\n"
                                except json.JSONDecodeError:
                                    pass
                                    
                if leaked or TARGET_KEYWORD in buffer or TARGET_KEYWORD.lower() in buffer.lower():
                    yield f"event: delta\ndata: {json.dumps({'text': '... [REDACTED: This detail is currently not available in our system.]'})}\n\n"
                elif buffer:
                    final_answer = buffer
                    if detected_lang:
                        final_answer = await v2_rag_engine.translate_from_english(buffer, detected_lang, model=model)
                    yield f"event: delta\ndata: {json.dumps({'text': final_answer})}\n\n"
                    
                yield f"event: done\ndata: {json.dumps({'session_id': session})}\n\n"
            except Exception as e:
                logger.error("V2 Stream Ollama Error: %s", str(e))
                yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
                
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream; charset=utf-8",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error("V2 Ask Query Stream Error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/is-file-present")
async def get_current_file():
    """Returns the list of files currently loaded in the V2 system."""
    v2_rag_engine = rag_engine()
    import os
    if not os.path.exists(v2_rag_engine.V2_UPLOAD_DIR):
        return {"status": False, "file name": None}
    files = [f for f in os.listdir(v2_rag_engine.V2_UPLOAD_DIR) if os.path.isfile(os.path.join(v2_rag_engine.V2_UPLOAD_DIR, f))]
    if len(files) > 0:
        return {"status": True, "file name": files[0]}
    return {"status": False, "file name": None}

# --- ADDED FOR ADMIN GLOBAL CONFIG API ---
ADMIN_CONFIG_FILE = "admin_db_config.json"
STATIC_MODEL_NAME = "qwen2.5-coder:7b"

class AdminDbConfigRequest(BaseModel):
    environment: str = "default"
    tables: list[str]

class GlobalQuestionRequest(BaseModel):
    environment: str = "default"
    question: str
    # Pass the same session_id across turns so follow-ups ("name of it") resolve.
    # Omit it and the request is treated as a fresh, standalone question.
    session_id: Optional[str] = None


# /env-list is an alias for /admin/environments — same handler, same response —
# so callers using either path get identical results.
@app.get("/env-list")
@app.get("/admin/environments")
def get_all_environments():
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"environments": []}
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"environments": list(data.keys())}
    except Exception:
        return {"environments": []}

@app.post("/admin/save-db-config")
def save_admin_db_config(req: AdminDbConfigRequest):
    if not os.path.exists(ADMIN_CONFIG_FILE):
        raise HTTPException(status_code=400, detail="No database connection found. Please connect first.")

    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read existing configuration: {e}")

    if req.environment not in config_data:
        raise HTTPException(status_code=400, detail=f"Environment '{req.environment}' not found. Please connect first.")

    parsed_tables = []
    for t in req.tables:
        try:
            parsed = _parse_table(t)
            parsed_tables.append(list(parsed))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid table format: {t}. Must be 'schema.table'")

    config_data[req.environment]["tables"] = parsed_tables

    try:
        with open(ADMIN_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {e}")

    return {"status": "success", "message": "Global configuration saved successfully."}

@app.get("/admin/get-db-config")
def get_admin_db_config(environment: str = "default"):
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"status": "not_configured", "config": None}
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            config = config_data.get(environment)
            if not config:
                return {"status": "not_configured", "config": None}
            config["password"] = "********"
            return {"status": "configured", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read configuration: {e}")

@app.get("/admin/check-db-status")
def check_db_status(environment: str = "default"):
    if not os.path.exists(ADMIN_CONFIG_FILE):
        return {"is_configured": False, "is_connected": False, "message": "No database configuration found."}
        
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            config = config_data.get(environment)
            if not config:
                return {"is_configured": False, "is_connected": False, "message": f"Environment '{environment}' not configured."}
            
        cipher = get_cipher()
        try:
            decrypted_password = cipher.decrypt(config["password"].encode("utf-8")).decode("utf-8")
        except Exception:
            decrypted_password = config["password"]
            
        conn = connect_mssql(
            server=config["server"],
            database=config["database"],
            auth_mode=config.get("auth_mode", "SQL Server Authentication"),
            username=config["username"],
            password=decrypted_password,
            driver=config.get("driver", "ODBC Driver 17 for SQL Server")
        )
        privileges = enforce_read_only(
            conn,
            server=config["server"],
            database=config["database"],
            username=config.get("username") or "",
            strict=False,          # status must report the problem, not fail on it
        )
        conn.close()

        return {
            "is_configured": True,
            "is_connected": True,
            "message": "Database is configured and connection is successful.",
            "login_name": privileges.get("login_name"),
            "read_only": (
                None if not privileges.get("known") else not privileges["write_capable"]
            ),
            "write_grants": privileges.get("grants", []),
            "privilege_message": describe_privileges(privileges),
        }
        
    except Exception as e:
        return {"is_configured": True, "is_connected": False, "message": f"Connection failed: {str(e)}"}

@app.post("/fetch-answer")
def ask_global_db_query(request: GlobalQuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    if not os.path.exists(ADMIN_CONFIG_FILE):
        raise HTTPException(status_code=400, detail="Database is not configured. Admin must save config first.")
        
    try:
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            config = config_data.get(request.environment)
            if not config:
                raise HTTPException(status_code=400, detail=f"Environment '{request.environment}' not found.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read admin config: {e}")
        
    conn = None
    try:
        logger.info(f"Connecting to {config['server']} - {config['database']}...")
        cipher = get_cipher()
        try:
            decrypted_password = cipher.decrypt(config["password"].encode("utf-8")).decode("utf-8")
        except Exception:
            decrypted_password = config["password"]
            
        conn = connect_mssql(
            server=config["server"],
            database=config["database"],
            auth_mode=config.get("auth_mode", "SQL Server Authentication"),
            username=config["username"],
            password=decrypted_password,
            driver=config.get("driver", "ODBC Driver 17 for SQL Server")
        )
        
        # The login must not be able to write. With REQUIRE_READONLY_DB=1 this
        # refuses the request outright; otherwise it warns once per connection.
        privileges = enforce_read_only(
            conn,
            server=config["server"],
            database=config["database"],
            username=config.get("username") or "",
        )

        logger.info("Extracting schema...")
        tables_tuple = [tuple(t) for t in config["tables"]]
        if not tables_tuple:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Environment '{request.environment}' has no tables configured. "
                    "Call /admin/save-db-config to select tables before asking questions."
                ),
            )
        schema_text = get_selected_schema_text(conn, tables_tuple)

        # Rules derived from the ticked tables themselves, so a newly selected
        # table is usable without anyone hand-authoring a rule for it. Cached:
        # the DISTINCT/GROUP BY probes are the same on every question.
        dynamic_rules, schema_profile = profile_and_render_cached(
            conn, tables_tuple, get_table_metadata,
            cache_key=(config["server"], config["database"], tuple(map(tuple, tables_tuple))),
        )

        db_identifier_env = f"{request.environment}_MS SQL_{config['database']}"

        # Record which database and skills answered this question — without
        # this, a mis-pointed environment produces a plausible wrong number
        # with no way to tell after the fact.
        active_skills = describe_active_skills(
            question, schema_text=schema_text, db_identifier=db_identifier_env
        )

        # Worked examples from the verified query library, filtered to this
        # database and to tables present in the schema.
        examples_text, examples_used = load_examples_for_prompt(
            question, db_identifier=db_identifier_env, schema_text=schema_text
        )

        # Prior turns for this session, so a follow-up can resolve "it"/"that".
        session_id = request.session_id or str(uuid.uuid4())
        history = get_db_chat_history(request.session_id) if request.session_id else []
        logger.info(
            "Answering against %s/%s (env=%s) tables=%s skills=%s",
            config["server"],
            config["database"],
            request.environment,
            tables_tuple,
            [s["id"] for s in active_skills],
        )

        logger.info("Generating SQL...")
        sql_query = generate_tsql(
            question,
            schema_text,
            model=STATIC_MODEL_NAME,
            db_identifier=db_identifier_env,
            dynamic_rules=dynamic_rules,
            examples_text=examples_text,
            history=history,
        )
        if not sql_query:
             raise HTTPException(status_code=500, detail="Failed to generate SQL.")
             
        is_safe, reason = validate_tsql(sql_query)
        if not is_safe:
             raise HTTPException(status_code=400, detail=f"Unsafe query blocked: {reason}")
             
        logger.info(f"Executing SQL: {sql_query}")
        try:
            columns, rows = execute_tsql(conn, sql_query)
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"SQL Execution Error: {e}")
             
        logger.info("Generating natural language answer...")
        answer = generate_answer_summary(question, sql_query, columns, rows, model=STATIC_MODEL_NAME, simple_mode=True)
        
        add_db_chat_turn(session_id, question, sql_query, answer)

        # Provenance is logged server-side on every request, so a wrong answer
        # can still be traced to the database that produced it — but it is not
        # returned to the caller, where a server IP and raw rows do not belong.
        logger.info(
            "answered env=%s server=%s db=%s tables=%s rows=%d skills=%s session=%s",
            request.environment, config["server"], config["database"],
            tables_tuple, len(rows), [s["id"] for s in active_skills], session_id,
        )

        response = {
            "question": question,
            "sql": sql_query,
            "answer": answer,
            "row_count": len(rows),
            "session_id": session_id,
        }

        # Diagnostics are an operator switch, never a caller's choice — a client
        # must not be able to ask the API to reveal the server IP, table names
        # or raw rows. Off unless FETCH_ANSWER_DEBUG is set on the host.
        if FETCH_ANSWER_DEBUG:
            response["debug"] = {
                "environment": request.environment,
                "server": config["server"],
                "database": config["database"],
                "tables": [".".join(t) for t in tables_tuple],
                "active_skills": [s["id"] for s in active_skills],
                "examples_used": [e["question"] for e in examples_used],
                "history_turns": len(history),
                "read_only_connection": (
                    None if not privileges.get("known") else not privileges["write_capable"]
                ),
                "schema_rules": {
                    "soft_deletes": schema_profile.get("soft_deletes", []),
                    "joins": schema_profile.get("joins", []),
                    "enums": schema_profile.get("enums", []),
                },
                "columns": columns,
                "rows": [list(r) for r in rows[:100]],
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing global question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
# -------------------------------
# Business Rules Admin APIs
# -------------------------------

class GenerateRuleRequest(BaseModel):
    question: str
    sql: str

class SaveRulesRequest(BaseModel):
    rules_text: str
    keywords: list[str] = ["*"]
    skill_name: str = "Custom Rule"
    # Which databases the rule applies to. "*" is the global scope; use a
    # database name (e.g. "Hims_Zrams") to keep domain rules off other schemas.
    scope: str = "*"
    # Tables the rule references. The rule stays dormant unless all of them are
    # in the schema handed to the model.
    requires_tables: list[str] = []
    priority: int = 200

@app.post("/admin/generate-rule")
def admin_generate_rule(req: GenerateRuleRequest):
    prompt = f"""You are a database business logic expert. 
Given an Example Question and the Correct SQL Query that answers it, extract the underlying business rule or logic into a single, concise English sentence.

You must return ONLY a valid JSON object with EXACTLY these three keys:
1. "generated_rule": The extracted business rule in plain English.
2. "skill_name": A short, 2-4 word title for this rule.
3. "keywords": A list of 3-5 important words from the question that should trigger this rule.

Example JSON output:
{{
  "generated_rule": "Busiest roads means AADT > 10000.",
  "skill_name": "Traffic Rules",
  "keywords": ["busiest", "traffic", "heavy"]
}}

Example Question:
{req.question}

Correct SQL Query:
{req.sql}
"""
    try:
        response_text = ask_ollama(prompt, model=STATIC_MODEL_NAME).strip()
        
        # In case the LLM wraps it in markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        import json
        result = json.loads(response_text.strip())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

SKILLS_FILE = skills_registry.SKILLS_FILE


def _read_skills_document() -> dict:
    """Read skills.json, normalising a legacy v1 document to the v2 shape."""
    if not os.path.exists(SKILLS_FILE):
        return {"version": 2, "scopes": {}}

    try:
        with open(SKILLS_FILE, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read rules: {e}")

    if not isinstance(doc, dict):
        return {"version": 2, "scopes": {}}

    if not isinstance(doc.get("scopes"), dict):
        # v1: {"global": [...]} — the flat list was applied to every database.
        doc = {"version": 2, "scopes": {"*": doc.get("global", []) or []}}

    doc.setdefault("version", 2)
    doc.setdefault("scopes", {})
    return doc


@app.get("/admin/get-business-rules")
def admin_get_business_rules(database_identifier: str = "", environment: str = "default"):
    """
    Rules that would apply to `database_identifier`.

    `rules_text` stays for the existing frontend; `skills` carries the
    structured registry (scope, keywords, table gating) the v2 format adds.
    """
    doc = _read_skills_document()

    identifier = database_identifier or ""
    applicable = skills_registry.resolve_scopes(identifier) if identifier else ["*"]

    skills_out = []
    for scope in applicable:
        for skill in doc["scopes"].get(scope, []):
            if isinstance(skill, dict):
                skills_out.append(dict(skill, scope=scope))

    return {
        "rules_text": "\n".join(s.get("instruction", "") for s in skills_out),
        "skills": skills_out,
        "scopes_applied": applicable,
        "available_scopes": sorted(doc["scopes"].keys()),
    }


@app.post("/admin/save-business-rules")
def admin_save_business_rules(req: SaveRulesRequest):
    """Append one skill to a scope. Default scope "*" applies to every database."""
    if not req.rules_text.strip():
        raise HTTPException(status_code=400, detail="rules_text cannot be empty.")

    doc = _read_skills_document()
    scope = (req.scope or "*").strip() or "*"
    doc["scopes"].setdefault(scope, [])

    keywords = [k.strip() for k in req.keywords if k and k.strip() and k.strip() != "*"]

    import uuid
    new_skill = {
        "id": str(uuid.uuid4())[:8],
        "name": req.skill_name,
        "enabled": True,
        "priority": req.priority,
        # No keywords means the rule is unconditional within its scope.
        "match": "any" if keywords else "always",
        "keywords": keywords,
        "requires_tables": [t.strip() for t in req.requires_tables if t and t.strip()],
        "instruction": req.rules_text.strip(),
    }
    doc["scopes"][scope].append(new_skill)

    try:
        with open(SKILLS_FILE, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=4, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save rules: {e}")

    return {"status": "success", "scope": scope, "skill": new_skill}


@app.post("/fetch-answer/reset-session")
def reset_chat_session(session_id: str):
    """Forget a conversation, so the next question starts with no history."""
    clear_db_chat_history(session_id)
    return {"status": "cleared", "session_id": session_id}


class QueryExample(BaseModel):
    question: str
    sql: str
    database: Optional[str] = None
    notes: Optional[str] = None


class ImportQueriesRequest(BaseModel):
    examples: list[QueryExample]
    # Which databases these apply to. "*" means every database.
    scope: str = "*"


@app.post("/admin/query-library/import")
def import_query_library(req: ImportQueriesRequest):
    """
    Bulk-load verified (question, SQL) pairs as few-shot examples.

    Every SQL is checked by the read-only guard; a pair containing a write is
    rejected with a reason rather than becoming an example the model imitates.
    """
    pairs = [e.model_dump() if hasattr(e, "model_dump") else e.dict() for e in req.examples]
    report = add_examples(pairs, scope=req.scope, source="api")
    report.pop("added_examples", None)
    return report


@app.post("/admin/query-library/upload")
async def upload_query_library(file: UploadFile = File(...), scope: str = Form("*")):
    """Same as /import, but takes the support team's .csv, .xlsx or .json file."""
    try:
        content = await file.read()
        rows = parse_upload(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read {file.filename}: {e}")

    report = add_examples(rows, scope=scope, source=f"upload:{file.filename}")
    report.pop("added_examples", None)
    return report


@app.get("/admin/query-library")
def get_query_library(scope: Optional[str] = None):
    """List the stored examples, optionally filtered to one scope."""
    doc = load_library()
    examples = doc["examples"]
    if scope:
        examples = [e for e in examples if e.get("scope") == scope]
    return {"stats": library_stats(), "examples": examples}


@app.delete("/admin/query-library/{example_id}")
def delete_query_example(example_id: str):
    doc = load_library()
    before = len(doc["examples"])
    doc["examples"] = [e for e in doc["examples"] if e.get("id") != example_id]
    if len(doc["examples"]) == before:
        raise HTTPException(status_code=404, detail=f"No example with id '{example_id}'.")
    save_library(doc)
    return {"status": "deleted", "id": example_id, "remaining": len(doc["examples"])}


@app.get("/admin/query-library/preview")
def preview_query_examples(question: str, database_identifier: str = "", top_k: int = 4):
    """Which examples a question would pull in, and their similarity scores."""
    return {
        "question": question,
        "examples": retrieve(question, database_identifier or None, top_k),
    }


@app.get("/admin/preview-skills")
def admin_preview_skills(question: str, database_identifier: str = "", schema_text: str = ""):
    """
    Which skills a question would activate, and why — so a wrong rule can be
    found without reading the generated SQL and guessing.
    """
    return {
        "question": question,
        "database_identifier": database_identifier,
        "scopes_applied": skills_registry.resolve_scopes(database_identifier),
        "active_skills": skills_registry.describe_active_skills(
            question,
            schema_text=schema_text or None,
            db_identifier=database_identifier,
        ),
    }

@app.get("/admin/get-all-tables")
def admin_get_all_tables(environment: str = "default"):
    if not os.path.exists(ADMIN_CONFIG_FILE):
        raise HTTPException(status_code=400, detail="Database is not configured.")
        
    try:
        import json
        with open(ADMIN_CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = json.load(f)
            config = config_data.get(environment)
            if not config:
                raise HTTPException(status_code=400, detail=f"Environment '{environment}' not found.")
            
        cipher = get_cipher()
        try:
            decrypted_password = cipher.decrypt(config["password"].encode("utf-8")).decode("utf-8")
        except Exception:
            decrypted_password = config["password"]
            
        conn = connect_mssql(
            server=config["server"],
            database=config["database"],
            auth_mode=config.get("auth_mode", "SQL Server Authentication"),
            username=config["username"],
            password=decrypted_password,
            driver=config.get("driver", "ODBC Driver 17 for SQL Server")
        )
        
        tables = get_all_tables(conn)
        conn.close()
        
        table_labels = [f"{s}.{t}" for s, t in tables]
        return {"status": "success", "tables": table_labels}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tables: {e}")
