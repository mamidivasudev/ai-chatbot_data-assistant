import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pyodbc
import uvicorn

# Import existing logic
from mssql_connector import connect_mssql
from mssql_schema_reader import get_selected_schema_text
from mssql_sql_generator import generate_tsql, generate_answer_summary
from mssql_executor import execute_tsql
from history_manager import get_business_rules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("static_qa_api")

app = FastAPI(title="Static QA API", version="1.0.0")

# ─────────────────────────────────────────────
# Static Configuration
# ─────────────────────────────────────────────
SERVER = "192.168.1.19"
DATABASE = "Hims_Zrams"
AUTH_MODE = "SQL Server Authentication"
USERNAME = "Sa"
PASSWORD = "Satra@123"
MODEL_NAME = "llama3:latest"

# Tables to always include in context
STATIC_TABLES = [
    ("dbo", "Roads"),
    ("dbo", "Road_Sections"),
    ("dbo", "Pavement_Condition"),
    ("dbo", "Traffic_Data")
]

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask_db_query")
def ask_static(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    conn = None
    try:
        # 1. Connect to Database
        logger.info(f"Connecting to {SERVER} - {DATABASE}...")
        conn = connect_mssql(
            server=SERVER,
            database=DATABASE,
            auth_mode=AUTH_MODE,
            username=USERNAME,
            password=PASSWORD
        )
        
        # 2. Get Schema for static tables
        logger.info("Extracting schema...")
        schema_text = get_selected_schema_text(conn, STATIC_TABLES)
        
        # 3. Get Business Rules
        db_identifier = f"MS SQL_{DATABASE}"
        business_rules = get_business_rules(db_identifier)
        
        # 4. Generate SQL
        logger.info("Generating SQL...")
        sql_query = generate_tsql(question, schema_text, business_rules, model=MODEL_NAME)
        if not sql_query:
             raise HTTPException(status_code=500, detail="Failed to generate SQL.")
             
        # 5. Execute SQL
        logger.info(f"Executing SQL: {sql_query}")
        try:
            columns, rows = execute_tsql(conn, sql_query)
        except Exception as e:
             raise HTTPException(status_code=400, detail=f"SQL Execution Error: {e}")
             
        # 6. Generate Answer Summary
        logger.info("Generating natural language answer...")
        answer = generate_answer_summary(question, sql_query, columns, rows, model=MODEL_NAME)
        
        return {
            "question": question,
            "sql_query": sql_query,
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
