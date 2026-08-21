# AI Chatbot & Database Assistant Architecture

This document provides a high-level overview of how the AI Data Assistant is structured, including the components, data flow, and text-to-SQL logic.

## 1. High-Level Architecture Diagram

```mermaid
graph TD
    %% Frontend Layer
    subgraph Frontend [Streamlit UI - mssql_app.py]
        UI_Chat[Chat Interface]
        UI_Sidebar[Sidebar: Environment, Credentials, Tables]
        UI_Rules[AI Skills & Rules Manager]
    end

    %% Backend Layer
    subgraph Backend [FastAPI Backend - fastapi_app.py]
        API_Conn[/connect]
        API_Schema[/schema]
        API_Ask[/fetch-answer]
        API_Admin[/admin endpoints]
    end

    %% File System State
    subgraph Storage [Local Configuration & State]
        JSON_DB[admin_db_config.json]
        JSON_Rules[business_rules.json]
        SQL_Hist[(chat_history.db)]
    end

    %% AI & Database Engine Layer
    subgraph Engine [Text-to-SQL & LLM Engine]
        Ollama[Ollama Client / Local LLMs]
        SQL_Gen[SQL Generator]
        SQL_Exec[SQL Executor]
        Schema_Read[Schema Reader]
    end

    %% External Systems
    DB[(Microsoft SQL Server\nDev / QA / Prod)]

    %% Connections
    UI_Chat -->|Ask Question| API_Ask
    UI_Sidebar -->|Save Config| API_Conn
    UI_Rules -->|Update Rules| API_Admin

    API_Conn <-->|Read/Write| JSON_DB
    API_Admin <-->|Read/Write| JSON_Rules
    API_Ask <-->|Log| SQL_Hist

    API_Ask --> Schema_Read
    API_Ask --> SQL_Gen
    SQL_Gen <--> Ollama
    SQL_Gen --> SQL_Exec
    Schema_Read --> DB
    SQL_Exec <--> DB
```

---

## 2. Component Breakdown

### A. The Frontend (Streamlit)
* **File:** `mssql_app.py`
* **Role:** The user interface. It provides an intuitive sidebar for selecting the **Environment** (`dev`, `qa`, `prod`), inputting credentials, and managing **Dynamic AI Skills (Business Rules)**. The main panel provides a chat interface to ask natural language questions.
* **Communication:** It makes standard HTTP POST/GET requests to the FastAPI backend.

### B. The Backend (FastAPI)
* **File:** `fastapi_app.py`
* **Role:** The central router. It receives requests from the Streamlit UI, reads the saved JSON configurations to figure out *which* database to talk to, and orchestrates the AI engines.
* **Multi-Environment Support:** The backend dynamically routes requests by checking the `environment` parameter sent from the UI, ensuring QA questions don't accidentally run against Dev databases.

### C. The Core Engine (Text-to-SQL)
* **`mssql_schema_reader.py`**: Automatically inspects the Microsoft SQL Server database to pull exact column names, foreign keys, and data types for the selected tables.
* **`ollama_client.py`**: Talks to your locally hosted LLM (e.g., `llama3:latest`).
* **`mssql_sql_generator.py`**: The "brain" of the operation. It takes:
  1. The user's question
  2. The database schema
  3. The custom business rules (from `business_rules.json`)
  ...and forces the AI to generate a strictly formatted Microsoft T-SQL query.
* **`mssql_executor.py`**: Takes the AI-generated SQL query, runs it securely against the SQL Server, and returns the raw data rows back to the user.

### D. Local Storage & Configuration
* **`admin_db_config.json`**: Stores server credentials, isolated by environment.
* **`business_rules.json`**: Stores custom rules and guidelines for the AI, isolated by Environment + Database Name (e.g., `dev_MS SQL_Hims_Zrams`).
* **`chat_history.db`**: A local SQLite database managed by `history_manager.py` to keep a log of user questions and AI answers.

---

## 3. Data Flow: "What happens when you ask a question?"

1. **User Types Question**: "How many State Highways are in Jhalawar?" in the Streamlit UI.
2. **API Call**: UI sends the question and the selected environment (e.g., `dev`) to FastAPI (`/fetch-answer`).
3. **Load Context**: FastAPI reads `admin_db_config.json` to get the Dev database password/IP, and reads `business_rules.json` to get the Dev business rules.
4. **Schema Retrieval**: The backend quickly hits the SQL Server to grab the current schema for the tables you selected.
5. **Prompt Construction**: The backend bundles the Question, the Schema, and the Business Rules into one massive prompt and sends it to **Ollama** (`llama3`).
6. **SQL Generation**: The AI generates `SELECT COUNT(...) FROM Roads WHERE District='Jhalawar' AND Road_Class='SH'`.
7. **Execution**: `mssql_executor.py` runs that exact SQL query on the Microsoft SQL Server.
8. **Result**: The numerical or tabular result is passed back through FastAPI to Streamlit, where it is displayed to the user as a clean table.
