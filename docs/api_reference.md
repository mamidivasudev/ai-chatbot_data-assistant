# MSSQL AI Assistant — REST API Reference

Base URL: `http://your-server:8000`

All endpoints except `/health` require a **Bearer JWT token** in the `Authorization` header.

**Database Connection Details**

| Field | Value |
|---|---|
| Server | `192.168.1.19` |
| Database | `Hims_Zrams` |
| Auth Mode | SQL Server Authentication |
| Username | `sa` |
| Password | `user` |

---

## Sequential Usage Flow

```
1. /health      → confirm service is running
2. /models      → pick an Ollama model
3. /connect     → verify DB credentials, get table list
4. /schema      → inspect schema for selected tables
5. /ask         → ask a natural-language question, get SQL + results + AI answer
```

---

## 1. Health Check

Confirm the API service is alive. No authentication required.

**Request**
```http
GET /health
```

**Response `200 OK`**
```json
{
  "status": "ok",
  "timestamp": "2026-06-01T08:30:00.000000+00:00"
}
```

---

## 2. List Ollama Models

Returns all locally installed Ollama models available for SQL generation.

**Request**
```http
GET /models
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Response `200 OK`**
```json
{
  "models": [
    "llama3:latest",
    "mistral:latest",
    "qwen2.5-coder:7b"
  ]
}
```

> Use any model name from this list in the `model` field of `/ask`.

---

## 3. Connect to Database

Tests the database connection and returns all available base tables.

**Request**
```http
POST /connect
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

```json
{
  "server": "192.168.1.19",
  "database": "Hims_Zrams",
  "auth_mode": "SQL Server Authentication",
  "username": "sa",
  "password": "user",
  "driver": "ODBC Driver 17 for SQL Server"
}
```

**Response `200 OK`**
```json
{
  "status": "connected",
  "database": "Hims_Zrams",
  "table_count": 1258,
  "tables": [
    "dbo.SysRoadSum",
    "dbo.RoadSegments",
    "dbo.Districts",
    "..."
  ]
}
```

**Response `503 Service Unavailable`** *(wrong credentials)*
```json
{
  "detail": "DB connection failed: [HY000] [Microsoft][ODBC SQL Server Driver] Login failed for user 'sa'."
}
```

---

## 4. Get Schema

Returns the full schema (columns, data types, PKs, FKs) for selected tables — useful for confirming what context the AI will use before asking a question.

**Request**
```http
POST /schema
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

```json
{
  "server": "192.168.1.19",
  "database": "Hims_Zrams",
  "auth_mode": "SQL Server Authentication",
  "username": "sa",
  "password": "user",
  "tables": [
    "dbo.SysRoadSum"
  ]
}
```

**Response `200 OK`**
```json
{
  "schema_text": "TABLE: [dbo].[SysRoadSum]\n  RoadID int  [PK, NOT NULL]\n  RoadName nvarchar(255)  [NOT NULL]\n  DistrictID int  [FK→[dbo].[Districts].DistrictID, NOT NULL]\n  LengthKm decimal  [NOT NULL]\n  SurfaceType nvarchar(50)\n  Condition nvarchar(20)\n  LastInspected date"
}
```

**Response `400 Bad Request`** *(wrong table format — missing schema prefix)*
```json
{
  "detail": "Table 'SysRoadSum' must be in 'schema.table' format."
}
```

---

## 5. Ask a Question

The core endpoint. Accepts a natural-language question, generates T-SQL, executes it against `Hims_Zrams`, and returns the results plus an AI-written plain-English answer.

**Request**
```http
POST /ask
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
```

```json
{
  "server": "192.168.1.19",
  "database": "Hims_Zrams",
  "auth_mode": "SQL Server Authentication",
  "username": "sa",
  "password": "user",
  "tables": [
    "dbo.SysRoadSum"
  ],
  "question": "How many roads are there in total?",
  "model": "llama3:latest"
}
```

**Response `200 OK`**
```json
{
  "question": "How many roads are there in total?",
  "sql": "SELECT COUNT(*) AS TotalRoads FROM [dbo].[SysRoadSum];",
  "columns": ["TotalRoads"],
  "rows": [
    [3847]
  ],
  "row_count": 1,
  "answer": "There are 3,847 roads recorded in the Hims_Zrams database."
}
```

**Response `400 Bad Request`** *(unsafe query blocked)*
```json
{
  "detail": "Unsafe query blocked: Blocked keyword detected: 'drop'"
}
```

**Response `401 Unauthorized`** *(missing or expired token)*
```json
{
  "detail": "Invalid or expired token: Signature verification failed."
}
```

**Response `500 Internal Server Error`** *(query execution error)*
```json
{
  "detail": "Invalid column name 'RoadNam'."
}
```

---

## 6. Summaries

AI summaries of dashboard charts, whole dashboards and documents (`summarize_api.py`, `summarizer.py`).

For charts the service computes every figure itself: totals, shares, "Good or better" / "Poor or worse" groups, trends, and target-vs-achieved gaps. The model only writes the sentences. Any number the model writes that isn't in those figures is rejected. If the model is down, the summary is built from the data by rule, and `source` says which happened:

| `source` | Meaning |
|---|---|
| `llm` | Written by the model; every number verified |
| `llm+rules` | Model sentences with unverifiable numbers were removed and refilled from the data |
| `rules` | Model unreachable or unusable; built directly from the data |

Documents are read in full, not just the top matching chunks that `/ask-your-query` uses. Long files are summarized in parts, then combined.

**Configuration** (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `gemini` |
| `OLLAMA_SUMMARY_MODEL` | `llama3:latest` | Any installed Ollama model |
| `OLLAMA_NUM_CTX` | `8192` | Context window per call. Raise it (e.g. `16384`) for models that support more, so long documents need fewer calls |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | — / `gemini-2.5-flash` | Only when `LLM_PROVIDER=gemini` |
| `SUMMARY_MAX_UPLOAD_MB` | `25` | Upload size limit |

`fastapi_app.py` pins `OLLAMA_HOST` to `http://127.0.0.1:11434`, so Ollama must run on the same machine as the API.

### 6.1 Status

```http
GET /summarize/status
```
```json
{ "provider": "ollama", "model": "llama3:latest", "max_input_chars": 19660,
  "reachable": true, "model_available": true, "installed_models": ["llama3:latest", "qwen2.5-coder:7b"] }
```

### 6.2 Summarize a chart

Send the chart's data rows exactly as amCharts holds them. `category_field` and `value_fields` are inferred when omitted; styling fields such as `color` are ignored.

```http
POST /summarize/chart
Content-Type: application/json
```
```json
{
  "title": "Roughness by category",
  "chart_type": "bar",
  "unit": "km",
  "filters": { "year": 2025, "module": "RIS" },
  "category_field": "category",
  "value_fields": ["length"],
  "series_names": { "length": "Length" },
  "data": [
    { "category": "Good", "length": 11249 },
    { "category": "Fair", "length": 6338.71 },
    { "category": "Very Good", "length": 1573.22 },
    { "category": "Poor", "length": 2132.71 },
    { "category": "Very Poor", "length": 1031.08 }
  ],
  "language": "English"
}
```

Optional: `groups` (e.g. `{"Needs work": ["Poor", "Very Poor"]}`; condition scales get "Good or better" / "Poor or worse" automatically), `model`, `refresh: true` to skip the cache.

**Response `200 OK`** *(illustrative wording)*
```json
{
  "headline": "Just over half the surveyed network, 11,249 km (50.4%), is in Good condition.",
  "points": ["12,822.22 km (57.4%) is Good or better.", "Poor or worse covers 3,163.79 km (14.2%)."],
  "watch": ["Fair roads account for 6,338.71 km (28.4%)."],
  "source": "llm",
  "warnings": [],
  "facts": { "kind": "categories", "series": [ { "total": 22324.72, "...": "..." } ] },
  "provider": "ollama", "model": "llama3:latest", "cached": false, "time_taken": 2.4
}
```

`422` if the data has no label field or no numeric field.

### 6.3 Summarize a dashboard

One summary across several charts. Charts that share a unit are also compared by total, e.g. roughness covers 81.3% of the road-type total, which points to roads without a survey.

```http
POST /summarize/dashboard
```
```json
{ "title": "RIS dashboard", "filters": { "year": 2025 },
  "charts": [ { "title": "Roadtype by category", "unit": "km", "data": [ ... ] },
              { "title": "Roughness by category", "unit": "km", "data": [ ... ] } ] }
```
Response: as for a chart, with `facts` as a list and `cross_chart` lines.

### 6.4 Summarize a document

Multipart form. Omit `file` to summarize the file currently loaded in SatBot (`/upload-file`).

```http
POST /summarize/document
Content-Type: multipart/form-data

file=<.pdf | .docx | .txt | .md | .csv | .xlsx | .xls>   language=English
```
```json
{
  "title": "...", "overview": "...", "key_points": ["..."], "action_items": ["..."],
  "unverified_numbers": [], "parts": 3, "truncated": false, "warnings": [],
  "filename": "report.pdf", "provider": "ollama", "model": "llama3:latest", "cached": false, "time_taken": 41.7
}
```
`unverified_numbers` lists figures in the summary that don't appear in the document. `415` unsupported type, `413` too large, `422` scanned PDF (no text layer; OCR not enabled yet), `503` model failure.

`POST /summarize/document-stream` takes the same form and returns server-sent events:

```
event: start     data: {"filename": "report.pdf", "characters": 48211}
event: progress  data: {"stage": "reading", "part": 1, "parts": 3}
event: progress  data: {"stage": "merging", "groups": 1}
event: progress  data: {"stage": "writing"}
event: result    data: { ...same body as /summarize/document... }
event: done      data: {}
```
On failure: `event: error  data: {"detail": "..."}`.

### 6.5 Checking a deployment

```bash
curl -s http://your-server:8000/summarize/status
curl -s -X POST http://your-server:8000/summarize/chart -H "Content-Type: application/json" \
     -d '{"title":"Roughness by category","unit":"km","data":[{"category":"Good","length":11249},{"category":"Fair","length":6338.71},{"category":"Poor","length":2132.71}]}'
curl -s -X POST http://your-server:8000/summarize/document -F "file=@report.pdf"
curl -N -X POST http://your-server:8000/summarize/document-stream -F "file=@report.pdf"
```
A chart summary with `"source": "rules"` means the model was not reached. Check `/summarize/status`.

Unit tests (no model needed): `pip install pytest` then `python -m pytest tests`.

---

## Request Fields Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `server` | string | Yes | `192.168.1.19` |
| `database` | string | Yes | `Hims_Zrams` |
| `auth_mode` | string | Yes | `"SQL Server Authentication"` |
| `username` | string | Yes | `sa` |
| `password` | string | Yes | `user` |
| `driver` | string | No | ODBC driver name; auto-detected if omitted |
| `tables` | string[] | Yes | Table names in `schema.table` format, e.g. `"dbo.SysRoadSum"` |
| `question` | string | Yes (`/ask`) | Natural-language question about the data |
| `model` | string | No | Ollama model name from `/models`; uses default if omitted |

---

## Authentication

All protected endpoints expect an **HS256 JWT** in the `Authorization` header.

```
Authorization: Bearer <token>
```

Generate a token for testing (Python):

```python
from jose import jwt
import datetime

token = jwt.encode(
    {
        "sub": "sa",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    },
    key="your-secret-key",
    algorithm="HS256"
)
print(token)
```

Set the secret on the server:
```powershell
$env:SECRET_KEY = "your-secret-key"
```

---

## Running the Server

```powershell
# Install dependencies
pip install -r requirements.txt

# Start (development)
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000

# Start (production — 4 workers)
gunicorn fastapi_app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

Interactive API docs (Swagger UI) available at: `http://your-server:8000/docs`

---

## Audit Log Table DDL

Run once on `Hims_Zrams` (or a dedicated audit database) before starting the server:

```sql
CREATE TABLE dbo.AiQueryAudit (
    id            BIGINT IDENTITY PRIMARY KEY,
    user_id       NVARCHAR(256)  NOT NULL,
    question      NVARCHAR(MAX)  NOT NULL,
    sql_query     NVARCHAR(MAX)  NOT NULL,
    tables_used   NVARCHAR(MAX)  NOT NULL,
    database_name NVARCHAR(256)  NOT NULL,
    row_count     INT            NOT NULL DEFAULT 0,
    created_at    DATETIME2      NOT NULL DEFAULT SYSUTCDATETIME()
);
```

Set audit DB env vars:
```powershell
$env:AUDIT_SERVER   = "192.168.1.19"
$env:AUDIT_DATABASE = "Hims_Zrams"
$env:AUDIT_UID      = "sa"
$env:AUDIT_PWD      = "user"
```
