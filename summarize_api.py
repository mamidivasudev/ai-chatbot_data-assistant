"""
/summarize endpoints: AI summaries of dashboard charts, whole dashboards and documents.

  GET  /summarize/status           provider, model, and whether the model is reachable
  POST /summarize/chart            one chart's data rows      -> headline, points, watch
  POST /summarize/dashboard        several charts at once     -> one summary across them
  POST /summarize/document         uploaded file (or SatBot's current file) -> summary
  POST /summarize/document-stream  same, as server-sent events with progress

See docs/api_reference.md, "Summaries", for payloads. The model is chosen by
llm_provider.get_provider (Ollama by default).
"""

import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections import OrderedDict
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import document_tables
import summarizer
import voice_intent as voice_intent_module
from llm_provider import LLMError, get_provider

logger = logging.getLogger("summarize_api")

router = APIRouter(prefix="/summarize", tags=["summarize"])

MAX_UPLOAD_BYTES = int(os.environ.get("SUMMARY_MAX_UPLOAD_MB", "25")) * 1024 * 1024
# The folder /upload-file keeps SatBot's current document in (v2_rag_engine.V2_UPLOAD_DIR).
RAG_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_file_RAG")
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls")
_LANGUAGE = re.compile(r"^[A-Za-z][A-Za-z ()-]{1,39}$")


# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────
class ChartPayload(BaseModel):
    title: str
    chart_type: Optional[str] = None                  # pie | donut | bar | column | line | area ...
    unit: Optional[str] = None                        # "km", "Nos" ...
    category_field: Optional[str] = None              # amCharts categoryField; inferred when omitted
    value_fields: Optional[list[str]] = None          # amCharts valueField(s); inferred when omitted
    series_names: Optional[dict[str, str]] = None     # value field -> display name
    groups: Optional[dict[str, list[str]]] = None     # {"Good or better": ["Very Good", "Good"]}
    filters: Optional[dict[str, Any]] = None          # {"year": 2025, "module": "RIS"}
    data: list[dict[str, Any]]


class ChartSummaryRequest(ChartPayload):
    language: str = "English"
    model: Optional[str] = None
    refresh: bool = False                             # skip the cache


class DashboardSummaryRequest(BaseModel):
    title: Optional[str] = None
    filters: Optional[dict[str, Any]] = None
    charts: list[ChartPayload]
    language: str = "English"
    model: Optional[str] = None
    refresh: bool = False


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _dump(model: BaseModel) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _provider(model: Optional[str]):
    try:
        return get_provider(model or None)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def _check_language(language: str) -> str:
    # The language is written into the system prompt, so only plain names pass.
    if not _LANGUAGE.match(language or ""):
        raise HTTPException(status_code=422, detail="language must be a plain language name, e.g. 'English' or 'Hindi'.")
    return language


# Same data, language and model always produce the same summary, so recent
# results are kept in memory (per worker). Rule-based results from a model
# outage are not kept, so the next request tries the model again.
_CACHE_SIZE = 256
_cache: "OrderedDict[str, dict]" = OrderedDict()


def _cache_key(kind: str, payload: Any, language: str, provider) -> str:
    raw = json.dumps([kind, payload, language, provider.name, provider.model], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cached(key: str, refresh: bool) -> Optional[dict]:
    if refresh or key not in _cache:
        return None
    _cache.move_to_end(key)
    return {**_cache[key], "cached": True}


def _remember(key: str, result: dict) -> None:
    if result.get("source") == "rules":
        return
    _cache[key] = result
    _cache.move_to_end(key)
    while len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)


def _read_document(path: str, filename: str) -> tuple[str, list[dict]]:
    """(text, status counts). Status counts come from status-like table columns; see document_tables."""
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(), []
    if name.endswith(".docx"):
        # Read in body order with header-labelled cells, not with SatBot's ingestion
        # parser, which moves tables to the end and drops the column names.
        try:
            text, tables = document_tables.read_docx(path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to read Word document: {exc}")
        return text, _safe_status_counts(tables, filename)
    try:
        import v2_rag_engine
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Document parsing is unavailable on this host: {exc}")
    try:
        if name.endswith(".pdf"):
            return v2_rag_engine.extract_text_from_pdf(path), []
        # Spreadsheet rows already read "Column: value | …" in SatBot's parser.
        text = v2_rag_engine.extract_text_from_excel(path, name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        tables = document_tables.read_sheet_tables(path, name)
    except Exception as exc:
        logger.warning("Could not read tables of %s for status counts: %s", filename, exc)
        tables = []
    return text, _safe_status_counts(tables, filename)


def _safe_status_counts(tables: list, filename: str) -> list[dict]:
    # The counts are a bonus: a quirk in one table must not cost the whole summary.
    try:
        return document_tables.status_counts(tables)
    except Exception as exc:
        logger.warning("Could not count statuses in %s: %s", filename, exc)
        return []


async def _load_document(file: Optional[UploadFile]) -> tuple[str, str, list[dict]]:
    """(filename, text, status counts) of the uploaded file, or of SatBot's current file when none is sent."""
    if file is not None and file.filename:
        filename = os.path.basename(file.filename)
        extension = os.path.splitext(filename)[1].lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=415, detail=f"Unsupported file type. Use one of: {', '.join(SUPPORTED_EXTENSIONS)}")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        with tempfile.TemporaryDirectory() as tmp:
            # Saved under a fixed name: the client's filename never becomes a path.
            path = os.path.join(tmp, "document" + extension)
            with open(path, "wb") as fh:
                fh.write(data)
            text, status = await run_in_threadpool(_read_document, path, filename)
    else:
        current = []
        if os.path.isdir(RAG_UPLOAD_DIR):
            current = [f for f in os.listdir(RAG_UPLOAD_DIR) if os.path.isfile(os.path.join(RAG_UPLOAD_DIR, f))]
        if not current:
            raise HTTPException(status_code=400, detail="Send a file, or upload one to SatBot first.")
        filename = current[0]
        text, status = await run_in_threadpool(_read_document, os.path.join(RAG_UPLOAD_DIR, filename), filename)

    # PDF text comes back with "## SECTION: Page N" markers even when a page
    # has almost no text, so measure what's left without them.
    content = re.sub(r"## SECTION: Page \d+", "", text or "").strip()
    if filename.lower().endswith(".pdf") and len(content) < 200:
        raise HTTPException(
            status_code=422,
            detail="This PDF has no text layer; it looks scanned. Scanned documents need OCR, which is not enabled yet.",
        )
    if not content:
        raise HTTPException(status_code=422, detail="The document has no readable text.")
    return filename, text, status


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@router.get("/status")
async def summarize_status():
    try:
        provider = get_provider()
    except LLMError as exc:
        return {"provider": os.environ.get("LLM_PROVIDER", "ollama"), "reachable": False, "detail": str(exc)}
    return {
        "provider": provider.name,
        "model": provider.model,
        "max_input_chars": provider.max_input_chars,
        **(await provider.status()),
    }


class VoiceIntentRequest(BaseModel):
    text: str                                         # what the recogniser heard, any language
    targets: list[str] = []                           # titles of the charts / documents on screen
    can_summarize_all: bool = True                    # the screen has a whole-dashboard summary
    language: str = "English"                         # the app's language, as a plain name
    model: Optional[str] = None


@router.post("/voice-intent")
async def voice_intent(req: VoiceIntentRequest):
    """Maps a spoken command (any language) to summarize_all / summarize <item> / read / stop / none."""
    language = _check_language(req.language)
    provider = _provider(req.model)
    try:
        return await voice_intent_module.resolve_intent(req.text, req.targets, provider, language=language,
                                                        can_summarize_all=req.can_summarize_all)
    except LLMError as exc:
        logger.warning("Voice intent: model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="The AI model could not be reached.")


@router.post("/chart")
async def summarize_chart(req: ChartSummaryRequest):
    language = _check_language(req.language)
    provider = _provider(req.model)
    chart = _dump(req)
    for option in ("language", "model", "refresh"):
        chart.pop(option)

    key = _cache_key("chart", chart, language, provider)
    cached = _cached(key, req.refresh)
    if cached:
        return cached
    started = time.time()
    try:
        result = await summarizer.summarize_chart(chart, provider, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result.update(provider=provider.name, model=provider.model, cached=False,
                  time_taken=round(time.time() - started, 2))
    _remember(key, result)
    return result


@router.post("/dashboard")
async def summarize_dashboard(req: DashboardSummaryRequest):
    language = _check_language(req.language)
    provider = _provider(req.model)
    dashboard = {"title": req.title, "filters": req.filters, "charts": [_dump(c) for c in req.charts]}

    key = _cache_key("dashboard", dashboard, language, provider)
    cached = _cached(key, req.refresh)
    if cached:
        return cached
    started = time.time()
    try:
        result = await summarizer.summarize_dashboard(dashboard, provider, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result.update(provider=provider.name, model=provider.model, cached=False,
                  time_taken=round(time.time() - started, 2))
    _remember(key, result)
    return result


@router.post("/document")
async def summarize_document(
    file: Optional[UploadFile] = File(None),
    language: str = Form("English"),
    model: Optional[str] = Form(None),
    refresh: bool = Form(False),
):
    language = _check_language(language)
    provider = _provider(model)
    filename, text, status = await _load_document(file)

    key = _cache_key("document", hashlib.sha256(text.encode("utf-8")).hexdigest(), language, provider)
    cached = _cached(key, refresh)
    if cached:
        return {**cached, "filename": filename}
    started = time.time()
    result = None
    try:
        async for event, data in summarizer.summarize_document_events(text, provider, language=language, status=status):
            if event == "result":
                result = data
    except LLMError as exc:
        logger.error("Document summary failed: %s", exc)
        raise HTTPException(status_code=503, detail="The AI model could not produce a summary. Try again shortly.")
    result.update(filename=filename, provider=provider.name, model=provider.model, cached=False,
                  time_taken=round(time.time() - started, 2))
    _remember(key, result)
    return result


@router.post("/document-stream")
async def summarize_document_stream(
    file: Optional[UploadFile] = File(None),
    language: str = Form("English"),
    model: Optional[str] = Form(None),
    refresh: bool = Form(False),
):
    """Events: start, progress (reading part i of n / merging / writing), result, done; or error."""
    language = _check_language(language)
    provider = _provider(model)
    # Read the upload before streaming starts; the request's file is closed afterwards.
    filename, text, status = await _load_document(file)
    key = _cache_key("document", hashlib.sha256(text.encode("utf-8")).hexdigest(), language, provider)

    async def events():
        yield _sse("start", {"filename": filename, "characters": len(text)})
        cached = _cached(key, refresh)
        if cached:
            yield _sse("result", {**cached, "filename": filename})
            yield _sse("done", {})
            return
        started = time.time()
        try:
            async for event, data in summarizer.summarize_document_events(text, provider, language=language, status=status):
                if event == "result":
                    data.update(filename=filename, provider=provider.name, model=provider.model, cached=False,
                                time_taken=round(time.time() - started, 2))
                    _remember(key, data)
                yield _sse(event, data)
        except LLMError as exc:
            logger.error("Document summary failed: %s", exc)
            yield _sse("error", {"detail": "The AI model could not produce a summary. Try again shortly."})
            return
        yield _sse("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream; charset=utf-8",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
