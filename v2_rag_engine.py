import os
import re
import uuid
import docx
import chromadb
from sentence_transformers import CrossEncoder
import httpx


# Configuration
V2_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_file_RAG")
V2_CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_chroma_db")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
INITIAL_TOP_K = 20
RERANKED_TOP_K = 6          # default number of chunks returned for a normal question
LIST_QUESTION_TOP_K = 15    # used when the question looks like it wants an enumerated list
RELEVANCE_THRESHOLD = -8.0  # ms-marco scores are usually logits, > 0 is very good, -5 is a relaxed cutoff
STRONG_RELEVANCE_THRESHOLD = -7.0  # below this, the top match is too weak to safely answer from

# Initialize clients lazily to save memory on startup
_chroma_client = None
_cross_encoder = None
_multilingual_ef = None

from chromadb.utils import embedding_functions

def get_embedding_function():
    global _multilingual_ef
    if _multilingual_ef is None:
        _multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return _multilingual_ef

def needs_translation(text: str) -> str:
    """Detects Telugu or Hindi (Devanagari) script and returns the language name."""
    for ch in text:
        code = ord(ch)
        if 0x0C00 <= code <= 0x0C7F:   # Telugu block
            return "Telugu"
        if 0x0900 <= code <= 0x097F:   # Devanagari (Hindi) block
            return "Hindi"
    return ""

async def translate_to_english(text: str, model: str = "llama3:latest") -> str:
    """Translates Telugu/Hindi query to English using Ollama before retrieval."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a professional translator. Translate the user's message to English. Respond with ONLY the English translation — no quotes, no explanation, no preamble."},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:11434/api/chat", json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            translated = data.get("message", {}).get("content", "").strip()
            return translated if translated else text
    except Exception:
        return text  # fall back to original query if translation fails

async def translate_from_english(text: str, target_language: str, model: str = "llama3:latest") -> str:
    """Translates an English answer back to the target language using Ollama."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"You are a professional translator. Translate the following English text to {target_language}. Respond with ONLY the {target_language} translation — no quotes, no explanation, no English words mixed in."},
            {"role": "user", "content": text}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://localhost:11434/api/chat", json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            translated = data.get("message", {}).get("content", "").strip()
            return translated if translated else text
    except Exception:
        return text  # fall back to English if translation fails


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        os.makedirs(V2_CHROMA_DB_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=V2_CHROMA_DB_DIR)
    return _chroma_client

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        # Cross-Encoder (Multilingual)
        _cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', max_length=512)
    return _cross_encoder


# ---------------------------------------------------------------------------
# FIX #1: extract_text_from_docx now also reads real Word tables.
# python-docx's doc.paragraphs does NOT include text inside table cells —
# those live separately in doc.tables. Any manual authored with genuine
# Word tables (BCI ranges, icon/function tables, batch-process tables, etc.)
# was previously being silently dropped from ingestion entirely.
# ---------------------------------------------------------------------------
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style.name.startswith("Heading"):
            parts.append(f"\n\n## SECTION: {text}\n")
        else:
            parts.append(text)

    # Pull in table content (row-by-row, pipe-separated so it still reads
    # naturally as text for embedding/retrieval purposes).
    for t_idx, table in enumerate(doc.tables):
        rows_text = []
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                rows_text.append(row_text)
        if rows_text:
            # Tag it as its own mini-section so it can be chunked/retrieved
            # as a coherent unit rather than getting merged into whatever
            # paragraph text happens to precede it.
            parts.append(f"\n\n## SECTION: Table {t_idx + 1}\n" + "\n".join(rows_text))

    return "\n".join(parts)


def extract_text_from_pdf(file_path):
    try:
        from pypdf import PdfReader
    except ImportError:
        import PyPDF2 as pypdf
        PdfReader = pypdf.PdfReader

    text_parts = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"## SECTION: Page {i+1}\n" + page_text.strip())
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")

def extract_text_from_excel(file_path, filename):
    import pandas as pd
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            sheets = {"Data": df}
        else:
            sheets = pd.read_excel(file_path, sheet_name=None)

        text_parts = []
        for sheet_name, df in sheets.items():
            text_parts.append(f"\n\n## SECTION: Excel Sheet '{sheet_name}'\n")
            # Convert each row to a textual format
            for index, row in df.iterrows():
                row_str = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                if row_str.strip():
                    text_parts.append(row_str)
        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to read Spreadsheet: {e}")

def create_overlapping_chunks(text, chunk_size, overlap):
    chunks = []
    sections = text.split("## SECTION: ")
    for idx, section_text in enumerate(sections):
        section_text = section_text.strip()
        if not section_text:
            continue

        section_title = section_text.split("\n")[0][:100] if idx > 0 else "Introduction"
        full_text = f"[{section_title}] " + section_text if idx > 0 else section_text

        start = 0
        text_len = len(full_text)

        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                while end > start and full_text[end] not in [' ', '\n', '\t']:
                    end -= 1
                if end == start:
                    end = start + chunk_size

            chunk = full_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap
    return chunks

def ingest_file_v2(file_path, filename):
    """Parses file, chunks it, and stores in ChromaDB."""
    print(f"V2 Ingesting: {filename}")
    text = ""
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif filename_lower.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif filename_lower.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls"):
        text = extract_text_from_excel(file_path, filename_lower)
    else:
        raise ValueError("Unsupported file type for V2. Please use .docx, .txt, .pdf, .xlsx, or .csv")

    if not text.strip():
        raise ValueError("File is empty or could not be read.")

    chunks = create_overlapping_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)

    client = get_chroma_client()

    # Wipe the entire collection to ensure strict single-document mode
    try:
        client.delete_collection(name="rrams_multilingual_v2")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="rrams_multilingual_v2",
        embedding_function=get_embedding_function()
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    # By default, ChromaDB uses all-MiniLM-L6-v2 for embeddings if we pass texts
    # Insert in batches to avoid ChromaDB batch size limits (max 166)
    batch_size = 150
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    return {"status": "success", "chunks_added": len(chunks)}


# ---------------------------------------------------------------------------
# FIX #2: detect "list all / name all / how many X are there" style
# questions and pull more chunks for them. These questions typically need
# every occurrence of a repeated heading/list-item pattern (e.g. all 8 NM
# tools, each of which became its own heading/section during ingestion), and
# the old fixed RERANKED_TOP_K=6 could never surface more than 6 of them no
# matter how well retrieval worked.
# ---------------------------------------------------------------------------
_LIST_QUESTION_PATTERNS = re.compile(
    r"\b(list all|name all|all of the|how many .*(are there|does|do)|"
    r"name (at least )?\d+|list the|enumerate|what are the (\w+\s){0,3}(tools|modules|"
    r"steps|types|charts|formats|groups|tables|kpis)\b)",
    re.IGNORECASE
)

def is_list_question(question: str) -> bool:
    return bool(_LIST_QUESTION_PATTERNS.search(question))


def retrieve_and_rerank(query, question: str = None):
    """
    Retrieves top K chunks, reranks them, and applies threshold.

    Returns a dict: {"chunks": [...], "top_score": float or None}
    top_score lets the caller decide whether the single best match is
    actually strong enough to answer from, instead of blindly trusting
    the LLM to say "not available" when it isn't.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="rrams_multilingual_v2",
        embedding_function=get_embedding_function()
    )

    if collection.count() == 0:
        return {"chunks": [], "top_score": None}

    top_k = LIST_QUESTION_TOP_K if (question and is_list_question(question)) else RERANKED_TOP_K

    # 1. Initial Vector Retrieval
    results = collection.query(
        query_texts=[query],
        n_results=min(INITIAL_TOP_K, collection.count())
    )

    candidates = results['documents'][0]

    if not candidates:
        return {"chunks": [], "top_score": None}

    # 2. Cross-Encoder Reranking
    encoder = get_cross_encoder()
    pairs = [[query, chunk] for chunk in candidates]
    scores = encoder.predict(pairs)

    # 3. Zip and Sort by Score
    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    top_score = float(scored_candidates[0][1]) if scored_candidates else None

    # 4. Apply Threshold and Select Top N
    final_chunks = []
    for chunk, score in scored_candidates:
        if score > RELEVANCE_THRESHOLD:
            final_chunks.append(chunk)
            if len(final_chunks) >= top_k:
                break

    return {"chunks": final_chunks, "top_score": top_score}
