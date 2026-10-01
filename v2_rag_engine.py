import os
import re
import json
import math
import uuid
import logging
import docx
import chromadb
from collections import Counter
from sentence_transformers import CrossEncoder
import httpx


logger = logging.getLogger("v2_rag_engine")

# Configuration
V2_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upload_file_RAG")
V2_CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_chroma_db")
V2_ACRONYM_FILE = os.path.join(V2_CHROMA_DB_DIR, "doc_acronyms.json")   # acronyms + term index
COLLECTION_NAME = "rrams_multilingual_v2"

# Smaller chunks. paraphrase-multilingual-MiniLM-L12-v2 only embeds the first
# ~128 tokens (~450-500 characters) of a chunk and silently ignores the rest,
# so large chunks hide most of their text from vector search. Each chunk is
# also prefixed with its "[module › section]" label.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
INITIAL_TOP_K = 30          # vector-search candidates
KEYWORD_TOP_K = 12          # keyword (BM25) candidates added on top of the vector ones
RERANKED_TOP_K = 6          # default number of chunks returned for a normal question
LIST_QUESTION_TOP_K = 15    # used when the question looks like it wants an enumerated list
MIN_FILTERED_RESULTS = 3    # if a module-filtered search finds fewer than this, fall back to a full search
MAX_EXACT_CHUNKS = 4        # max definition/heading chunks injected for "what is X?" questions
SECTION_EXPAND_MAX = 3000   # a picked chunk is replaced by its WHOLE section if the section is at most this long
MAX_CONTEXT_CHARS = 12000   # total size of what is sent to the LLM (llama3 has an 8k-token window)
DUPLICATE_SIMILARITY = 0.85 # sections this similar across modules (e.g. the same Error Log list) are merged into one
RELEVANCE_THRESHOLD = -8.0  # ms-marco scores are usually logits, > 0 is very good, -5 is a relaxed cutoff

# fastapi_app.py refuses to answer when the best reranker score is below this.
# -1.0 was too strict: real in-document questions such as "what is
# maintenance?" (-1.47) and "what is inspection?" (-2.25) were being turned
# away without ever reaching the LLM. Clearly unrelated questions still score
# far lower (e.g. "what is a high maintenance girl?" scored -6.2), and the LLM
# prompt itself falls back to the "not available" message when the records
# don't contain the answer.
STRONG_RELEVANCE_THRESHOLD = -3.0

# When a "what is X?" question names a term that is defined (or is a section
# heading) in the document, retrieval reports at least this score, so the
# answer is never refused on reranker score alone.
EXACT_TERM_SCORE = 5.0

# Initialize clients lazily to save memory on startup
_chroma_client = None
_cross_encoder = None
_multilingual_ef = None
_acronym_cache = None
_bm25_cache = None

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
# Section markers used between extraction and chunking.
# Every extractor emits "## SECTION: <label>" lines. A label may carry a
# module tag ("@@MODULE=TIS@@") which the chunker strips off and stores as
# chunk metadata, so retrieval can filter by module.
# ---------------------------------------------------------------------------
SECTION_MARKER = "## SECTION: "
_MODULE_TAG_RE = re.compile(r"@@MODULE=(.*?)@@")

# Headings TYPED into the document as plain text ("## 1.3 Navigate to TIS").
_TYPED_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# A module title line such as
#   "Traffic Information System (TIS) — GujRAMS Module Reference"
_MODULE_TITLE_RE = re.compile(
    r"^(?P<name>[A-Z][^()\n]{2,80}?)\s*\((?P<abbr>[A-Z][A-Z0-9&]{1,7})\)\s*[—–-]\s*.*\bmodule\b",
    re.IGNORECASE,
)

# Glossary lines: "Maintenance — Scheduling and carrying out repair ..."
#                 "KPI (Key Performance Indicator) — A summary metric ..."
_DEFINITION_RE = re.compile(r"^(?P<term>[A-Z][^—–|:\n]{0,80}?)\s+[—–]\s+(?P<definition>.{15,})$")
_NOT_A_TERM_START = re.compile(r"^(annexure|appendix|note|caution|step|steps|example|section|table|figure|chapter)\b", re.IGNORECASE)

# "Full Name (ABBR)"  and  "ABBR (Full Name)"
_ACRONYM_DEF_RE = re.compile(
    r"\b([A-Z][A-Za-z&/-]+(?:\s+(?:of|and|for|the|&|[A-Z][A-Za-z&/-]+)){1,7})\s*\(([A-Z][A-Z0-9]{1,7})\)"
)
_ACRONYM_REV_RE = re.compile(
    r"\b([A-Z][A-Z0-9]{1,7})\s*\(([A-Z][A-Za-z&/-]+(?:\s+(?:of|and|for|the|&|[A-Z][A-Za-z&/-]+)){1,7})\)"
)

_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "for", "to", "in", "on", "at", "by", "with",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "when", "where", "do", "does", "did",
    "can", "could", "should", "would", "will", "i", "me", "my", "we", "you", "your",
    "from", "as", "about", "into", "than", "then", "there", "their", "its", "if",
    "tell", "explain", "define", "describe", "please", "give", "show", "mean", "means",
}

# Acronyms that are also ordinary English words; only expanded in capitals.
_COMMON_WORDS = {
    "IT", "IS", "AS", "OR", "AT", "ON", "IN", "NO", "US", "AN", "TO", "BE", "WE",
    "IF", "OF", "DO", "GO", "SO", "UP", "ME", "MY", "BY", "HE", "OK", "ALL", "AND",
    "FOR", "THE", "ARE", "ANY", "CAN", "NOT", "HOW", "WHO", "MAP", "USE", "SET",
}


def _structure_blocks(blocks):
    """
    Turns a sequence of (text, heading_level, is_table) blocks into
    sectioned text. Each section label is a breadcrumb such as
        "TIS › 1.3 Navigate to TIS"
    so every chunk says which module and sub-section it came from.
    """
    parts = []
    current_module = ""
    current_module_abbr = ""
    heading_stack = {}           # level -> heading text

    def breadcrumb():
        crumbs = []
        if current_module_abbr:
            crumbs.append(current_module_abbr)
        elif current_module:
            crumbs.append(current_module)
        levels = sorted(heading_stack)
        for lvl in levels[-2:]:
            crumbs.append(heading_stack[lvl])
        return " › ".join(crumbs) if crumbs else "Introduction"

    def open_section(label):
        tag = f"@@MODULE={current_module_abbr or current_module}@@"
        parts.append(f"\n\n{SECTION_MARKER}{label}{tag}\n")

    for text, style_level, is_table in blocks:
        text = text.strip()
        if not text:
            continue

        if is_table:
            parts.append(text)
            continue

        # 1) Module title line -> new module, reset headings
        m_title = _MODULE_TITLE_RE.match(text) if len(text) <= 160 else None
        if m_title:
            current_module = f"{m_title.group('name').strip()} ({m_title.group('abbr')})"
            current_module_abbr = m_title.group("abbr")
            heading_stack = {}
            open_section(current_module)
            parts.append(text)
            continue

        # 2) Word heading style
        if style_level:
            if style_level == 1 and not current_module_abbr:
                current_module = text[:100]
                heading_stack = {}
            else:
                heading_stack = {k: v for k, v in heading_stack.items() if k < style_level}
                heading_stack[style_level] = text[:100]
            open_section(breadcrumb())
            parts.append(text)
            continue

        # 3) Typed markdown heading
        m_head = _TYPED_HEADING_RE.match(text)
        if m_head:
            level = len(m_head.group(1))
            title = m_head.group(2).strip()
            heading_stack = {k: v for k, v in heading_stack.items() if k < level}
            heading_stack[level] = title[:100]
            open_section(breadcrumb())
            parts.append(title)
            continue

        parts.append(text)

    return "\n".join(parts)


def _table_to_text(table):
    rows_text = []
    for row in table.rows:
        cells = []
        for c in row.cells:
            ct = c.text.strip()
            # merged cells repeat in python-docx; skip consecutive duplicates
            if ct and (not cells or cells[-1] != ct):
                cells.append(ct)
        if cells:
            rows_text.append(" | ".join(cells))
    return "\n".join(rows_text)


def _heading_level(para):
    try:
        name = para.style.name or ""
    except Exception:
        return 0
    if name == "Title":
        return 1
    m = re.match(r"Heading\s*(\d)", name)
    return int(m.group(1)) if m else 0


def extract_text_from_docx(file_path):
    """Reads paragraphs AND tables in document order."""
    doc = docx.Document(file_path)
    blocks = []

    if hasattr(doc, "iter_inner_content"):
        items = doc.iter_inner_content()
    else:
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        items = []
        for child in doc.element.body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                items.append(Paragraph(child, doc))
            elif tag == "tbl":
                items.append(Table(child, doc))

    for item in items:
        if hasattr(item, "rows"):          # table
            t = _table_to_text(item)
            if t:
                blocks.append((t, 0, True))
        else:                              # paragraph
            blocks.append((item.text, _heading_level(item), False))

    return _structure_blocks(blocks)


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
                text_parts.append(f"{SECTION_MARKER}Page {i+1}\n" + page_text.strip())
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
            text_parts.append(f"\n\n{SECTION_MARKER}Excel Sheet '{sheet_name}'\n")
            for index, row in df.iterrows():
                row_str = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
                if row_str.strip():
                    text_parts.append(row_str)
        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to read Spreadsheet: {e}")


def _iter_sections(text):
    """Yields (label, module, body) for every section of extracted text."""
    sections = text.split(SECTION_MARKER)
    for idx, section_text in enumerate(sections):
        section_text = section_text.strip()
        if not section_text:
            continue
        module = ""
        if idx > 0:
            first_line, _, body = section_text.partition("\n")
            m = _MODULE_TAG_RE.search(first_line)
            if m:
                module = m.group(1).strip()
                first_line = _MODULE_TAG_RE.sub("", first_line)
            label = first_line.strip()[:120] or "Introduction"
        else:
            label, body = "Introduction", section_text
        body = body.strip()
        if body:
            yield label, module, body


def _split_long_text(body, max_len, overlap):
    """Splits body into pieces of at most max_len chars, breaking on whitespace."""
    pieces = []
    start = 0
    n = len(body)
    while start < n:
        end = start + max_len
        if end < n:
            cut = body.rfind("\n", start + max_len // 2, end)
            if cut == -1:
                cut = body.rfind(" ", start + max_len // 2, end)
            if cut != -1:
                end = cut
        else:
            end = n
        piece = body[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        sp = body.find(" ", next_start, end)
        next_start = sp + 1 if sp != -1 else next_start
        start = next_start if next_start > start else end
    return pieces


def create_overlapping_chunks(text, chunk_size, overlap, with_sections=False):
    """
    Every chunk carries its "[module › section]" label at the START.
    Returns a list of (chunk_text, module) tuples, or
    (chunk_text, module, section_id) tuples when with_sections=True.
    """
    chunks = []
    for sid, (label, module, body) in enumerate(_iter_sections(text)):
        prefix = f"[{label}] "
        room = max(200, chunk_size - len(prefix))
        for piece in _split_long_text(body, room, overlap):
            chunks.append((prefix + piece, module, sid) if with_sections else (prefix + piece, module))
    return chunks


def build_sections(text):
    """section_id -> {"label", "module", "body"} (used to hand the LLM whole sections)."""
    return {str(sid): {"label": label, "module": module, "body": body}
            for sid, (label, module, body) in enumerate(_iter_sections(text))}


# ---------------------------------------------------------------------------
# Term index: glossary definitions + section headings.
#
# A glossary of ~20 "Term — definition" lines was being cut into 3-4 chunks
# of mixed definitions, so "what is maintenance?" was compared against a
# chunk that was mostly about Report / Survey Year / Condition and scored
# too low to be answered. Each definition now becomes its OWN chunk, and
# every defined term and section heading is indexed so "what is X?"
# questions go straight to the right text.
# ---------------------------------------------------------------------------
def _normalize_term(term):
    t = term.lower()
    t = re.sub(r"[“”\"'’`]", "", t)
    t = re.sub(r"^\s*(annexure\s+[a-z]\s*[—–-]\s*)", "", t)
    t = re.sub(r"^\s*[\d.]+\s*", "", t)              # "2.2 Traffic Volume Count"
    t = re.sub(r"^(the|a|an)\s+", "", t)
    t = re.sub(r"[^\w&/ -]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _term_keys(term):
    """'KPI (Key Performance Indicator)' -> {'kpi', 'key performance indicator', ...}"""
    keys = set()
    inside = re.findall(r"\(([^)]*)\)", term)
    outside = re.sub(r"\([^)]*\)", " ", term)
    for piece in [outside] + inside:
        k = _normalize_term(piece)
        if k:
            keys.add(k)
            if len(k) > 4 and k.endswith("s") and not k.endswith("ss"):
                keys.add(k[:-1])                    # simple singular
    full = _normalize_term(term)
    if full:
        keys.add(full)
    return keys


def _is_glossary_term(term):
    term = term.strip()
    if not term or _NOT_A_TERM_START.match(term):
        return False
    words = re.sub(r"\([^)]*\)", " ", term).split()
    if not words or len(words) > 6:
        return False
    for w in words:
        wl = w.lower()
        if wl in {"of", "and", "for", "the", "&", "or", "to", "in", "a", "an", "with"}:
            continue
        if not (w[0].isupper() or w[0].isdigit()):
            return False                            # sentence, not a term
    return True


def extract_definitions(text):
    """Returns a list of {"term", "keys", "module", "chunk"} for glossary lines."""
    defs = []
    for label, module, body in _iter_sections(text):
        for line in body.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if _MODULE_TITLE_RE.match(line):
                continue
            m = _DEFINITION_RE.match(line)
            if not m or not _is_glossary_term(m.group("term")):
                continue
            term = m.group("term").strip()
            where = module or label.split(" › ")[0]
            chunk = f"[{where} › Definition: {term}] {line}"
            defs.append({"term": term, "keys": sorted(_term_keys(term)), "module": module, "chunk": chunk})
    return defs


def build_term_index(text, section_chunks, definitions):
    """
    key -> list of {"chunk", "module", "kind"} where kind is "definition" or "heading".
    Heading entries point at the first chunk of that section.
    """
    index = {}

    def add(key, entry):
        lst = index.setdefault(key, [])
        if entry["chunk"] not in (e["chunk"] for e in lst):
            lst.append(entry)

    for d in definitions:
        for k in d["keys"]:
            add(k, {"chunk": d["chunk"], "module": d["module"], "kind": "definition"})

    seen_labels = set()
    for chunk, module in section_chunks:
        m = re.match(r"^\[([^\]]+)\]", chunk)
        if not m:
            continue
        label = m.group(1)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        last = label.split(" › ")[-1]
        for k in _term_keys(last):
            if len(k) >= 3:
                add(k, {"chunk": chunk, "module": module, "kind": "heading"})
    return index


def build_acronym_map(text):
    """Collects 'Full Name (ABBR)' and 'ABBR (Full Name)' definitions found in the document."""
    acronyms = {}
    clean = _MODULE_TAG_RE.sub("", text)
    minor = {"of", "and", "for", "the", "&"}

    def fit(full, abbr):
        words = [w for w in re.split(r"[\s/-]+", full.strip()) if w]
        # shortest trailing run of words whose initials spell the acronym
        for k in range(1, len(words) + 1):
            cand = words[-k:]
            with_minor = "".join(w[0].upper() for w in cand)
            without_minor = "".join(w[0].upper() for w in cand if w.lower() not in minor)
            if abbr in (with_minor, without_minor):
                return " ".join(cand)
        return ""

    for full, abbr in _ACRONYM_DEF_RE.findall(clean):
        name = fit(full, abbr)
        if name and abbr not in acronyms:
            acronyms[abbr] = name
    for abbr, full in _ACRONYM_REV_RE.findall(clean):
        words = full.split()
        # initials of the leading words should spell the acronym
        for k in range(len(words), 0, -1):
            cand = words[:k]
            with_minor = "".join(w[0].upper() for w in cand)
            without_minor = "".join(w[0].upper() for w in cand if w.lower() not in minor)
            if abbr in (with_minor, without_minor):
                if abbr not in acronyms:
                    acronyms[abbr] = " ".join(cand)
                break
    return acronyms


def _save_index(acronyms, modules, terms, sections=None):
    global _acronym_cache
    data = {"acronyms": acronyms, "modules": sorted(m for m in modules if m), "terms": terms,
            "sections": sections or {}}
    try:
        os.makedirs(V2_CHROMA_DB_DIR, exist_ok=True)
        with open(V2_ACRONYM_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    _acronym_cache = data


def _load_acronyms():
    global _acronym_cache
    if _acronym_cache is None:
        try:
            with open(V2_ACRONYM_FILE, "r", encoding="utf-8") as f:
                _acronym_cache = json.load(f)
        except Exception:
            _acronym_cache = {"acronyms": {}, "modules": [], "terms": {}}
    return _acronym_cache


def expand_query(query):
    """
    "what is tis?" -> "what is tis? TIS (Traffic Information System)".
    Returns (expanded_query, [modules mentioned in the query]).
    """
    data = _load_acronyms()
    acronyms = data.get("acronyms", {})
    modules = set(data.get("modules", []))

    additions = []
    mentioned_modules = []
    for abbr, full in acronyms.items():
        if abbr in _COMMON_WORDS or len(abbr) < 3:
            hit = re.search(rf"\b{re.escape(abbr)}\b", query)
        else:
            hit = re.search(rf"\b{re.escape(abbr)}\b", query, re.IGNORECASE)
        if hit:
            if full.lower() not in query.lower():
                additions.append(f"{abbr} ({full})")
            if abbr in modules:
                mentioned_modules.append(abbr)

    for abbr, full in acronyms.items():
        if abbr in modules and abbr not in mentioned_modules and full.lower() in query.lower():
            mentioned_modules.append(abbr)

    expanded = query + (" " + " ".join(additions) if additions else "")
    return expanded, mentioned_modules


_DEF_QUESTION_PREFIX = re.compile(
    r"^(please\s+)?(can you\s+|could you\s+)?(tell me\s+)?("
    r"what\s+(is|are|was|were)\s+(meant\s+by\s+|the\s+meaning\s+of\s+|the\s+definition\s+of\s+)?|"
    r"what\s+do\s+you\s+mean\s+by\s+|what\s+does\s+|"
    r"define\s+|explain\s+|describe\s+|meaning\s+of\s+|definition\s+of\s+|"
    r"tell\s+me\s+about\s+|about\s+)",
    re.IGNORECASE,
)
_DEF_QUESTION_SUFFIX = re.compile(r"\s+(mean|means|stand\s+for|stands\s+for|refer\s+to|refers\s+to)$", re.IGNORECASE)


def lookup_terms(query, mentioned_modules=None):
    """
    For "what is X?" / "define X" / "X" style questions, returns the
    definition / heading chunks for X if X is a term in the document.
    "what is maintenance and planning?" returns both definitions.
    Returns [] when the question is not about a known term.
    """
    data = _load_acronyms()
    terms = data.get("terms", {})
    if not terms:
        return []
    acronyms = data.get("acronyms", {})
    modules = data.get("modules", [])
    mentioned_modules = mentioned_modules or []

    q = query.strip().strip("?.!").strip()
    is_def_question = bool(_DEF_QUESTION_PREFIX.match(q))
    q = _DEF_QUESTION_PREFIX.sub("", q)
    q = _DEF_QUESTION_SUFFIX.sub("", q).strip()

    # drop a trailing module qualifier: "maintenance in BMS", "... under the TIS module"
    for abbr in modules:
        names = [re.escape(abbr)] + ([re.escape(acronyms[abbr])] if abbr in acronyms else [])
        q = re.sub(rf"\s+(in|of|for|under|from)\s+(the\s+)?({'|'.join(names)})(\s+module)?$", "", q, flags=re.IGNORECASE)

    phrase = _normalize_term(q)
    if not phrase:
        return []

    def find(key):
        if key in terms:
            return terms[key]
        if len(key) > 4 and key.endswith("s") and key[:-1] in terms:
            return terms[key[:-1]]
        up = key.upper()
        if up in acronyms:                          # "tis" -> "traffic information system"
            full = _normalize_term(acronyms[up])
            if full in terms:
                return terms[full]
        return None

    parts = [phrase]
    if find(phrase) is None:
        if not is_def_question:
            return []
        parts = [p for p in re.split(r"\s*(?:,|\band\b|&|\bvs\b|\bversus\b|\bor\b)\s*", phrase) if p]
        if len(parts) < 2:
            return []

    picked = []
    for p in parts:
        entries = find(p)
        if entries is None:
            return []                               # every part must be a real term
        entries = sorted(entries, key=lambda e: (e["kind"] != "definition",
                                                 bool(mentioned_modules) and e["module"] not in mentioned_modules))
        if mentioned_modules:
            preferred = [e for e in entries if e["module"] in mentioned_modules]
            entries = preferred or entries
        defs = [e for e in entries if e["kind"] == "definition"]
        chosen = defs[:2] if defs else entries[:3]
        picked.extend(e["chunk"] for e in chosen)

    out = []
    for c in picked:
        if c not in out:
            out.append(c)
    return out[:MAX_EXACT_CHUNKS]


# ---------------------------------------------------------------------------
# Keyword (BM25) retrieval alongside vector search. The multilingual
# embedding model is weak on exact domain words (abutment, chainage, PCU,
# BCI ...); a plain keyword score makes sure chunks that literally contain
# the asked-about words always reach the reranker.
# ---------------------------------------------------------------------------
def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _get_bm25(collection):
    global _bm25_cache
    if _bm25_cache is None or _bm25_cache.get("count") != collection.count():
        data = collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{} for _ in docs]
        toks = [_tokens(d) for d in docs]
        df = Counter()
        for t in toks:
            df.update(set(t))
        avgdl = (sum(len(t) for t in toks) / len(toks)) if toks else 1.0
        _bm25_cache = {"count": collection.count(), "docs": docs, "metas": metas,
                       "tf": [Counter(t) for t in toks], "len": [len(t) for t in toks],
                       "df": df, "avgdl": avgdl or 1.0, "N": len(docs)}
    return _bm25_cache


def keyword_search(collection, query, n, modules=None, k1=1.5, b=0.75):
    idx = _get_bm25(collection)
    q = set(_tokens(query))
    if not q or not idx["N"]:
        return []
    scores = []
    for i in range(idx["N"]):
        if modules and (idx["metas"][i] or {}).get("module") not in modules:
            continue
        tf, dl, s = idx["tf"][i], idx["len"][i], 0.0
        for w in q:
            f = tf.get(w, 0)
            if not f:
                continue
            idf = math.log(1 + (idx["N"] - idx["df"][w] + 0.5) / (idx["df"][w] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / idx["avgdl"]))
        if s > 0:
            scores.append((s, i))
    scores.sort(reverse=True)
    return [idx["docs"][i] for _, i in scores[:n]]


def ingest_file_v2(file_path, filename):
    """Parses file, chunks it, and stores in ChromaDB."""
    global _bm25_cache
    print(f"V2 Ingesting: {filename}")
    text = ""
    filename_lower = filename.lower()

    if filename_lower.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif filename_lower.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif filename_lower.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read()
        text = _structure_blocks((line, 0, False) for line in raw.splitlines())
    elif filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls"):
        text = extract_text_from_excel(file_path, filename_lower)
    else:
        raise ValueError("Unsupported file type for V2. Please use .docx, .txt, .pdf, .xlsx, or .csv")

    if not text.strip():
        raise ValueError("File is empty or could not be read.")

    chunk_triples = create_overlapping_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP, with_sections=True)
    section_chunks = [(c, m) for c, m, _ in chunk_triples]
    definitions = extract_definitions(text)

    chunks = [c for c, _, _ in chunk_triples] + [d["chunk"] for d in definitions]
    modules = [m for _, m, _ in chunk_triples] + [d["module"] for d in definitions]
    kinds = ["text"] * len(chunk_triples) + ["definition"] * len(definitions)
    section_ids = [str(s) for _, _, s in chunk_triples] + [""] * len(definitions)

    # de-duplicate identical chunk texts (Chroma allows it, but it wastes top-K slots)
    seen, uniq = set(), []
    for c, m, k, s in zip(chunks, modules, kinds, section_ids):
        if c not in seen:
            seen.add(c)
            uniq.append((c, m, k, s))
    chunks = [u[0] for u in uniq]
    modules = [u[1] for u in uniq]
    kinds = [u[2] for u in uniq]
    section_ids = [u[3] for u in uniq]

    _save_index(build_acronym_map(text), set(modules),
                build_term_index(text, section_chunks, definitions),
                build_sections(text))

    client = get_chroma_client()

    # Wipe the entire collection to ensure strict single-document mode
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function()
    )

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"filename": filename, "chunk_index": i, "module": modules[i], "kind": kinds[i],
                  "section": section_ids[i]}
                 for i in range(len(chunks))]

    # Insert in batches to avoid ChromaDB batch size limits (max 166)
    batch_size = 150
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

    _bm25_cache = None
    return {"status": "success", "chunks_added": len(chunks)}


# ---------------------------------------------------------------------------
# Detect "list all / name all / how many X are there" style questions and
# pull more chunks for them.
# ---------------------------------------------------------------------------
_LIST_QUESTION_PATTERNS = re.compile(
    r"\b(list all|name all|all of the|how many .*(are there|does|do)|"
    r"name (at least )?\d+|list the|enumerate|what are the (\w+\s){0,3}(tools|modules|"
    r"steps|types|charts|formats|groups|tables|kpis|reports)\b)",
    re.IGNORECASE
)

def is_list_question(question: str) -> bool:
    return bool(_LIST_QUESTION_PATTERNS.search(question))


def retrieve_and_rerank(query, question: str = None):
    """
    Retrieves candidates (exact term matches + vector search + keyword
    search), reranks them, and applies threshold.

    Returns a dict: {"chunks": [...], "top_score": float or None}
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function()
    )

    total = collection.count()
    if total == 0:
        return {"chunks": [], "top_score": None}

    top_k = LIST_QUESTION_TOP_K if (question and is_list_question(question)) else RERANKED_TOP_K

    expanded_query, mentioned_modules = expand_query(query)
    n_results = min(INITIAL_TOP_K, total)

    # 0. Exact term match for "what is X?" questions
    exact_chunks = lookup_terms(query, mentioned_modules)
    if exact_chunks:
        logger.info("EXACT_TERM_MATCH | query=%r | chunks=%d", query, len(exact_chunks))

    # 1. Vector retrieval (module-filtered first when the question names a module)
    candidates = []
    if mentioned_modules:
        where = ({"module": mentioned_modules[0]} if len(mentioned_modules) == 1
                 else {"module": {"$in": mentioned_modules}})
        try:
            res = collection.query(query_texts=[expanded_query], n_results=n_results, where=where)
            candidates = res["documents"][0] if res and res.get("documents") else []
        except Exception:
            candidates = []

    if len(candidates) < MIN_FILTERED_RESULTS:
        res = collection.query(query_texts=[expanded_query], n_results=n_results)
        full = res["documents"][0] if res and res.get("documents") else []
        candidates = candidates + [c for c in full if c not in candidates]

    # 2. Keyword retrieval
    try:
        kw = keyword_search(collection, expanded_query, KEYWORD_TOP_K, modules=mentioned_modules or None)
        if mentioned_modules and len(kw) < MIN_FILTERED_RESULTS:
            kw += [c for c in keyword_search(collection, expanded_query, KEYWORD_TOP_K) if c not in kw]
    except Exception as ex:
        logger.warning("keyword search failed: %s", ex)
        kw = []

    seen = set()
    merged = []
    for c in exact_chunks + candidates + kw:
        if c not in seen:
            seen.add(c)
            merged.append(c)
    candidates = merged

    if not candidates:
        return {"chunks": [], "top_score": None}

    # 3. Cross-Encoder Reranking (with the expanded query)
    encoder = get_cross_encoder()
    pairs = [[expanded_query, chunk] for chunk in candidates]
    scores = [float(s) for s in encoder.predict(pairs)]

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    top_score = scored_candidates[0][1] if scored_candidates else None

    # 4. Exact term chunks always go first; then the best-scoring rest
    final_chunks = list(exact_chunks)
    for chunk, score in scored_candidates:
        if len(final_chunks) >= max(top_k, len(exact_chunks)):
            break
        if chunk in final_chunks:
            continue
        if score > RELEVANCE_THRESHOLD:
            final_chunks.append(chunk)

    if exact_chunks:
        top_score = max(top_score if top_score is not None else EXACT_TERM_SCORE, EXACT_TERM_SCORE)

    # keep going down the ranked list so merged/expanded sections don't leave empty slots
    ranked = list(final_chunks) + [c for c, s in scored_candidates
                                   if s > RELEVANCE_THRESHOLD and c not in final_chunks]
    try:
        final_chunks = _assemble_context(collection, ranked, max(top_k, len(exact_chunks)))
    except Exception as ex:
        logger.warning("section assembly failed, using raw chunks: %s", ex)

    return {"chunks": final_chunks, "top_score": top_score}


# ---------------------------------------------------------------------------
# Whole-section context + cross-module de-duplication.
#
# A list like "Possible errors and how to resolve them" (6 errors) was split
# over 4 chunks, and the same list exists in BMS, TIS and RIS — 12 nearly
# identical chunks competing for 6 slots. The LLM typically received the
# first part of the list three times and never saw the rest.
# Now: when a chunk is picked, the LLM gets its ENTIRE section (if the
# section is small enough), and sections that are the same across modules
# are sent once, labelled "[BMS / TIS / RIS › ...]".
# ---------------------------------------------------------------------------
def _chunk_meta_map(collection):
    idx = _get_bm25(collection)
    if "meta_by_doc" not in idx:
        idx["meta_by_doc"] = {d: (m or {}) for d, m in zip(idx["docs"], idx["metas"])}
    return idx["meta_by_doc"]


def _normalize_for_compare(body, modules):
    t = body.lower()
    for m in modules:
        t = re.sub(rf"\b{re.escape(m.lower())}\b", "<m>", t)
    t = re.sub(r"\b\d+(\.\d+)*\b", "<n>", t)
    return re.sub(r"\s+", " ", t).strip()


def _assemble_context(collection, ranked_chunks, top_k):
    from difflib import SequenceMatcher

    data = _load_acronyms()
    sections = data.get("sections", {})
    modules = data.get("modules", [])
    meta_by_doc = _chunk_meta_map(collection) if sections else {}

    units = []            # {"key", "label_modules", "label_rest", "body", "norm"}
    used_sections = set()

    def add_unit(key, module, label, body):
        rest = label.split(" › ", 1)[1] if (module and label.startswith(module + " › ")) else label
        norm = _normalize_for_compare(body, modules)
        for u in units:
            if u["key"] == key:
                return
            # same content in another module -> merge labels instead of repeating text
            if module and u["modules"] and module not in u["modules"] and len(norm) > 80:
                if SequenceMatcher(None, u["norm"], norm).quick_ratio() >= DUPLICATE_SIMILARITY and \
                   SequenceMatcher(None, u["norm"], norm).ratio() >= DUPLICATE_SIMILARITY:
                    u["modules"].append(module)
                    return
        units.append({"key": key, "modules": [module] if module else [], "rest": rest,
                      "label": label, "body": body, "norm": norm})

    for chunk in ranked_chunks:
        meta = meta_by_doc.get(chunk, {})
        sid = meta.get("section") or ""
        sec = sections.get(sid) if sid else None
        if sec and len(sec["body"]) <= SECTION_EXPAND_MAX:
            if sid in used_sections:
                continue
            used_sections.add(sid)
            add_unit("s" + sid, sec["module"], sec["label"], sec["body"])
        else:
            m = re.match(r"^\[([^\]]+)\]\s*", chunk)
            label = m.group(1) if m else ""
            body = chunk[m.end():] if m else chunk
            module = meta.get("module", "")
            add_unit("c" + chunk, module, label, body)
        if len(units) >= top_k:
            break

    out, total = [], 0
    for u in units:
        if len(u["modules"]) > 1:
            label = f"{' / '.join(u['modules'])} › {u['rest']}"
            note = f"(This section is the same in the {', '.join(u['modules'])} modules.)\n"
        else:
            label, note = u["label"], ""
        text = f"[{label}] {note}{u['body']}"
        if out and total + len(text) > MAX_CONTEXT_CHARS:
            break
        out.append(text[:MAX_CONTEXT_CHARS])
        total += len(text)
    return out
