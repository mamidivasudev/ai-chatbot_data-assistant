"""
Verified query library — few-shot examples for SQL generation.

A support team's (question, SQL) pairs are the highest-value input this system
can take, but they must not become entries in skills.json. A hundred rules
would exceed the rule budget, contradict each other, and fire on the wrong
questions. Worked examples behave better: the model is shown three to five
queries that solve questions resembling the one asked, and generalises from
them. This is the standard approach for text-to-SQL (DAIL-SQL / DIN-SQL).

So the library is stored whole and retrieved selectively:

  * IMPORT    — bulk load from CSV, Excel or JSON. Every SQL is run through the
                read-only guard, so a pair containing UPDATE is rejected at
                import rather than becoming an example the model imitates.
  * RETRIEVE  — IDF-weighted token overlap plus a phrase bonus. Deterministic,
                instant, and no model to load for a library this size.
  * RENDER    — top-k pairs as examples, under a character budget.

Scope keys work as in skills.py: "*" is global, a database name limits the
examples to that database.
"""

import json
import logging
import math
import os
import re
import uuid
from datetime import date

from sql_guard import validate_read_only

logger = logging.getLogger("query_library")

LIBRARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query_library.json")

GLOBAL_SCOPE = "*"

# How many examples reach the prompt, and the ceiling on their combined size.
DEFAULT_TOP_K = 4
MAX_EXAMPLE_CHARS = 6000
# Below this score an example is more distracting than helpful.
MIN_SCORE = 0.08

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on", "for",
    "to", "and", "or", "how", "what", "which", "there", "their", "me", "show",
    "give", "list", "get", "find", "tell", "do", "does", "did", "i", "we", "you",
    "please", "can", "with", "by", "from", "that", "this", "it", "as", "at",
    "all", "any", "much", "many",
}


# ---------------------------------------------------------------------------
# Tokenising
# ---------------------------------------------------------------------------
def _singular(token):
    if len(token) <= 3 or token.endswith("ss"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def tokenise(text, keep_stopwords=False):
    raw = [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]
    out = [_singular(t) for t in raw]
    if keep_stopwords:
        return out
    return [t for t in out if t not in _STOPWORDS]


def _bigrams(tokens):
    return {f"{a}_{b}" for a, b in zip(tokens, tokens[1:])}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def _empty_document():
    return {"version": 1, "examples": []}


def load_library(path=LIBRARY_FILE):
    if not os.path.exists(path):
        return _empty_document()
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as exc:
        logger.warning("Could not read query library: %s", exc)
        return _empty_document()

    if not isinstance(doc, dict) or not isinstance(doc.get("examples"), list):
        return _empty_document()
    doc.setdefault("version", 1)
    return doc


def save_library(doc, path=LIBRARY_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
def _normalise_sql(sql):
    return re.sub(r"\s+", " ", str(sql)).strip().rstrip(";")


def add_examples(pairs, scope=GLOBAL_SCOPE, source="manual", path=LIBRARY_FILE):
    """
    Add (question, sql) pairs to the library.

    Returns a report: how many were added, and every rejection with a reason,
    so a support team's spreadsheet can be corrected and re-submitted rather
    than failing silently.
    """
    doc = load_library(path)
    existing = doc["examples"]

    # Deduplicate on the question plus the normalised SQL.
    seen = {
        (tuple(tokenise(e.get("question", ""))), _normalise_sql(e.get("sql", "")).lower())
        for e in existing
    }

    added, rejected = [], []

    for index, pair in enumerate(pairs):
        if isinstance(pair, dict):
            question = pair.get("question") or pair.get("Question") or ""
            sql = pair.get("sql") or pair.get("SQL") or pair.get("query") or ""
            row_scope = (pair.get("database") or pair.get("scope") or scope) or scope
            notes = pair.get("notes") or pair.get("Notes") or ""
        else:
            try:
                question, sql = pair[0], pair[1]
            except Exception:
                rejected.append({"row": index + 1, "reason": "unrecognised row format"})
                continue
            row_scope, notes = scope, ""

        question = str(question).strip()
        sql = _normalise_sql(sql)

        if not question:
            rejected.append({"row": index + 1, "reason": "question is empty"})
            continue
        if not sql:
            rejected.append({"row": index + 1, "question": question, "reason": "sql is empty"})
            continue

        # A write in the library would teach the model to emit writes.
        is_safe, reason = validate_read_only(sql)
        if not is_safe:
            rejected.append({
                "row": index + 1, "question": question,
                "reason": f"not a read-only query: {reason}",
            })
            continue

        key = (tuple(tokenise(question)), sql.lower())
        if key in seen:
            rejected.append({"row": index + 1, "question": question, "reason": "duplicate"})
            continue
        seen.add(key)

        example = {
            "id": uuid.uuid4().hex[:8],
            "question": question,
            "sql": sql,
            "scope": str(row_scope) or GLOBAL_SCOPE,
            "notes": str(notes),
            "source": source,
            "added_on": date.today().isoformat(),
            "enabled": True,
        }
        existing.append(example)
        added.append(example)

    save_library(doc, path)
    return {
        "added": len(added),
        "rejected": len(rejected),
        "total": len(existing),
        "rejections": rejected,
        "added_examples": added,
    }


def parse_upload(file_bytes, filename):
    """Read (question, sql) rows from a CSV, Excel or JSON upload."""
    name = (filename or "").lower()

    if name.endswith(".json"):
        data = json.loads(file_bytes.decode("utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("examples") or data.get("queries") or []
        return list(data)

    import io
    import pandas as pd

    if name.endswith(".csv"):
        frame = pd.read_csv(io.BytesIO(file_bytes))
    elif name.endswith((".xlsx", ".xls")):
        frame = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Use .csv, .xlsx, .xls or .json")

    # Accept whatever casing the spreadsheet uses.
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "question" not in frame.columns:
        raise ValueError(f"No 'question' column found. Columns present: {list(frame.columns)}")

    sql_column = next((c for c in ("sql", "query", "sql_query") if c in frame.columns), None)
    if not sql_column:
        raise ValueError(f"No 'sql' column found. Columns present: {list(frame.columns)}")

    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "question": row.get("question"),
            "sql": row.get(sql_column),
            "database": row.get("database") or row.get("scope"),
            "notes": row.get("notes"),
        })
    return rows


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def _scope_matches(example_scope, db_identifier):
    if not example_scope or example_scope == GLOBAL_SCOPE:
        return True
    if not db_identifier:
        return False
    want = tokenise(example_scope, keep_stopwords=True)
    have = tokenise(db_identifier, keep_stopwords=True)
    if not want:
        return True
    return any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


# ---------------------------------------------------------------------------
# Semantic scoring (optional)
#
# Lexical overlap cannot tell that "biggest road by length" and "the longest
# road" are the same request — they share no distinctive word. Embeddings can.
# The model is the multilingual one the RAG engine already downloads, which also
# means a Hindi or Telugu question matches an English example.
#
# Entirely optional: if sentence-transformers is unavailable or the model fails
# to load, retrieval falls back to lexical scoring alone.
# ---------------------------------------------------------------------------
SEMANTIC_MODEL = os.environ.get(
    "QUERY_LIBRARY_EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
# 0 disables semantic scoring; 1 uses it alone. Hybrid beats either in practice.
SEMANTIC_WEIGHT = float(os.environ.get("QUERY_LIBRARY_SEMANTIC_WEIGHT", "0.6"))

_embedder = None
_embedder_failed = False
_embedding_cache = {}


def _get_embedder():
    global _embedder, _embedder_failed
    if _embedder is not None or _embedder_failed:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(SEMANTIC_MODEL)
        logger.info("Query library semantic scoring enabled (%s)", SEMANTIC_MODEL)
    except Exception as exc:
        _embedder_failed = True
        logger.info("Semantic scoring unavailable, using lexical only (%s)", exc)
    return _embedder


def _cosine_scores(question, example_questions):
    """Return {index: cosine} or None when embeddings are unavailable."""
    if SEMANTIC_WEIGHT <= 0 or not example_questions:
        return None

    embedder = _get_embedder()
    if embedder is None:
        return None

    try:
        import numpy as np

        missing = [q for q in example_questions if q not in _embedding_cache]
        if missing:
            vectors = embedder.encode(missing, normalize_embeddings=True)
            for text, vector in zip(missing, vectors):
                _embedding_cache[text] = np.asarray(vector, dtype="float32")

        matrix = np.stack([_embedding_cache[q] for q in example_questions])
        query_vector = np.asarray(
            embedder.encode([question], normalize_embeddings=True)[0], dtype="float32"
        )
        # Vectors are normalised, so the dot product is the cosine.
        return {i: float(s) for i, s in enumerate(matrix @ query_vector)}
    except Exception as exc:
        logger.debug("Semantic scoring failed, falling back to lexical: %s", exc)
        return None


def _build_idf(examples):
    """Rarer words identify a question better than common ones."""
    total = len(examples) or 1
    counts = {}
    for example in examples:
        for token in set(example["_tokens"]):
            counts[token] = counts.get(token, 0) + 1
    return {t: math.log(1 + total / c) for t, c in counts.items()}, total


def retrieve(question, db_identifier=None, top_k=DEFAULT_TOP_K,
             schema_text=None, path=LIBRARY_FILE):
    """
    Return the examples most similar to `question`, best first.

    When `schema_text` is given, an example is skipped if it references a table
    absent from the schema — showing the model a query against a table it has
    no schema for invites it to use that table anyway.
    """
    doc = load_library(path)
    candidates = []

    for example in doc["examples"]:
        if not example.get("enabled", True):
            continue
        if not _scope_matches(example.get("scope"), db_identifier):
            continue
        entry = dict(example)
        entry["_tokens"] = tokenise(entry.get("question", ""))
        if entry["_tokens"]:
            candidates.append(entry)

    if not candidates:
        return []

    if schema_text:
        schema_lower = schema_text.lower()
        kept = []
        for entry in candidates:
            tables = _tables_in_sql(entry["sql"])
            if tables and not all(
                re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", schema_lower)
                for t in tables
            ):
                continue
            kept.append(entry)
        # Only apply the filter if it leaves something to work with.
        candidates = kept or candidates

    idf, _ = _build_idf(candidates)
    q_tokens = tokenise(question)
    if not q_tokens:
        return []
    q_set = set(q_tokens)
    q_bigrams = _bigrams(q_tokens)
    q_weight = sum(idf.get(t, 1.0) for t in q_set) or 1.0

    semantic = _cosine_scores(question, [e.get("question", "") for e in candidates])

    scored = []
    for index, entry in enumerate(candidates):
        e_set = set(entry["_tokens"])
        shared = q_set & e_set

        lexical = 0.0
        if shared:
            overlap = sum(idf.get(t, 1.0) for t in shared)
            # Normalise by both sides so a long example cannot win on length alone.
            e_weight = sum(idf.get(t, 1.0) for t in e_set) or 1.0
            lexical = overlap / math.sqrt(q_weight * e_weight)
            # Consecutive word matches are strong evidence ("total network length").
            lexical += 0.12 * len(q_bigrams & _bigrams(entry["_tokens"]))

        if semantic is None:
            score = lexical
            parts = {"lexical": round(lexical, 4)}
        else:
            # Hybrid: semantic catches paraphrases, lexical anchors exact terms
            # like column names and road-class codes.
            sem = max(0.0, semantic.get(index, 0.0))
            score = SEMANTIC_WEIGHT * sem + (1.0 - SEMANTIC_WEIGHT) * lexical
            parts = {"lexical": round(lexical, 4), "semantic": round(sem, 4)}

        if score >= MIN_SCORE:
            scored.append((score, entry, parts))

    scored.sort(key=lambda triple: (-triple[0], triple[1]["id"]))

    out = []
    for score, entry, parts in scored[:top_k]:
        result = {k: v for k, v in entry.items() if not k.startswith("_")}
        result["score"] = round(score, 4)
        result["score_parts"] = parts
        out.append(result)
    return out


_TABLE_RE = re.compile(r"(?:from|join)\s+((?:\[[^\]]+\]|\w+)(?:\s*\.\s*(?:\[[^\]]+\]|\w+))*)", re.I)


def _tables_in_sql(sql):
    """Table names referenced after FROM/JOIN, brackets stripped."""
    names = set()
    for match in _TABLE_RE.finditer(sql or ""):
        parts = [p.strip().strip("[]") for p in match.group(1).split(".")]
        if parts and parts[-1]:
            names.add(parts[-1].lower())
    return names


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_examples(examples):
    """Render retrieved examples as a few-shot block for the prompt."""
    if not examples:
        return ""

    blocks, used = [], 0
    for example in examples:
        block = f"Question: {example['question']}\nT-SQL:\n{example['sql']}"
        if example.get("notes"):
            block += f"\n-- note: {example['notes']}"
        if used + len(block) > MAX_EXAMPLE_CHARS:
            break
        blocks.append(block)
        used += len(block)

    if not blocks:
        return ""

    rule = "=" * 60
    body = "\n\n".join(blocks)
    return (
        f"\n\n{rule}\nVERIFIED EXAMPLES (reviewed by the support team)\n{rule}\n"
        "These questions were answered correctly by the SQL shown. Follow the same "
        "table choices, joins, filters and conventions when the question is similar. "
        "Adapt them to the question asked — do not copy one verbatim if it does not fit.\n\n"
        f"{body}\n{rule}\n"
    )


def load_examples_for_prompt(question, db_identifier=None, schema_text=None,
                             top_k=DEFAULT_TOP_K, path=LIBRARY_FILE):
    """Retrieve and render in one call. Returns (text, examples_used)."""
    examples = retrieve(question, db_identifier, top_k, schema_text, path)
    return render_examples(examples), examples


def library_stats(path=LIBRARY_FILE):
    doc = load_library(path)
    examples = doc["examples"]
    scopes = {}
    for example in examples:
        scope = example.get("scope") or GLOBAL_SCOPE
        scopes[scope] = scopes.get(scope, 0) + 1
    return {
        "total": len(examples),
        "enabled": sum(1 for e in examples if e.get("enabled", True)),
        "by_scope": scopes,
    }
