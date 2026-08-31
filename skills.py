"""
Skill registry — selects which business rules get injected into the SQL prompt.

A "skill" is a named block of instructions that is injected only when it is
relevant to the question AND applicable to the connected database. Three gates
decide that, in order:

  1. SCOPE   — the skill's scope must match the connected database.
               "*" is the global scope and always applies.
  2. TRIGGER — the question must match the skill's keywords (word-boundary,
               not substring — so "sh" no longer fires on "show me").
  3. SCHEMA  — every table in `requires_tables` must appear in the schema text
               actually handed to the model. A skill that tells the model to
               join tables it has no schema for produces invalid SQL.

Skills are then ordered by priority (low number = injected first) and capped by
a character budget so a large registry cannot crowd the schema out of context.

File format (skills.json), version 2:

    {
      "version": 2,
      "scopes": {
        "*": [ <skill>, ... ],
        "Hims_Zrams": [ <skill>, ... ]
      }
    }

    <skill> = {
      "id":             "unique_slug",         # required, unique within scope
      "name":           "Human readable",
      "enabled":        true,
      "priority":       100,                   # lower = earlier
      "match":          "always" | "any" | "all",
      "keywords":       ["road", "total length"],
      "requires_tables": ["LinkMaster"],
      "instruction":    "- rule one\n- rule two"
    }

Version 1 files ({"global": [...]}, keyword "*" meaning always-on) are migrated
on read, so existing deployments keep working.
"""

import json
import os
import re
from functools import lru_cache

SKILLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills.json")

GLOBAL_SCOPE = "*"

# Cap on injected rule text. Skills are added in priority order until the budget
# is spent, so the highest-priority rules survive when a registry grows large.
MAX_SKILL_CHARS = 8000

_MATCH_ALWAYS = "always"
_MATCH_ANY = "any"
_MATCH_ALL = "all"


# ---------------------------------------------------------------------------
# Loading / migration
# ---------------------------------------------------------------------------
def _normalise_skill(raw, index):
    """Coerce one raw dict into the canonical skill shape."""
    keywords = raw.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [k.strip() for k in keywords.split(",")]
    keywords = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]

    # v1 used the pseudo-keyword "*" to mean "always on".
    match = raw.get("match")
    if not match:
        match = _MATCH_ALWAYS if "*" in keywords else _MATCH_ANY
    keywords = [k for k in keywords if k != "*"]

    requires = raw.get("requires_tables") or []
    if isinstance(requires, str):
        requires = [requires]
    requires = [t.strip() for t in requires if isinstance(t, str) and t.strip()]

    return {
        "id": str(raw.get("id") or f"skill_{index}"),
        "name": str(raw.get("name") or raw.get("id") or f"Skill {index}"),
        "enabled": bool(raw.get("enabled", True)),
        "priority": int(raw.get("priority", 100)),
        "match": match if match in (_MATCH_ALWAYS, _MATCH_ANY, _MATCH_ALL) else _MATCH_ANY,
        "keywords": keywords,
        "requires_tables": requires,
        "instruction": str(raw.get("instruction") or raw.get("rule_text") or "").strip(),
    }


def _load_registry(path, mtime):
    """Read skills.json and return {scope: [skill, ...]}. `mtime` busts the cache."""
    del mtime  # only present so lru_cache reloads when the file changes
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    # v2: {"version": 2, "scopes": {...}}   v1: {"global": [...]}
    raw_scopes = data.get("scopes")
    if not isinstance(raw_scopes, dict):
        raw_scopes = {GLOBAL_SCOPE: data.get("global", [])}

    registry = {}
    for scope, items in raw_scopes.items():
        if not isinstance(items, list):
            continue
        skills, seen = [], set()
        for i, raw in enumerate(items):
            if not isinstance(raw, dict):
                continue
            skill = _normalise_skill(raw, i)
            if not skill["instruction"] or skill["id"] in seen:
                continue
            seen.add(skill["id"])
            skills.append(skill)
        registry[str(scope)] = skills
    return registry


@lru_cache(maxsize=8)
def _cached_registry(path, mtime):
    return _load_registry(path, mtime)


def load_registry(path=SKILLS_FILE):
    """Load the registry, re-reading only when skills.json changes on disk."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    return _cached_registry(path, mtime)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _word_match(needle, haystack_lower):
    """Word-boundary match, so 'sh' does not fire on 'show' or 'washed'."""
    return re.search(r"(?<!\w)" + re.escape(needle.lower()) + r"(?!\w)", haystack_lower) is not None


def _tokens(text):
    """Split an identifier into comparable tokens: 'Dev_MS SQL_Hims_Zrams' -> [dev, ms, sql, hims, zrams]."""
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if t]


def _singular(token):
    """
    Crude, dependency-free singulariser so a keyword matches both numbers.

    Word-boundary matching alone would make the keyword "road" miss "roads" and
    "bridges" miss "bridge", which is how the old substring matching happened to
    work. Short tokens are left alone so codes like 'sh', 'nh' and 'is' survive.
    """
    if len(token) <= 3 or token.endswith("ss"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def _norm_tokens(text):
    """Tokenise and singularise, for number-insensitive keyword matching."""
    return [_singular(t) for t in _tokens(text)]


def _keyword_match(keyword, question_tokens):
    """
    True when the keyword's tokens appear as a consecutive run in the question.

    Token-based, so 'sh' matches "SH roads" but not "show me"; and
    number-insensitive, so 'road' matches "roads" and 'bridges' matches "bridge".
    """
    want = _norm_tokens(keyword)
    if not want:
        return False
    span = len(want)
    return any(
        question_tokens[i:i + span] == want
        for i in range(len(question_tokens) - span + 1)
    )


def resolve_scopes(db_identifier):
    """
    Scopes that apply to this connection, most general first.

    The global scope always applies. A named scope applies when its tokens
    appear as a consecutive run inside the identifier's tokens, so a scope keyed
    "Hims_Zrams" matches "Development_MS SQL_Hims_Zrams" but not the lookalike
    "Development_MS SQL_HIMS_RRAMS".
    """
    scopes = [GLOBAL_SCOPE]
    if not db_identifier:
        return scopes

    ident = _tokens(db_identifier)
    for scope in load_registry():
        if scope == GLOBAL_SCOPE:
            continue
        want = _tokens(scope)
        if not want:
            continue
        if any(ident[i:i + len(want)] == want for i in range(len(ident) - len(want) + 1)):
            scopes.append(scope)
    return scopes


def _triggers(skill, question_tokens):
    if skill["match"] == _MATCH_ALWAYS:
        return True
    keywords = skill["keywords"]
    if not keywords:
        # No keywords and not explicitly always-on: treat as always-on rather
        # than silently never firing.
        return True
    if skill["match"] == _MATCH_ALL:
        return all(_keyword_match(k, question_tokens) for k in keywords)
    return any(_keyword_match(k, question_tokens) for k in keywords)


def _schema_supports(skill, schema_lower):
    """A skill that names tables must not fire unless those tables are in scope."""
    if not skill["requires_tables"]:
        return True
    if schema_lower is None:
        # No schema supplied — cannot verify, so let the skill through rather
        # than silently dropping domain rules for callers that pass no schema.
        return True
    return all(_word_match(t, schema_lower) for t in skill["requires_tables"])


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_skills(question, schema_text=None, db_identifier=None):
    """Return the ordered list of skills that should be injected."""
    question_tokens = _norm_tokens(question or "")
    schema_lower = schema_text.lower() if isinstance(schema_text, str) else None

    registry = load_registry()
    selected, seen = [], set()

    for scope in resolve_scopes(db_identifier):
        for skill in registry.get(scope, []):
            if not skill["enabled"] or skill["id"] in seen:
                continue
            if not _triggers(skill, question_tokens):
                continue
            if not _schema_supports(skill, schema_lower):
                continue
            seen.add(skill["id"])
            selected.append(dict(skill, scope=scope))

    selected.sort(key=lambda s: (s["priority"], s["id"]))
    return selected


def render_skills(skills):
    """Render selected skills into the prompt block."""
    if not skills:
        return ""

    blocks, names, used = [], [], 0
    for skill in skills:
        block = f"### {skill['name']}\n{skill['instruction']}"
        if used + len(block) > MAX_SKILL_CHARS:
            break
        blocks.append(block)
        names.append(skill["name"])
        used += len(block)

    if not blocks:
        return ""

    rule = "=" * 60
    body = "\n\n".join(blocks)
    return f"\n\n{rule}\nACTIVE SKILLS: {', '.join(names)}\n{rule}\n{body}\n{rule}\n"


def load_skills(question, schema_text=None, db_identifier=None):
    """
    Public entry point: return the rule text to splice into the SQL prompt.

    Returns "" when nothing applies, so callers can inject it unconditionally.
    """
    return render_skills(select_skills(question, schema_text, db_identifier))


# ---------------------------------------------------------------------------
# Editing (used by the admin API and the Streamlit skills editor)
#
# skills.json is the single source of business rules. The older
# business_rules.json is no longer read by anything.
# ---------------------------------------------------------------------------
def read_document(path=SKILLS_FILE):
    """Read skills.json, normalising a legacy v1 document to the v2 shape."""
    if not os.path.exists(path):
        return {"version": 2, "scopes": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception:
        return {"version": 2, "scopes": {}}

    if not isinstance(doc, dict):
        return {"version": 2, "scopes": {}}
    if not isinstance(doc.get("scopes"), dict):
        doc = {"version": 2, "scopes": {GLOBAL_SCOPE: doc.get("global", []) or []}}
    doc.setdefault("version", 2)
    doc.setdefault("scopes", {})
    return doc


def write_document(doc, path=SKILLS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=4, ensure_ascii=False)
    _cached_registry.cache_clear()


def list_skills(scope, path=SKILLS_FILE):
    """Raw (unnormalised) skills stored under one scope, for editing."""
    return read_document(path).get("scopes", {}).get(scope, [])


def save_scope_skills(scope, skills, path=SKILLS_FILE):
    """Replace every skill in one scope. Other scopes are left untouched."""
    doc = read_document(path)
    doc.setdefault("scopes", {})[scope] = list(skills)
    write_document(doc, path)
    return doc["scopes"][scope]


def list_scopes(path=SKILLS_FILE):
    return sorted(read_document(path).get("scopes", {}).keys())


def describe_active_skills(question, schema_text=None, db_identifier=None):
    """Which skills fired, and why — for logging and for the admin UI."""
    return [
        {"id": s["id"], "name": s["name"], "scope": s["scope"], "priority": s["priority"]}
        for s in select_skills(question, schema_text, db_identifier)
    ]
