"""
Read-only SQL guard.

The previous validator scanned the raw query for banned words. That fails in
both directions: `SELECT * INTO Backup FROM T` writes a table without using a
banned word, `DR/**/OP TABLE T` hides one inside a comment, and a perfectly
ordinary `WHERE Remarks LIKE '%update%'` was rejected because the word appeared
inside a string literal.

The fix is to normalise before matching. A single pass rewrites comments,
string literals and quoted identifiers to inert placeholders, so keyword
matching sees only executable SQL. On the normalised text the guard then:

  1. rejects more than one statement — a read never needs a second one, and
     this kills the whole stacked-statement class at once;
  2. requires the statement to start with SELECT (or WITH ... SELECT, so CTEs
     keep working);
  3. rejects any write, DDL, permission, backup, execution or out-of-process
     construct.

This is a guard, not a substitute for a read-only database login. Connect the
service with an account that only has SELECT and the guard becomes a second
line of defence rather than the only one.
"""

import re

# Statements that write, change structure, change permissions, execute code, or
# reach outside the database. Matched with word boundaries on normalised SQL.
_BLOCKED_KEYWORDS = [
    # data modification
    "insert", "update", "delete", "merge", "upsert", "truncate",
    "writetext", "updatetext",
    # SELECT ... INTO materialises a new table; no read query needs INTO
    "into",
    # structure
    "drop", "alter", "create", "rename",
    # permissions
    "grant", "revoke", "deny",
    # execution
    "exec", "execute", "call",
    # server / maintenance
    "backup", "restore", "dbcc", "shutdown", "reconfigure", "kill",
    "checkpoint", "waitfor",
    # out-of-process / exfiltration
    "openrowset", "openquery", "opendatasource", "openxml", "bulk",
    # transaction control (a read needs none, and these enable staged writes)
    "commit", "rollback", "savepoint",
]

_BLOCKED_RE = re.compile(
    r"(?<!\w)(" + "|".join(_BLOCKED_KEYWORDS) + r")(?!\w)",
    re.IGNORECASE,
)

# Stored-procedure prefixes (xp_cmdshell, sp_executesql, ...).
_PROC_PREFIX_RE = re.compile(r"(?<!\w)(x|s)p_\w+", re.IGNORECASE)

_STARTS_SELECT_RE = re.compile(r"^\s*select(?!\w)", re.IGNORECASE)
_STARTS_WITH_CTE_RE = re.compile(r"^\s*with(?!\w)", re.IGNORECASE)
_HAS_SELECT_RE = re.compile(r"(?<!\w)select(?!\w)", re.IGNORECASE)


def normalise(sql):
    """
    Replace comments, string literals and quoted identifiers with placeholders.

    Returns text positionally similar to the input (so error messages stay
    meaningful) but containing only executable SQL, never user data.
    """
    out = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # -- line comment
        if ch == "-" and nxt == "-":
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
            continue

        # /* block comment */ (SQL Server allows nesting)
        if ch == "/" and nxt == "*":
            depth = 1
            i += 2
            while i < n and depth:
                if sql[i] == "/" and i + 1 < n and sql[i + 1] == "*":
                    depth += 1
                    i += 2
                elif sql[i] == "*" and i + 1 < n and sql[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            out.append(" ")
            continue

        # 'string literal' with '' escape
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("'?'")
            continue

        # [bracket identifier] with ]] escape
        if ch == "[":
            i += 1
            while i < n:
                if sql[i] == "]":
                    if i + 1 < n and sql[i + 1] == "]":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("qid")
            continue

        # "quoted identifier"
        if ch == '"':
            i += 1
            while i < n and sql[i] != '"':
                i += 1
            i += 1
            out.append("qid")
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def split_statements(normalised_sql):
    """Non-empty statements, using the normalised text so ';' inside a literal is ignored."""
    return [s.strip() for s in normalised_sql.split(";") if s.strip()]


def validate_read_only(sql, extra_blocked=()):
    """
    Return (is_safe, reason). Allows a single SELECT or WITH...SELECT statement.

    `extra_blocked` adds dialect-specific keywords (e.g. Postgres COPY).
    """
    if not sql or not sql.strip():
        return False, "Query is empty."

    normalised = normalise(sql)
    statements = split_statements(normalised)

    if not statements:
        return False, "Query contains no statement."
    if len(statements) > 1:
        return False, (
            f"Only one statement is allowed; found {len(statements)}. "
            "Multiple statements are not permitted on a read-only endpoint."
        )

    statement = statements[0]

    if _STARTS_WITH_CTE_RE.match(statement):
        if not _HAS_SELECT_RE.search(statement):
            return False, "A WITH clause must lead to a SELECT."
    elif not _STARTS_SELECT_RE.match(statement):
        return False, "Query must start with SELECT (or WITH ... SELECT)."

    match = _BLOCKED_RE.search(statement)
    if match:
        return False, f"Blocked keyword detected: '{match.group()}'"

    match = _PROC_PREFIX_RE.search(statement)
    if match:
        return False, f"Stored procedure calls are not permitted: '{match.group()}'"

    for keyword in extra_blocked:
        if re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", statement, re.IGNORECASE):
            return False, f"Blocked keyword detected: '{keyword}'"

    return True, ""
