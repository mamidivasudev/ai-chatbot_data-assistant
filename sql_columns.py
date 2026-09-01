"""
Column reference checker.

Asking the model to fix its own SQL after an error does not reliably work: at
temperature 0 it regenerates the identical query, error feedback and all. So
invalid columns are caught deterministically here, before the query reaches the
database, by checking every referenced column against the schema the model was
actually given.

The check is deliberately conservative — it only reports an identifier as
unknown when it is confident. A false positive would reject a valid query, so
aliases, SQL keywords, function names and string literals are all excluded.
"""

import re

# Words that can appear where a column would, but are not columns.
_SQL_WORDS = {
    "select", "from", "where", "group", "by", "order", "having", "join", "inner",
    "left", "right", "full", "outer", "cross", "apply", "on", "as", "and", "or",
    "not", "in", "like", "between", "is", "null", "case", "when", "then", "else",
    "end", "distinct", "top", "percent", "with", "union", "all", "except",
    "intersect", "asc", "desc", "over", "partition", "rows", "range", "offset",
    "fetch", "next", "first", "only", "exists", "any", "some", "cast", "convert",
    "try_cast", "try_convert", "coalesce", "isnull", "nullif", "iif",
    "count", "sum", "avg", "min", "max", "stdev", "var", "abs", "round", "ceiling",
    "floor", "len", "ltrim", "rtrim", "trim", "upper", "lower", "substring",
    "replace", "concat", "concat_ws", "format", "str", "left", "right", "charindex",
    "patindex", "stuff", "reverse", "getdate", "sysdatetime", "dateadd", "datediff",
    "datepart", "datename", "year", "month", "day", "eomonth", "row_number", "rank",
    "dense_rank", "ntile", "lag", "lead", "int", "bigint", "smallint", "tinyint",
    "bit", "decimal", "numeric", "float", "real", "money", "varchar", "nvarchar",
    "char", "nchar", "date", "datetime", "datetime2", "time", "value", "values",
}


def parse_schema(schema_text):
    """
    Extract table and column names from the schema block given to the model.

    Expects the layout produced by mssql_schema_reader.get_table_schema_text:

        TABLE: [dbo].[RoadMaster]
          Id INT  [PK, NOT NULL]
          RoadCode VARCHAR(20)
    """
    tables, columns = set(), set()
    for line in (schema_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("TABLE:"):
            for part in re.findall(r"\[([^\]]+)\]|(\w+)", stripped[6:]):
                name = part[0] or part[1]
                if name:
                    tables.add(name.lower())
            continue
        # Column lines are indented under their table.
        if line[:1] in (" ", "\t"):
            match = re.match(r"[\s]*\[?([A-Za-z_]\w*)\]?\s", line)
            if match:
                columns.add(match.group(1).lower())
    return tables, columns


def _strip_for_analysis(sql):
    """
    Remove comments and string literals, but keep bracketed identifiers.

    sql_guard.normalise() rewrites [dbo].[RoadMaster] to a placeholder, which is
    right for keyword safety but useless here — the column names are exactly
    what needs to stay visible. Brackets are unwrapped instead of blanked.
    """
    out, i, n = [], 0, len(sql)
    while i < n:
        ch, nxt = sql[i], (sql[i + 1] if i + 1 < n else "")
        if ch == "-" and nxt == "-":
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
        elif ch == "/" and nxt == "*":
            depth, i = 1, i + 2
            while i < n and depth:
                if sql[i] == "/" and i + 1 < n and sql[i + 1] == "*":
                    depth, i = depth + 1, i + 2
                elif sql[i] == "*" and i + 1 < n and sql[i + 1] == "/":
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append("''")
        elif ch == "[":
            j = sql.find("]", i)
            if j == -1:
                out.append(" ")
                i = n
            else:
                out.append(sql[i + 1:j])   # keep the identifier, drop the brackets
                i = j + 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _declared_aliases(sql):
    """Aliases the query itself defines, which are legitimate references."""
    aliases = set()
    # AS alias
    for m in re.finditer(r"\bas\s+([A-Za-z_]\w*)", sql, re.I):
        aliases.add(m.group(1).lower())
    # FROM/JOIN table alias (with or without AS)
    for m in re.finditer(
        r"\b(?:from|join)\s+\w+(?:\s*\.\s*\w+)*\s+(?:as\s+)?([A-Za-z_]\w*)",
        sql, re.I,
    ):
        aliases.add(m.group(1).lower())
    # Bare select-list alias: "COUNT(*) c," or "SUM(x) total FROM".
    # The preceding token must not be a keyword, or "SELECT District," would
    # register District as an alias and hide the very error we are looking for.
    for m in re.finditer(r"(\)|\w+)\s+([A-Za-z_]\w*)\s*(?=,|\bfrom\b)", sql, re.I):
        previous, alias = m.group(1), m.group(2)
        if previous != ")" and previous.lower() in _SQL_WORDS:
            continue
        aliases.add(alias.lower())
    return aliases - _SQL_WORDS


def find_unknown_columns(sql, schema_text):
    """
    Column references in `sql` that do not exist in `schema_text`.

    Only qualified references (alias.Column) and GROUP BY / ORDER BY terms are
    checked — the positions where a hallucinated column reliably shows up, and
    where the parse is unambiguous enough to avoid false positives.
    """
    if not sql or not schema_text:
        return []

    tables, columns = parse_schema(schema_text)
    if not columns:
        return []

    clean = _strip_for_analysis(sql)
    aliases = _declared_aliases(clean)
    known = columns | tables | aliases | _SQL_WORDS

    unknown = []

    # alias.Column  ->  the column half must exist
    for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\.\s*([A-Za-z_]\w*)\b", clean):
        column = m.group(2)
        if column == "*":
            continue
        if column.lower() not in known:
            unknown.append(column)

    # GROUP BY / ORDER BY term lists
    for m in re.finditer(r"\b(?:group|order)\s+by\s+([^)]+?)(?=\b(?:having|order|union|option|offset|fetch)\b|$)",
                         clean, re.I | re.S):
        for term in m.group(1).split(","):
            term = term.strip().rstrip(";")
            term = re.sub(r"\b(asc|desc)\b", "", term, flags=re.I).strip()
            if not term or "(" in term:      # skip expressions
                continue
            name = term.split(".")[-1].strip("[] ")
            if not re.fullmatch(r"[A-Za-z_]\w*", name or ""):
                continue
            if name.lower() not in known:
                unknown.append(name)

    # Preserve order, drop duplicates.
    seen, out = set(), []
    for name in unknown:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out
