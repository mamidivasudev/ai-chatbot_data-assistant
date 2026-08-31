"""
Schema profiler — derives query rules from the tables the admin actually ticked.

The static registry in skills.py encodes domain knowledge a human wrote down.
This module derives the rest from the live schema, so a newly ticked table is
usable without anyone authoring a rule for it:

  * ALLOWED TABLES  — the exact list the model may query, so it cannot retarget
                      to a similarly named table it was never granted.
  * SOFT DELETE     — an IsDelete/IsActive style column means COUNT(*) is wrong
                      by default; the filter is stated explicitly.
  * JOIN PATHS      — taken from real foreign keys, not guessed from names.
  * COLUMN VALUES   — low-cardinality columns are sampled, so the model learns
                      RoadClass IN ('SH','NH','MDR',...) instead of needing a
                      hand-written "State Highway = SH" rule.

Everything here is best-effort: profiling is wrapped so a slow or permission-
denied column degrades the prompt rather than failing the request.
"""

import logging
import re

logger = logging.getLogger("schema_profiler")

# Types that must never be sampled: blobs, geometry and large text would dump
# megabytes into the prompt (the Shape column on a road table is raw WKB).
_UNSAMPLEABLE_TYPES = {
    "image", "varbinary", "binary", "geometry", "geography", "hierarchyid",
    "text", "ntext", "xml", "timestamp", "rowversion", "sql_variant", "uniqueidentifier",
}

_SAMPLEABLE_TYPES = {"varchar", "nvarchar", "char", "nchar"}

# A column is a useful enum if it has at most this many distinct values.
MAX_DISTINCT_VALUES = 25
# Don't sample wide free-text columns — they are descriptions, not categories.
MAX_SAMPLE_COLUMN_WIDTH = 100
# Bound the work per request regardless of how many tables were ticked.
MAX_COLUMNS_TO_SAMPLE = 40

_SOFT_DELETE_PATTERNS = [
    (re.compile(r"^is_?deleted?$", re.I), 0, "not deleted"),
    (re.compile(r"^deleted$", re.I), 0, "not deleted"),
    (re.compile(r"^is_?active$", re.I), 1, "active"),
    (re.compile(r"^active$", re.I), 1, "active"),
]


def _detect_soft_delete(column_name):
    """Return (keep_value, meaning) when a column looks like a soft-delete flag."""
    for pattern, keep_value, meaning in _SOFT_DELETE_PATTERNS:
        if pattern.match(column_name):
            return keep_value, meaning
    return None, None


def _quote(identifier):
    """Bracket-quote an identifier, escaping ']' so it cannot break out."""
    return "[" + str(identifier).replace("]", "]]") + "]"


def _flag_distribution(conn, schema, table, column_name):
    """Return {value: count} for a candidate flag column."""
    col = _quote(column_name)
    sql = (
        f"SELECT {col} AS v, COUNT(*) AS c "
        f"FROM {_quote(schema)}.{_quote(table)} GROUP BY {col}"
    )
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    finally:
        cursor.close()
    return {(None if r[0] is None else int(r[0])): int(r[1]) for r in rows}


def _validate_soft_delete(distribution, keep_value):
    """
    Decide whether a name-matched flag column actually filters anything.

    A column named IsDelete does not necessarily mean what it says. In one live
    database every row has IsDelete = 1, so emitting "WHERE IsDelete = 0" would
    silently return zero rows for every question — worse than no rule at all.
    A flag is only trustworthy when both values are actually present.
    """
    if not distribution:
        return False, "no rows"

    present = {v for v in distribution if v is not None}
    if len(present) < 2:
        only = next(iter(present), None)
        if only == keep_value:
            return False, f"every row already has {keep_value}; filter is redundant"
        return False, (
            f"every row has {only}, not {keep_value}; the column does not mark "
            "deleted rows in this database"
        )

    if distribution.get(keep_value, 0) == 0:
        return False, f"no rows have {keep_value}; filter would return nothing"

    return True, "both values present"


def _is_sampleable(column):
    col_type = (column.get("type") or "").lower()
    if col_type in _UNSAMPLEABLE_TYPES or col_type not in _SAMPLEABLE_TYPES:
        return False
    max_length = column.get("max_length")
    # -1 is MAX (varchar(max)) — always free text.
    if max_length is None or max_length == -1 or max_length > MAX_SAMPLE_COLUMN_WIDTH:
        return False
    return True


def _sample_column(conn, schema, table, column_name):
    """
    Return the distinct values of a column when there are few enough to matter.

    Fetches one more than the cap so a column that blows the cap is discarded
    without a second round trip.
    """
    col = _quote(column_name)
    sql = (
        f"SELECT DISTINCT TOP {MAX_DISTINCT_VALUES + 1} {col} "
        f"FROM {_quote(schema)}.{_quote(table)} WHERE {col} IS NOT NULL"
    )
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    if len(rows) > MAX_DISTINCT_VALUES:
        return None

    values = []
    for row in rows:
        value = str(row[0]).strip()
        if value:
            values.append(value)
    return sorted(set(values)) or None


def profile_tables(conn, selected_tables, metadata_reader, sample_values=True):
    """
    Build a profile of the ticked tables.

    `metadata_reader(conn, schema, table) -> {"columns": [...], "pks": [...], "fks": [...]}`
    is injected so this module stays independent of the MSSQL reader.
    """
    profile = {
        "tables": [], "soft_deletes": [], "joins": [], "enums": [],
        # Columns that look like delete/active flags but do not discriminate.
        # Surfaced so an admin can see the assumption was checked and rejected.
        "inert_flags": [],
    }
    sampled = 0

    for schema, table in selected_tables:
        try:
            meta = metadata_reader(conn, schema, table)
        except Exception as exc:
            logger.warning("Could not profile [%s].[%s]: %s", schema, table, exc)
            continue

        columns = meta.get("columns") or []
        profile["tables"].append({
            "schema": schema,
            "table": table,
            "label": f"[{schema}].[{table}]",
            "columns": [c.get("name") for c in columns],
        })

        for fk in meta.get("fks") or []:
            profile["joins"].append({
                "from": f"[{schema}].[{table}].[{fk['column']}]",
                "to": f"[{fk['ref_schema']}].[{fk['ref_table']}].[{fk['ref_column']}]",
            })

        for column in columns:
            name = column.get("name")
            if not name:
                continue

            keep_value, meaning = _detect_soft_delete(name)
            if keep_value is not None:
                # Never assert a filter on the strength of the column name
                # alone — verify it against the data first.
                try:
                    distribution = _flag_distribution(conn, schema, table, name)
                except Exception as exc:
                    logger.debug("Flag check [%s].[%s].[%s] failed: %s", schema, table, name, exc)
                    profile["inert_flags"].append({
                        "table": f"[{schema}].[{table}]", "column": name,
                        "reason": "could not be verified against the data",
                    })
                    continue

                trustworthy, reason = _validate_soft_delete(distribution, keep_value)
                entry = {
                    "table": f"[{schema}].[{table}]",
                    "column": name,
                    "keep_value": keep_value,
                    "meaning": meaning,
                    "distribution": {str(k): v for k, v in distribution.items()},
                    "reason": reason,
                }
                if trustworthy:
                    profile["soft_deletes"].append(entry)
                else:
                    logger.info(
                        "Ignoring soft-delete column [%s].[%s].[%s]: %s",
                        schema, table, name, reason,
                    )
                    profile["inert_flags"].append(entry)
                continue

            if not sample_values or sampled >= MAX_COLUMNS_TO_SAMPLE:
                continue
            if not _is_sampleable(column):
                continue

            sampled += 1
            try:
                values = _sample_column(conn, schema, table, name)
            except Exception as exc:
                logger.debug("Sampling [%s].[%s].[%s] failed: %s", schema, table, name, exc)
                continue

            if values:
                profile["enums"].append({
                    "table": f"[{schema}].[{table}]",
                    "column": name,
                    "values": values,
                })

    return profile


def build_dynamic_rules(profile):
    """Render a profile into a rules block for the SQL prompt."""
    if not profile or not profile.get("tables"):
        return ""

    sections = []

    labels = [t["label"] for t in profile["tables"]]
    sections.append(
        "### Allowed Tables\n"
        "- You may ONLY query these tables. They are the tables the administrator selected:\n"
        + "\n".join(f"  - {label}" for label in labels)
        + "\n- If the question needs a table that is not in this list, return exactly: "
          "SELECT 'SCHEMA_INSUFFICIENT' AS Error"
    )

    if profile.get("soft_deletes"):
        lines = [
            f"  - {sd['table']}: add WHERE [{sd['column']}] = {sd['keep_value']} to keep only {sd['meaning']} rows"
            for sd in profile["soft_deletes"]
        ]
        sections.append(
            "### Soft-Delete Filters (MANDATORY)\n"
            "- These tables keep deleted or inactive rows in place. A plain COUNT(*) or "
            "SELECT over them is wrong because it includes those rows.\n"
            + "\n".join(lines)
            + "\n- Apply these filters to every query unless the user explicitly asks for "
              "deleted or inactive records."
        )

    if profile.get("joins"):
        lines = [f"  - {j['from']} = {j['to']}" for j in profile["joins"]]
        sections.append(
            "### Verified Join Paths\n"
            "- These are the real foreign keys. Join only on these unless the question forces otherwise:\n"
            + "\n".join(lines)
        )

    if profile.get("enums"):
        lines = []
        for enum in profile["enums"]:
            values = ", ".join(f"'{v}'" for v in enum["values"])
            lines.append(f"  - {enum['table']}.[{enum['column']}] IN ({values})")
        sections.append(
            "### Known Column Values\n"
            "- These columns only ever contain the values below. Filter using these exact "
            "literals — never invent a value or spell one out in full when a code is used:\n"
            + "\n".join(lines)
        )

    rule = "=" * 60
    body = "\n\n".join(sections)
    return f"\n\n{rule}\nSCHEMA-DERIVED RULES (from the selected tables)\n{rule}\n{body}\n{rule}\n"


def profile_and_render(conn, selected_tables, metadata_reader, sample_values=True):
    """Profile the selected tables and render the rules. Never raises."""
    try:
        profile = profile_tables(conn, selected_tables, metadata_reader, sample_values)
        return build_dynamic_rules(profile), profile
    except Exception as exc:
        logger.warning("Schema profiling failed, continuing without dynamic rules: %s", exc)
        return "", {}
