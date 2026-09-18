"""
Structure-aware reading of Word files and spreadsheets for document summaries.

SatBot's ingestion parser (v2_rag_engine) appends Word tables after all the
paragraphs and drops empty cells, so a row arrives as "Rename panel | Done" with
nothing saying which column is the status. For trackers, change requests and
inspection lists that loses the most useful fact of all — how many items are done.

This module reads Word files in body order with every table cell labelled by its
column header ("Dev Status: Pending"), and counts the values of status-like
columns in Word tables and spreadsheets, so those totals are computed here rather
than estimated by the model.
"""

import re
from collections import Counter
from typing import Optional

# Header of a column whose values are statuses ("Status", "Dev Status", "Progress").
_STATUS_HEADER = re.compile(r"\b(status|state|progress|completion)\b", re.IGNORECASE)
# A status word inside a cell, so "Done (20-06)" counts as Done and "In-progress" as In progress.
_STATUS_WORD = re.compile(
    r"\b(done|completed?|closed|resolved|fixed|approved|implemented|pending|open|not[\s-]started|"
    r"in[\s-]progress|ongoing|wip|on[\s-]hold|blocked|rejected|dropped|deferred)\b",
    re.IGNORECASE,
)
MAX_STATUS_VALUES = 10  # more distinct values than this is free text, not a status
STATUS_VALUE_SHARE = 0.6  # a column counts as status when this share of its cells hold a status word

Table = list[list[str]]  # rows of cell texts; the header is found by _split_header


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _table_rows(table) -> Table:
    # Merged cells are returned once per grid column, which keeps every row aligned
    # with the header; the repeats are dropped when the rows are rendered.
    rows = [[_clean(cell.text) for cell in row.cells] for row in table.rows]
    return [row for row in rows if any(row)]


def _split_header(rows: Table) -> tuple[list[str], list[str], Table]:
    """(title lines, header, body). Word tables often open with a merged title row
    ("1. Cell Properties Panel", repeated across every cell); the header is the first
    of the top rows with at least two different cells."""
    for i, row in enumerate(rows[:3]):
        if len({cell for cell in row if cell}) >= 2:
            titles = [next(cell for cell in r if cell) for r in rows[:i]]
            return titles, row, rows[i + 1:]
    return [], rows[0], rows[1:]


def _status_label(value: str) -> str:
    """The status word in a cell, tidied ("in-progress" → "In progress"); else the cell as is."""
    match = _STATUS_WORD.search(value)
    if not match:
        return value
    word = re.sub(r"[\s-]+", " ", match.group(0)).lower()
    return word[0].upper() + word[1:]


def _render_table(rows: Table, number: int) -> str:
    titles, header, body = _split_header(rows)
    lines = [f"\n## Table {number}" + (f": {' / '.join(titles)}" if titles else "")]
    if not body:
        return "\n".join(lines + [" | ".join(cell for cell in header if cell)])
    for row in body:
        pairs, last = [], None
        for i, value in enumerate(row):
            if not value:
                continue
            label = header[i] if i < len(header) else ""
            pair = f"{label}: {value}" if label and label != value else value
            if pair != last:
                pairs.append(pair)
            last = pair
        if pairs:
            lines.append("- " + " | ".join(pairs))
    return "\n".join(lines)


def read_docx(path: str) -> tuple[str, list[Table]]:
    """Text in reading order, tables in place with labelled cells; plus the raw tables."""
    import docx
    from docx.table import Table as DocxTable
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    parts: list[str] = []
    tables: list[Table] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = _clean(paragraph.text)
            if text:
                style = getattr(paragraph.style, "name", "") or ""
                parts.append(f"\n## {text}" if style.startswith("Heading") else text)
        elif tag == "tbl":
            rows = _table_rows(DocxTable(child, document))
            if rows:
                tables.append(rows)
                parts.append(_render_table(rows, len(tables)))
    return "\n".join(parts), tables


def read_sheet_tables(path: str, filename: str) -> list[Table]:
    """Each sheet (or the CSV) as a table whose first row is the column names."""
    import pandas as pd

    if filename.lower().endswith(".csv"):
        sheets = {"Data": pd.read_csv(path, dtype=str)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
    tables = []
    for frame in sheets.values():
        frame = frame.fillna("")
        tables.append([[_clean(str(c)) for c in frame.columns]]
                      + [[_clean(str(v)) for v in row] for row in frame.values.tolist()])
    return tables


def _count(values: list[str]) -> Optional[list[dict]]:
    groups: dict[str, list[str]] = {}
    for value in values:
        groups.setdefault(value.lower() or "(blank)", []).append(value or "Not set")
    if not groups or len(groups) > MAX_STATUS_VALUES:
        return None
    # Every row different means free text (remarks, descriptions), not a status.
    if len(groups) == len(values) and len(values) > 3:
        return None
    counts = [{"label": Counter(labels).most_common(1)[0][0], "count": len(labels)} for labels in groups.values()]
    return sorted(counts, key=lambda c: -c["count"])


def status_counts(tables: list[Table]) -> list[dict]:
    """Counts per status-like column, summed across tables that share the column name.

    [{"column": "Dev Status", "total": 20, "counts": [{"label": "Done", "count": 14}, ...]}]
    """
    columns: dict[str, dict] = {}
    for rows in tables:
        _, header, body = _split_header(rows)
        if len(body) < 2:
            continue
        for index, name in enumerate(header):
            cells = [row[index] for row in body if index < len(row)]
            filled = [cell for cell in cells if cell]
            with_status = sum(1 for cell in filled if _STATUS_WORD.search(cell))
            # By its header ("Dev Status"), or by its contents when the header says
            # something else ("Remarks" holding Done / Pending / In progress).
            by_header = bool(name) and bool(_STATUS_HEADER.search(name))
            by_values = len(filled) >= 3 and with_status / len(filled) >= STATUS_VALUE_SHARE
            if not (by_header or by_values) or header.index(name) != index:
                continue  # the second test skips merged-cell repeats of the same column
            values = [_status_label(cell) for cell in cells]
            if _count(values) is None:
                continue
            entry = columns.setdefault(name.lower() or "status", {"column": name or "Status", "values": []})
            entry["values"] += values
    result = []
    for entry in columns.values():
        counts = _count(entry["values"])
        if counts:
            result.append({"column": entry["column"], "total": len(entry["values"]), "counts": counts})
    return result


def render_status_facts(status: list[dict]) -> str:
    """The counts as prompt text, e.g. "- Dev Status: 20 items — Done 14, Pending 6"."""
    lines = [
        f"- {s['column']}: {s['total']} items — " + ", ".join(f"{c['label']} {c['count']}" for c in s["counts"])
        for s in status
    ]
    return "Counted from the document's tables (exact):\n" + "\n".join(lines)
