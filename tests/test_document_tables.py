"""
Tests for structure-aware document reading and status counts (document_tables.py),
and for how the document summary uses them.

Run from the repo root:  python -m pytest tests
"""

import asyncio
import io
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

docx = pytest.importorskip("docx")
pd = pytest.importorskip("pandas")

import document_tables
import summarize_api
import summarizer

HEADER = ["S.No", "Change", "Dev Status", "Target Date"]


def change_request_docx(path):
    """Shaped like a UI change request tracker: two sections, a status table in each."""
    document = docx.Document()
    document.add_heading("Cell Properties panel", level=1)
    document.add_paragraph("Changes to the panel on the right of the builder.")
    rows = [["1.1", "Rename 'Cell Properties' to 'Visual Properties'", "Done", "20-06-2026"],
            ["1.2", "Remove the Identity heading", "done", "20-06-2026"],
            ["1.3", "Show Row, Column and Size inputs", "Pending", "23-06-2026"]]
    add_table(document, rows)
    document.add_paragraph("Section 2 covers the dashboard properties panel.")
    add_table(document, [["2.1", "Rename 'Grid & Settings' to 'Dashboard Properties'", "Pending", "23-06-2026"],
                         ["2.2", "Show only 'Drag' instead of 'Click or Drag'", "In progress", "23-06-2026"]])
    document.save(path)


def add_table(document, rows):
    table = document.add_table(rows=1 + len(rows), cols=len(HEADER))
    for row, values in zip(table.rows, [HEADER, *rows]):
        for cell, value in zip(row.cells, values):
            cell.text = value


def test_docx_keeps_tables_in_place_with_labelled_cells(tmp_path):
    path = tmp_path / "cr.docx"
    change_request_docx(path)
    text, tables = document_tables.read_docx(str(path))

    assert "Dev Status: Pending" in text and "Target Date: 23-06-2026" in text
    # The paragraph between the tables stays between them.
    assert text.index("Remove the Identity heading") < text.index("Section 2 covers") < text.index("Grid & Settings")
    assert "## Cell Properties panel" in text
    assert len(tables) == 2 and tables[0][0] == HEADER


def test_status_counts_merge_tables_and_casing(tmp_path):
    path = tmp_path / "cr.docx"
    change_request_docx(path)
    _, tables = document_tables.read_docx(str(path))

    assert document_tables.status_counts(tables) == [{
        "column": "Dev Status", "total": 5,
        "counts": [{"label": "Done", "count": 2}, {"label": "Pending", "count": 2}, {"label": "In progress", "count": 1}],
    }]


def test_title_row_and_status_found_by_values(tmp_path):
    """A merged title row above the header, and the status in a "Remarks" column."""
    document = docx.Document()
    table = document.add_table(rows=6, cols=3)
    title = table.rows[0].cells[0].merge(table.rows[0].cells[2])
    title.text = "1. Cell Properties Panel"
    rows = [["S.No", "Change", "Remarks"], ["1.1", "Rename panel", "Done (20-06)"],
            ["1.2", "Remove Identity heading", "done"], ["1.3", "Show Row and Column inputs", "In-progress"],
            ["1.4", "Remove Duplicate button", "Pending - UI"]]
    for row, values in zip(table.rows[1:], rows):
        for cell, value in zip(row.cells, values):
            cell.text = value
    path = tmp_path / "cr.docx"
    document.save(path)

    text, tables = document_tables.read_docx(str(path))
    assert "## Table 1: 1. Cell Properties Panel" in text
    assert "Change: Rename panel | Remarks: Done (20-06)" in text
    assert document_tables.status_counts(tables) == [{
        "column": "Remarks", "total": 4,
        "counts": [{"label": "Done", "count": 2}, {"label": "In progress", "count": 1}, {"label": "Pending", "count": 1}],
    }]


def test_free_text_status_column_is_ignored():
    table = [["Item", "Status notes"], ["a", "waiting on vendor"], ["b", "tested in QA"],
             ["c", "moved to next sprint"], ["d", "needs design sign-off"]]
    assert document_tables.status_counts([table]) == []


def test_spreadsheet_status_counts(tmp_path):
    path = tmp_path / "roads.xlsx"
    pd.DataFrame({"Road": ["NH-8", "SH-12", "MDR-4", "ODR-9", "VR-2"],
                  "Progress": ["Completed", "Completed", "Ongoing", "Completed", "Ongoing"]}).to_excel(path, index=False)

    status = document_tables.status_counts(document_tables.read_sheet_tables(str(path), "roads.xlsx"))
    assert status == [{"column": "Progress", "total": 5,
                       "counts": [{"label": "Completed", "count": 3}, {"label": "Ongoing", "count": 2}]}]


def test_render_status_facts():
    status = [{"column": "Dev Status", "total": 20, "counts": [{"label": "Done", "count": 14}, {"label": "Pending", "count": 6}]}]
    assert document_tables.render_status_facts(status) == (
        "Counted from the document's tables (exact):\n- Dev Status: 20 items — Done 14, Pending 6")


def test_labels_and_number_fragments_are_dropped():
    items = ["Deadlines: Finish the Visual Properties panel by 23-06-2026.",
             "Required actions or pending decisions: 2.{6,7,8}, 3 {1,2,3}, 1.{2},{8},{13}",
             "Remove Duplicate button"]
    assert summarizer._as_sentences(items) == ["Finish the Visual Properties panel by 23-06-2026.",
                                               "Remove Duplicate button"]


class Recorder:
    """Returns `reply` for every call and keeps the prompts."""
    name, model, max_input_chars = "recorder", "recorder-model", 20000

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    async def complete(self, system, user, **kwargs):
        self.prompts.append(user)
        return self.reply


def test_status_counts_reach_the_model_and_the_result():
    status = [{"column": "Dev Status", "total": 20, "counts": [{"label": "Done", "count": 14}, {"label": "Pending", "count": 6}]}]
    reply = json.dumps({"title": "UI change request", "overview": "14 of the 20 changes are done and 6 are pending.",
                        "key_points": ["The Visual Properties panel was renamed and simplified."],
                        "action_items": ["Deadlines: Finish the 6 pending changes by 23-06-2026."]})
    provider = Recorder(reply)

    async def run():
        text = "Change request for the builder. Target date 23-06-2026."
        return [e async for e in summarizer.summarize_document_events(text, provider, status=status)]

    result = asyncio.run(run())[-1][1]
    assert "Counted from the document's tables (exact):" in provider.prompts[0]
    assert result["status"] == status
    assert result["unverified_numbers"] == []  # 14, 20 and 6 come from the counts
    assert result["action_items"] == ["Finish the 6 pending changes by 23-06-2026."]


def test_document_endpoint_returns_status_for_docx(monkeypatch, tmp_path):
    path = tmp_path / "cr.docx"
    change_request_docx(path)
    reply = json.dumps({"title": "UI change request", "overview": "2 changes are done.", "key_points": [], "action_items": []})
    summarize_api._cache.clear()
    monkeypatch.setattr(summarize_api, "get_provider", lambda model=None: Recorder(reply))
    app = FastAPI()
    app.include_router(summarize_api.router)

    response = TestClient(app).post("/summarize/document", files={
        "file": ("CR.docx", path.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})

    assert response.status_code == 200
    assert response.json()["status"][0]["total"] == 5
