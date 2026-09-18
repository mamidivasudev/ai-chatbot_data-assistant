"""
Tests for GET /diagnostics/ollama against a local stub that answers like Ollama.

Run from the repo root:  python -m pytest tests
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import diagnostics_api

DEAD = "http://127.0.0.1:1"  # nothing listens on port 1


class StubOllama(BaseHTTPRequestHandler):
    models = ["llama3:latest", "qwen2.5-coder:7b"]

    def log_message(self, *args):
        pass

    def _send(self, body: dict):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            self._send({"version": "0.5.7"})
        else:
            self._send({"models": [{"model": m} for m in self.models]})

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"response": "OK"})


@pytest.fixture(scope="module")
def stub():
    server = HTTPServer(("127.0.0.1", 0), StubOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def check(monkeypatch, configured, others=(), **params):
    monkeypatch.setenv("OLLAMA_HOST", configured)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    # Only the hosts under test, so a real Ollama on this machine can't change the outcome.
    monkeypatch.setattr(diagnostics_api, "_candidates", lambda configured, in_docker: [configured, *others])
    app = FastAPI()
    app.include_router(diagnostics_api.router)
    response = TestClient(app).get("/diagnostics/ollama", params=params)
    assert response.status_code == 200
    return response.json()


def test_reachable_with_model_and_test_prompt(monkeypatch, stub):
    body = check(monkeypatch, stub, test_model=True)

    assert body["verdict"] == f"OK: Ollama answers at {stub} and llama3:latest is installed."
    assert body["checks"][0]["version"] == "0.5.7"
    assert body["generation"]["ok"] is True and body["generation"]["reply"] == "OK"
    assert body["api_machine"]["hostname"]


def test_points_to_the_address_that_answers(monkeypatch, stub):
    body = check(monkeypatch, DEAD, others=[stub])

    assert body["checks"][0]["reachable"] is False
    assert f"Ollama answers at {stub}, but this API is configured for {DEAD}" in body["verdict"]


def test_nothing_reachable(monkeypatch):
    body = check(monkeypatch, DEAD)

    assert body["model_available"] is False
    assert "Ollama is not reachable from this API" in body["verdict"]


def test_model_not_installed(monkeypatch, stub):
    body = check(monkeypatch, stub, model="aya:8b")

    assert body["verdict"].endswith("Run on this machine: ollama pull aya:8b")


def test_candidates_are_deduplicated_and_docker_aware(monkeypatch):
    monkeypatch.setenv("OLLAMA_CHECK_HOSTS", "10.0.0.7:11434, http://127.0.0.1:11434/")
    hosts = diagnostics_api._candidates("http://127.0.0.1:11434", in_docker=True)

    assert hosts == ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434",
                     "http://host.docker.internal:11434", "http://10.0.0.7:11434"]


def test_proxy_credentials_are_masked():
    assert diagnostics_api._mask("http://user:secret@proxy:8080") == "http://***@proxy:8080"
