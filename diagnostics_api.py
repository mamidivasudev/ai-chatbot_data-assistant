"""
GET /diagnostics/ollama — can this API process reach Ollama, and where?

Ollama's port is never exposed outside the server, so the API is the only place
it can be checked from. This reports which machine the API is running on, tries
every likely Ollama address from there, lists installed models, can run a tiny
test prompt, and ends with a one-line verdict saying what to fix.

The addresses tried come from server configuration only (OLLAMA_HOST, the usual
loopback names, OLLAMA_CHECK_HOSTS), never from the request, so the endpoint
cannot be used to make the server call arbitrary URLs.
"""

import os
import platform
import re
import socket
import time
from typing import Optional

import httpx
from fastapi import APIRouter, Query

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])

_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
               "http_proxy", "https_proxy", "all_proxy", "no_proxy")


def _api_machine() -> dict:
    hostname = socket.gethostname()
    try:
        addresses = sorted(set(socket.gethostbyname_ex(hostname)[2]))
    except OSError:
        addresses = []
    release = platform.release().lower()
    return {
        "hostname": hostname,
        "addresses": addresses,
        "os": f"{platform.system()} {platform.release()}",
        # Inside a container or WSL, 127.0.0.1 is not the GPU machine itself.
        "in_docker": os.path.exists("/.dockerenv"),
        "in_wsl": "microsoft" in release or "wsl" in release,
    }


def _candidates(configured: str, in_docker: bool) -> list[str]:
    hosts = [configured, "http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"]
    if in_docker:
        hosts.append("http://host.docker.internal:11434")
    hosts += [h.strip() for h in os.environ.get("OLLAMA_CHECK_HOSTS", "").split(",") if h.strip()]
    seen, unique = set(), []
    for host in hosts:
        host = host.rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        if host not in seen:
            seen.add(host)
            unique.append(host)
    return unique


def _probe(client: httpx.Client, host: str) -> dict:
    started = time.perf_counter()
    try:
        version = client.get(f"{host}/api/version").json().get("version")
        tags = client.get(f"{host}/api/tags")
        tags.raise_for_status()
        models = [m.get("model") or m.get("name") for m in tags.json().get("models", [])]
        return {"host": host, "reachable": True, "ms": round((time.perf_counter() - started) * 1000),
                "version": version, "models": models, "error": None}
    except (httpx.HTTPError, ValueError) as exc:
        return {"host": host, "reachable": False, "ms": round((time.perf_counter() - started) * 1000),
                "version": None, "models": [], "error": f"{type(exc).__name__}: {exc}"}


def _tagged(model: str) -> str:
    # Ollama lists "llama3" as "llama3:latest".
    return model if ":" in model else f"{model}:latest"


def _mask(value: str) -> str:
    # Proxy URLs can carry credentials (http://user:pass@proxy:8080).
    return re.sub(r"//[^@/]+@", "//***@", value)


@router.get("/ollama")
def ollama_check(
    test_model: bool = Query(False, description="Also send a one-word prompt to the model and time it."),
    model: Optional[str] = Query(None, description="Model to look for; defaults to the summary model."),
):
    machine = _api_machine()
    configured = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = model or os.environ.get("OLLAMA_SUMMARY_MODEL", "llama3:latest")

    # Same httpx defaults (proxy settings included) as the real model calls use.
    with httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0)) as client:
        checks = [_probe(client, host) for host in _candidates(configured, machine["in_docker"])]

    configured_check = checks[0]
    working = next((c for c in checks if c["reachable"]), None)
    model_available = bool(working) and _tagged(model) in working["models"]

    generation = None
    if test_model and working and model_available:
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=180.0) as client:
                reply = client.post(f"{working['host']}/api/generate", json={
                    "model": model, "prompt": "Reply with the single word OK.",
                    "stream": False, "options": {"num_predict": 5, "temperature": 0},
                })
                reply.raise_for_status()
            generation = {"ok": True, "ms": round((time.perf_counter() - started) * 1000),
                          "reply": reply.json().get("response", "").strip()[:50]}
        except (httpx.HTTPError, ValueError) as exc:
            generation = {"ok": False, "ms": round((time.perf_counter() - started) * 1000),
                          "error": f"{type(exc).__name__}: {exc}"}

    if configured_check["reachable"] and model_available:
        verdict = f"OK: Ollama answers at {configured} and {model} is installed."
    elif configured_check["reachable"]:
        verdict = (f"Ollama answers at {configured}, but {model} is not installed. "
                   f"Run on this machine: ollama pull {model}")
    elif working:
        verdict = (f"Ollama answers at {working['host']}, but this API is configured for {configured}. "
                   f"Set OLLAMA_HOST (fastapi_app.py line 29) to {working['host']} and restart.")
    else:
        where = "a container" if machine["in_docker"] else "WSL" if machine["in_wsl"] else f"machine {machine['hostname']}"
        verdict = (f"Ollama is not reachable from this API, which runs on {where}. Either Ollama is not running "
                   f"there (start it with: ollama serve), or it runs on another machine; list that address in "
                   f"OLLAMA_CHECK_HOSTS to test it, then set OLLAMA_HOST to it.")
    if generation and not generation["ok"]:
        verdict += f" The test prompt failed: {generation['error']}"

    return {
        "verdict": verdict,
        "api_machine": machine,
        "configured_host": configured,
        "model": model,
        "model_available": model_available,
        "checks": checks,
        "generation": generation,
        "proxy_env": {name: _mask(os.environ[name]) for name in _PROXY_VARS if os.environ.get(name)},
    }
