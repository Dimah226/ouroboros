#!/usr/bin/env python3
"""
test_ollama.py — Zero-dependency script to test local Ollama installation.

Usage:
    python scripts/test_ollama.py
    python scripts/test_ollama.py --model qwen3:8b
    python scripts/test_ollama.py --host http://localhost:11434

No pip installs required. Uses only Python standard library.
"""

import json
import sys
import time
import argparse
import urllib.request
import urllib.error
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_HOST  = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"
TIMEOUT_S     = 30   # seconds to wait for a response

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(url: str) -> dict | None:
    """HTTP GET → parsed JSON, or None on error."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _post(url: str, payload: dict) -> dict | None:
    """HTTP POST JSON → parsed JSON, or None on error."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read().decode())
    except urllib.error.URLError as e:
        print(f"  ✗ Connection error: {e.reason}")
        return None
    except json.JSONDecodeError:
        print("  ✗ Server returned invalid JSON.")
        return None


def _ok(msg: str):  print(f"  ✓ {msg}")
def _fail(msg: str): print(f"  ✗ {msg}")
def _info(msg: str): print(f"  · {msg}")

# ── Test steps ─────────────────────────────────────────────────────────────────

def test_reachable(host: str) -> bool:
    """Check that Ollama is listening at all."""
    print("\n[1/4] Connectivity check …")
    result = _get(f"{host}/api/version")
    if result and "version" in result:
        _ok(f"Ollama v{result['version']} is running at {host}")
        return True
    _fail(f"Cannot reach {host}")
    print()
    print("  → Is Ollama installed?  https://ollama.com/download/windows")
    print("  → Is it running?        Check the system tray icon, or run:")
    print("                            ollama serve")
    return False


def list_models(host: str) -> list[str]:
    """Return a list of locally available model names."""
    print("\n[2/4] Available models …")
    result = _get(f"{host}/api/tags")
    if not result or "models" not in result:
        _fail("Could not retrieve model list.")
        return []

    models = result["models"]
    if not models:
        _fail("No models installed yet.")
        print("  → Pull one with:  ollama pull qwen3:8b")
        return []

    names = [m["name"] for m in models]
    for m in models:
        size_gb = m.get("size", 0) / 1e9
        _ok(f"{m['name']}  ({size_gb:.1f} GB)")
    return names


def test_generation(host: str, model: str) -> bool:
    """Send a short prompt and measure response time + tokens/sec."""
    print(f"\n[3/4] Generation test  (model: {model}) …")
    _info("Sending prompt — this may take a few seconds on first run …")

    prompt  = "Réponds en une phrase courte : quelle est la capitale de la France ?"
    t0      = time.perf_counter()
    result  = _post(f"{host}/api/generate",
                    {"model": model, "prompt": prompt, "stream": False})
    elapsed = time.perf_counter() - t0

    if not result:
        _fail("No response from model.")
        print(f"  → Is the model installed?  ollama pull {model}")
        return False

    response_text = result.get("response", "").strip()
    eval_count    = result.get("eval_count", 0)          # tokens generated
    eval_ns       = result.get("eval_duration", 0)       # nanoseconds

    tps = eval_count / (eval_ns / 1e9) if eval_ns else 0

    _ok(f"Response received in {elapsed:.1f}s")
    _info(f"Tokens generated : {eval_count}")
    _info(f"Speed            : {tps:.1f} tokens/sec")
    print()
    print(f"  Prompt   : {prompt}")
    print(f"  Response : {response_text}")
    return True


def test_api_endpoint(host: str, model: str) -> bool:
    """Test the /api/chat endpoint (OpenAI-compatible style)."""
    print(f"\n[4/4] Chat endpoint test …")
    result = _post(f"{host}/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
        "stream": False,
    })

    if not result:
        _fail("/api/chat endpoint failed.")
        return False

    content = result.get("message", {}).get("content", "").strip()
    _ok(f"/api/chat works  →  model replied: '{content}'")
    return True

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test local Ollama installation.")
    parser.add_argument("--host",  default=DEFAULT_HOST,  help="Ollama base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to test")
    args = parser.parse_args()

    host  = args.host.rstrip("/")
    model = args.model

    print("=" * 60)
    print(f"  Ollama connectivity test — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Host  : {host}")
    print(f"  Model : {model}")
    print("=" * 60)

    # ── Step 1: reachable?
    if not test_reachable(host):
        sys.exit(1)

    # ── Step 2: model list
    available = list_models(host)

    # ── Step 3 & 4: generation + chat (only if model present)
    model_present = any(model in n for n in available)
    if not model_present and available:
        fallback = available[0]
        print(f"\n  ⚠ Model '{model}' not found locally. Falling back to '{fallback}'.")
        model = fallback
    elif not model_present:
        print(f"\n  ⚠ No models available. Pull one first:")
        print(f"      ollama pull {DEFAULT_MODEL}")
        sys.exit(1)

    gen_ok  = test_generation(host, model)
    chat_ok = test_api_endpoint(host, model)

    # ── Summary
    print("\n" + "=" * 60)
    all_ok = gen_ok and chat_ok
    if all_ok:
        print("  🎉 ALL TESTS PASSED — Ollama is ready!")
        print()
        print("  To use it as an API endpoint from Ouroboros:")
        print(f"    Base URL : {host}/v1")
        print(f"    Model    : {model}")
        print("    (No API key required for local Ollama)")
    else:
        print("  ⚠  Some tests failed — see details above.")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
