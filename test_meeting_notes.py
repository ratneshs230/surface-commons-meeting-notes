"""Integration tests for /commons/meeting — run against a live surface-api at
127.0.0.1:3000 with a service-key + a seeded agent.

Mirrors tests/commons/test_pdf_toolkit.py: real HTTP, no mocks. The
`transcribe` test depends on faster-whisper being installed in the venv;
`summarize` and `action_items` depend on the route_think substrate
(claude_cli fallback ensures it always answers).
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest


FIX = Path(__file__).parent / "fixtures"
BASE = os.environ.get("SURFACE_BASE_URL", "http://127.0.0.1:3000")
SVC_KEY = os.environ.get("SURFACE_SERVICE_KEY") or os.environ.get("SERVICE_API_KEY")
AGENT_ID = os.environ.get("SURFACE_TEST_AGENT_ID")


SAMPLE_TRANSCRIPT = (
    "Ratnesh: Welcome everyone. Today we need to decide on the launch date for "
    "the Meeting Notes Commons tool.\n"
    "Priya: I think May 15th is realistic. Engineering is on track.\n"
    "Ratnesh: Agreed. May 15th it is. Priya, please send the launch checklist "
    "to the team by Wednesday.\n"
    "Priya: Will do.\n"
    "Ratnesh: We also need to figure out pricing. Let's table that for next week.\n"
    "Priya: Sounds good. I'll prep three pricing options for the next sync."
)


def _client() -> httpx.Client:
    if not SVC_KEY or not AGENT_ID:
        pytest.skip("SURFACE_SERVICE_KEY + SURFACE_TEST_AGENT_ID required")
    return httpx.Client(
        base_url=BASE,
        headers={"X-Service-Key": SVC_KEY, "X-Agent-Id": AGENT_ID},
        timeout=180.0,
    )


def test_health_public():
    r = httpx.get(f"{BASE}/commons/meeting/health", timeout=10)
    assert r.status_code == 200
    j = r.json()
    assert j["service"] == "commons.meeting"
    assert "available" in j
    assert "sc_cost" in j
    assert j["sc_cost"]["transcribe"] == 8
    assert j["sc_cost"]["summarize"] == 10
    assert j["sc_cost"]["action_items"] == 5


def test_transcribe():
    """faster-whisper must be available — synthetic 10s sine-wave WAV."""
    with _client() as c:
        files = [("file", ("sample.wav", (FIX / "sample_10s.wav").read_bytes(), "audio/wav"))]
        r = c.post("/commons/meeting/transcribe", files=files, data={"language": "en"})
        if r.status_code == 501:
            pytest.skip("faster-whisper not installed on this VM")
        assert r.status_code == 200, r.text
        j = r.json()
        assert "transcript" in j
        assert "segments" in j
        assert isinstance(j["segments"], list)
        assert j["sc_cost"] == 8
        # 10-second sine isn't speech, so transcript may be empty or hallucinated.
        # We only assert structure here.
        assert j["duration_s"] >= 9.0


def test_summarize():
    with _client() as c:
        r = c.post(
            "/commons/meeting/summarize",
            json={"transcript": SAMPLE_TRANSCRIPT, "context": "weekly product sync"},
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert "tldr_3lines" in j
        assert "key_decisions" in j
        assert "action_items" in j
        assert "follow_ups" in j
        assert j["sc_cost"] == 10
        # Some shape sanity — a real LLM should produce at least one bullet.
        assert isinstance(j["tldr_3lines"], list)
        assert isinstance(j["action_items"], list)


def test_action_items():
    with _client() as c:
        r = c.post(
            "/commons/meeting/action_items",
            json={"transcript": SAMPLE_TRANSCRIPT},
        )
        assert r.status_code == 200, r.text
        j = r.json()
        assert "action_items" in j
        assert "count" in j
        assert j["sc_cost"] == 5
        assert isinstance(j["action_items"], list)
        # Each entry must have the schema fields.
        for it in j["action_items"]:
            assert "owner" in it
            assert "action" in it
            assert "deadline" in it


def test_summarize_empty_transcript_rejected():
    with _client() as c:
        r = c.post("/commons/meeting/summarize", json={"transcript": "   "})
        assert r.status_code == 400


def test_auth_rejected_without_key():
    r = httpx.post(
        f"{BASE}/commons/meeting/summarize",
        json={"transcript": "hello"},
        timeout=10,
    )
    assert r.status_code in (401, 422)
