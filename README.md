# Meeting Notes — Surface Commons

Open-source replacement for **Otter.ai / Fireflies.ai**. Three endpoints, SC-billed per call.

| endpoint | cost | input | output |
|---|---|---|---|
| `POST /commons/meeting/transcribe`   | 8 SC  | audio (mp3/wav/m4a/flac/ogg/webm) ≤ 25 MB | `{transcript, segments[], duration_s, language}` |
| `POST /commons/meeting/summarize`    | 10 SC | `{transcript, context?}` | `{tldr_3lines[], key_decisions[], action_items[], follow_ups[]}` |
| `POST /commons/meeting/action_items` | 5 SC  | `{transcript}` | `[{owner, action, deadline}]` |
| `GET  /commons/meeting/health`       | free  | — | capability + cost table |

## Stack

- **Transcription** — [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) `base.en`, CPU/int8, VAD-filtered.
- **Summarize / action items** — Surface's `route_think` substrate. Falls back through anthropic → gemini → ollama → `claude_cli` so the path always answers even with one provider down.
- **Audit** — every call writes a row to `tool_calls` (input summary, latency, output size, sc_cost) and an SC debit to `sc_transactions`.

## Why

Otter Pro is ₹1,500/mo. This is `~₹3` per meeting (8 + 10 + 5 SC = 23 SC ≈ ₹3 at 1 SC = ₹0.13). Source is MIT, runs on a single CPU box, no vendor lock-in.

## Source-of-truth

This repo is a **mirror** of [`Surface-V2/app/api/commons/meeting_notes.py`](https://github.com/ratneshs230/Surface-V2/blob/master/app/api/commons/meeting_notes.py). PRs accepted upstream.

Part of the [Surface Commons](https://commons.agentnation.in) guild.

## Files in this mirror

- `meeting_notes.py` — the FastAPI router (4 endpoints)
- `test_meeting_notes.py` — 6 integration tests, run against a live surface-api
- `sample_10s.wav` — 10-sec synthetic sine fixture
- `__init__.py` — package marker
