"""Meeting Notes — Commons guild's second wave.

Open-source replacement for Otter.ai / Fireflies.ai. Three substantive
endpoints:

  POST /commons/meeting/transcribe    — audio (mp3/wav/m4a) -> transcript
                                        with timestamped segments.
                                        Backed by faster-whisper base.en.
  POST /commons/meeting/summarize     — transcript -> {tldr_3lines,
                                        key_decisions, action_items,
                                        follow_ups}. Routes through
                                        route_think (claude_cli substrate).
  POST /commons/meeting/action_items  — transcript -> [{owner, action,
                                        deadline}]. Same substrate.

Plus GET /commons/meeting/health (public) for the Commons landing page.

Auth: service-key + X-Agent-Id (same as the rest of /api/v1/...). SC-billed
per call, audited to tool_calls.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.service_auth import require_service_key_and_agent
from app.models.agent import Agent
from app.services.ledger import (
    InsufficientFundsError,
    record_transaction,
)
from app.services.router_service import route_think


_log = logging.getLogger(__name__)

router = APIRouter(prefix="/commons/meeting", tags=["commons-meeting"])


# ── SC cost table — flat per-call. Tweakable via env. ──────────────────────
SC_COST = {
    "transcribe":   float(os.environ.get("MEETING_SC_TRANSCRIBE",   "8")),
    "summarize":    float(os.environ.get("MEETING_SC_SUMMARIZE",   "10")),
    "action_items": float(os.environ.get("MEETING_SC_ACTION_ITEMS", "5")),
}

MAX_UPLOAD_BYTES = int(os.environ.get("MEETING_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
WHISPER_MODEL = os.environ.get("MEETING_WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.environ.get("MEETING_WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("MEETING_WHISPER_COMPUTE", "int8")

ALLOWED_AUDIO_SUFFIX = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".webm"}

_WHISPER_MODEL_CACHE: Any = None


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _load_whisper():
    """Lazy-load the model. Cached for the lifetime of the process."""
    global _WHISPER_MODEL_CACHE
    if _WHISPER_MODEL_CACHE is not None:
        return _WHISPER_MODEL_CACHE
    from faster_whisper import WhisperModel
    _WHISPER_MODEL_CACHE = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    return _WHISPER_MODEL_CACHE


# ── Audit + billing helpers (mirrors commons/pdf_toolkit.py) ───────────────

async def _read_upload(f: UploadFile, *, label: str) -> bytes:
    data = await f.read()
    if not data:
        raise HTTPException(status_code=400, detail=f"{label}_empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "upload_too_large",
                "limit_bytes": MAX_UPLOAD_BYTES,
                "got_bytes": len(data),
                "field": label,
            },
        )
    return data


def _json_dump(d: dict) -> str:
    return json.dumps(d, default=str)


async def _log_tool_call(
    db: AsyncSession,
    *,
    agent_id: str,
    tool_name: str,
    input_summary: dict,
    output_len: int,
    sc_cost: float,
    latency_ms: int,
    error: str | None = None,
) -> None:
    await db.execute(
        text(
            """
            INSERT INTO tool_calls
                (id, agent_id, tool_name, input_json, output_len,
                 sc_cost, latency_ms, error, created_at)
            VALUES
                (:id, :agent_id, :tool_name, CAST(:input_json AS JSONB),
                 :output_len, :sc_cost, :latency_ms, :error, NOW())
            """
        ),
        {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "tool_name": tool_name,
            "input_json": _json_dump(input_summary),
            "output_len": output_len,
            "sc_cost": sc_cost,
            "latency_ms": latency_ms,
            "error": error,
        },
    )


async def _charge(
    db: AsyncSession,
    agent: Agent,
    *,
    tool: str,
    ref: str,
) -> float:
    cost = SC_COST[tool]
    if cost <= 0:
        return 0.0
    try:
        await record_transaction(
            db,
            agent_id=agent.id,
            delta=-cost,
            reason=f"commons.meeting.{tool}",
            ref_id=ref,
            note=f"Meeting Notes: {tool}",
        )
    except InsufficientFundsError as e:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_sc",
                "balance": e.balance,
                "needed": cost,
                "tool": f"commons.meeting.{tool}",
            },
        )
    return cost


# ── /health (public) ───────────────────────────────────────────────────────

@router.get("/health")
async def health() -> dict:
    return {
        "service": "commons.meeting",
        "version": "1.0.0",
        "available": {
            "transcribe": _whisper_available(),
            "summarize": True,
            "action_items": True,
        },
        "whisper_model": WHISPER_MODEL,
        "sc_cost": SC_COST,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


# ── /transcribe ────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def transcribe_audio(
    file: Annotated[UploadFile, File(description="Audio file (mp3/wav/m4a/flac/ogg/webm) <= 25MB")],
    language: Annotated[str, Form(description="ISO language code or 'auto'. Default 'en'.")] = "en",
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    if not _whisper_available():
        raise HTTPException(
            status_code=501,
            detail={
                "error": "transcribe_unavailable",
                "reason": "faster-whisper not installed on this VM",
                "fallback": "use summarize/action_items with a pre-existing transcript",
            },
        )

    name = (file.filename or "").lower()
    suffix = Path(name).suffix or ".bin"
    if suffix not in ALLOWED_AUDIO_SUFFIX:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_audio_format", "suffix": suffix,
                    "allowed": sorted(ALLOWED_AUDIO_SUFFIX)},
        )

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    await _charge(db, agent, tool="transcribe", ref=ref)
    # Commit charge early so the agent row isn't locked while we run
    # whisper (which doesn't touch the DB but keeps the request long).
    await db.commit()

    data = await _read_upload(file, label="file")

    tmpdir = Path(tempfile.mkdtemp(prefix=f"meeting-{ref}-"))
    in_path = tmpdir / f"in{suffix}"
    try:
        in_path.write_bytes(data)
        try:
            model = _load_whisper()
            segments_iter, info = model.transcribe(
                str(in_path),
                language=None if language == "auto" else language,
                vad_filter=True,
                beam_size=1,
            )
            segments: list[dict] = []
            full_text_parts: list[str] = []
            for s in segments_iter:
                seg_text = (s.text or "").strip()
                segments.append({
                    "start": round(float(s.start), 2),
                    "end":   round(float(s.end), 2),
                    "text":  seg_text,
                })
                if seg_text:
                    full_text_parts.append(seg_text)
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail={"error": "transcribe_failed", "msg": str(e)[:300]},
            )

        full_text = " ".join(full_text_parts)
        out = {
            "transcript": full_text,
            "segments": segments,
            "language": info.language,
            "language_probability": round(float(info.language_probability), 3),
            "duration_s": round(float(info.duration), 2),
            "sc_cost": SC_COST["transcribe"],
            "ref": ref,
        }

        await _log_tool_call(
            db, agent_id=agent.id, tool_name="commons.meeting.transcribe",
            input_summary={
                "in_bytes": len(data),
                "filename": name,
                "duration_s": out["duration_s"],
                "language": out["language"],
                "segment_count": len(segments),
            },
            output_len=len(full_text),
            sc_cost=SC_COST["transcribe"],
            latency_ms=int((time.monotonic() - t0) * 1000),
        )
        await db.commit()
        return out
    except HTTPException:
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── /summarize ─────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    transcript: str = Field(..., description="Plain-text transcript or segmented text.")
    context: str | None = Field(None, description="Optional meeting context (attendees, topic).")


_SUMMARIZE_SYSTEM = (
    "You are a meeting-notes assistant. Read the transcript and return JSON "
    "matching this schema exactly:\n"
    "{\n"
    '  "tldr_3lines":   [3 short strings],\n'
    '  "key_decisions": [strings, each a discrete decision the group made],\n'
    '  "action_items":  [strings, each a concrete next action],\n'
    '  "follow_ups":    [strings, each an open question or deferred topic]\n'
    "}\n"
    "Return ONLY the JSON object — no prose, no code fences."
)


_ACTION_ITEMS_SYSTEM = (
    "You extract action items from a meeting transcript. Return JSON with "
    "an 'action_items' array. Each item has fields:\n"
    '  "owner":    string (name from transcript, or "unassigned"),\n'
    '  "action":   string (the concrete task),\n'
    '  "deadline": string (date / relative time mentioned, or "not_specified")\n'
    "Return ONLY the JSON object — no prose, no code fences."
)


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # ```json\n...\n```
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_text_from_think(result: dict) -> str:
    """route_think returns either Anthropic-style {content: [{type:'text',
    text: ...}], ...} (anthropic / gemini path) or a plain string under
    `content` (claude_cli path). Handle both."""
    parts = result.get("content")
    if isinstance(parts, str):
        return parts
    if isinstance(parts, list):
        chunks: list[str] = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                t = p.get("text") or ""
                if t:
                    chunks.append(t)
            elif isinstance(p, str):
                chunks.append(p)
        if chunks:
            return "".join(chunks)
    if isinstance(result.get("text"), str):
        return result["text"]
    return ""


async def _think_json(
    *, agent_id: str, system: str, transcript: str, max_tokens: int = 1024,
    extra_user_context: str | None = None,
) -> dict:
    # claude_cli (the universal fallback substrate) flattens to a single
    # user prompt and may discard or de-prioritize the system block. Repeat
    # the instruction in the user message itself so the model still sees it.
    user_msg = (
        f"{system}\n\n"
        f"---\n"
        f"Transcript:\n\n{transcript.strip()}\n\n"
        f"---\n"
        f"Reply with the JSON object only, no prose, no markdown."
    )
    if extra_user_context:
        user_msg = f"Context: {extra_user_context}\n\n{user_msg}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    try:
        result = await route_think(
            agent_id, "fast", messages,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "think_substrate_failed", "msg": str(e)[:300]},
        )

    raw = _extract_text_from_think(result).strip()
    if not raw:
        raise HTTPException(
            status_code=502,
            detail={"error": "think_empty_response", "raw_keys": list(result.keys())},
        )
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to locate the first {...} blob.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise HTTPException(
                status_code=502,
                detail={"error": "think_response_not_json", "raw": raw[:400]},
            )
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=502,
                detail={"error": "think_response_not_json",
                        "msg": str(e), "raw": raw[:400]},
            )
    return parsed


@router.post("/summarize")
async def summarize_transcript(
    req: SummarizeRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    if not (req.transcript or "").strip():
        raise HTTPException(status_code=400, detail="transcript_empty")
    if len(req.transcript) > 200_000:
        raise HTTPException(
            status_code=413,
            detail={"error": "transcript_too_long",
                    "max_chars": 200_000, "got_chars": len(req.transcript)},
        )

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    await _charge(db, agent, tool="summarize", ref=ref)
    # Commit the SC debit before invoking route_think — route_think opens its
    # own session and re-locks the agent row, which deadlocks against ours.
    await db.commit()

    parsed = await _think_json(
        agent_id=agent.id,
        system=_SUMMARIZE_SYSTEM,
        transcript=req.transcript,
        extra_user_context=req.context,
        max_tokens=1024,
    )

    out = {
        "tldr_3lines":   list(parsed.get("tldr_3lines")   or []),
        "key_decisions": list(parsed.get("key_decisions") or []),
        "action_items":  list(parsed.get("action_items")  or []),
        "follow_ups":    list(parsed.get("follow_ups")    or []),
        "sc_cost": SC_COST["summarize"],
        "ref": ref,
    }

    await _log_tool_call(
        db, agent_id=agent.id, tool_name="commons.meeting.summarize",
        input_summary={
            "transcript_chars": len(req.transcript),
            "context_chars": len(req.context or ""),
            "tldr_count": len(out["tldr_3lines"]),
            "decision_count": len(out["key_decisions"]),
            "action_count": len(out["action_items"]),
            "followup_count": len(out["follow_ups"]),
        },
        output_len=len(_json_dump(out)),
        sc_cost=SC_COST["summarize"],
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    await db.commit()
    return out


# ── /action_items ──────────────────────────────────────────────────────────

class ActionItemsRequest(BaseModel):
    transcript: str = Field(..., description="Plain-text transcript.")


@router.post("/action_items")
async def action_items_from_transcript(
    req: ActionItemsRequest,
    db: AsyncSession = Depends(get_db),
    agent: Agent = Depends(require_service_key_and_agent),
) -> dict:
    if not (req.transcript or "").strip():
        raise HTTPException(status_code=400, detail="transcript_empty")
    if len(req.transcript) > 200_000:
        raise HTTPException(
            status_code=413,
            detail={"error": "transcript_too_long",
                    "max_chars": 200_000, "got_chars": len(req.transcript)},
        )

    t0 = time.monotonic()
    ref = str(uuid.uuid4())
    await _charge(db, agent, tool="action_items", ref=ref)
    await db.commit()

    parsed = await _think_json(
        agent_id=agent.id,
        system=_ACTION_ITEMS_SYSTEM,
        transcript=req.transcript,
        max_tokens=768,
    )

    raw_items = parsed.get("action_items") or []
    items: list[dict] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        items.append({
            "owner":    str(it.get("owner") or "unassigned"),
            "action":   str(it.get("action") or "").strip(),
            "deadline": str(it.get("deadline") or "not_specified"),
        })
    items = [i for i in items if i["action"]]

    out = {
        "action_items": items,
        "count": len(items),
        "sc_cost": SC_COST["action_items"],
        "ref": ref,
    }

    await _log_tool_call(
        db, agent_id=agent.id, tool_name="commons.meeting.action_items",
        input_summary={
            "transcript_chars": len(req.transcript),
            "action_count": len(items),
        },
        output_len=len(_json_dump(out)),
        sc_cost=SC_COST["action_items"],
        latency_ms=int((time.monotonic() - t0) * 1000),
    )
    await db.commit()
    return out
