"""De-identified structured events for pilot shadow-mode monitoring."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any


_LOGGER = logging.getLogger("flynnmed.shadow")
_VALID_OUTCOMES = {"accepted", "edited", "rejected", "not_reviewed"}


def enabled() -> bool:
    return os.getenv("SHADOW_MODE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _safe_event(trace: dict[str, Any], clinician_outcome: str) -> dict[str, Any]:
    if clinician_outcome not in _VALID_OUTCOMES:
        raise ValueError(f"Invalid clinician outcome: {clinician_outcome}")
    timings = trace.get("stage_timings_ms") or {}
    duration_seconds = sum(
        float(value) for value in timings.values() if isinstance(value, (int, float))
    ) / 1000.0
    sources = trace.get("sources") or []
    return {
        "event_id": str(trace.get("trace_id") or "unknown"),
        "occurred_at": str(trace.get("created_at") or datetime.now(timezone.utc).isoformat()),
        "release_id": os.getenv("FLYNNMED_RELEASE_ID", "unfrozen-development"),
        "role": str(trace.get("role_key") or "unknown"),
        "intent_category": str(trace.get("intent_category") or "unknown"),
        "risk_level": str(trace.get("risk_level") or "unknown"),
        "retrieval_mode": str(trace.get("retrieval_mode") or "unknown"),
        "displayed_source_count": len(sources),
        "citation_resolution_rate": None,
        "duration_seconds": round(duration_seconds, 3),
        "clinician_outcome": clinician_outcome,
        "safety_event": False,
    }


def emit_shadow_event(
    trace: dict[str, Any], clinician_outcome: str = "not_reviewed"
) -> dict[str, Any] | None:
    """Emit one JSON log event without question, answer, username, or patient ID."""
    if not enabled():
        return None
    event = _safe_event(trace, clinician_outcome)
    _LOGGER.info("FLYNNMED_SHADOW %s", json.dumps(event, sort_keys=True))
    return event
