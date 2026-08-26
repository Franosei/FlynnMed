import json
import logging

from backend.shadow_monitor import emit_shadow_event


def test_shadow_event_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SHADOW_MODE_ENABLED", raising=False)
    assert emit_shadow_event({"trace_id": "trace-123456789"}) is None


def test_shadow_event_excludes_patient_content(monkeypatch, caplog):
    monkeypatch.setenv("SHADOW_MODE_ENABLED", "true")
    monkeypatch.setenv("FLYNNMED_RELEASE_ID", "pilot-v1")
    trace = {
        "trace_id": "trace-123456789",
        "created_at": "2026-08-25T10:00:00+00:00",
        "question": "private patient question",
        "answer_preview": "private answer",
        "personal_context": [{"mrn": "SECRET"}],
        "role_key": "patient",
        "intent_category": "symptom_triage",
        "risk_level": "urgent",
        "retrieval_mode": "agentic_multi_source",
        "sources": [{"source_id": "S1"}],
        "stage_timings_ms": {"prepare_bundle": 1000, "generation": 2000},
    }

    with caplog.at_level(logging.INFO, logger="flynnmed.shadow"):
        event = emit_shadow_event(trace)

    rendered = json.dumps(event)
    assert event["release_id"] == "pilot-v1"
    assert event["duration_seconds"] == 3.0
    assert "private patient question" not in rendered
    assert "private answer" not in rendered
    assert "SECRET" not in rendered
