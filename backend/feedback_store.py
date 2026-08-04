"""
Response feedback store -- persists anonymised quality-signal rows to Neon PostgreSQL.

Privacy contract:
  - No question text, no answer text, no username, no PII stored.
  - Only: rating + clinical/governance metadata from the trace dict.
  - interaction_id is a deterministic UUID derived from the trace_id string
    so rows can be grouped by interaction without exposing any content.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional

from backend.config import psycopg_database_url


FEEDBACK_TABLE = "response_feedback"

_table_ready = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _derive_interaction_id(trace_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, trace_id or "unknown")


def _connect():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "DATABASE_URL is set but psycopg is not installed. "
            "Add `psycopg[binary]` to requirements.txt."
        ) from exc
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set -- feedback cannot be saved.")
    return psycopg.connect(psycopg_database_url(database_url))


def _ensure_table() -> None:
    global _table_ready
    if _table_ready:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
                    id UUID PRIMARY KEY,
                    interaction_id UUID NOT NULL,
                    session_id UUID,
                    rating VARCHAR(20) NOT NULL,

                    intent_detected VARCHAR(100),
                    risk_level VARCHAR(50),
                    user_role VARCHAR(50),
                    clinical_pathway VARCHAR(100),
                    triage_urgency VARCHAR(50),
                    suggested_next_step VARCHAR(100),

                    evidence_tier_summary VARCHAR(100),
                    num_sources INTEGER,

                    policy_gates_applied TEXT,
                    alignment_passed BOOLEAN,
                    contradiction_flag BOOLEAN,

                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        conn.commit()
    _table_ready = True


def _evidence_tier_summary(sources: list) -> str:
    if not sources:
        return "no_sources"
    tiers = {s.get("evidence_tier", 3) for s in sources}
    if tiers == {1}:
        return "tier1_only"
    if tiers == {2}:
        return "tier2_only"
    if tiers == {3}:
        return "tier3_only"
    return "mixed"


def _alignment_flags(claim_alignment: list) -> tuple[Optional[bool], Optional[bool]]:
    if not claim_alignment:
        return None, None
    unsupported = [c for c in claim_alignment if c.get("status") != "supported"]
    alignment_passed = len(unsupported) == 0
    contradiction_flag = not alignment_passed
    return alignment_passed, contradiction_flag


def save_feedback(
    rating: str,
    trace: Dict,
    session_id: Optional[str] = None,
) -> bool:
    """
    Persist one anonymised feedback row.
    Returns True on success, False if DATABASE_URL is not configured.
    """
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        return False

    try:
        _ensure_table()

        sources = trace.get("sources", [])
        claim_alignment = trace.get("claim_alignment", [])
        alignment_passed, contradiction_flag = _alignment_flags(claim_alignment)
        gates = trace.get("policy_gates_applied", [])
        gates_str = ", ".join(
            g.get("gate_name", "") for g in gates if g.get("gate_name")
        ) if gates else None

        pathway_decision = trace.get("pathway_decision", {})
        triage_urgency = (
            pathway_decision.get("urgency_level")
            or trace.get("risk_level", "")
        ) or None
        suggested_next_step = pathway_decision.get("next_step") or None

        trace_id = trace.get("trace_id", "")
        interaction_id = _derive_interaction_id(trace_id)

        try:
            session_uuid = uuid.UUID(str(session_id)) if session_id else None
        except (ValueError, AttributeError):
            session_uuid = None

        row_id = uuid.uuid4()

        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {FEEDBACK_TABLE} (
                        id, interaction_id, session_id, rating,
                        intent_detected, risk_level, user_role, clinical_pathway,
                        triage_urgency, suggested_next_step,
                        evidence_tier_summary, num_sources,
                        policy_gates_applied, alignment_passed, contradiction_flag,
                        created_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s
                    )
                    """,
                    (
                        row_id,
                        interaction_id,
                        session_uuid,
                        rating,
                        trace.get("intent_category") or None,
                        trace.get("risk_level") or None,
                        trace.get("role_key") or None,
                        trace.get("pathway_used") or None,
                        triage_urgency,
                        suggested_next_step,
                        _evidence_tier_summary(sources),
                        len(sources),
                        gates_str,
                        alignment_passed,
                        contradiction_flag,
                        _utc_now(),
                    ),
                )
            conn.commit()
        return True

    except Exception as exc:
        print(f"[feedback_store] Failed to save feedback: {exc}")
        return False
