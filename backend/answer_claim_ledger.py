"""
Evidence Ledger Phase 4 write path: persists what
backend/summarizer.py's check_claim_source_alignment/apply_claim_corrections
already compute on every answer (see backend/rag_system.py's
_finalize_answer_payload for the call site) as first-class AnswerClaim rows,
with best-effort links to the EvidenceClaim/PatientFact rows Phases 2/3
already persist.

Same short-lived-session, never-blocks-the-answer discipline as
backend/evidence_ledger.py and backend/patient_fact_ledger.py -- a
persistence failure must never prevent the answer already generated from
being returned to the user.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_session_factory
from backend.models.answer_claim import ANSWER_CLAIM_RULE_VERSION, ANSWER_CLAIM_STATUSES, AnswerClaim
from backend.models.evidence import EvidenceClaim
from backend.models.patient_fact import PatientFact


def _parse_uuid(value: Any) -> Optional[UUID]:
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, AttributeError):
        return None


def persist_answer_claim(
    db: Session,
    trace_id: str,
    patient_id: Optional[UUID],
    *,
    claim_text: str,
    status: str,
    requires_evidence: bool,
    source_ids: Optional[List[str]] = None,
    evidence_claim_ids: Optional[List[str]] = None,
    patient_fact_ids: Optional[List[str]] = None,
) -> Optional[AnswerClaim]:
    """Always inserts a new row -- this is an append-only audit log, not a
    deduplicated fact table (see backend/models/answer_claim.py's
    docstring): a repeated question should produce a fresh AnswerClaim per
    answer instance, not reuse one from a prior answer."""
    claim_text = (claim_text or "").strip()
    if not claim_text or not trace_id:
        return None

    status = (status or "unknown").strip().lower()
    if status not in ANSWER_CLAIM_STATUSES:
        status = "unknown"

    claim = AnswerClaim(
        trace_id=trace_id[:64],
        patient_id=patient_id,
        claim_text=claim_text[:4000],
        status=status,
        requires_evidence=bool(requires_evidence),
        source_ids=list(source_ids or []),
        evidence_claim_ids=list(evidence_claim_ids or []),
        patient_fact_ids=list(patient_fact_ids or []),
        rule_version=ANSWER_CLAIM_RULE_VERSION,
    )
    db.add(claim)
    db.flush()
    return claim


def _lookup_evidence_claim_ids(db: Session, passage_ids: List[UUID]) -> List[UUID]:
    """Best-effort match: real, never a false positive (a DB-verified
    passage_id equality), but often empty -- see
    backend/models/answer_claim.py's docstring and the Phase 4 plan for why
    check_claim_source_alignment's passage_ids and EvidenceClaim.passage_id
    frequently reference different EvidencePassage rows for the same
    source."""
    if not passage_ids:
        return []
    rows = db.execute(
        select(EvidenceClaim.id).where(EvidenceClaim.passage_id.in_(passage_ids))
    ).scalars().all()
    return list(rows)


def _latest_patient_facts(db: Session, patient_id: Optional[UUID]) -> List[PatientFact]:
    """All PatientFact rows for this patient, deduped to the most-recently-
    created row per case-insensitive label -- a patient's dose can have
    several historical rows after an edit (Phase 3's content-hash-on-edit
    design), only the current one should be eligible for text matching."""
    if patient_id is None:
        return []
    rows = db.execute(
        select(PatientFact)
        .where(PatientFact.patient_id == patient_id)
        .order_by(PatientFact.created_at.desc())
    ).scalars().all()
    latest_by_label: Dict[str, PatientFact] = {}
    for fact in rows:
        key = fact.label.strip().lower()
        if key and key not in latest_by_label:
            latest_by_label[key] = fact
    return list(latest_by_label.values())


def persist_answer_claims_for_bundle(
    trace_id: str,
    patient_id: Any,
    *,
    claim_alignment: List[Dict],
    uncited_supported_claims: Optional[List[Dict]] = None,
    claim_correction_applied: bool = False,
) -> None:
    """Classifies and persists each claim from check_claim_source_alignment.
    No-op if claim_alignment is empty. Never raises -- persistence must not
    block the answer already generated from this same data."""
    if not claim_alignment or not trace_id:
        return

    parsed_patient_id = _parse_uuid(patient_id)
    uncited_supported_claims = uncited_supported_claims or []

    session_factory = get_session_factory()
    with session_factory() as db:
        patient_facts = _latest_patient_facts(db, parsed_patient_id)

        for c in claim_alignment:
            try:
                claim_text = str(c.get("claim", ""))
                if not claim_text.strip():
                    continue

                raw_status = c.get("status")
                requires_evidence = bool(c.get("requires_evidence"))
                is_unsupported = raw_status == "general_knowledge" and requires_evidence
                is_uncited_supported = any(c is u for u in uncited_supported_claims)

                if is_unsupported:
                    status = "unsupported_hedged" if claim_correction_applied else "unsupported_uncorrected"
                elif is_uncited_supported:
                    status = "supported_citation_added" if claim_correction_applied else "supported_cited"
                elif raw_status == "supported":
                    status = "supported_cited"
                elif raw_status == "general_knowledge":
                    status = "general_knowledge_no_evidence_required"
                else:
                    status = "unknown"

                passage_ids = [
                    parsed for p in (c.get("passage_ids") or []) if (parsed := _parse_uuid(p)) is not None
                ]
                evidence_claim_ids = _lookup_evidence_claim_ids(db, passage_ids)
                matched_facts = [
                    f for f in patient_facts if f.label.strip().lower() in claim_text.lower()
                ]

                persist_answer_claim(
                    db, trace_id, parsed_patient_id,
                    claim_text=claim_text,
                    status=status,
                    requires_evidence=requires_evidence,
                    source_ids=c.get("source_ids", []),
                    evidence_claim_ids=[str(i) for i in evidence_claim_ids],
                    patient_fact_ids=[str(f.id) for f in matched_facts],
                )
            except Exception as exc:
                print(f"[AnswerClaimLedger] claim persist failed: {exc}")

        db.commit()
