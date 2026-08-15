"""
Evidence Ledger Phase 4 write path: persists what
backend/summarizer.py's check_claim_source_alignment/apply_claim_corrections
already compute on every answer (see backend/rag_system.py's
_finalize_answer_payload for the call site) as first-class AnswerClaim rows,
with best-effort links to the EvidenceClaim/PatientFact rows Phases 2/3
already persist.

Also the traceability write path for modules other than Health Chat --
Safety Review (persist_safety_review_evidence) and Trial Finder
(persist_trial_finder_matches) -- both deterministic, so neither needs an
LLM claim-check pass; they classify directly. Care Plans reuses
check_claim_source_alignment itself (see backend/care_plan_agent.py) and
calls persist_answer_claims_for_bundle the same way Health Chat does, just
tagged module="care_plan".

Same short-lived-session, never-blocks-the-answer discipline as
backend/evidence_ledger.py and backend/patient_fact_ledger.py -- a
persistence failure must never prevent the answer already generated from
being returned to the user.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_session_factory
from backend.models.answer_claim import ANSWER_CLAIM_RULE_VERSION, ANSWER_CLAIM_STATUSES, AnswerClaim
from backend.models.evidence import EvidenceClaim
from backend.models.patient_fact import PatientFact

# Scoped anaphora heuristic (Evidence Ledger v2): a claim referencing a fact
# only by a generic noun phrase ("this medication", "your allergy") instead
# of its actual label. Only applied when the patient has exactly one fact in
# that category -- with more than one, which fact the phrase refers to is
# genuinely ambiguous, and a wrong link is worse than no link. True
# coreference resolution across multiple same-category facts is out of scope.
_GENERIC_REFERENCE_TERMS = {
    "medication": ("this medication", "this medicine", "this drug", "the medication", "the medicine", "the drug"),
    "allergy": ("this allergy", "your allergy", "the allergy"),
    "condition": ("this condition", "your condition", "the condition"),
    "vital": ("this reading", "this result", "your result", "the reading"),
    "symptom": ("this symptom", "your symptom", "the symptom"),
}


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
    module: str = "health_chat",
    llm_only_support: bool = False,
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
        module=(module or "health_chat")[:32],
        llm_only_support=bool(llm_only_support),
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
    """All non-retracted PatientFact rows for this patient, deduped to the
    most-recently-created row per case-insensitive label -- a patient's dose
    can have several historical rows after an edit (see
    backend/patient_fact_ledger.py's supersession chain), only the current
    one should be eligible for text matching. Retracted facts (deleted
    medications, disproved allergies) are excluded -- an answer claim should
    never appear to cite a fact the patient's record no longer asserts."""
    if patient_id is None:
        return []
    # Latest row per label first, THEN drop retracted ones -- filtering out
    # retracted rows before dedup would let the older, already-superseded
    # row underneath resurface as "latest" (retracting never mutates the
    # original row in place, see backend/patient_fact_ledger.py).
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
    return [fact for fact in latest_by_label.values() if fact.status != "retracted"]


def _match_patient_facts(claim_text: str, patient_facts: List[PatientFact]) -> List[PatientFact]:
    """Word-boundary label match (fixes the old raw-substring false positive,
    e.g. label "iron" no longer matching inside "ironic") plus a scoped
    generic-reference heuristic for indirect mentions ("this antibiotic")
    when exactly one fact of that category exists. Still a partial fix --
    true coreference resolution across multiple same-category facts isn't
    attempted."""
    lowered_claim = claim_text.lower()
    matched: List[PatientFact] = []

    for fact in patient_facts:
        label = fact.label.strip().lower()
        if label and re.search(rf"\b{re.escape(label)}\b", lowered_claim):
            matched.append(fact)

    if matched:
        return matched

    facts_by_category: Dict[str, List[PatientFact]] = {}
    for fact in patient_facts:
        facts_by_category.setdefault(fact.category, []).append(fact)

    for category, terms in _GENERIC_REFERENCE_TERMS.items():
        candidates = facts_by_category.get(category, [])
        if len(candidates) != 1:
            continue  # ambiguous with 0 or 2+ facts in this category -- skip
        if any(term in lowered_claim for term in terms):
            matched.append(candidates[0])

    return matched


def persist_answer_claims_for_bundle(
    trace_id: str,
    patient_id: Any,
    *,
    claim_alignment: List[Dict],
    uncited_supported_claims: Optional[List[Dict]] = None,
    claim_correction_applied: bool = False,
    module: str = "health_chat",
    answer_blocked: bool = False,
) -> None:
    """Classifies and persists each claim from check_claim_source_alignment.
    No-op if claim_alignment is empty and the answer wasn't blocked. Never
    raises -- persistence must not block the answer already generated from
    this same data.

    answer_blocked (fail-closed policy, Evidence Ledger v2): when
    verification failed twice and backend/rag_system.py replaced the answer
    with the safe fallback message, individual claims couldn't be computed --
    persist one "unsupported_blocked" row instead of per-claim rows, so the
    audit trail still shows the block happened.
    """
    if answer_blocked:
        if not trace_id:
            return
        session_factory = get_session_factory()
        with session_factory() as db:
            try:
                persist_answer_claim(
                    db, trace_id, _parse_uuid(patient_id),
                    claim_text="<answer blocked pending verification>",
                    status="unsupported_blocked",
                    requires_evidence=True,
                    module=module,
                )
                db.commit()
            except Exception as exc:
                print(f"[AnswerClaimLedger] blocked-claim persist failed: {exc}")
        return

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
                llm_only_support = c.get("deterministic_corroboration") == "failed"

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
                matched_facts = _match_patient_facts(claim_text, patient_facts)

                persist_answer_claim(
                    db, trace_id, parsed_patient_id,
                    claim_text=claim_text,
                    status=status,
                    requires_evidence=requires_evidence,
                    source_ids=c.get("source_ids", []),
                    evidence_claim_ids=[str(i) for i in evidence_claim_ids],
                    patient_fact_ids=[str(f.id) for f in matched_facts],
                    module=module,
                    llm_only_support=llm_only_support,
                )
            except Exception as exc:
                print(f"[AnswerClaimLedger] claim persist failed: {exc}")

        db.commit()


def persist_safety_review_evidence(reviews: List[Dict], patient_id: Any) -> None:
    """Traceability for Safety Review (backend/safety_review.py): fully
    deterministic rule engine, no LLM judgment involved, and every review
    object already carries its own inline evidence (claim/source_title/
    source_url/passage -- see _base_review's `evidence` field). Persists one
    SourceArtifact per hardcoded source_url (deduped as usual) and one
    AnswerClaim per review, classified directly as "supported_cited" since
    the evidence is code-verified rather than checked by an LLM.

    Scoped and deduped deliberately, unlike the normal append-only
    AnswerClaim convention: build_safety_reviews (and therefore this
    function's caller) runs on nearly every app-wide request via api.py's
    _snapshot(), not once per distinct answer, so (a) only emergency/urgent
    reviews are persisted -- the ones that actually matter for an audit
    trail -- and (b) a review already recorded for this trace (keyed by
    review_id, which is itself a content hash of the rule + underlying
    facts -- see _review_id) is skipped rather than re-inserted on every
    poll. Never raises -- same never-blocks discipline as the rest of this
    module."""
    if not reviews:
        return
    parsed_patient_id = _parse_uuid(patient_id)
    trace_prefix = f"safety-review-{parsed_patient_id or 'unscoped'}"

    from backend.evidence_ledger import persist_evidence_passage, persist_source_artifact

    session_factory = get_session_factory()
    with session_factory() as db:
        for review in reviews:
            if review.get("priority") not in ("emergency", "urgent"):
                continue
            try:
                evidence_items = review.get("evidence") or []
                if not evidence_items:
                    continue
                primary = evidence_items[0]
                source_url = str(primary.get("source_url") or "").strip()
                passage_text = str(primary.get("passage") or "").strip()
                claim_text = str(primary.get("claim") or review.get("what_changed") or "").strip()
                if not claim_text:
                    continue

                trace_id = f"{trace_prefix}-{review.get('review_id', '')}"[:64]
                already_recorded = db.execute(
                    select(AnswerClaim.id).where(
                        AnswerClaim.trace_id == trace_id,
                        AnswerClaim.module == "safety_review",
                    )
                ).scalar_one_or_none()
                if already_recorded is not None:
                    continue

                source_ids: List[str] = []
                if source_url and passage_text:
                    artifact = persist_source_artifact(
                        db,
                        {
                            "url": source_url,
                            "title": str(primary.get("source_title") or ""),
                            "detail_snippet": passage_text,
                            "source_type": "official_guidance",
                        },
                    )
                    if artifact is not None:
                        persist_evidence_passage(
                            db, artifact, passage_text, "passage cited by a Safety Review rule"
                        )
                        source_ids = [str(artifact.id)]

                persist_answer_claim(
                    db, trace_id, parsed_patient_id,
                    claim_text=claim_text,
                    status="supported_cited",
                    requires_evidence=True,
                    source_ids=source_ids,
                    module="safety_review",
                )
            except Exception as exc:
                print(f"[AnswerClaimLedger] safety review evidence persist failed: {exc}")
        db.commit()


def persist_trial_finder_matches(trace_id: str, patient_id: Any, matched_trials: List[Dict]) -> None:
    """Traceability for Trial Finder (backend/clinical_trials.py): matching
    is deterministic ClinicalTrials.gov API filtering, not LLM narrative
    generation, so there's no claim text to run check_claim_source_alignment
    against. Persists one lightweight AnswerClaim per matched trial instead,
    recording why it was surfaced -- no SourceArtifact needed, a
    ClinicalTrials.gov entry is a record ID, not a text passage to hash.
    Never raises."""
    if not matched_trials or not trace_id:
        return
    parsed_patient_id = _parse_uuid(patient_id)
    session_factory = get_session_factory()
    with session_factory() as db:
        for trial in matched_trials:
            try:
                nct_id = str(trial.get("nct_id") or trial.get("id") or "").strip()
                title = str(trial.get("title") or "Untitled trial").strip()
                if not nct_id:
                    continue
                persist_answer_claim(
                    db, trace_id, parsed_patient_id,
                    claim_text=f"Matched trial: {title} ({nct_id})",
                    status="supported_cited",
                    requires_evidence=True,
                    source_ids=[nct_id],
                    module="trial_finder",
                )
            except Exception as exc:
                print(f"[AnswerClaimLedger] trial finder match persist failed: {exc}")
        db.commit()
