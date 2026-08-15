"""
Evidence Ledger Phase 4 tests: AnswerClaim persistence, classification, and
best-effort EvidenceClaim/PatientFact linkage. Follows
test_patient_fact_ledger.py's exact conventions (skipif-gated on a live
Postgres with migrations applied, rollback-isolated db_session fixture, real
Account/Patient rows for the FK).
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.answer_claim_ledger import (
    persist_answer_claim,
    persist_answer_claims_for_bundle,
)
from backend.db import get_session_factory
from backend.evidence_ledger import (
    persist_evidence_claim,
    persist_evidence_passage,
    persist_source_artifact,
)
from backend.models.account import Account, AccountKind
from backend.models.patient import Patient
from backend.mrn import generate_mrn
from backend.patient_fact_ledger import persist_patient_fact


def _db_available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="requires a live Postgres (DATABASE_URL) with migrations applied",
)


@pytest.fixture()
def db_session():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _account(db_session) -> Account:
    username = f"answer-claim-{uuid.uuid4().hex[:8]}"
    account = Account(
        id=uuid.uuid4(),
        username=username,
        email=f"{username}@example.com",
        display_name="Test Patient",
        password_hash="x",
        password_algo="argon2id",
        account_kind=AccountKind.patient,
        role_label="Patient / Individual",
        clinical_role="Patient / Individual",
        organization="",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    return account


def _patient(db_session) -> Patient:
    account = _account(db_session)
    patient = Patient(
        id=uuid.uuid4(),
        account_id=account.id,
        patient_id=generate_mrn(),
        biological_sex="",
    )
    db_session.add(patient)
    db_session.flush()
    return patient


def _unique_url() -> str:
    return f"https://test.example/answer-claim-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# persist_answer_claim
# ---------------------------------------------------------------------------

def test_persist_answer_claim_basic_insert(db_session):
    patient = _patient(db_session)

    claim = persist_answer_claim(
        db_session, "trace-abc123", patient.id,
        claim_text="Metformin is first-line therapy for type 2 diabetes.",
        status="supported_cited", requires_evidence=True,
        source_ids=["S1"], evidence_claim_ids=[], patient_fact_ids=[],
    )

    assert claim is not None
    assert claim.trace_id == "trace-abc123"
    assert claim.patient_id == patient.id
    assert claim.status == "supported_cited"
    assert claim.requires_evidence is True
    assert claim.rule_version


def test_persist_answer_claim_coerces_unrecognised_status(db_session):
    patient = _patient(db_session)

    claim = persist_answer_claim(
        db_session, "trace-abc123", patient.id,
        claim_text="Some claim.", status="not_a_real_status", requires_evidence=False,
    )

    assert claim is not None
    assert claim.status == "unknown"


def test_persist_answer_claim_creates_two_rows_for_identical_repeat(db_session):
    """AnswerClaim is an append-only audit log, not a dedup'd fact table --
    the same claim text/status from two separate answer instances (e.g. the
    same question asked twice) must produce two distinct rows."""
    patient = _patient(db_session)

    first = persist_answer_claim(
        db_session, "trace-one", patient.id,
        claim_text="Ibuprofen can interact with warfarin.",
        status="supported_cited", requires_evidence=True,
    )
    second = persist_answer_claim(
        db_session, "trace-two", patient.id,
        claim_text="Ibuprofen can interact with warfarin.",
        status="supported_cited", requires_evidence=True,
    )

    assert first is not None
    assert second is not None
    assert first.id != second.id


def test_persist_answer_claim_returns_none_without_claim_text(db_session):
    patient = _patient(db_session)
    assert persist_answer_claim(
        db_session, "trace-abc", patient.id, claim_text="", status="supported_cited", requires_evidence=False
    ) is None


# ---------------------------------------------------------------------------
# persist_answer_claims_for_bundle -- classification
# ---------------------------------------------------------------------------

def _get_status(patient_id: str, trace_id: str, claim_text: str) -> str:
    session_factory = get_session_factory()
    with session_factory() as db:
        row = db.execute(
            text(
                "SELECT status FROM answer_claims WHERE patient_id = :pid "
                "AND trace_id = :tid AND claim_text = :ct"
            ),
            {"pid": patient_id, "tid": trace_id, "ct": claim_text},
        ).mappings().one()
        return row["status"]


def test_persist_answer_claims_for_bundle_classifies_unsupported_hedged():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        db.commit()
        patient_id = str(patient.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    claim = {
        "claim": "This herbal supplement cures the common cold.",
        "status": "general_knowledge", "requires_evidence": True,
        "source_ids": [], "passage_ids": [],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[claim], uncited_supported_claims=[], claim_correction_applied=True,
    )

    assert _get_status(patient_id, trace_id, claim["claim"]) == "unsupported_hedged"


def test_persist_answer_claims_for_bundle_classifies_unsupported_uncorrected():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        db.commit()
        patient_id = str(patient.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    claim = {
        "claim": "This herbal supplement cures the common cold.",
        "status": "general_knowledge", "requires_evidence": True,
        "source_ids": [], "passage_ids": [],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[claim], uncited_supported_claims=[], claim_correction_applied=False,
    )

    assert _get_status(patient_id, trace_id, claim["claim"]) == "unsupported_uncorrected"


def test_persist_answer_claims_for_bundle_classifies_supported_citation_added():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        db.commit()
        patient_id = str(patient.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    claim = {
        "claim": "Metformin reduces HbA1c.",
        "status": "supported", "requires_evidence": True,
        "source_ids": ["S1"], "passage_ids": [],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[claim], uncited_supported_claims=[claim], claim_correction_applied=True,
    )

    assert _get_status(patient_id, trace_id, claim["claim"]) == "supported_citation_added"


def test_persist_answer_claims_for_bundle_classifies_supported_cited():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        db.commit()
        patient_id = str(patient.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    claim = {
        "claim": "Metformin reduces HbA1c.",
        "status": "supported", "requires_evidence": True,
        "source_ids": ["S1"], "passage_ids": [],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[claim], uncited_supported_claims=[], claim_correction_applied=False,
    )

    assert _get_status(patient_id, trace_id, claim["claim"]) == "supported_cited"


def test_persist_answer_claims_for_bundle_classifies_general_knowledge_no_evidence_required():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        db.commit()
        patient_id = str(patient.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    claim = {
        "claim": "Drinking water is generally healthy.",
        "status": "general_knowledge", "requires_evidence": False,
        "source_ids": [], "passage_ids": [],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[claim], uncited_supported_claims=[], claim_correction_applied=False,
    )

    assert _get_status(patient_id, trace_id, claim["claim"]) == "general_knowledge_no_evidence_required"


def test_persist_answer_claims_for_bundle_never_raises_on_empty_or_missing_patient():
    persist_answer_claims_for_bundle("trace-x", None, claim_alignment=[])
    persist_answer_claims_for_bundle("trace-x", None, claim_alignment=[{"claim": "x", "status": "supported"}])


# ---------------------------------------------------------------------------
# evidence_claim_ids / patient_fact_ids resolution
# ---------------------------------------------------------------------------

def test_persist_answer_claims_for_bundle_resolves_evidence_claim_ids():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        artifact = persist_source_artifact(
            db, {"url": _unique_url(), "title": "Trial guidance", "detail_snippet": "Metformin reduced HbA1c more than placebo."},
        )
        passage = persist_evidence_passage(db, artifact, "Metformin reduced HbA1c more than placebo", "passage 1 of 1")
        evidence_claim = persist_evidence_claim(
            db, artifact, passage,
            claim=type("C", (), {
                "claim_text": "Metformin reduced HbA1c more than placebo",
                "population": "", "intervention": "Metformin", "comparator": "Placebo",
                "outcome": "HbA1c reduction", "study_design": "rct", "certainty": "high",
            })(),
        )
        db.commit()
        patient_id = str(patient.id)
        passage_id = str(passage.id)
        expected_evidence_claim_id = str(evidence_claim.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    matched_claim = {
        "claim": "Metformin reduced HbA1c more than placebo.",
        "status": "supported", "requires_evidence": True,
        "source_ids": ["S1"], "passage_ids": [passage_id],
    }
    unmatched_claim = {
        "claim": "This is unrelated.",
        "status": "supported", "requires_evidence": True,
        "source_ids": ["S2"], "passage_ids": [str(uuid.uuid4())],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[matched_claim, unmatched_claim], uncited_supported_claims=[], claim_correction_applied=False,
    )

    with session_factory() as db:
        rows = db.execute(
            text("SELECT claim_text, evidence_claim_ids FROM answer_claims WHERE trace_id = :tid ORDER BY claim_text"),
            {"tid": trace_id},
        ).mappings().all()
    by_text = {r["claim_text"]: r["evidence_claim_ids"] for r in rows}
    assert by_text[matched_claim["claim"]] == [expected_evidence_claim_id]
    assert by_text[unmatched_claim["claim"]] == []


def test_persist_answer_claims_for_bundle_resolves_patient_fact_ids():
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        fact = persist_patient_fact(
            db, patient.id, category="allergy", label="Penicillin", value="Rash",
        )
        db.commit()
        patient_id = str(patient.id)
        expected_fact_id = str(fact.id)

    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    matched_claim = {
        "claim": "This patient has a recorded Penicillin allergy.",
        "status": "supported", "requires_evidence": True,
        "source_ids": [], "passage_ids": [],
    }
    unmatched_claim = {
        "claim": "This claim mentions no known allergies.",
        "status": "general_knowledge", "requires_evidence": False,
        "source_ids": [], "passage_ids": [],
    }
    persist_answer_claims_for_bundle(
        trace_id, patient_id,
        claim_alignment=[matched_claim, unmatched_claim], uncited_supported_claims=[], claim_correction_applied=False,
    )

    with session_factory() as db:
        rows = db.execute(
            text("SELECT claim_text, patient_fact_ids FROM answer_claims WHERE trace_id = :tid ORDER BY claim_text"),
            {"tid": trace_id},
        ).mappings().all()
    by_text = {r["claim_text"]: r["patient_fact_ids"] for r in rows}
    assert by_text[matched_claim["claim"]] == [expected_fact_id]
    assert by_text[unmatched_claim["claim"]] == []
