"""
Evidence Ledger v2 (#11) regression test: get_trace_patient_ids must
distinguish "no AnswerClaim rows at all" (None -- genuinely not found) from
"rows exist but every one has patient_id=None" (empty set -- a
patient-agnostic trace, e.g. a clinician's own self-service question,
since clinician accounts have no Patient row). The original ownership
check in backend/api.py's get_evidence_trace instead went through
UserStore.get_interaction_traces, which always returns [] for clinician
accounts -- breaking the "Reasoning lineage" panel for every clinician
account, its only intended audience.
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.answer_claim_ledger import persist_answer_claim
from backend.db import get_session_factory
from backend.evidence_trace import get_trace_patient_ids
from backend.models.account import Account, AccountKind
from backend.models.patient import Patient
from backend.mrn import generate_mrn


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


def _patient(db_session) -> Patient:
    username = f"evidence-trace-{uuid.uuid4().hex[:8]}"
    account = Account(
        id=uuid.uuid4(), username=username, email=f"{username}@example.com",
        display_name="Test Patient", password_hash="x", password_algo="argon2id",
        account_kind=AccountKind.patient, role_label="Patient / Individual",
        clinical_role="Patient / Individual", organization="", is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    patient = Patient(id=uuid.uuid4(), account_id=account.id, patient_id=generate_mrn(), biological_sex="")
    db_session.add(patient)
    db_session.flush()
    return patient


def test_get_trace_patient_ids_returns_none_for_unknown_trace():
    assert get_trace_patient_ids(f"trace-{uuid.uuid4().hex[:8]}") is None


def test_get_trace_patient_ids_returns_empty_set_for_patient_agnostic_trace(db_session):
    """The exact scenario a clinician's self-service Health Chat question
    produces -- claims with no patient_id, since clinician accounts have no
    Patient row (backend/repositories/sql_user_store.py's _get_patient)."""
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    persist_answer_claim(
        db_session, trace_id, None,
        claim_text="General clinical guidance claim.", status="supported_cited", requires_evidence=True,
    )
    db_session.commit()

    assert get_trace_patient_ids(trace_id) == set()


def test_get_trace_patient_ids_returns_the_referenced_patient(db_session):
    patient = _patient(db_session)
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"
    persist_answer_claim(
        db_session, trace_id, patient.id,
        claim_text="Patient-specific claim.", status="supported_cited", requires_evidence=True,
    )
    db_session.commit()

    assert get_trace_patient_ids(trace_id) == {patient.id}
