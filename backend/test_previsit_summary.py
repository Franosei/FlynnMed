"""Integration coverage for the clinician pre-visit summary + inline
patient-scoped chat workflow. Follows test_clinician_access.py's exact
conventions (skipif-gated on a live Postgres, db_session fixture, account/
patient factories) -- reusing those factories directly rather than
re-deriving them.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

import backend.api as api
from backend.clinician_access import (
    decide_access_request,
    request_patient_access,
    revoke_access,
)
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.audit import AuditAction, AuditLogEntry, AuditOutcome
from backend.models.patient import Condition, Patient, PreVisitChatMessage, PreVisitSummary
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


def _account(db_session, kind: AccountKind, prefix: str) -> Account:
    username = f"{prefix}-{uuid.uuid4().hex[:8]}"
    account = Account(
        id=uuid.uuid4(),
        username=username,
        email=f"{username}@example.com",
        display_name=prefix.title(),
        password_hash="x",
        password_algo="argon2id",
        account_kind=kind,
        role_label=("Doctor / Physician" if kind == AccountKind.clinician else "Patient / Individual"),
        clinical_role=("Doctor / Physician" if kind == AccountKind.clinician else "Patient / Individual"),
        organization="Test Clinic" if kind == AccountKind.clinician else "",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    return account


def _patient(db_session, account: Account) -> Patient:
    patient = Patient(
        id=uuid.uuid4(),
        account_id=account.id,
        patient_id=generate_mrn(),
        biological_sex="",
    )
    db_session.add(patient)
    db_session.flush()
    return patient


def _grant_active_access(db_session, clinician: Account, patient: Patient, include_chat_history=False):
    requested = request_patient_access(
        db_session,
        clinician.username,
        patient.patient_id,
        "Pre-visit review",
        include_chat_history=include_chat_history,
    )
    decide_access_request(db_session, patient.account.username, requested["grant_id"], approve=True)
    return requested["grant_id"]


@pytest.fixture(autouse=True)
def _mock_summary_generation():
    # generate_previsit_chart_summary makes a real LLM call -- mocked so
    # these tests are fast and don't depend on network/API-key availability,
    # consistent with how test_summarizer.py/test_clinical_orchestrator.py
    # mock llm.client.chat.completions.create elsewhere in this codebase.
    with patch.object(api, "generate_previsit_chart_summary", return_value="Mocked AI-drafted summary."):
        yield


def _condition(db_session, patient: Patient):
    db_session.add(Condition(id=uuid.uuid4(), patient_id=patient.id, name="Hypertension", status="active"))
    db_session.flush()


class _FakeSummaryPayload:
    def __init__(self, summary_text: str):
        self.summary_text = summary_text


def test_no_grant_denies_generate_and_chat(db_session):
    # These call the FastAPI endpoint functions directly, not the underlying
    # require_active_previsit_access -- the endpoints catch AccessWorkflowError
    # and convert it to HTTPException(403) (see api.py's _access_error), so
    # that's what propagates here, not AccessWorkflowError itself.
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)

    with pytest.raises(HTTPException, match="No active access") as exc_info:
        api.generate_previsit_summary(patient.patient_id, username=clinician.username, db=db_session)
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException, match="No active access") as exc_info:
        api.save_previsit_summary_draft(
            patient.patient_id,
            _FakeSummaryPayload("edited text"),
            username=clinician.username,
            db=db_session,
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException, match="No active access") as exc_info:
        api.release_previsit_summary(
            patient.patient_id,
            _FakeSummaryPayload("final text"),
            username=clinician.username,
            db=db_session,
        )
    assert exc_info.value.status_code == 403


def test_generate_populates_authorship_and_defaults_to_draft(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)
    _grant_active_access(db_session, clinician, patient)

    result = api.generate_previsit_summary(patient.patient_id, username=clinician.username, db=db_session)

    assert result["status"] == "draft"
    assert result["generation_trigger"] == "ai_generated"
    assert result["summary_text"] == "Mocked AI-drafted summary."
    # The fix for ClinicalNote.edited_by_account_id being a dead/unpopulated
    # field: authorship must always be populated, never blank.
    assert result["authored_by_display_name"] == clinician.display_name
    assert result["authored_by_clinical_role"]
    assert result["released_at"] == ""


def test_release_is_the_only_path_that_writes_a_released_row(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)
    _grant_active_access(db_session, clinician, patient)

    api.generate_previsit_summary(patient.patient_id, username=clinician.username, db=db_session)
    api.save_previsit_summary_draft(
        patient.patient_id, _FakeSummaryPayload("clinician-edited text"), username=clinician.username, db=db_session
    )

    rows = db_session.execute(
        select(PreVisitSummary).where(PreVisitSummary.patient_id == patient.id)
    ).scalars().all()
    assert len(rows) == 2
    assert all(r.status == "draft" for r in rows)

    released = api.release_previsit_summary(
        patient.patient_id, _FakeSummaryPayload("final released text"), username=clinician.username, db=db_session
    )
    assert released["status"] == "released"
    assert released["released_at"]
    assert released["released_by_display_name"] == clinician.display_name

    rows = db_session.execute(
        select(PreVisitSummary).where(PreVisitSummary.patient_id == patient.id)
    ).scalars().all()
    assert sum(1 for r in rows if r.status == "released") == 1
    assert sum(1 for r in rows if r.status == "draft") == 2


def test_patient_read_returns_only_released_rows(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)
    _grant_active_access(db_session, clinician, patient)

    api.generate_previsit_summary(patient.patient_id, username=clinician.username, db=db_session)
    api.release_previsit_summary(
        patient.patient_id, _FakeSummaryPayload("released text"), username=clinician.username, db=db_session
    )

    result = api.list_my_previsit_summaries(username=patient_account.username, db=db_session)
    assert len(result["summaries"]) == 1
    assert result["summaries"][0]["status"] == "released"
    assert result["summaries"][0]["summary_text"] == "released text"


def test_continuity_of_care_second_clinician_sees_first_clinicians_history(db_session):
    """
    The behavior explicitly reconfirmed with the user: past summaries (draft
    and released) and the reasoning behind them are visible to ANY future
    clinician granted access to this patient, not just the original author.
    """
    clinician_a = _account(db_session, AccountKind.clinician, "doctor-a")
    clinician_b = _account(db_session, AccountKind.clinician, "doctor-b")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)

    grant_a = _grant_active_access(db_session, clinician_a, patient)
    api.generate_previsit_summary(patient.patient_id, username=clinician_a.username, db=db_session)
    api.release_previsit_summary(
        patient.patient_id, _FakeSummaryPayload("clinician A's released summary"), username=clinician_a.username, db=db_session
    )
    revoke_access(db_session, patient_account.username, grant_a)

    _grant_active_access(db_session, clinician_b, patient)

    summary_for_b = api.clinician_patient_summary(patient.patient_id, username=clinician_b.username, db=db_session)

    authors = {item["authored_by_display_name"] for item in summary_for_b["previsit_summaries"]}
    assert clinician_a.display_name in authors


def test_audit_rows_recorded_for_generate_and_release(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)
    _grant_active_access(db_session, clinician, patient)

    api.generate_previsit_summary(patient.patient_id, username=clinician.username, db=db_session)
    api.release_previsit_summary(
        patient.patient_id, _FakeSummaryPayload("final text"), username=clinician.username, db=db_session
    )

    audit_rows = db_session.execute(
        select(AuditLogEntry).where(AuditLogEntry.patient_id == patient.id)
    ).scalars().all()
    actions = [row.action for row in audit_rows]
    assert AuditAction.clinician_generate_previsit_summary in actions
    assert AuditAction.clinician_release_previsit_summary in actions
    for row in audit_rows:
        if row.action in (
            AuditAction.clinician_generate_previsit_summary,
            AuditAction.clinician_release_previsit_summary,
        ):
            assert row.outcome == AuditOutcome.success
            assert row.consent_grant_id is not None


class _NoCommitSessionWrapper:
    """_save_previsit_chat_message intentionally opens its own DB session
    (it's called from inside a StreamingResponse generator in production, see
    its docstring) -- but that means it can't see this test's fixture rows,
    which live in db_session's own uncommitted transaction. Wrapping db_session
    itself (with commit() downgraded to flush()) keeps the write in the same
    transaction so the test's own subsequent read can see it, while the
    db_session fixture's rollback still discards everything at the end --
    same test-isolation guarantee as every other test in this file, without
    ever really committing to the database."""

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def add(self, obj):
        self._session.add(obj)

    def commit(self):
        self._session.flush()


def test_previsit_chat_message_persisted_with_authorship(db_session, monkeypatch):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _condition(db_session, patient)
    grant_id = _grant_active_access(db_session, clinician, patient)

    monkeypatch.setattr(api, "get_session_factory", lambda: (lambda: _NoCommitSessionWrapper(db_session)))

    api._save_previsit_chat_message(
        patient_id=patient.id,
        role="clinician",
        content="What were the last three BP readings?",
        authored_by_account_id=clinician.id,
        authored_by_display_name=clinician.display_name,
        authored_by_clinical_role="Doctor / Physician",
        consent_grant_id=uuid.UUID(grant_id),
    )

    rows = db_session.execute(
        select(PreVisitChatMessage).where(PreVisitChatMessage.patient_id == patient.id)
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].role == "clinician"
    assert rows[0].authored_by_display_name == clinician.display_name
