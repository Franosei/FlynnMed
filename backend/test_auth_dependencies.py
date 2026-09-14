"""Integration tests for backend/auth/dependencies.py against a real Postgres
instance with migrations applied. Skips entirely if DATABASE_URL isn't set or
unreachable -- these run for real in CI (which starts a Postgres service and
runs `alembic upgrade head` before tests) and locally once `docker compose up
-d db && alembic upgrade head` has been run.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from backend.auth.dependencies import (
    authorize_patient_access,
    current_account,
    require_clinician,
    require_patient,
)
from backend.auth.sessions import issue_session
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.audit import AuditAction, AuditLogEntry, AuditOutcome
from backend.models.consent import ConsentGrant, ConsentScope, ConsentStatus
from backend.models.patient import Patient


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
    not _db_available(), reason="requires a live Postgres (DATABASE_URL) with migrations applied"
)


@pytest.fixture()
def db_session():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _make_account(db_session, kind: AccountKind, username: str) -> Account:
    account = Account(
        id=uuid.uuid4(),
        username=username,
        email=f"{username}@example.com",
        display_name=username,
        password_hash="x",
        password_algo="argon2id",
        account_kind=kind,
        email_verified=True,
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    return account


def _make_patient(db_session, account: Account) -> Patient:
    from backend.mrn import generate_mrn

    patient = Patient(id=uuid.uuid4(), account_id=account.id, patient_id=generate_mrn(), biological_sex="")
    db_session.add(patient)
    db_session.flush()
    return patient


def _bearer_for_db(db_session, account: Account) -> str:
    return f"Bearer {issue_session(db_session, account).access_token}"


def test_current_account_happy_path(db_session):
    account = _make_account(db_session, AccountKind.patient, "cad-happy")
    resolved = current_account(authorization=_bearer_for_db(db_session, account), db=db_session)
    assert resolved.id == account.id


def test_current_account_rejects_missing_header(db_session):
    with pytest.raises(HTTPException) as exc:
        current_account(authorization="", db=db_session)
    assert exc.value.status_code == 401


def test_current_account_rejects_inactive_account(db_session):
    account = _make_account(db_session, AccountKind.patient, "cad-inactive")
    account.is_active = False
    db_session.flush()
    with pytest.raises(HTTPException) as exc:
        current_account(authorization=_bearer_for_db(db_session, account), db=db_session)
    assert exc.value.status_code == 401


def test_require_patient_rejects_clinician_account(db_session):
    account = _make_account(db_session, AccountKind.clinician, "rp-clinician")
    with pytest.raises(HTTPException) as exc:
        require_patient(account=account, db=db_session)
    assert exc.value.status_code == 403


def test_require_clinician_rejects_patient_account(db_session):
    account = _make_account(db_session, AccountKind.patient, "rc-patient")
    with pytest.raises(HTTPException) as exc:
        require_clinician(account=account)
    assert exc.value.status_code == 403


class _FakeClient:
    host = "127.0.0.1"


class _FakeRequest:
    client = _FakeClient()


def test_authorize_patient_access_denies_unknown_mrn(db_session):
    clinician = _make_account(db_session, AccountKind.clinician, "apa-unknown-mrn")
    dependency = authorize_patient_access(ConsentScope.previsit_summary, AuditAction.clinician_read_previsit_summary)

    with pytest.raises(HTTPException) as exc:
        dependency(patient_id="FM-0000-0000", request=_FakeRequest(), clinician=clinician, db=db_session)
    assert exc.value.status_code == 403

    logged = db_session.execute(
        select(AuditLogEntry).where(AuditLogEntry.actor_account_id == clinician.id)
    ).scalars().all()
    assert any(entry.outcome == AuditOutcome.denied for entry in logged)


def test_authorize_patient_access_denies_without_grant(db_session):
    patient_account = _make_account(db_session, AccountKind.patient, "apa-nogrant-patient")
    patient = _make_patient(db_session, patient_account)
    clinician = _make_account(db_session, AccountKind.clinician, "apa-nogrant-clinician")
    dependency = authorize_patient_access(ConsentScope.previsit_summary, AuditAction.clinician_read_previsit_summary)

    with pytest.raises(HTTPException) as exc:
        dependency(patient_id=patient.patient_id, request=_FakeRequest(), clinician=clinician, db=db_session)
    assert exc.value.status_code == 403


def test_authorize_patient_access_denies_wrong_scope(db_session):
    patient_account = _make_account(db_session, AccountKind.patient, "apa-scope-patient")
    patient = _make_patient(db_session, patient_account)
    clinician = _make_account(db_session, AccountKind.clinician, "apa-scope-clinician")
    db_session.add(
        ConsentGrant(
            id=uuid.uuid4(),
            patient_id=patient.id,
            clinician_account_id=clinician.id,
            status=ConsentStatus.active,
            scope=[ConsentScope.chat_history.value],  # granted chat_history, not previsit_summary
            requested_at=datetime.now(timezone.utc),
        )
    )
    db_session.flush()
    dependency = authorize_patient_access(ConsentScope.previsit_summary, AuditAction.clinician_read_previsit_summary)

    with pytest.raises(HTTPException) as exc:
        dependency(patient_id=patient.patient_id, request=_FakeRequest(), clinician=clinician, db=db_session)
    assert exc.value.status_code == 403


def test_authorize_patient_access_denies_expired_grant(db_session):
    patient_account = _make_account(db_session, AccountKind.patient, "apa-expired-patient")
    patient = _make_patient(db_session, patient_account)
    clinician = _make_account(db_session, AccountKind.clinician, "apa-expired-clinician")
    db_session.add(
        ConsentGrant(
            id=uuid.uuid4(),
            patient_id=patient.id,
            clinician_account_id=clinician.id,
            status=ConsentStatus.active,
            scope=[ConsentScope.previsit_summary.value],
            requested_at=datetime.now(timezone.utc) - timedelta(days=30),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    db_session.flush()
    dependency = authorize_patient_access(ConsentScope.previsit_summary, AuditAction.clinician_read_previsit_summary)

    with pytest.raises(HTTPException) as exc:
        dependency(patient_id=patient.patient_id, request=_FakeRequest(), clinician=clinician, db=db_session)
    assert exc.value.status_code == 403


def test_authorize_patient_access_grants_with_valid_active_scope_and_audits_success(db_session):
    patient_account = _make_account(db_session, AccountKind.patient, "apa-valid-patient")
    patient = _make_patient(db_session, patient_account)
    clinician = _make_account(db_session, AccountKind.clinician, "apa-valid-clinician")
    grant = ConsentGrant(
        id=uuid.uuid4(),
        patient_id=patient.id,
        clinician_account_id=clinician.id,
        status=ConsentStatus.active,
        scope=[ConsentScope.previsit_summary.value],
        requested_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )
    db_session.add(grant)
    db_session.flush()
    dependency = authorize_patient_access(ConsentScope.previsit_summary, AuditAction.clinician_read_previsit_summary)

    context = dependency(patient_id=patient.patient_id, request=_FakeRequest(), clinician=clinician, db=db_session)
    assert context.patient.id == patient.id
    assert context.grant.id == grant.id

    logged = db_session.execute(
        select(AuditLogEntry).where(AuditLogEntry.actor_account_id == clinician.id)
    ).scalars().all()
    assert any(entry.outcome == AuditOutcome.success and entry.consent_grant_id == grant.id for entry in logged)


def test_authorize_patient_access_denies_revoked_grant(db_session):
    patient_account = _make_account(db_session, AccountKind.patient, "apa-revoked-patient")
    patient = _make_patient(db_session, patient_account)
    clinician = _make_account(db_session, AccountKind.clinician, "apa-revoked-clinician")
    db_session.add(
        ConsentGrant(
            id=uuid.uuid4(),
            patient_id=patient.id,
            clinician_account_id=clinician.id,
            status=ConsentStatus.revoked,
            scope=[ConsentScope.previsit_summary.value],
            requested_at=datetime.now(timezone.utc),
            revoked_at=datetime.now(timezone.utc),
        )
    )
    db_session.flush()
    dependency = authorize_patient_access(ConsentScope.previsit_summary, AuditAction.clinician_read_previsit_summary)

    with pytest.raises(HTTPException) as exc:
        dependency(patient_id=patient.patient_id, request=_FakeRequest(), clinician=clinician, db=db_session)
    assert exc.value.status_code == 403
