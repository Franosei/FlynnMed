"""Integration coverage for the consent-based clinician workflow."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.clinician_access import (
    AccessWorkflowError,
    access_overview,
    authorized_patient_summary,
    decide_access_request,
    request_patient_access,
    revoke_access,
)
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.patient import ChatMessage, Condition, Patient
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
        role_label=(
            "Doctor / Physician"
            if kind == AccountKind.clinician
            else "Patient / Individual"
        ),
        clinical_role=(
            "Doctor / Physician"
            if kind == AccountKind.clinician
            else "Patient / Individual"
        ),
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


def test_consent_lifecycle_controls_patient_summary_and_chat(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    db_session.add(
        Condition(
            id=uuid.uuid4(),
            patient_id=patient.id,
            name="Hypertension",
            status="active",
        )
    )
    db_session.add(
        ChatMessage(
            id=uuid.uuid4(),
            patient_id=patient.id,
            role="user",
            content="My home BP is high.",
            timestamp=datetime.now(timezone.utc),
        )
    )
    db_session.flush()

    with pytest.raises(AccessWorkflowError, match="No active access"):
        authorized_patient_summary(db_session, clinician.username, patient.patient_id)

    requested = request_patient_access(
        db_session,
        clinician.username,
        patient.patient_id,
        "Medication review",
        include_chat_history=True,
    )
    assert requested["status"] == "pending"
    assert access_overview(db_session, patient_account.username)["pending_count"] == 1

    approved = decide_access_request(
        db_session, patient_account.username, requested["grant_id"], approve=True
    )
    assert approved["status"] == "active"
    summary = authorized_patient_summary(
        db_session, clinician.username, patient.patient_id
    )
    assert summary["conditions"][0]["name"] == "Hypertension"
    assert summary["chat_history_authorized"] is True
    assert summary["chat_history"][0]["content"] == "My home BP is high."

    revoked = revoke_access(db_session, patient_account.username, requested["grant_id"])
    assert revoked["status"] == "revoked"
    with pytest.raises(AccessWorkflowError, match="No active access"):
        authorized_patient_summary(db_session, clinician.username, patient.patient_id)


def test_patient_cannot_request_cross_patient_access(db_session):
    patient_account = _account(db_session, AccountKind.patient, "patient-request")
    patient = _patient(db_session, patient_account)
    with pytest.raises(AccessWorkflowError, match="Clinician account required"):
        request_patient_access(
            db_session,
            patient_account.username,
            patient.patient_id,
            "Not permitted",
        )
