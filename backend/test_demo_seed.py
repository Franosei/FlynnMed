"""Integration checks for the public fictional demo seed."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import OperationalError

from backend.auth.passwords import verify_password
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.consent import ConsentGrant, ConsentScope, ConsentStatus
from backend.models.patient import ChatMessage, Condition, Patient
from backend.scripts.seed_demo_accounts import (
    AISHA_MRN,
    AISHA_PASSWORD,
    AISHA_USERNAME,
    CLINICIAN_PASSWORD,
    CLINICIAN_USERNAME,
    MICHAEL_MRN,
    MICHAEL_PASSWORD,
    MICHAEL_USERNAME,
    PATIENT_MRN,
    PATIENT_PASSWORD,
    PATIENT_USERNAME,
    seed_demo_accounts,
)


def _db_available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="requires migrated PostgreSQL")


def test_demo_seed_is_idempotent_and_login_ready():
    factory = get_session_factory()
    with factory() as db:
        first = seed_demo_accounts(db)
        jane = db.execute(select(Account).where(Account.username == PATIENT_USERNAME)).scalar_one()
        patient = db.execute(select(Patient).where(Patient.account_id == jane.id)).scalar_one()
        condition_count = len(
            db.scalars(select(Condition).where(Condition.patient_id == patient.id)).all()
        )
        second = seed_demo_accounts(db)
        assert first == second
        assert second["active_patient_count"] == 3
        assert len(
            db.scalars(select(Condition).where(Condition.patient_id == patient.id)).all()
        ) == condition_count

    with factory() as db:
        jane = db.execute(select(Account).where(Account.username == PATIENT_USERNAME)).scalar_one()
        omar = db.execute(select(Account).where(Account.username == CLINICIAN_USERNAME)).scalar_one()
        patient = db.execute(select(Patient).where(Patient.account_id == jane.id)).scalar_one()
        grant = db.execute(
            select(ConsentGrant).where(
                ConsentGrant.patient_id == patient.id,
                ConsentGrant.clinician_account_id == omar.id,
                ConsentGrant.status == ConsentStatus.active,
            )
        ).scalar_one()

        assert jane.account_kind == AccountKind.patient
        assert omar.account_kind == AccountKind.clinician
        assert patient.patient_id == PATIENT_MRN
        assert verify_password(PATIENT_PASSWORD, jane.password_hash, jane.password_algo, jane.password_salt)
        assert verify_password(CLINICIAN_PASSWORD, omar.password_hash, omar.password_algo, omar.password_salt)
        assert set(grant.scope) == {
            ConsentScope.previsit_summary.value,
            ConsentScope.chat_history.value,
        }
        condition_names = set(
            db.scalars(select(Condition.name).where(Condition.patient_id == patient.id)).all()
        )
        assert {"Asthma", "Migraine without aura", "Iron-deficiency anaemia"} <= condition_names
        assert db.scalar(
            select(func.count()).select_from(ChatMessage).where(
                ChatMessage.patient_id == patient.id,
                ChatMessage.role == "user",
            )
        ) >= 5

        extra_patients = (
            (MICHAEL_USERNAME, MICHAEL_PASSWORD, MICHAEL_MRN),
            (AISHA_USERNAME, AISHA_PASSWORD, AISHA_MRN),
        )
        for username, password, mrn in extra_patients:
            account = db.scalar(select(Account).where(Account.username == username))
            extra_patient = db.scalar(select(Patient).where(Patient.account_id == account.id))
            assert verify_password(password, account.password_hash, account.password_algo, account.password_salt)
            assert extra_patient.patient_id == mrn
            assert db.scalar(
                select(func.count()).select_from(ConsentGrant).where(
                    ConsentGrant.patient_id == extra_patient.id,
                    ConsentGrant.clinician_account_id == omar.id,
                    ConsentGrant.status == ConsentStatus.active,
                )
            ) == 1
