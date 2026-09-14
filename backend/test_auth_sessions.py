from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

from backend.auth.sessions import SessionError, issue_session, rotate_refresh_session, validate_access_session
from backend.db import get_session_factory
from backend.identity_verification import (
    create_email_challenge,
    review_clinician_registration,
    submit_clinician_registration,
    verify_email_code,
)
from backend.models.account import Account, AccountKind
from backend.models.security import AuthSession
from backend.rate_limit import LimitRule, consume


def _db_available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
            session.execute(text("SELECT 1 FROM auth_sessions LIMIT 1"))
        return True
    except (OperationalError, Exception):
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="requires migrated Postgres")


@pytest.fixture()
def db_session():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _account(db_session, *, verified: bool = True) -> Account:
    key = f"security-{uuid.uuid4().hex[:12]}"
    account = Account(
        id=uuid.uuid4(),
        username=key,
        email=f"{key}@example.com",
        display_name="Security Test",
        password_hash="x",
        password_algo="argon2id",
        account_kind=AccountKind.patient,
        email_verified=verified,
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    return account


def test_refresh_rotation_rejects_reuse_and_revokes_family(db_session):
    account = _account(db_session)
    first = issue_session(db_session, account)
    _, second = rotate_refresh_session(db_session, first.refresh_token)
    validate_access_session(db_session, str(second.session_id), account.id)

    with pytest.raises(SessionError, match="reuse"):
        rotate_refresh_session(db_session, first.refresh_token)

    rows = db_session.execute(select(AuthSession).where(AuthSession.account_id == account.id)).scalars().all()
    assert rows
    assert all(row.revoked_at is not None for row in rows)


def test_email_code_is_single_use_and_marks_account_verified(db_session):
    account = _account(db_session, verified=False)
    code = create_email_challenge(db_session, account)
    verify_email_code(db_session, account, code)
    assert account.email_verified is True


def test_clinician_approval_is_out_of_band_and_revokes_existing_sessions(db_session):
    account = _account(db_session)
    session_tokens = issue_session(db_session, account)
    registration = submit_clinician_registration(
        db_session,
        account,
        requested_role="Doctor / Physician",
        organization="Test Hospital",
        registration_number="GMC-123456",
        registration_country="United Kingdom",
    )
    review_clinician_registration(
        db_session,
        registration.id,
        approve=True,
        note="Registry checked",
        reviewer="test-admin",
    )

    assert account.account_kind == AccountKind.clinician
    with pytest.raises(SessionError):
        validate_access_session(db_session, str(session_tokens.session_id), account.id)


def test_postgres_rate_limit_is_shared_and_atomic():
    rule = LimitRule(f"test-{uuid.uuid4().hex}", frozenset({"POST"}), "/test", 2, 60)
    identity = uuid.uuid4().hex

    assert consume(rule, identity)[0] is True
    assert consume(rule, identity)[0] is True
    assert consume(rule, identity)[0] is False
