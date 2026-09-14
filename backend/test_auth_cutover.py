"""Integration tests for the C-01 auth cutover: backend/api.py's
`current_username`/`_mint_token_for` (replacing the legacy hand-rolled HMAC
token -- `_token_secret`/`_create_token`/`_read_token`/`current_user`) against
a real Postgres instance with migrations applied. Skips entirely if
DATABASE_URL isn't set or unreachable -- see backend/test_auth_dependencies.py
for the same pattern and rationale.

These test `current_username`/`_mint_token_for` directly rather than through
`fastapi.testclient.TestClient` against a full business route: this test
suite's conftest.py deliberately forces DATA_BACKEND=legacy for every test
(so ordinary tests stay deterministic against the JSON store), which means
`UserStore.create_user`/`UserStore.authenticate` -- and therefore the real
`/api/auth/login`/`/api/auth/signup` HTTP routes -- cannot reach the SQL
`Account` rows these tests create. Testing the dependency functions directly
(the same pattern backend/test_auth_dependencies.py already uses for
`current_account`/`require_patient`/`require_clinician`) is both accurate to
what changed and avoids that unrelated data-layer mismatch.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.api import _mint_token_for, current_username
from backend.auth.jwt import create_access_token, decode_access_token
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind


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


def _unique_username(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_account(db_session, kind: AccountKind, username: str, is_active: bool = True) -> Account:
    account = Account(
        id=uuid.uuid4(),
        username=username,
        email=f"{username}@example.com",
        display_name=username,
        password_hash="x",
        password_algo="argon2id",
        account_kind=kind,
        email_verified=True,
        is_active=is_active,
    )
    db_session.add(account)
    db_session.flush()
    return account


def _legacy_style_token(username: str, secret: str = "flynnmed-local-dev-secret") -> str:
    """Builds a token in exactly the shape the now-deleted _create_token
    produced, so we can prove it is no longer accepted."""

    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    payload = {"sub": username.strip().lower(), "exp": int(time.time()) + 3600}
    payload_part = b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(secret.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_part}.{b64(signature)}"


# ── _mint_token_for ──────────────────────────────────────────────────────────

def test_mint_token_for_issues_decodable_jwt_bound_to_account(db_session):
    username = _unique_username("mint")
    account = _make_account(db_session, AccountKind.patient, username)

    token = _mint_token_for(db_session, username)
    payload = decode_access_token(token)

    assert payload.account_id == str(account.id)
    assert payload.account_kind == AccountKind.patient.value


def test_mint_token_for_unknown_username_raises_401(db_session):
    with pytest.raises(HTTPException) as exc:
        _mint_token_for(db_session, _unique_username("no-such-account"))
    assert exc.value.status_code == 401


# ── current_username ─────────────────────────────────────────────────────────

def test_current_username_happy_path(db_session):
    username = _unique_username("cu-happy")
    account = _make_account(db_session, AccountKind.patient, username)

    assert current_username(account=account) == username


def test_current_username_rejects_disabled_account(db_session):
    # current_username itself just reads account.username -- the is_active
    # check happens one layer down in current_account (already covered by
    # backend/test_auth_dependencies.py::test_current_account_rejects_inactive_account).
    # This test confirms the account produced by a disabled-account token
    # never reaches current_username in the first place, by exercising the
    # full dependency chain current_username relies on.
    from backend.auth.dependencies import current_account

    username = _unique_username("cu-disabled")
    _make_account(db_session, AccountKind.patient, username, is_active=False)
    token = _mint_token_for(db_session, username)

    with pytest.raises(HTTPException) as exc:
        current_account(authorization=f"Bearer {token}", db=db_session)
    assert exc.value.status_code == 401


def test_forged_legacy_token_is_rejected(db_session):
    """A token in the exact shape/secret the deleted legacy _create_token
    produced (including the hardcoded fallback secret) must not be accepted
    anywhere -- decode_access_token only understands real JWTs."""
    username = _unique_username("forged")
    _make_account(db_session, AccountKind.patient, username)
    forged = _legacy_style_token(username)

    with pytest.raises(Exception):
        decode_access_token(forged)


def test_expired_jwt_is_rejected(db_session):
    from backend.auth.dependencies import current_account

    username = _unique_username("expired")
    account = _make_account(db_session, AccountKind.patient, username)
    from backend.auth.sessions import issue_session
    session = issue_session(db_session, account)
    expired_token = create_access_token(
        str(account.id), account.account_kind.value, session_id=str(session.session_id), ttl_seconds=-10
    )

    with pytest.raises(HTTPException) as exc:
        current_account(authorization=f"Bearer {expired_token}", db=db_session)
    assert exc.value.status_code == 401


def test_malformed_bearer_token_is_rejected(db_session):
    from backend.auth.dependencies import current_account

    with pytest.raises(HTTPException) as exc:
        current_account(authorization="Bearer not-a-real-token", db=db_session)
    assert exc.value.status_code == 401


def test_missing_authorization_header_is_rejected(db_session):
    from backend.auth.dependencies import current_account

    with pytest.raises(HTTPException) as exc:
        current_account(authorization="", db=db_session)
    assert exc.value.status_code == 401
