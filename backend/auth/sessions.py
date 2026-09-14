"""Rotating, server-revocable refresh sessions."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.auth.jwt import create_access_token
from backend.config import jwt_secret_key
from backend.models.account import Account
from backend.models.security import AuthSession

REFRESH_TTL = timedelta(days=30)


class SessionError(Exception):
    pass


@dataclass(frozen=True)
class SessionTokens:
    access_token: str
    refresh_token: str
    session_id: uuid.UUID


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _digest(value: str, purpose: str = "refresh") -> str:
    return hmac.new(
        jwt_secret_key().encode("utf-8"),
        f"{purpose}:{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def fingerprint(value: str) -> str:
    return _digest(value or "", "fingerprint") if value else ""


def issue_session(
    db: Session,
    account: Account,
    *,
    request_ip: str = "",
    user_agent: str = "",
    family_id: uuid.UUID | None = None,
) -> SessionTokens:
    now = _now()
    session_id = uuid.uuid4()
    refresh_token = secrets.token_urlsafe(48)
    row = AuthSession(
        id=session_id,
        family_id=family_id or uuid.uuid4(),
        account_id=account.id,
        refresh_token_hash=_digest(refresh_token),
        expires_at=now + REFRESH_TTL,
        created_at=now,
        created_ip_hash=fingerprint(request_ip),
        user_agent_hash=fingerprint(user_agent),
    )
    db.add(row)
    db.flush()
    return SessionTokens(
        access_token=create_access_token(
            str(account.id),
            account.account_kind.value,
            session_id=str(session_id),
        ),
        refresh_token=refresh_token,
        session_id=session_id,
    )


def validate_access_session(db: Session, session_id: str, account_id: uuid.UUID) -> AuthSession:
    try:
        parsed_session_id = uuid.UUID(session_id)
    except (TypeError, ValueError) as exc:
        raise SessionError("Invalid session.") from exc
    row = db.get(AuthSession, parsed_session_id)
    if (
        row is None
        or row.account_id != account_id
        or row.revoked_at is not None
        or row.expires_at <= _now()
    ):
        raise SessionError("Invalid session.")
    return row


def rotate_refresh_session(
    db: Session,
    refresh_token: str,
    *,
    request_ip: str = "",
    user_agent: str = "",
) -> tuple[Account, SessionTokens]:
    now = _now()
    token_hash = _digest(refresh_token)
    row = db.execute(
        select(AuthSession).where(AuthSession.refresh_token_hash == token_hash).with_for_update()
    ).scalar_one_or_none()
    if row is None:
        raise SessionError("Invalid refresh session.")
    if row.revoked_at is not None:
        # Reuse of a rotated token revokes the whole family, containing theft.
        db.execute(
            update(AuthSession)
            .where(AuthSession.family_id == row.family_id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        db.commit()
        raise SessionError("Refresh token reuse detected.")
    if row.expires_at <= now:
        row.revoked_at = now
        db.commit()
        raise SessionError("Refresh session expired.")

    account = db.get(Account, row.account_id)
    if account is None or not account.is_active:
        row.revoked_at = now
        db.flush()
        raise SessionError("Invalid refresh session.")

    replacement = issue_session(
        db,
        account,
        request_ip=request_ip,
        user_agent=user_agent,
        family_id=row.family_id,
    )
    row.revoked_at = now
    row.last_used_at = now
    row.replaced_by_session_id = replacement.session_id
    db.flush()
    return account, replacement


def revoke_session(db: Session, session_id: str, account_id: uuid.UUID) -> None:
    row = validate_access_session(db, session_id, account_id)
    row.revoked_at = _now()
    db.flush()


def revoke_refresh_token(db: Session, refresh_token: str, account_id: uuid.UUID | None = None) -> None:
    row = db.execute(
        select(AuthSession).where(AuthSession.refresh_token_hash == _digest(refresh_token)).with_for_update()
    ).scalar_one_or_none()
    if row is None or (account_id is not None and row.account_id != account_id):
        return
    if row.revoked_at is None:
        row.revoked_at = _now()
    db.flush()


def revoke_all_account_sessions(db: Session, account_id: uuid.UUID) -> None:
    db.execute(
        update(AuthSession)
        .where(AuthSession.account_id == account_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=_now())
    )
    db.flush()
