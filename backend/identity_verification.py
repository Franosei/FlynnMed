"""Server-side email and clinician identity verification workflows."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from backend.auth.sessions import revoke_all_account_sessions
from backend.config import jwt_secret_key
from backend.email_service import send_verification_email
from backend.models.account import Account, AccountKind
from backend.models.security import ClinicianRegistration, EmailVerificationChallenge

CODE_TTL = timedelta(minutes=15)
MAX_ATTEMPTS = 5


class VerificationError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _code_digest(account_id: uuid.UUID, code: str) -> str:
    return hmac.new(
        jwt_secret_key().encode("utf-8"),
        f"email-verification:{account_id}:{code}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def create_email_challenge(db: Session, account: Account) -> str:
    """Invalidate older codes and persist a new short-lived, hashed code."""
    now = _now()
    db.execute(
        update(EmailVerificationChallenge)
        .where(
            EmailVerificationChallenge.account_id == account.id,
            EmailVerificationChallenge.consumed_at.is_(None),
        )
        .values(consumed_at=now)
    )
    code = f"{secrets.randbelow(1_000_000):06d}"
    db.add(
        EmailVerificationChallenge(
            id=uuid.uuid4(),
            account_id=account.id,
            code_hash=_code_digest(account.id, code),
            expires_at=now + CODE_TTL,
            attempts=0,
            created_at=now,
        )
    )
    db.flush()
    return code


def deliver_email_challenge(account: Account, code: str) -> None:
    send_verification_email(account.email, account.display_name, code)


def verify_email_code(db: Session, account: Account, code: str) -> None:
    if account.email_verified:
        return
    now = _now()
    challenge = db.execute(
        select(EmailVerificationChallenge)
        .where(
            EmailVerificationChallenge.account_id == account.id,
            EmailVerificationChallenge.consumed_at.is_(None),
        )
        .order_by(EmailVerificationChallenge.created_at.desc())
        .with_for_update()
    ).scalars().first()
    if challenge is None or challenge.expires_at <= now:
        raise VerificationError("Verification code expired. Request a new code.")
    if challenge.attempts >= MAX_ATTEMPTS:
        challenge.consumed_at = now
        db.commit()
        raise VerificationError("Too many attempts. Request a new code.")
    challenge.attempts += 1
    if not hmac.compare_digest(challenge.code_hash, _code_digest(account.id, code.strip())):
        if challenge.attempts >= MAX_ATTEMPTS:
            challenge.consumed_at = now
        db.commit()
        raise VerificationError("The verification code is incorrect.")
    challenge.consumed_at = now
    account.email_verified = True
    db.flush()


def submit_clinician_registration(
    db: Session,
    account: Account,
    *,
    requested_role: str,
    organization: str,
    registration_number: str,
    registration_country: str,
    require_verified_email: bool = True,
) -> ClinicianRegistration:
    if require_verified_email and not account.email_verified:
        raise VerificationError("Verify your email before applying for clinician access.")
    values = [requested_role, organization, registration_number, registration_country]
    if any(not value.strip() for value in values):
        raise VerificationError("Role, organization, registration number, and country are required.")
    row = db.execute(
        select(ClinicianRegistration).where(ClinicianRegistration.account_id == account.id)
    ).scalar_one_or_none()
    if row and row.status == "approved":
        raise VerificationError("This clinician registration is already approved.")
    if row is None:
        row = ClinicianRegistration(id=uuid.uuid4(), account_id=account.id)
        db.add(row)
    row.requested_role = requested_role.strip()
    row.organization = organization.strip()
    row.registration_number = registration_number.strip()
    row.registration_country = registration_country.strip()
    row.status = "pending"
    row.review_note = ""
    row.reviewed_at = None
    row.reviewed_by = ""
    db.flush()
    return row


def review_clinician_registration(
    db: Session,
    registration_id: uuid.UUID,
    *,
    approve: bool,
    note: str,
    reviewer: str,
) -> ClinicianRegistration:
    row = db.execute(
        select(ClinicianRegistration)
        .where(ClinicianRegistration.id == registration_id)
        .with_for_update()
    ).scalar_one_or_none()
    if row is None:
        raise VerificationError("Clinician registration not found.")
    if row.status != "pending":
        raise VerificationError("Clinician registration has already been reviewed.")
    account = db.get(Account, row.account_id)
    if account is None or not account.is_active:
        raise VerificationError("Account is unavailable.")
    row.status = "approved" if approve else "rejected"
    row.review_note = note.strip()
    row.reviewed_at = _now()
    row.reviewed_by = reviewer
    if approve:
        account.account_kind = AccountKind.clinician
        account.clinical_role = row.requested_role
        account.role_label = row.requested_role
        account.organization = row.organization
    revoke_all_account_sessions(db, account.id)
    db.flush()
    return row


def require_clinician_admin_key(candidate: str) -> None:
    expected = os.getenv("CLINICIAN_VERIFICATION_ADMIN_KEY", "")
    if not expected or not hmac.compare_digest(candidate or "", expected):
        raise VerificationError("Clinician verification administrator credentials are invalid.")
