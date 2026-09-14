"""FastAPI auth/authorization dependencies for the SQL-backed path.

Not wired into backend/api.py's routes yet -- PR6 (cutover) replaces every
`username: str = Depends(current_user)` with `patient: Patient =
Depends(require_patient)` in one coordinated deploy, alongside running the
migration. Until then this module is exercised only by its own tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.jwt import TokenError, decode_access_token
from backend.auth.sessions import SessionError, validate_access_session
from backend.db import get_db
from backend.models.account import Account, AccountKind
from backend.models.audit import AuditAction, AuditLogEntry, AuditOutcome
from backend.models.consent import ConsentGrant, ConsentScope, ConsentStatus
from backend.models.patient import Patient


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def current_account(authorization: str = Header(default=""), db: Session = Depends(get_db)) -> Account:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing access token.")

    try:
        payload = decode_access_token(token)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail="Sign in again to continue.") from exc

    try:
        account_id = uuid.UUID(payload.account_id)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Sign in again to continue.") from exc

    account = db.get(Account, account_id)
    if account is None or not account.is_active or account.account_kind.value != payload.account_kind:
        raise HTTPException(status_code=401, detail="Sign in again to continue.")
    try:
        validate_access_session(db, payload.session_id, account.id)
    except SessionError as exc:
        raise HTTPException(status_code=401, detail="Sign in again to continue.") from exc
    return account


def require_patient(account: Account = Depends(current_account), db: Session = Depends(get_db)) -> Patient:
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="Verify your email address to continue.")
    if account.account_kind != AccountKind.patient:
        raise HTTPException(status_code=403, detail="Patient account required.")
    patient = db.execute(select(Patient).where(Patient.account_id == account.id)).scalar_one_or_none()
    if patient is None:
        raise HTTPException(status_code=403, detail="Patient account required.")
    return patient


def require_clinician(account: Account = Depends(current_account)) -> Account:
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="Verify your email address to continue.")
    if account.account_kind != AccountKind.clinician:
        raise HTTPException(status_code=403, detail="Clinician account required.")
    return account


@dataclass(frozen=True)
class AuthorizedPatientContext:
    patient: Patient
    clinician: Account
    grant: ConsentGrant


def _write_audit(
    db: Session,
    *,
    actor: Account,
    patient_id: Optional[uuid.UUID],
    action: AuditAction,
    outcome: AuditOutcome,
    resource_type: str,
    resource_id: Optional[str],
    consent_grant_id: Optional[uuid.UUID],
    request_ip: Optional[str],
) -> None:
    db.add(
        AuditLogEntry(
            actor_account_id=actor.id,
            actor_role_at_time=actor.account_kind.value,
            patient_id=patient_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            outcome=outcome,
            consent_grant_id=consent_grant_id,
            request_ip=request_ip,
        )
    )
    db.flush()


def authorize_patient_access(required_scope: ConsentScope, action: AuditAction):
    """Factory for the single choke point every cross-patient read must go
    through: `Depends(authorize_patient_access(ConsentScope.previsit_summary,
    AuditAction.clinician_read_previsit_summary))`. No route may accept a
    `patient_id` path param without depending on this. Unknown MRNs and
    denied grants return the identical 403 (no existence leak); both
    denials and successes are audit-logged before any data is returned."""

    def _dependency(
        patient_id: str,
        request: Request,
        clinician: Account = Depends(require_clinician),
        db: Session = Depends(get_db),
    ) -> AuthorizedPatientContext:
        request_ip = request.client.host if request.client else None
        patient = db.execute(select(Patient).where(Patient.patient_id == patient_id)).scalar_one_or_none()

        if patient is None:
            _write_audit(
                db,
                actor=clinician,
                patient_id=None,
                action=action,
                outcome=AuditOutcome.denied,
                resource_type="patient",
                resource_id=patient_id,
                consent_grant_id=None,
                request_ip=request_ip,
            )
            raise HTTPException(status_code=403, detail="No active access grant for this patient.")

        grant = db.execute(
            select(ConsentGrant).where(
                ConsentGrant.patient_id == patient.id,
                ConsentGrant.clinician_account_id == clinician.id,
                ConsentGrant.status == ConsentStatus.active,
            )
        ).scalar_one_or_none()

        grant_valid = (
            grant is not None
            and (grant.expires_at is None or grant.expires_at > _utc_now())
            and required_scope.value in (grant.scope or [])
        )

        if not grant_valid:
            _write_audit(
                db,
                actor=clinician,
                patient_id=patient.id,
                action=action,
                outcome=AuditOutcome.denied,
                resource_type="patient",
                resource_id=patient_id,
                consent_grant_id=grant.id if grant else None,
                request_ip=request_ip,
            )
            raise HTTPException(status_code=403, detail="No active access grant for this patient.")

        _write_audit(
            db,
            actor=clinician,
            patient_id=patient.id,
            action=action,
            outcome=AuditOutcome.success,
            resource_type="patient",
            resource_id=patient_id,
            consent_grant_id=grant.id,
            request_ip=request_ip,
        )
        return AuthorizedPatientContext(patient=patient, clinician=clinician, grant=grant)

    return _dependency
