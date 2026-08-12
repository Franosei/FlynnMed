from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import BigInteger, DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.models.base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuditAction(str, enum.Enum):
    login = "login"
    patient_read_self = "patient_read_self"
    clinician_access_requested = "clinician_access_requested"
    clinician_access_granted = "clinician_access_granted"
    clinician_access_denied = "clinician_access_denied"
    clinician_access_revoked = "clinician_access_revoked"
    clinician_read_previsit_summary = "clinician_read_previsit_summary"
    clinician_read_gp_prep = "clinician_read_gp_prep"
    clinician_read_chat_history = "clinician_read_chat_history"
    clinician_read_clinical_notes = "clinician_read_clinical_notes"
    clinician_generate_previsit_summary = "clinician_generate_previsit_summary"
    clinician_edit_previsit_summary_draft = "clinician_edit_previsit_summary_draft"
    clinician_release_previsit_summary = "clinician_release_previsit_summary"
    clinician_chat_previsit = "clinician_chat_previsit"
    clinician_generate_medication_proposal = "clinician_generate_medication_proposal"
    clinician_edit_medication_proposal_draft = "clinician_edit_medication_proposal_draft"
    clinician_release_medication_proposal = "clinician_release_medication_proposal"


class AuditOutcome(str, enum.Enum):
    success = "success"
    denied = "denied"
    error = "error"


class AuditLogEntry(Base):
    """Append-only cross-patient access log. This is distinct from
    backend/audit_models.py, which records per-response AI governance
    metadata (risk level, moderation, evidence tiers) -- not "who accessed
    whose record." No `updated_at` column and no ORM update path is exposed
    on this model; immutability is additionally enforced at the DB role
    level (the app's runtime DB role is granted INSERT, SELECT only on this
    table -- see migrations/versions/0001_initial_schema.py)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    # Snapshot, not a live FK-joined lookup -- roles can change after the fact.
    actor_role_at_time: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    patient_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="SET NULL"), nullable=True, index=True
    )
    action: Mapped[AuditAction] = mapped_column(Enum(AuditAction, name="audit_action"), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    resource_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    outcome: Mapped[AuditOutcome] = mapped_column(Enum(AuditOutcome, name="audit_outcome"), nullable=False)
    consent_grant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("consent_grants.id", ondelete="SET NULL"), nullable=True
    )
    request_ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
