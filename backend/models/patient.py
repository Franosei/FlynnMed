from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin


class Patient(Base, TimestampMixin):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    # The MRN -- durable, clinician-facing lookup key. See backend/mrn.py.
    patient_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False, index=True)

    date_of_birth: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    biological_sex: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    dob_recorded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Free-form AI-maintained rolling summary -- shape tracks prompt engineering,
    # not query needs, so it stays JSONB rather than normalized columns.
    longitudinal_memory: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Single most-recent-result caches (not history) -- legacy
    # user_store.py's `last_video_generated_at` (rate-limit timestamp) and
    # `last_trial_search` (cached clinical-trials search result) fields.
    last_video_generated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_trial_search: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    account: Mapped["Account"] = relationship(back_populates="patient")  # noqa: F821
    medications: Mapped[list["Medication"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    conditions: Mapped[list["Condition"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    allergies: Mapped[list["Allergy"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    vitals: Mapped[list["VitalsEntry"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    symptom_logs: Mapped[list["SymptomLog"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    chat_messages: Mapped[list["ChatMessage"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    care_plans: Mapped[list["CarePlan"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    clinical_notes: Mapped[list["ClinicalNote"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    uploads: Mapped[list["Upload"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    triage_summaries: Mapped[list["TriageSummary"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    interaction_traces: Mapped[list["InteractionTrace"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    previsit_summaries: Mapped[list["PreVisitSummary"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    previsit_chat_messages: Mapped[list["PreVisitChatMessage"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    proposed_medications: Mapped[list["ProposedMedication"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Medication(Base, TimestampMixin):
    __tablename__ = "medications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dose: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    schedule: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    reason: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    started_on: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="medications")


class Condition(Base, TimestampMixin):
    __tablename__ = "conditions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    recorded_on: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="conditions")


class Allergy(Base, TimestampMixin):
    __tablename__ = "allergies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    reaction: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    allergy_type: Mapped[str] = mapped_column(String(32), nullable=False, default="other")
    confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="allergies")


class VitalsEntry(Base, TimestampMixin):
    __tablename__ = "vitals_entries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    recorded_on: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(String(64), nullable=False)
    unit: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="vitals")

    __table_args__ = (Index("ix_vitals_patient_recorded", "patient_id", "recorded_on"),)


class SymptomLog(Base, TimestampMixin):
    __tablename__ = "symptom_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    symptom: Mapped[str] = mapped_column(String(255), nullable=False)
    logged_for: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    severity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    triggers: Mapped[str] = mapped_column(Text, nullable=False, default="")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="symptom_logs")


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Grab-bag of evolving AI-shaped fields (personal_context, triage_summary,
    # trace, image/video refs, feedback) -- always read/written as one unit.
    message_metadata: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    patient: Mapped["Patient"] = relationship(back_populates="chat_messages")

    __table_args__ = (Index("ix_chat_patient_timestamp", "patient_id", "timestamp"),)


class CarePlan(Base, TimestampMixin):
    __tablename__ = "care_plans"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    condition: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    # goals/daily_tasks/weekly_tasks/medication_reminders/lab_reminders/
    # escalation_thresholds/missed_care_checklist/after_visit_notes/lifestyle --
    # exactly the sub-keys care_plan_store.py's _stamp_ids walks today.
    body: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    clinical_context: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    validation: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    gp_prep_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="care_plans")


class ClinicalNote(Base, TimestampMixin):
    __tablename__ = "clinical_notes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    subjective: Mapped[str] = mapped_column(Text, nullable=False, default="")
    objective: Mapped[str] = mapped_column(Text, nullable=False, default="")
    assessment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    plan: Mapped[str] = mapped_column(Text, nullable=False, default="")
    urgency_level: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    requires_gp_visit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gp_visit_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    edited_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    email_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    email_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="clinical_notes")


class Upload(Base, TimestampMixin):
    __tablename__ = "uploads"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    summary_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    patient: Mapped["Patient"] = relationship(back_populates="uploads")
    document_summary: Mapped[Optional["DocumentSummary"]] = relationship(
        back_populates="upload", uselist=False, cascade="all, delete-orphan"
    )


class DocumentSummary(Base, TimestampMixin):
    __tablename__ = "document_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    upload_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("uploads.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    upload: Mapped["Upload"] = relationship(back_populates="document_summary")


class TriageSummary(Base, TimestampMixin):
    __tablename__ = "triage_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    urgency_level: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    next_step: Mapped[str] = mapped_column(Text, nullable=False, default="")
    what_to_monitor: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    pathway_label: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    decision_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    immediate_actions: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    escalation_triggers: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    communication_points: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    rule_hits: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    guideline_references: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    logic_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    patient: Mapped["Patient"] = relationship(back_populates="triage_summaries")


class InteractionTrace(Base, TimestampMixin):
    __tablename__ = "interaction_traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Grab-bag payload (question, answer_preview, sources, and any other ad
    # hoc fields the trace carries) -- same rationale as ChatMessage.message_metadata.
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    patient: Mapped["Patient"] = relationship(back_populates="interaction_traces")


class PreVisitSummary(Base, TimestampMixin):
    """
    Append-only: every generate/edit/release inserts a new row rather than
    updating one in place, so the full history survives for future reference
    and for continuity of care across clinicians. "Current draft" is the
    latest status="draft" row; "current released" is the latest
    status="released" row; full history is just patient_id + created_at order.

    authored_by_* / released_by_* are snapshot columns, not re-derived via the
    account FK -- unlike ClinicalNote.edited_by_account_id (which the SQL
    repository never actually populates), these are always written at insert
    time so "who wrote this" survives even if the account is later renamed or
    deleted (the FK itself is ON DELETE SET NULL).
    """
    __tablename__ = "previsit_summaries"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")  # draft | released
    generation_trigger: Mapped[str] = mapped_column(String(32), nullable=False, default="ai_generated")
    # ai_generated | clinician_edited | released
    summary_text: Mapped[str] = mapped_column(Text, nullable=False, default="")

    authored_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    authored_by_display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    authored_by_clinical_role: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    authored_by_organization: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    consent_grant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("consent_grants.id", ondelete="SET NULL"), nullable=True
    )

    released_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    released_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    released_by_display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    released_by_clinical_role: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="previsit_summaries")

    __table_args__ = (Index("ix_previsit_summaries_patient_created", "patient_id", "created_at"),)


class PreVisitChatMessage(Base, TimestampMixin):
    """
    The clinician-scoped, patient-specific exploratory chat used to inform a
    PreVisitSummary. Kept separate from PreVisitSummary rows (not nested under
    one) because the transcript is continuous per patient across multiple
    generate/regenerate cycles, not scoped to a single summary version.
    """
    __tablename__ = "previsit_chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # clinician | assistant
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")

    authored_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    authored_by_display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    authored_by_clinical_role: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    consent_grant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("consent_grants.id", ondelete="SET NULL"), nullable=True
    )
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    patient: Mapped["Patient"] = relationship(back_populates="previsit_chat_messages")

    __table_args__ = (Index("ix_previsit_chat_patient_timestamp", "patient_id", "timestamp"),)


class ProposedMedication(Base, TimestampMixin):
    """
    Append-only, same shape/philosophy as PreVisitSummary: every generate/
    edit/release inserts a new row rather than updating one in place, so the
    full history survives for audit and continuity of care. "Current draft"
    is the latest status="draft" row; "current released" is the latest
    status="released" row.

    status stays binary (draft/released) -- safety_check is not a workflow
    gate state, it's data always computed and attached to every draft row
    (both ai_generated and clinician_edited), never a separate status value.
    Deliberately no denormalized "has conflicts" boolean column: that would
    be able to drift from safety_check itself, which is exactly the kind of
    footgun a safety feature shouldn't have -- callers derive it from
    safety_check on the fly instead.
    """
    __tablename__ = "proposed_medications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")  # draft | released
    generation_trigger: Mapped[str] = mapped_column(String(32), nullable=False, default="ai_generated")
    # ai_generated | clinician_edited | released

    clinical_situation_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    candidate_medication_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    candidate_dose_frequency: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    rationale_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Source citations (combined_sources shape: source_id, title, url, ...).
    citations: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # {"allergy_flags": [...], "interaction_flags": [...],
    #  "unresolved_medications": [...], "checked_at": iso-str}
    safety_check: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    # Populated only when released despite an unresolved safety_check flag.
    override_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    trace_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    authored_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    authored_by_display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    authored_by_clinical_role: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    authored_by_organization: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    consent_grant_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("consent_grants.id", ondelete="SET NULL"), nullable=True
    )

    released_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    released_by_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True
    )
    released_by_display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    released_by_clinical_role: Mapped[str] = mapped_column(String(255), nullable=False, default="")

    patient: Mapped["Patient"] = relationship(back_populates="proposed_medications")

    __table_args__ = (Index("ix_proposed_medications_patient_created", "patient_id", "created_at"),)
