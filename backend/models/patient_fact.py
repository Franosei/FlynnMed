"""
Evidence Ledger Phase 3: PatientFact -- the patient-side counterpart to
SourceArtifact/EvidencePassage/EvidenceClaim (backend/models/evidence.py). A
snapshot of one patient fact (a condition, medication, allergy, vital, or
symptom) as it stood when an answer was generated, with explicit status
(confirmed/suspected/inferred/unknown) and source provenance -- so a future
AnswerClaim (Phase 4) can say "this used your recorded penicillin allergy",
not just silently read from a live, mutable Condition/Medication/Allergy/
VitalsEntry row that could since have been edited.

Deliberately additive: does NOT add provenance columns to Condition/
Medication/Allergy/VitalsEntry/SymptomLog themselves, or touch their write
paths (backend/api.py, backend/repositories/sql_user_store.py) -- those
tables remain the live, editable source of truth a PatientFact snapshots
from. See backend/patient_fact_ledger.py for the write path.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin

# App-level constrained vocabularies (plain string columns + a Python set
# check at write time, matching STUDY_DESIGNS/CERTAINTY_LEVELS in
# backend/models/evidence.py and ProposedMedication's status/
# generation_trigger convention). An unrecognised value is coerced rather
# than rejected outright.
CATEGORIES = {"condition", "medication", "allergy", "vital", "symptom", "unknown"}
# "inferred" is reserved for a future conversation-derived-fact pipeline --
# not populated by this phase (see backend/patient_fact_ledger.py).
# "retracted" marks a fact whose label was live in a previous snapshot but is
# absent from the patient's current structured record (deleted, or an
# allergy/condition later disproved) -- see persist_patient_facts_for_bundle's
# retraction pass.
FACT_STATUSES = {"confirmed", "suspected", "inferred", "retracted", "unknown"}
# "clinician_entered" is reserved -- no write path today distinguishes
# clinician-entered from patient-entered data (see backend/api.py's
# save_condition/save_medication/save_allergy/save_vitals/add_symptom, all
# behind a single current_user dependency with no role split).
# "document_extracted" is populated for facts persisted during a
# document-analysis chat turn (see backend/rag_system.py's
# stream_document_analysis_events path).
FACT_SOURCES = {
    "structured_patient_record",
    "conversation_inferred",
    "clinician_entered",
    "document_extracted",
}


class PatientFact(Base, TimestampMixin):
    __tablename__ = "patient_facts"
    __table_args__ = (
        UniqueConstraint("patient_id", "fact_hash", name="uq_patient_fact_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True
    )
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    unit: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="confirmed")
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="structured_patient_record")
    # Loose reference back to the originating Condition/Medication/Allergy/
    # VitalsEntry/SymptomLog row -- no FK constraint, since that row can be
    # deleted by the patient while this snapshot (and any AnswerClaim citing
    # it) must remain valid, matching SourceArtifact's immutable-snapshot
    # precedent for external sources.
    source_record_type: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    source_record_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    # Best-effort "when was this recorded" -- the source row's created_at,
    # not a clinically-verified date (no such date exists on the source
    # tables today).
    recorded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    fact_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Explicit supersession chain: set when this row's insert represents an
    # edit (changed value/status) or retraction of an earlier row for the
    # same (patient_id, category, label) -- see
    # backend/patient_fact_ledger.py::persist_patient_fact. SET NULL rather
    # than CASCADE so deleting an old snapshot never cascades into deleting
    # its successor.
    previous_fact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_facts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    patient: Mapped["Patient"] = relationship()  # noqa: F821
