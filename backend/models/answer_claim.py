"""
Evidence Ledger Phase 4: AnswerClaim -- the final object in the five-object
evidence-lineage model. Persists what backend/summarizer.py's
check_claim_source_alignment/apply_claim_corrections already compute on
every answer (see backend/answer_claim_ledger.py for the write path), so a
specific sentence shown to the user can eventually be traced to exactly
which EvidenceClaim/PatientFact rows backed it, instead of that
verification living only inside an unstructured trace JSON blob.

Structurally different from SourceArtifact/EvidencePassage/EvidenceClaim/
PatientFact: those are deduplicated fact/evidence graph nodes (the same
content reuses one row across many answers). AnswerClaim is an APPEND-ONLY
AUDIT LOG -- one row per claim per answer instance, even for a repeated
question -- because a true audit trail records what happened each time, not
just the current state of a fact. There is deliberately no content-hash
UniqueConstraint here.
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin

ANSWER_CLAIM_STATUSES = {
    "supported_cited",                          # supported, already carried its [S#] marker
    "supported_citation_added",                 # supported, marker inserted by apply_claim_corrections
    "unsupported_hedged",                        # flagged unsupported, text was rewritten to hedge it
    "unsupported_uncorrected",                   # flagged unsupported, but no rewrite happened (correction call failed/no-op)
    "general_knowledge_no_evidence_required",    # general knowledge that never needed a citation -- not a violation
    "unknown",
}
# Bump when the classification logic in backend/answer_claim_ledger.py changes.
ANSWER_CLAIM_RULE_VERSION = "2026.08-phase4-v1"


class AnswerClaim(Base, TimestampMixin):
    __tablename__ = "answer_claims"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Loose reference to InteractionTrace.trace_id (backend/models/patient.py) --
    # not a real FK. InteractionTrace is only ever saved for self-service patient
    # chats (UserStore.save_interaction_trace resolves the *acting* username to a
    # Patient row, which is None for a clinician account), so an AnswerClaim from a
    # clinician's patient-scoped chat would otherwise have nothing to link against.
    # Same loose-reference convention as ProposedMedication.trace_id.
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=True, index=True
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, default="unknown")
    requires_evidence: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    evidence_claim_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    patient_fact_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    rule_version: Mapped[str] = mapped_column(String(32), nullable=False, default=ANSWER_CLAIM_RULE_VERSION)

    patient: Mapped[Optional["Patient"]] = relationship()  # noqa: F821
