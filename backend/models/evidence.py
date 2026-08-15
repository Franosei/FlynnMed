"""
Evidence Ledger Phase 1: persisted source identity + passage-level citation
linking. A SourceArtifact is an immutable snapshot of one retrieved
NHS/PubMed source at one point in time -- refetching the same URL with
unchanged content reuses the existing row (deduped on url+content_hash);
a genuinely changed source (a guideline update) creates a new row rather
than overwriting the old one, so a citation always points at the exact
version it was actually checked against.

An EvidencePassage is one verbatim excerpt from a SourceArtifact that an
extracted clinical fact was actually grounded in -- see
backend/evidence_ledger.py for the write path and backend/evidence_extractor.py
for where exact_quote text originates.

Phase 2 adds EvidenceClaim: a normalised clinical claim (Population,
Intervention, Comparator, Outcome), tagged with study design and certainty,
linked to the passage it was extracted from -- so a claim can eventually be
shown as "this is a high-certainty RCT finding", not just "this came from a
Tier 1 source". Only populated when the source actually states a genuine
comparative/interventional finding (see evidence_extractor.py's prompt) --
most sources (a plain NHS "how to take this medicine" page, for instance)
won't have one, and that's correct, not a gap.

Deliberately does not yet include PatientFact/AnswerClaim -- those are later
phases (patient-fact provenance, and full answer-to-evidence audit records
respectively).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin

# App-level constrained vocabularies (plain string columns + a Python set
# check at write time, matching this codebase's existing status/
# generation_trigger string-column convention on ProposedMedication -- not a
# DB CHECK constraint or enum type). An unrecognised value from the
# extraction LLM is coerced to "unknown" rather than rejected outright.
STUDY_DESIGNS = {
    "systematic_review",
    "meta_analysis",
    "rct",
    "cohort_study",
    "case_control",
    "case_report",
    "clinical_guideline",
    "narrative_review",
    "expert_opinion",
    "unknown",
}
# GRADE framework terminology -- the real-world evidence-based-medicine
# standard "study design and certainty" maps onto.
CERTAINTY_LEVELS = {"high", "moderate", "low", "very_low", "unknown"}
# Deterministic risk-of-bias tag, assigned from study_design alone (see
# backend/evidence_quality.py::assign_risk_of_bias) -- independent of the
# LLM-self-reported certainty above, not a substitute for a full RoB2/GRADE
# assessment (no imprecision/indirectness/inconsistency/publication-bias
# domains are evaluated).
RISK_OF_BIAS_LEVELS = {"low", "some_concerns", "high", "unclear"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceArtifact(Base, TimestampMixin):
    __tablename__ = "source_artifacts"
    __table_args__ = (
        UniqueConstraint("url", "content_hash", name="uq_source_artifact_url_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    url: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    publisher: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    # Best-effort, not authoritative -- most retrieved sources (NHS, PMC) don't
    # publish a machine-readable jurisdiction; left blank when unknown.
    jurisdiction: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    # The source's own stated publication/update date, often just a year --
    # kept as a string rather than a Date since precision varies by source.
    published_date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utc_now
    )
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    # Immutable as-fetched text this row's hash was computed from. Never
    # updated in place -- a changed source gets a new row (new content_hash).
    stored_snapshot_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Always False today -- no fetch-the-entire-document step exists anywhere
    # in this codebase (only domain-specific search + paragraph/section
    # fetch). Present so downstream/UI consumers can be honest about "this is
    # an excerpt" rather than silently implying full-document capture.
    is_full_document: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    passages: Mapped[list["EvidencePassage"]] = relationship(
        back_populates="source_artifact", cascade="all, delete-orphan"
    )
    claims: Mapped[list["EvidenceClaim"]] = relationship(
        back_populates="source_artifact", cascade="all, delete-orphan"
    )


class EvidencePassage(Base, TimestampMixin):
    __tablename__ = "evidence_passages"
    __table_args__ = (
        UniqueConstraint(
            "source_artifact_id", "passage_hash", name="uq_passage_source_hash"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("source_artifacts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    exact_text: Mapped[str] = mapped_column(Text, nullable=False)
    # A coarse "passage N of M extracted from this source" locator, not a true
    # page/paragraph reference -- NHS/PubMed excerpts aren't parsed into
    # page/section structure yet. Stated Phase 1 limitation, not a bug.
    locator: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    passage_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    source_artifact: Mapped["SourceArtifact"] = relationship(back_populates="passages")


class EvidenceClaim(Base, TimestampMixin):
    __tablename__ = "evidence_claims"
    __table_args__ = (
        UniqueConstraint(
            "source_artifact_id", "claim_hash", name="uq_claim_source_hash"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("source_artifacts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # SET NULL rather than CASCADE -- a claim can outlive the specific
    # passage row it was first linked to (e.g. a future re-extraction pass)
    # without the claim itself needing to disappear.
    passage_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence_passages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    population: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    intervention: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    comparator: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    outcome: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    study_design: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    certainty: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    risk_of_bias: Mapped[str] = mapped_column(String(16), nullable=False, default="unclear")
    claim_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    source_artifact: Mapped["SourceArtifact"] = relationship(back_populates="claims")
    passage: Mapped[Optional["EvidencePassage"]] = relationship()
