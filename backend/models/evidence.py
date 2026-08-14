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

Deliberately does not yet include EvidenceClaim/PatientFact/AnswerClaim --
those are later phases (normalised PICO claims, patient-fact provenance,
and full answer-to-evidence audit records respectively).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin


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

    passages: Mapped[list["EvidencePassage"]] = relationship(
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
