"""
Evidence Ledger v2: EvidenceContradiction -- a scoped, same-intervention
pairwise disagreement between two SourceArtifact rows, detected by
backend/contradiction_detector.py. Deliberately narrow: this is not a
general reconciliation engine (no jurisdiction-aware comparison, no
narrative synthesis beyond a single pairwise judgment per intervention
group) -- see contradiction_detector.py's module docstring for what's
actually being detected and what isn't.
"""
from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.models.base import Base, TimestampMixin


class EvidenceContradiction(Base, TimestampMixin):
    __tablename__ = "evidence_contradictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_a_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_artifacts.id", ondelete="CASCADE"), nullable=False
    )
    source_b_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("source_artifacts.id", ondelete="CASCADE"), nullable=False
    )
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    claim_a: Mapped[str] = mapped_column(Text, nullable=False)
    claim_b: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    source_a: Mapped["SourceArtifact"] = relationship(foreign_keys=[source_a_id])  # noqa: F821
    source_b: Mapped["SourceArtifact"] = relationship(foreign_keys=[source_b_id])  # noqa: F821
