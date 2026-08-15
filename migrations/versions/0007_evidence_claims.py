"""Evidence Ledger Phase 2: evidence_claims table.

A normalised clinical claim (Population, Intervention, Comparator, Outcome),
tagged with study design and certainty, linked to the passage it was
extracted from -- so a claim can eventually be shown as "this is a
high-certainty RCT finding", not just "this came from a Tier 1 source". Only
populated when a source states a genuine comparative/interventional finding;
most sources won't have one. See backend/models/evidence.py and
backend/evidence_ledger.py.

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evidence_claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "passage_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("evidence_passages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("population", sa.String(512), nullable=False),
        sa.Column("intervention", sa.String(512), nullable=False),
        sa.Column("comparator", sa.String(512), nullable=False),
        sa.Column("outcome", sa.String(512), nullable=False),
        sa.Column("study_design", sa.String(64), nullable=False),
        sa.Column("certainty", sa.String(32), nullable=False),
        sa.Column("claim_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_artifact_id", "claim_hash", name="uq_claim_source_hash"
        ),
    )
    op.create_index(
        "ix_evidence_claims_source_artifact_id", "evidence_claims", ["source_artifact_id"]
    )
    op.create_index("ix_evidence_claims_passage_id", "evidence_claims", ["passage_id"])
    op.create_index("ix_evidence_claims_claim_hash", "evidence_claims", ["claim_hash"])


def downgrade() -> None:
    op.drop_table("evidence_claims")
