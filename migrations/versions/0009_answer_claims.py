"""Evidence Ledger Phase 4: answer_claims table.

An append-only audit log of each claim check_claim_source_alignment found in
a generated answer, with its final status classification and best-effort
links to the EvidenceClaim/PatientFact rows that may have backed it. Unlike
source_artifacts/evidence_passages/evidence_claims/patient_facts, this table
is NOT deduplicated -- every answer instance gets its own rows, even for a
repeated question. See backend/models/answer_claim.py and
backend/answer_claim_ledger.py.

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "answer_claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("claim_text", sa.Text, nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("requires_evidence", sa.Boolean, nullable=False),
        sa.Column("source_ids", postgresql.JSONB, nullable=False),
        sa.Column("evidence_claim_ids", postgresql.JSONB, nullable=False),
        sa.Column("patient_fact_ids", postgresql.JSONB, nullable=False),
        sa.Column("rule_version", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_answer_claims_trace_id", "answer_claims", ["trace_id"])
    op.create_index("ix_answer_claims_patient_id", "answer_claims", ["patient_id"])


def downgrade() -> None:
    op.drop_table("answer_claims")
