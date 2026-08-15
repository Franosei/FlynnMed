"""Evidence Ledger Phase 3: patient_facts table.

A snapshot of one patient fact (condition/medication/allergy/vital/symptom)
as it stood when an answer was generated, with an explicit status
(confirmed/suspected/inferred/unknown) and source. The patient-side
counterpart to source_artifacts/evidence_passages/evidence_claims. See
backend/models/patient_fact.py and backend/patient_fact_ledger.py.

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "patient_facts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("category", sa.String(32), nullable=False),
        sa.Column("label", sa.String(255), nullable=False),
        sa.Column("value", sa.String(512), nullable=False),
        sa.Column("unit", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("source_record_type", sa.String(32), nullable=False),
        sa.Column("source_record_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("fact_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("patient_id", "fact_hash", name="uq_patient_fact_hash"),
    )
    op.create_index("ix_patient_facts_patient_id", "patient_facts", ["patient_id"])
    op.create_index("ix_patient_facts_fact_hash", "patient_facts", ["fact_hash"])


def downgrade() -> None:
    op.drop_table("patient_facts")
