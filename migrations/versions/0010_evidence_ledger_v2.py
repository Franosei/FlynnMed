"""Evidence Ledger v2: risk-of-bias, full-document flag, fact supersession,
module-tagged claims, and cross-source contradictions.

Closes several known Evidence Ledger limitations without a new subsystem:
- evidence_claims.risk_of_bias: deterministic RoB tag (backend/evidence_quality.py),
  independent of the LLM-self-reported study_design/certainty.
- source_artifacts.is_full_document: honesty flag -- always false today, since
  no fetch-the-entire-document step exists yet (see backend/evidence_ledger.py).
- patient_facts.previous_fact_id: explicit supersession chain, replacing the
  old created_at-ordering heuristic in backend/answer_claim_ledger.py.
- answer_claims.module / .llm_only_support: which subsystem (health_chat/
  safety_review/care_plan/trial_finder) wrote this claim, and whether a
  deterministic corroboration check overrode the LLM's own "supported" call.
- evidence_contradictions: new table for backend/contradiction_detector.py's
  same-intervention cross-source disagreement pairs.

Revision ID: 0010
Revises: 0009
Create Date: 2026-08-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "evidence_claims",
        sa.Column("risk_of_bias", sa.String(16), nullable=False, server_default="unclear"),
    )
    op.add_column(
        "source_artifacts",
        sa.Column("is_full_document", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "patient_facts",
        sa.Column(
            "previous_fact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patient_facts.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_patient_facts_previous_fact_id", "patient_facts", ["previous_fact_id"]
    )
    op.add_column(
        "answer_claims",
        sa.Column("module", sa.String(32), nullable=False, server_default="health_chat"),
    )
    op.add_column(
        "answer_claims",
        sa.Column("llm_only_support", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index("ix_answer_claims_module", "answer_claims", ["module"])

    op.create_table(
        "evidence_contradictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column(
            "source_a_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_b_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("topic", sa.Text, nullable=False),
        sa.Column("claim_a", sa.Text, nullable=False),
        sa.Column("claim_b", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_evidence_contradictions_trace_id", "evidence_contradictions", ["trace_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_evidence_contradictions_trace_id", table_name="evidence_contradictions")
    op.drop_table("evidence_contradictions")
    op.drop_index("ix_answer_claims_module", table_name="answer_claims")
    op.drop_column("answer_claims", "llm_only_support")
    op.drop_column("answer_claims", "module")
    op.drop_index("ix_patient_facts_previous_fact_id", table_name="patient_facts")
    op.drop_column("patient_facts", "previous_fact_id")
    op.drop_column("source_artifacts", "is_full_document")
    op.drop_column("evidence_claims", "risk_of_bias")
