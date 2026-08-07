"""previsit_summaries + previsit_chat_messages tables for the clinician
pre-visit summary workflow (auto-drafted chart summary + inline patient-
scoped chat, with an explicit clinician-controlled release gate before a
patient ever sees it), plus the new AuditAction values it emits.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NEW_AUDIT_ACTIONS = (
    "clinician_generate_previsit_summary",
    "clinician_edit_previsit_summary_draft",
    "clinician_release_previsit_summary",
    "clinician_chat_previsit",
)


def upgrade() -> None:
    # New enum values on the existing audit_action type. Postgres requires
    # this outside the DDL that uses the new value in the same transaction,
    # which table creation below does not do, so this is safe as written.
    for value in _NEW_AUDIT_ACTIONS:
        op.execute(f"ALTER TYPE audit_action ADD VALUE IF NOT EXISTS '{value}'")

    op.create_table(
        "previsit_summaries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("generation_trigger", sa.String(32), nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column(
            "authored_by_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("authored_by_display_name", sa.String(255), nullable=False),
        sa.Column("authored_by_clinical_role", sa.String(255), nullable=False),
        sa.Column("authored_by_organization", sa.String(255), nullable=False),
        sa.Column(
            "consent_grant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("consent_grants.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "released_by_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("released_by_display_name", sa.String(255), nullable=False),
        sa.Column("released_by_clinical_role", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_previsit_summaries_patient_id", "previsit_summaries", ["patient_id"])
    op.create_index(
        "ix_previsit_summaries_patient_created", "previsit_summaries", ["patient_id", "created_at"]
    )

    op.create_table(
        "previsit_chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "authored_by_account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("authored_by_display_name", sa.String(255), nullable=False),
        sa.Column("authored_by_clinical_role", sa.String(255), nullable=False),
        sa.Column(
            "consent_grant_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("consent_grants.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column("sources", postgresql.JSONB, nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_previsit_chat_messages_patient_id", "previsit_chat_messages", ["patient_id"])
    op.create_index(
        "ix_previsit_chat_patient_timestamp", "previsit_chat_messages", ["patient_id", "timestamp"]
    )


def downgrade() -> None:
    op.drop_table("previsit_chat_messages")
    op.drop_table("previsit_summaries")
    # Postgres has no ADD-VALUE-reversal (DROP VALUE doesn't exist without
    # recreating the enum type from scratch) -- the audit_action values added
    # in upgrade() intentionally remain after downgrade. This mirrors the
    # standard, well-known Postgres enum limitation, not an oversight.
