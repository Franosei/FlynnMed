"""Evidence Ledger Phase 1: source_artifacts and evidence_passages tables.

An immutable, deduplicated record of retrieved sources (identity, version,
retrieval time, content hash) and the exact passages within them that
extracted clinical facts were actually grounded in -- so a citation can be
verified down to a specific passage of a specific source version, not just
"this whole page". See backend/models/evidence.py and
backend/evidence_ledger.py.

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "source_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("title", sa.String(1024), nullable=False),
        sa.Column("publisher", sa.String(256), nullable=False),
        sa.Column("jurisdiction", sa.String(64), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("published_date", sa.String(32), nullable=True),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_hash", sa.String(128), nullable=False),
        sa.Column("stored_snapshot_text", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("url", "content_hash", name="uq_source_artifact_url_hash"),
    )
    op.create_index("ix_source_artifacts_url", "source_artifacts", ["url"])
    op.create_index("ix_source_artifacts_content_hash", "source_artifacts", ["content_hash"])

    op.create_table(
        "evidence_passages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_artifact_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("source_artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("exact_text", sa.Text, nullable=False),
        sa.Column("locator", sa.String(256), nullable=False),
        sa.Column("passage_hash", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "source_artifact_id", "passage_hash", name="uq_passage_source_hash"
        ),
    )
    op.create_index(
        "ix_evidence_passages_source_artifact_id", "evidence_passages", ["source_artifact_id"]
    )
    op.create_index("ix_evidence_passages_passage_hash", "evidence_passages", ["passage_hash"])


def downgrade() -> None:
    op.drop_table("evidence_passages")
    op.drop_table("source_artifacts")
