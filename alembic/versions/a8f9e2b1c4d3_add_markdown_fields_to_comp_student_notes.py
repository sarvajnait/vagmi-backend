"""add markdown fields to comp_student_notes

Revision ID: b2c3d4e5f6a7
Revises: e1f2a3b4c5d6
Branch Labels: None
Depends On: None

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a8f9e2b1c4d3"
down_revision: Union[str, Sequence[str], None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_cols = {c["name"] for c in inspector.get_columns("comp_student_notes")}

    if "content" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("content", sa.Text(), nullable=True))
    if "content_status" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("content_status", sa.String(), nullable=True))
    if "is_published" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("is_published", sa.Boolean(), nullable=False, server_default="false"))
    if "version" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("version", sa.Integer(), nullable=False, server_default="1"))
    if "word_count" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("word_count", sa.Integer(), nullable=True))
    if "read_time_min" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("read_time_min", sa.Integer(), nullable=True))
    if "source" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("source", sa.String(), nullable=True))
    if "language" not in existing_cols:
        op.add_column("comp_student_notes", sa.Column("language", sa.String(), nullable=False, server_default="en"))

    # Make file_url nullable for future manual markdown entry
    op.alter_column("comp_student_notes", "file_url", nullable=True)


def downgrade() -> None:
    op.drop_column("comp_student_notes", "content")
    op.drop_column("comp_student_notes", "content_status")
    op.drop_column("comp_student_notes", "is_published")
    op.drop_column("comp_student_notes", "version")
    op.drop_column("comp_student_notes", "word_count")
    op.drop_column("comp_student_notes", "read_time_min")
    op.drop_column("comp_student_notes", "source")
    op.drop_column("comp_student_notes", "language")
    op.alter_column("comp_student_notes", "file_url", nullable=False)
