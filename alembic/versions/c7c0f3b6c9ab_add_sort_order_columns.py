"""add sort_order columns for content ordering

Revision ID: c7c0f3b6c9ab
Revises: b5d01a74a710
Create Date: 2025-12-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c7c0f3b6c9ab"
down_revision: Union[str, Sequence[str], None] = "b5d01a74a710"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_textbooks", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("additional_notes", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("llm_images", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("llm_notes", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("qa_patterns", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("student_textbooks", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("student_notes", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column("student_videos", sa.Column("sort_order", sa.Integer(), nullable=True))
    op.add_column(
        "previous_year_question_papers",
        sa.Column("sort_order", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("previous_year_question_papers", "sort_order")
    op.drop_column("student_videos", "sort_order")
    op.drop_column("student_notes", "sort_order")
    op.drop_column("student_textbooks", "sort_order")
    op.drop_column("qa_patterns", "sort_order")
    op.drop_column("llm_notes", "sort_order")
    op.drop_column("llm_images", "sort_order")
    op.drop_column("additional_notes", "sort_order")
    op.drop_column("llm_textbooks", "sort_order")
