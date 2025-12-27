"""add original_filename columns for file resources

Revision ID: f3b7a9b0e4c1
Revises: 65150b1c1b0f
Create Date: 2025-12-29 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f3b7a9b0e4c1"
down_revision: Union[str, Sequence[str], None] = "65150b1c1b0f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_textbooks", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column("llm_images", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column("llm_notes", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column("qa_patterns", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column("student_textbooks", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column("student_notes", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column("student_videos", sa.Column("original_filename", sa.String(), nullable=True))
    op.add_column(
        "previous_year_question_papers",
        sa.Column("original_filename", sa.String(), nullable=True),
    )

    conn = op.get_bind()
    tables = [
        "llm_textbooks",
        "llm_images",
        "llm_notes",
        "qa_patterns",
        "student_textbooks",
        "student_notes",
        "student_videos",
        "previous_year_question_papers",
    ]
    for table in tables:
        conn.execute(
            sa.text(
                f"""
                UPDATE {table}
                SET original_filename = split_part(file_url, '/', -1)
                WHERE original_filename IS NULL AND file_url IS NOT NULL
                """
            )
        )


def downgrade() -> None:
    op.drop_column("previous_year_question_papers", "original_filename")
    op.drop_column("student_videos", "original_filename")
    op.drop_column("student_notes", "original_filename")
    op.drop_column("student_textbooks", "original_filename")
    op.drop_column("qa_patterns", "original_filename")
    op.drop_column("llm_notes", "original_filename")
    op.drop_column("llm_images", "original_filename")
    op.drop_column("llm_textbooks", "original_filename")
