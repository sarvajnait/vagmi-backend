"""add_audio_fields_to_student_content

Revision ID: d6bda523532f
Revises: c9e0a304ea23
Create Date: 2026-02-15 18:41:02.004070

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd6bda523532f'
down_revision: Union[str, Sequence[str], None] = 'c9e0a304ea23'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('student_notes', sa.Column('audio_url', sa.String(), nullable=True))
    op.add_column('student_notes', sa.Column('audio_status', sa.String(length=20), nullable=True))
    op.add_column('student_textbooks', sa.Column('audio_url', sa.String(), nullable=True))
    op.add_column('student_textbooks', sa.Column('audio_status', sa.String(length=20), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('student_textbooks', 'audio_status')
    op.drop_column('student_textbooks', 'audio_url')
    op.drop_column('student_notes', 'audio_status')
    op.drop_column('student_notes', 'audio_url')
