"""add previous year question papers table

Revision ID: b5d01a74a710
Revises: 8a55f61eb0da
Create Date: 2025-12-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = 'b5d01a74a710'
down_revision: Union[str, Sequence[str], None] = '8a55f61eb0da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'previous_year_question_papers',
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('subject_id', sa.Integer(), nullable=False),
        sa.Column('title', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('num_pages', sa.Integer(), nullable=False),
        sa.Column('file_url', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column('is_premium', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('previous_year_question_papers')
