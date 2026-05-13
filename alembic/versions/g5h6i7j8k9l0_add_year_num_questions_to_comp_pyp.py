"""add year and num_questions to comp_previous_year_papers

Revision ID: g5h6i7j8k9l0
Revises: f4e5d6c7b8a9
Create Date: 2026-05-13

"""
from alembic import op
import sqlalchemy as sa

revision = 'g5h6i7j8k9l0'
down_revision = 'f4e5d6c7b8a9'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('comp_previous_year_papers', sa.Column('year', sa.Integer(), nullable=True))
    op.add_column('comp_previous_year_papers', sa.Column('num_questions', sa.Integer(), nullable=True))
    # Make num_pages nullable to match updated model
    op.alter_column('comp_previous_year_papers', 'num_pages', nullable=True)


def downgrade():
    op.alter_column('comp_previous_year_papers', 'num_pages', nullable=False)
    op.drop_column('comp_previous_year_papers', 'num_questions')
    op.drop_column('comp_previous_year_papers', 'year')
