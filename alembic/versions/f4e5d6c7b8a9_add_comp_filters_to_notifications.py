"""add comp filters to notifications

Revision ID: f4e5d6c7b8a9
Revises: e3f4a5b6c7d8
Create Date: 2026-05-11

"""
from alembic import op
import sqlalchemy as sa

revision = 'f4e5d6c7b8a9'
down_revision = 'e3f4a5b6c7d8'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('notifications', sa.Column('exam_id', sa.Integer(), sa.ForeignKey('exams.id', ondelete='SET NULL'), nullable=True))
    op.add_column('notifications', sa.Column('comp_medium_id', sa.Integer(), sa.ForeignKey('comp_exam_mediums.id', ondelete='SET NULL'), nullable=True))
    op.add_column('notifications', sa.Column('level_id', sa.Integer(), sa.ForeignKey('comp_levels.id', ondelete='SET NULL'), nullable=True))


def downgrade():
    op.drop_column('notifications', 'level_id')
    op.drop_column('notifications', 'comp_medium_id')
    op.drop_column('notifications', 'exam_id')
