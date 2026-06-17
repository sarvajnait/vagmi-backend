"""add comp subscription support

Revision ID: h1i2j3k4l5m6
Revises: g5h6i7j8k9l0
Create Date: 2026-06-14

"""
from alembic import op
import sqlalchemy as sa


revision = 'h1i2j3k4l5m6'
down_revision = 'g5h6i7j8k9l0'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('subscription_plans', sa.Column('plan_type', sa.String(), nullable=False, server_default='academic'))
    op.add_column('subscription_plans', sa.Column('level_id', sa.Integer(), sa.ForeignKey('comp_levels.id', ondelete='SET NULL'), nullable=True))
    op.alter_column('subscription_plans', 'class_level_id', nullable=True)
    op.alter_column('subscription_plans', 'board_id', nullable=True)
    op.alter_column('subscription_plans', 'medium_id', nullable=True)


def downgrade():
    op.alter_column('subscription_plans', 'medium_id', nullable=False)
    op.alter_column('subscription_plans', 'board_id', nullable=False)
    op.alter_column('subscription_plans', 'class_level_id', nullable=False)
    op.drop_column('subscription_plans', 'level_id')
    op.drop_column('subscription_plans', 'plan_type')
