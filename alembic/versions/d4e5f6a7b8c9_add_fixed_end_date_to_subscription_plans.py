"""add_fixed_end_date_to_subscription_plans

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-02-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'subscription_plans',
        sa.Column('fixed_end_date', sa.Date(), nullable=True),
    )
    # Set May 30 2026 on all existing plans
    op.execute("UPDATE subscription_plans SET fixed_end_date = '2026-05-30'")


def downgrade() -> None:
    op.drop_column('subscription_plans', 'fixed_end_date')
