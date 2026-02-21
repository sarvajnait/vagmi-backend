"""add_razorpay_orders_and_duration_days

Revision ID: a1b2c3d4e5f6
Revises: d6bda523532f
Create Date: 2026-02-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'd6bda523532f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add duration_days to subscription_plans
    op.add_column(
        'subscription_plans',
        sa.Column('duration_days', sa.Integer(), nullable=False, server_default='30'),
    )

    # Create razorpay_orders table
    op.create_table(
        'razorpay_orders',
        sa.Column('id', sa.Integer(), nullable=False, primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('user.id'), nullable=False),
        sa.Column('plan_id', sa.Integer(), sa.ForeignKey('subscription_plans.id'), nullable=False),
        sa.Column('razorpay_order_id', sa.String(), nullable=False, unique=True),
        sa.Column('amount', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(), nullable=False, server_default='INR'),
        sa.Column('status', sa.String(), nullable=False, server_default='created'),
        sa.Column('receipt', sa.String(), nullable=False),
        sa.Column('razorpay_payment_id', sa.String(), nullable=True),
        sa.Column('razorpay_signature', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index('ix_razorpay_orders_razorpay_order_id', 'razorpay_orders', ['razorpay_order_id'])


def downgrade() -> None:
    op.drop_index('ix_razorpay_orders_razorpay_order_id', table_name='razorpay_orders')
    op.drop_table('razorpay_orders')
    op.drop_column('subscription_plans', 'duration_days')
