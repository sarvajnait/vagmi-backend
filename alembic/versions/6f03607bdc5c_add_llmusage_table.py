"""add llmusage table

Revision ID: 6f03607bdc5c
Revises: b5d01a74a710
Create Date: 2025-12-25 23:02:57.549378

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel

# revision identifiers, used by Alembic.
revision: str = '6f03607bdc5c'
down_revision: Union[str, Sequence[str], None] = 'b5d01a74a710'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("llmusage"):
        op.create_table(
            'llmusage',
            sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.Column('model_name', sqlmodel.sql.sqltypes.AutoString(), nullable=True),
            sa.Column('input_tokens', sa.Integer(), nullable=False),
            sa.Column('output_tokens', sa.Integer(), nullable=False),
            sa.Column('total_tokens', sa.Integer(), nullable=False),
            sa.Column('input_token_details', sa.JSON(), nullable=True),
            sa.Column('output_token_details', sa.JSON(), nullable=True),
            sa.Column('id', sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
            sa.PrimaryKeyConstraint('id')
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("llmusage"):
        op.drop_table('llmusage')
