"""merge multiple heads

Revision ID: 0f63a19ca7aa
Revises: 20260118_213804, 7d5c1a9e4b11
Create Date: 2026-01-19 22:03:43.939159

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0f63a19ca7aa'
down_revision: Union[str, Sequence[str], None] = ('20260118_213804', '7d5c1a9e4b11')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
