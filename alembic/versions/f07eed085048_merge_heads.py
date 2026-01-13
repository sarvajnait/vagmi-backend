"""merge heads

Revision ID: f07eed085048
Revises: 6ee9de4a0e5e, 9c2b7d4e1a23
Create Date: 2026-01-10 23:57:12.697915

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f07eed085048'
down_revision: Union[str, Sequence[str], None] = ('6ee9de4a0e5e', '9c2b7d4e1a23')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
