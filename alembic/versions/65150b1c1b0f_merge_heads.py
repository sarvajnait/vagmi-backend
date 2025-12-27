"""merge heads

Revision ID: 65150b1c1b0f
Revises: 6f03607bdc5c, c7c0f3b6c9ab
Create Date: 2025-12-27 23:39:19.566334

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '65150b1c1b0f'
down_revision: Union[str, Sequence[str], None] = ('6f03607bdc5c', 'c7c0f3b6c9ab')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
