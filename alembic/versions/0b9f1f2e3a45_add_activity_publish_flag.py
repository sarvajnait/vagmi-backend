"""add activity publish flag

Revision ID: 0b9f1f2e3a45
Revises: f07eed085048
Create Date: 2026-01-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "0b9f1f2e3a45"
down_revision: Union[str, Sequence[str], None] = "f07eed085048"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("chapter_activities"):
        op.add_column(
            "chapter_activities",
            sa.Column(
                "is_published",
                sa.Boolean(),
                server_default=sa.text("true"),
                nullable=False,
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("chapter_activities"):
        op.drop_column("chapter_activities", "is_published")
