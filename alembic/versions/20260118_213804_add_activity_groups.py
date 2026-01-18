"""add activity groups and answer descriptions

Revision ID: 20260118_213804
Revises: c7c0f3b6c9ab
Create Date: 2026-01-18 21:38:04.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260118_213804"
down_revision: Union[str, Sequence[str], None] = "c7c0f3b6c9ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create activity_groups table
    op.create_table(
        "activity_groups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("chapter_id", sa.Integer(), nullable=False),
        sa.Column("timer_seconds", sa.Integer(), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    # Add answer_description column to chapter_activities
    op.add_column("chapter_activities", sa.Column("answer_description", sa.String(), nullable=True))

    # Add activity_group_id column to chapter_activities (nullable for migration)
    op.add_column("chapter_activities", sa.Column("activity_group_id", sa.Integer(), nullable=True))

    # Create a default activity group for each chapter that has activities
    # and assign all existing activities to these default groups
    op.execute("""
        INSERT INTO activity_groups (name, chapter_id, sort_order, created_at)
        SELECT
            'Default Activities' as name,
            chapter_id,
            1 as sort_order,
            NOW() as created_at
        FROM chapter_activities
        GROUP BY chapter_id
    """)

    # Update all existing activities to belong to the default group
    op.execute("""
        UPDATE chapter_activities ca
        SET activity_group_id = (
            SELECT id
            FROM activity_groups ag
            WHERE ag.chapter_id = ca.chapter_id
            AND ag.name = 'Default Activities'
            LIMIT 1
        )
    """)

    # Now make activity_group_id NOT NULL and add foreign key
    op.alter_column("chapter_activities", "activity_group_id", nullable=False)
    op.create_foreign_key(
        "fk_chapter_activities_activity_group_id",
        "chapter_activities",
        "activity_groups",
        ["activity_group_id"],
        ["id"],
        ondelete="CASCADE"
    )


def downgrade() -> None:
    # Remove foreign key and activity_group_id column
    op.drop_constraint("fk_chapter_activities_activity_group_id", "chapter_activities", type_="foreignkey")
    op.drop_column("chapter_activities", "activity_group_id")

    # Remove answer_description column
    op.drop_column("chapter_activities", "answer_description")

    # Drop activity_groups table
    op.drop_table("activity_groups")
