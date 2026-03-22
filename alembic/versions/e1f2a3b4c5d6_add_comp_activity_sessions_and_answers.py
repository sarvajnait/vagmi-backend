"""add comp activity play sessions and answers tables

Revision ID: e1f2a3b4c5d6
Revises: a570eb3d3480
Branch Labels: None
Depends On: None

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, Sequence[str], None] = "a570eb3d3480"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("comp_activity_play_sessions"):
        op.create_table(
            "comp_activity_play_sessions",
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("comp_chapter_id", sa.Integer(), nullable=True),
            sa.Column("sub_chapter_id", sa.Integer(), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="in_progress"),
            sa.Column("total_questions", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("correct_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("score", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["comp_chapter_id"], ["comp_chapters.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["sub_chapter_id"], ["comp_sub_chapters.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_comp_activity_play_sessions_comp_chapter_id", "comp_activity_play_sessions", ["comp_chapter_id"])
        op.create_index("ix_comp_activity_play_sessions_sub_chapter_id", "comp_activity_play_sessions", ["sub_chapter_id"])

    if not inspector.has_table("comp_activity_answers"):
        op.create_table(
            "comp_activity_answers",
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("session_id", sa.Integer(), nullable=False),
            sa.Column("activity_id", sa.Integer(), nullable=False),
            sa.Column("selected_option_index", sa.Integer(), nullable=True),
            sa.Column("is_correct", sa.Boolean(), nullable=True),
            sa.Column("score", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("answered_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
            sa.ForeignKeyConstraint(["session_id"], ["comp_activity_play_sessions.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["activity_id"], ["comp_chapter_activities.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("session_id", "activity_id"),
        )


def downgrade() -> None:
    op.drop_table("comp_activity_answers")
    op.drop_table("comp_activity_play_sessions")
