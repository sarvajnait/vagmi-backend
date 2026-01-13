"""add activity tables

Revision ID: 9c2b7d4e1a23
Revises: f3b7a9b0e4c1
Create Date: 2026-01-10 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "9c2b7d4e1a23"
down_revision: Union[str, Sequence[str], None] = "f3b7a9b0e4c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("chapter_activities"):
        op.create_table(
            "chapter_activities",
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("chapter_id", sa.Integer(), nullable=False),
            sa.Column("type", sa.String(length=20), nullable=False),
            sa.Column("question_text", sa.String(), nullable=False),
            sa.Column("options", sa.ARRAY(sa.String()), nullable=True),
            sa.Column("correct_option_index", sa.Integer(), nullable=True),
            sa.Column("answer_text", sa.String(), nullable=True),
            sa.Column("answer_image_url", sa.String(), nullable=True),
            sa.Column("sort_order", sa.Integer(), nullable=True),
            sa.Column("id", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not inspector.has_table("activity_play_sessions"):
        op.create_table(
            "activity_play_sessions",
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("chapter_id", sa.Integer(), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("total_questions", sa.Integer(), nullable=False),
            sa.Column("correct_count", sa.Integer(), nullable=False),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("id", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )

    if not inspector.has_table("activity_answers"):
        op.create_table(
            "activity_answers",
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("session_id", sa.Integer(), nullable=False),
            sa.Column("activity_id", sa.Integer(), nullable=False),
            sa.Column("selected_option_index", sa.Integer(), nullable=True),
            sa.Column("submitted_answer_text", sa.String(), nullable=True),
            sa.Column("is_correct", sa.Boolean(), nullable=True),
            sa.Column("score", sa.Integer(), nullable=False),
            sa.Column("answered_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
            sa.Column("id", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(["activity_id"], ["chapter_activities.id"]),
            sa.ForeignKeyConstraint(["session_id"], ["activity_play_sessions.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("session_id", "activity_id"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("activity_answers"):
        op.drop_table("activity_answers")
    if inspector.has_table("activity_play_sessions"):
        op.drop_table("activity_play_sessions")
    if inspector.has_table("chapter_activities"):
        op.drop_table("chapter_activities")
