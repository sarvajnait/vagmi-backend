"""add student features: streak, wrong answers, study time, notifications, activity group session

Revision ID: c1d2e3f4a5b6
Revises: b5c6d7e8f9a0
Create Date: 2026-05-03 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "b5c6d7e8f9a0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Streak tables ─────────────────────────────────────────────────────────
    op.create_table(
        "user_streaks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("current_streak", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("longest_streak", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_activity_date", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )
    op.create_index("ix_user_streaks_user_id", "user_streaks", ["user_id"])

    op.create_table(
        "user_streak_days",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("activity_date", sa.Date(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "activity_date"),
    )
    op.create_index("ix_user_streak_days_user_id", "user_streak_days", ["user_id"])

    op.create_table(
        "user_milestones",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("milestone_days", sa.Integer(), nullable=False),
        sa.Column("achieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "milestone_days"),
    )
    op.create_index("ix_user_milestones_user_id", "user_milestones", ["user_id"])

    # ── Wrong answer notebook ─────────────────────────────────────────────────
    op.create_table(
        "wrong_answer_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("activity_id", sa.Integer(), nullable=False),
        sa.Column("comp_chapter_id", sa.Integer(), nullable=True),
        sa.Column("activity_group_id", sa.Integer(), nullable=True),
        sa.Column("times_attempted", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("last_wrong_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("is_mastered", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["activity_id"], ["comp_chapter_activities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["comp_chapter_id"], ["comp_chapters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["activity_group_id"], ["comp_activity_groups.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "activity_id"),
    )
    op.create_index("ix_wrong_answer_entries_user_id", "wrong_answer_entries", ["user_id"])
    op.create_index("ix_wrong_answer_entries_activity_id", "wrong_answer_entries", ["activity_id"])
    op.create_index("ix_wrong_answer_entries_comp_chapter_id", "wrong_answer_entries", ["comp_chapter_id"])
    op.create_index("ix_wrong_answer_entries_activity_group_id", "wrong_answer_entries", ["activity_group_id"])

    # ── Study time ────────────────────────────────────────────────────────────
    op.create_table(
        "study_time_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("logged_date", sa.Date(), nullable=False),
        sa.Column("duration_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "logged_date"),
    )
    op.create_index("ix_study_time_logs_user_id", "study_time_logs", ["user_id"])

    # ── Notification inbox ────────────────────────────────────────────────────
    op.create_table(
        "user_notifications",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("body", sa.String(), nullable=False),
        sa.Column("notif_type", sa.String(length=50), nullable=False),
        sa.Column("icon_emoji", sa.String(length=10), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_notifications_user_id", "user_notifications", ["user_id"])
    op.create_index("ix_user_notifications_user_is_read", "user_notifications", ["user_id", "is_read"])

    # ── activity_group_id on play sessions ────────────────────────────────────
    op.add_column(
        "comp_activity_play_sessions",
        sa.Column("activity_group_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_comp_play_sessions_activity_group_id",
        "comp_activity_play_sessions",
        "comp_activity_groups",
        ["activity_group_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_comp_play_sessions_activity_group_id",
        "comp_activity_play_sessions",
        ["activity_group_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_comp_play_sessions_activity_group_id", table_name="comp_activity_play_sessions")
    op.drop_constraint("fk_comp_play_sessions_activity_group_id", "comp_activity_play_sessions", type_="foreignkey")
    op.drop_column("comp_activity_play_sessions", "activity_group_id")

    op.drop_index("ix_user_notifications_user_is_read", table_name="user_notifications")
    op.drop_index("ix_user_notifications_user_id", table_name="user_notifications")
    op.drop_table("user_notifications")

    op.drop_index("ix_study_time_logs_user_id", table_name="study_time_logs")
    op.drop_table("study_time_logs")

    op.drop_index("ix_wrong_answer_entries_activity_group_id", table_name="wrong_answer_entries")
    op.drop_index("ix_wrong_answer_entries_comp_chapter_id", table_name="wrong_answer_entries")
    op.drop_index("ix_wrong_answer_entries_activity_id", table_name="wrong_answer_entries")
    op.drop_index("ix_wrong_answer_entries_user_id", table_name="wrong_answer_entries")
    op.drop_table("wrong_answer_entries")

    op.drop_index("ix_user_milestones_user_id", table_name="user_milestones")
    op.drop_table("user_milestones")

    op.drop_index("ix_user_streak_days_user_id", table_name="user_streak_days")
    op.drop_table("user_streak_days")

    op.drop_index("ix_user_streaks_user_id", table_name="user_streaks")
    op.drop_table("user_streaks")
