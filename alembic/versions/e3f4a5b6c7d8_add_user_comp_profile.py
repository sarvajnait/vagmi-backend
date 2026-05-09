"""add user comp profile

Revision ID: e3f4a5b6c7d8
Revises: c1d2e3f4a5b6
Create Date: 2026-05-09

"""
from alembic import op
import sqlalchemy as sa

revision = 'e3f4a5b6c7d8'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_comp_profiles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, nullable=False, unique=True),
        sa.Column("exam_id", sa.Integer, nullable=True),
        sa.Column("comp_medium_id", sa.Integer, nullable=True),
        sa.Column("level_id", sa.Integer, nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["exam_id"], ["exams.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["comp_medium_id"], ["comp_exam_mediums.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["level_id"], ["comp_levels.id"], ondelete="SET NULL"),
        sa.Column("exam_date", sa.Date, nullable=True),
        sa.Column("daily_commitment_hours", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now(), nullable=False),
    )
    op.create_index("ix_user_comp_profiles_user_id", "user_comp_profiles", ["user_id"])


def downgrade():
    op.drop_index("ix_user_comp_profiles_user_id", table_name="user_comp_profiles")
    op.drop_table("user_comp_profiles")
