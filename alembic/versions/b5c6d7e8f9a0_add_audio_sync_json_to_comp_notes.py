"""add audio_sync_json to comp_student_notes

Revision ID: b5c6d7e8f9a0
Revises: a8f9e2b1c4d3
Create Date: 2026-04-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector

revision = 'b5c6d7e8f9a0'
down_revision = 'a8f9e2b1c4d3'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    existing = [c["name"] for c in inspector.get_columns("comp_student_notes")]
    if "audio_sync_json" not in existing:
        op.add_column("comp_student_notes", sa.Column("audio_sync_json", sa.Text(), nullable=True))


def downgrade():
    op.drop_column("comp_student_notes", "audio_sync_json")
