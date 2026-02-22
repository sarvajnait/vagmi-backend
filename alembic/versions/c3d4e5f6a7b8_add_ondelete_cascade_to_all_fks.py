"""add_ondelete_cascade_to_all_fks

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-02-22 14:00:00.000000

Adds ON DELETE CASCADE (or SET NULL) to all foreign keys that were
previously missing it, preventing FK violation errors on parent deletes.
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _recreate_fk(table, column, ref_table, ref_column, ondelete, constraint_name=None):
    """Drop existing FK and recreate with new ondelete rule."""
    # Find and drop existing constraint
    op.execute(f"""
        DO $$
        DECLARE r RECORD;
        BEGIN
          FOR r IN (
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_name = '{table}'
              AND kcu.column_name = '{column}'
          ) LOOP
            EXECUTE 'ALTER TABLE {table} DROP CONSTRAINT ' || quote_ident(r.constraint_name);
          END LOOP;
        END $$;
    """)
    name = constraint_name or f"fk_{table}_{column}"
    op.execute(
        f"ALTER TABLE {table} ADD CONSTRAINT {name} "
        f"FOREIGN KEY ({column}) REFERENCES {ref_table} ({ref_column}) ON DELETE {ondelete}"
    )


def upgrade() -> None:
    # academic_hierarchy
    _recreate_fk("boards", "class_level_id", "class_levels", "id", "CASCADE", "fk_boards_class_level_id")
    _recreate_fk("mediums", "board_id", "boards", "id", "CASCADE", "fk_mediums_board_id")
    _recreate_fk("subjects", "medium_id", "mediums", "id", "CASCADE", "fk_subjects_medium_id")
    _recreate_fk("chapters", "subject_id", "subjects", "id", "CASCADE", "fk_chapters_subject_id")

    # llm_resources
    _recreate_fk("llm_textbooks", "chapter_id", "chapters", "id", "CASCADE", "fk_llm_textbooks_chapter_id")
    _recreate_fk("additional_notes", "chapter_id", "chapters", "id", "CASCADE", "fk_additional_notes_chapter_id")
    _recreate_fk("llm_images", "chapter_id", "chapters", "id", "CASCADE", "fk_llm_images_chapter_id")
    _recreate_fk("llm_notes", "chapter_id", "chapters", "id", "CASCADE", "fk_llm_notes_chapter_id")
    _recreate_fk("qa_patterns", "chapter_id", "chapters", "id", "CASCADE", "fk_qa_patterns_chapter_id")

    # student_content
    _recreate_fk("student_textbooks", "chapter_id", "chapters", "id", "CASCADE", "fk_student_textbooks_chapter_id")
    _recreate_fk("student_notes", "chapter_id", "chapters", "id", "CASCADE", "fk_student_notes_chapter_id")
    _recreate_fk("student_videos", "chapter_id", "chapters", "id", "CASCADE", "fk_student_videos_chapter_id")
    _recreate_fk("previous_year_question_papers", "subject_id", "subjects", "id", "CASCADE", "fk_pyqp_subject_id")

    # activities
    _recreate_fk("topics", "chapter_id", "chapters", "id", "CASCADE", "fk_topics_chapter_id")
    _recreate_fk("activity_groups", "chapter_id", "chapters", "id", "CASCADE", "fk_activity_groups_chapter_id")
    _recreate_fk("chapter_activities", "chapter_id", "chapters", "id", "CASCADE", "fk_chapter_activities_chapter_id")
    _recreate_fk("chapter_activities", "activity_group_id", "activity_groups", "id", "CASCADE", "fk_chapter_activities_group_id")
    _recreate_fk("activity_play_sessions", "chapter_id", "chapters", "id", "CASCADE", "fk_play_sessions_chapter_id")
    _recreate_fk("activity_play_sessions", "user_id", "user", "id", "CASCADE", "fk_play_sessions_user_id")
    _recreate_fk("activity_answers", "session_id", "activity_play_sessions", "id", "CASCADE", "fk_activity_answers_session_id")
    _recreate_fk("activity_answers", "activity_id", "chapter_activities", "id", "CASCADE", "fk_activity_answers_activity_id")

    # chapter_artifacts (already fixed in previous migration but ensure)
    _recreate_fk("chapter_artifacts", "chapter_id", "chapters", "id", "CASCADE", "fk_chapter_artifacts_chapter_id")

    # notifications (SET NULL — optional filters)
    _recreate_fk("notifications", "class_level_id", "class_levels", "id", "SET NULL", "fk_notifications_class_level_id")
    _recreate_fk("notifications", "board_id", "boards", "id", "SET NULL", "fk_notifications_board_id")
    _recreate_fk("notifications", "medium_id", "mediums", "id", "SET NULL", "fk_notifications_medium_id")


def downgrade() -> None:
    # Re-add without ondelete (back to plain FK)
    tables_cols = [
        ("boards", "class_level_id", "class_levels", "id", "fk_boards_class_level_id"),
        ("mediums", "board_id", "boards", "id", "fk_mediums_board_id"),
        ("subjects", "medium_id", "mediums", "id", "fk_subjects_medium_id"),
        ("chapters", "subject_id", "subjects", "id", "fk_chapters_subject_id"),
        ("llm_textbooks", "chapter_id", "chapters", "id", "fk_llm_textbooks_chapter_id"),
        ("additional_notes", "chapter_id", "chapters", "id", "fk_additional_notes_chapter_id"),
        ("llm_images", "chapter_id", "chapters", "id", "fk_llm_images_chapter_id"),
        ("llm_notes", "chapter_id", "chapters", "id", "fk_llm_notes_chapter_id"),
        ("qa_patterns", "chapter_id", "chapters", "id", "fk_qa_patterns_chapter_id"),
        ("student_textbooks", "chapter_id", "chapters", "id", "fk_student_textbooks_chapter_id"),
        ("student_notes", "chapter_id", "chapters", "id", "fk_student_notes_chapter_id"),
        ("student_videos", "chapter_id", "chapters", "id", "fk_student_videos_chapter_id"),
        ("previous_year_question_papers", "subject_id", "subjects", "id", "fk_pyqp_subject_id"),
        ("topics", "chapter_id", "chapters", "id", "fk_topics_chapter_id"),
        ("activity_groups", "chapter_id", "chapters", "id", "fk_activity_groups_chapter_id"),
        ("chapter_activities", "chapter_id", "chapters", "id", "fk_chapter_activities_chapter_id"),
        ("chapter_activities", "activity_group_id", "activity_groups", "id", "fk_chapter_activities_group_id"),
        ("activity_play_sessions", "chapter_id", "chapters", "id", "fk_play_sessions_chapter_id"),
        ("activity_play_sessions", "user_id", "user", "id", "fk_play_sessions_user_id"),
        ("activity_answers", "session_id", "activity_play_sessions", "id", "fk_activity_answers_session_id"),
        ("activity_answers", "activity_id", "chapter_activities", "id", "fk_activity_answers_activity_id"),
        ("chapter_artifacts", "chapter_id", "chapters", "id", "fk_chapter_artifacts_chapter_id"),
        ("notifications", "class_level_id", "class_levels", "id", "fk_notifications_class_level_id"),
        ("notifications", "board_id", "boards", "id", "fk_notifications_board_id"),
        ("notifications", "medium_id", "mediums", "id", "fk_notifications_medium_id"),
    ]
    for table, col, ref_table, ref_col, name in tables_cols:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {name}")
        op.execute(
            f"ALTER TABLE {table} ADD CONSTRAINT {name} "
            f"FOREIGN KEY ({col}) REFERENCES {ref_table} ({ref_col})"
        )
