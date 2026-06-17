"""Unit tests for comp + academic subscription plan logic.

Tests mock the async DB session entirely — no database required.
Run with: pytest tests/test_subscriptions.py -v
"""
import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.api.v1.subscriptions import _validate_plan_scope, _enrich_plan
from app.services.subscriptions import get_active_subscription_summary
from app.models.subscriptions import SubscriptionPlan, SubscriptionPlanRead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session(get_map=None, exec_first=None):
    """Build a mock AsyncSession.

    get_map: dict mapping (Model, id) -> instance returned by session.get()
    exec_first: value returned by session.exec(...).first()
    """
    session = AsyncMock()

    async def mock_get(model, id_):
        if get_map and (model, id_) in get_map:
            return get_map[(model, id_)]
        return None

    session.get.side_effect = mock_get

    result_mock = MagicMock()
    result_mock.first.return_value = exec_first
    session.exec.return_value = result_mock

    return session


def make_academic_plan(**kwargs):
    defaults = dict(
        id=1, plan_type="academic", name="Test Plan",
        class_level_id=1, board_id=2, medium_id=3,
        level_id=None, amount_inr=99, duration_days=30,
        fixed_end_date=None, is_active=True, description=None,
        created_at=date.today(), updated_at=date.today(),
    )
    defaults.update(kwargs)
    plan = MagicMock(spec=SubscriptionPlan)
    for k, v in defaults.items():
        setattr(plan, k, v)
    plan.model_dump.return_value = defaults
    return plan


def make_comp_plan(**kwargs):
    defaults = dict(
        id=2, plan_type="comp", name="Comp Plan",
        class_level_id=None, board_id=None, medium_id=None,
        level_id=10, amount_inr=149, duration_days=30,
        fixed_end_date=None, is_active=True, description=None,
        created_at=date.today(), updated_at=date.today(),
    )
    defaults.update(kwargs)
    plan = MagicMock(spec=SubscriptionPlan)
    for k, v in defaults.items():
        setattr(plan, k, v)
    plan.model_dump.return_value = defaults
    return plan


def named(name):
    obj = MagicMock()
    obj.name = name
    return obj


# ---------------------------------------------------------------------------
# _validate_plan_scope — academic
# ---------------------------------------------------------------------------

class TestValidateAcademic:
    @pytest.mark.asyncio
    async def test_valid_academic_plan(self):
        from app.models import ClassLevel, Board, Medium
        session = make_session(
            get_map={
                (ClassLevel, 1): named("Class 10"),
                (Board, 2): named("CBSE"),
                (Medium, 3): named("English"),
            },
            exec_first=None,  # no duplicate
        )
        # Should not raise
        await _validate_plan_scope(
            "academic",
            {"class_level_id": 1, "board_id": 2, "medium_id": 3},
            session,
        )

    @pytest.mark.asyncio
    async def test_academic_missing_class(self):
        session = make_session()
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope(
                "academic",
                {"class_level_id": None, "board_id": 2, "medium_id": 3},
                session,
            )
        assert exc_info.value.status_code == 400
        assert "require class" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_academic_invalid_board(self):
        from app.models import ClassLevel, Board, Medium
        session = make_session(
            get_map={
                (ClassLevel, 1): named("Class 10"),
                # Board 2 missing → returns None
                (Medium, 3): named("English"),
            },
        )
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope(
                "academic",
                {"class_level_id": 1, "board_id": 2, "medium_id": 3},
                session,
            )
        assert exc_info.value.status_code == 400
        assert "Invalid class/board/medium" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_academic_duplicate_rejected(self):
        from app.models import ClassLevel, Board, Medium
        session = make_session(
            get_map={
                (ClassLevel, 1): named("Class 10"),
                (Board, 2): named("CBSE"),
                (Medium, 3): named("English"),
            },
            exec_first=make_academic_plan(),  # existing plan found
        )
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope(
                "academic",
                {"class_level_id": 1, "board_id": 2, "medium_id": 3},
                session,
            )
        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_academic_duplicate_excluded_on_update(self):
        """Editing a plan should exclude itself from the uniqueness check."""
        from app.models import ClassLevel, Board, Medium
        session = make_session(
            get_map={
                (ClassLevel, 1): named("Class 10"),
                (Board, 2): named("CBSE"),
                (Medium, 3): named("English"),
            },
            exec_first=None,  # no OTHER plan found (self excluded)
        )
        # Should not raise when updating plan id=1
        await _validate_plan_scope(
            "academic",
            {"class_level_id": 1, "board_id": 2, "medium_id": 3},
            session,
            exclude_plan_id=1,
        )


# ---------------------------------------------------------------------------
# _validate_plan_scope — comp
# ---------------------------------------------------------------------------

class TestValidateComp:
    @pytest.mark.asyncio
    async def test_valid_comp_plan(self):
        from app.models.competitive_hierarchy import Level
        session = make_session(
            get_map={(Level, 10): named("Prelims")},
            exec_first=None,
        )
        await _validate_plan_scope("comp", {"level_id": 10}, session)

    @pytest.mark.asyncio
    async def test_comp_missing_level(self):
        session = make_session()
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope("comp", {"level_id": None}, session)
        assert exc_info.value.status_code == 400
        assert "require a level" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_comp_invalid_level(self):
        from app.models.competitive_hierarchy import Level
        session = make_session(
            get_map={},  # Level 10 not found
        )
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope("comp", {"level_id": 10}, session)
        assert exc_info.value.status_code == 400
        assert "Invalid level" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_comp_duplicate_rejected(self):
        from app.models.competitive_hierarchy import Level
        session = make_session(
            get_map={(Level, 10): named("Prelims")},
            exec_first=make_comp_plan(),
        )
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope("comp", {"level_id": 10}, session)
        assert exc_info.value.status_code == 400
        assert "already exists" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_unknown_plan_type(self):
        session = make_session()
        with pytest.raises(HTTPException) as exc_info:
            await _validate_plan_scope("unknown", {}, session)
        assert exc_info.value.status_code == 400
        assert "plan_type" in exc_info.value.detail


# ---------------------------------------------------------------------------
# _enrich_plan
# ---------------------------------------------------------------------------

class TestEnrichPlan:
    @pytest.mark.asyncio
    async def test_enrich_academic(self):
        from app.models import ClassLevel, Board, Medium
        plan = make_academic_plan()
        session = make_session(
            get_map={
                (ClassLevel, 1): named("Class 10"),
                (Board, 2): named("CBSE"),
                (Medium, 3): named("English"),
            }
        )
        result = await _enrich_plan(plan, session)
        assert isinstance(result, SubscriptionPlanRead)
        assert result.class_level_name == "Class 10"
        assert result.board_name == "CBSE"
        assert result.medium_name == "English"
        assert result.level_name is None
        assert result.exam_name is None

    @pytest.mark.asyncio
    async def test_enrich_comp(self):
        from app.models.competitive_hierarchy import Level, CompExamMedium, Exam
        plan = make_comp_plan()

        level_obj = MagicMock()
        level_obj.name = "Prelims"
        level_obj.medium_id = 20

        comp_medium_obj = MagicMock()
        comp_medium_obj.name = "English"
        comp_medium_obj.exam_id = 30

        exam_obj = MagicMock()
        exam_obj.name = "UPSC"

        session = make_session(
            get_map={
                (Level, 10): level_obj,
                (CompExamMedium, 20): comp_medium_obj,
                (Exam, 30): exam_obj,
            }
        )
        result = await _enrich_plan(plan, session)
        assert isinstance(result, SubscriptionPlanRead)
        assert result.level_name == "Prelims"
        assert result.comp_medium_name == "English"
        assert result.exam_name == "UPSC"
        assert result.class_level_name is None
        assert result.board_name is None


# ---------------------------------------------------------------------------
# get_active_subscription_summary
# ---------------------------------------------------------------------------

class TestSubscriptionSummary:
    @pytest.mark.asyncio
    async def test_no_active_subscription_returns_none(self):
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.first.return_value = None
        session.exec.return_value = result_mock

        result = await get_active_subscription_summary(session, user_id=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_academic_subscription_summary(self):
        session = AsyncMock()
        sub = MagicMock()
        sub.id = 5
        sub.plan_id = 1
        sub.starts_at = date(2026, 1, 1)
        sub.ends_at = date(2026, 6, 1)
        sub.status = "active"

        plan = make_academic_plan(name="CBSE Plan")

        result_mock = MagicMock()
        result_mock.first.return_value = (sub, plan)
        session.exec.return_value = result_mock

        summary = await get_active_subscription_summary(session, user_id=1)

        assert summary is not None
        assert summary["plan_type"] == "academic"
        assert summary["class_level_id"] == 1
        assert summary["board_id"] == 2
        assert summary["medium_id"] == 3
        assert summary["level_id"] is None
        assert summary["is_active"] is True
        assert summary["plan_name"] == "CBSE Plan"

    @pytest.mark.asyncio
    async def test_comp_subscription_summary(self):
        session = AsyncMock()
        sub = MagicMock()
        sub.id = 7
        sub.plan_id = 2
        sub.starts_at = date(2026, 1, 1)
        sub.ends_at = date(2026, 6, 1)
        sub.status = "active"

        plan = make_comp_plan(name="UPSC Prelims Plan")

        result_mock = MagicMock()
        result_mock.first.return_value = (sub, plan)
        session.exec.return_value = result_mock

        summary = await get_active_subscription_summary(session, user_id=2)

        assert summary is not None
        assert summary["plan_type"] == "comp"
        assert summary["level_id"] == 10
        assert summary["class_level_id"] is None
        assert summary["board_id"] is None
        assert summary["medium_id"] is None
        assert summary["plan_name"] == "UPSC Prelims Plan"
