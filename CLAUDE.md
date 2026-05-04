# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Run dev server
uvicorn app.main:app --reload

# Production (matches Procfile)
gunicorn -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080 --worker-tmp-dir /dev/shm app.main:app

# Database migrations
alembic upgrade head          # apply all
alembic revision -m "desc"    # create new migration
alembic downgrade -1          # rollback one

# Lint
ruff check .
```

## Required Environment Variables

```env
POSTGRES_URL=postgresql+psycopg://user:password@host:port/db
GOOGLE_API_KEY=                    # Gemini for AI generation
DO_SPACES_ACCESS_KEY=              # DigitalOcean Spaces
DO_SPACES_SECRET_KEY=
DO_SPACES_BUCKET=
DO_SPACES_REGION=
JWT_SECRET_KEY=
MSG91_AUTH_KEY=                    # OTP via SMS
MSG91_TEMPLATE_ID=
RAZORPAY_KEY_ID=
RAZORPAY_KEY_SECRET=
FIREBASE_PROJECT_ID=               # FCM push notifications
FIREBASE_SERVICE_ACCOUNT_JSON=     # JSON string of Firebase service account
APP_ENV=development                # development | staging | production
```

## Architecture

### Dual Content Hierarchy

The codebase manages two parallel, independent content hierarchies:

**Academic** (school/board prep): `Board → ClassLevel → Medium → Subject → Chapter`
- Models: `app/models/academic_hierarchy.py`
- Activities: `app/models/activities.py` (`ActivityGroup`, `ChapterActivity`, `ActivityPlaySession`, `ActivityAnswer`)
- APIs: `app/api/v1/{boards,class_levels,subjects,chapters,activities,activity_groups,topics}.py`

**Competitive** (RRB, SSC, UPSC etc.): `ExamCategory → Exam → CompExamMedium → Level → CompSubject → CompChapter → SubChapter`
- Models: `app/models/competitive_hierarchy.py`, `app/models/comp_activities.py`
- Activities: `CompActivityGroup`, `CompChapterActivity`, `CompActivityPlaySession`, `CompActivityAnswer`
- APIs: `app/api/v1/comp_hierarchy.py`, `app/api/v1/comp_activities.py`
- Student-facing: `app/api/v1/comp_student.py` (progress, streak, wrong answers, performance)

Both hierarchies have parallel LLM resources (textbooks, notes, images, QA patterns, artifacts) and student content (notes, videos, textbooks, previous year papers).

### Router Layout (`app/api/v1/api.py`)

| Prefix | Router File | Purpose |
|--------|-------------|---------|
| `/comp` | `comp_hierarchy.py` | Comp exam hierarchy CRUD |
| `/comp/llm` | `comp_llm_resources.py` | LLM resources for comp |
| `/comp/student` | `comp_student_content.py` | Admin-managed student content |
| `/comp/student` | `comp_student.py` | Student-facing progress APIs |
| `/comp` | `comp_activities.py` | MCQ sessions, answers, AI gen |
| `/comp` | `mock_tests.py` | Mock test management |

### Model Conventions

All table models extend `BaseModel` (adds `created_at`, `updated_at`) plus `SQLModel`:

```python
class MyModel(BaseModel, table=True):
    __tablename__ = "my_models"
    id: Optional[int] = Field(default=None, primary_key=True)
    foreign_id: int = Field(
        sa_column=Column(Integer, ForeignKey("other.id", ondelete="CASCADE"), nullable=False, index=True)
    )
```

`async_session_maker` is configured with `expire_on_commit=False` — but background service functions that call `await db.commit()` will still expire ORM objects in routes that share the same session. Always snapshot `.dict()` before any subsequent commit if you need the data afterward.

### Auth Dependencies

```python
from app.api.v1.auth import get_current_user          # User JWT
from app.api.v1.admin.auth import get_current_user as get_current_admin  # Admin JWT
```

Routes requiring a student use `Depends(get_current_user)` → returns `User`.
Routes requiring an admin use `Depends(get_current_admin)` → returns `Admin`.

### Background Job Pattern

Long-running AI tasks (topic generation, activity generation, audio, PDF processing) use `ActivityGenerationJob`:

1. Create a `ActivityGenerationJob` row with `job_type` + `payload` dict
2. Call `background_tasks.add_task(enqueue_activity_job, job.id)` — runs in a thread
3. `app/services/activity_jobs.py` dispatches to the right `_run_*_job()` handler
4. Frontend polls `GET /comp/ai/jobs/{job_id}` for status

For background service code (outside a request), use `async_session_maker` directly:
```python
async with async_session_maker() as session:
    ...
```

### File Storage

All uploads go to DigitalOcean Spaces:
```python
from app.utils.files import upload_to_do, delete_from_do, delete_prefix_from_do
file_url = upload_to_do(upload_file, "comp/chapters/{id}/student-content/notes")
```

### Sort Ordering Convention

Most list endpoints use this helper to put NULL `sort_order` last:
```python
from sqlalchemy import case
def sort_ordering(model):
    return [case((model.sort_order == None, 1), else_=0), model.sort_order, model.created_at]
```

### New Student-Facing Features (recently added)

The following models track per-user competitive exam progress:
- `UserStreak`, `UserStreakDay`, `UserMilestone` — streak calendar and milestone badges
- `WrongAnswerEntry` — wrong answer notebook per user+activity
- `StudyTimeLog` — screen time logged by Flutter app per day

When a `CompActivityPlaySession` completes (in `submit_comp_answer`), it automatically calls `process_session_answers()` and `record_streak_activity()` from the respective services.

### Push Notifications

FCM is wired up in `app/services/fcm_service.py`. Users register their token via `POST /users/fcm-token`. Admin broadcast: `POST /admin/notifications/send` (fans out to every targeted user's inbox). Per-user inbox: `GET /comp/student/notifications` with Today/Yesterday/Earlier grouping, unread count, and mark-read endpoints. Notifications are also triggered automatically on milestone achieved and wrong-answer count ≥ 5.
