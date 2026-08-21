"""SQL-backed equivalent of backend/care_plan_store.py's CarePlanStore, with
an identical public method surface so backend/care_plan_store.py can
dispatch to this behind the DATA_BACKEND flag. See backend/repositories/
sql_user_store.py's module docstring for the general design rationale
(username-keyed calls, UUID ids returned as strings, no-Patient-row
accounts behave like "not found").
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.patient import CarePlan, Patient

_SUB_ITEM_KEYS = (
    "goals", "daily_tasks", "weekly_tasks", "medication_reminders",
    "lab_reminders", "escalation_thresholds", "missed_care_checklist",
)
_PROMOTED_FIELDS = {"id", "condition", "status", "clinical_context", "validation", "gp_prep_summary", "created_at", "updated_at"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _today() -> str:
    return date.today().isoformat()


def _iso(value) -> Optional[str]:
    return value.isoformat() if value else None


def _stamp_ids(body: Dict) -> Dict:
    for key in _SUB_ITEM_KEYS:
        for item in body.get(key, []):
            if not item.get("id"):
                item["id"] = uuid.uuid4().hex[:12]
    return body


def _get_patient(db, username: str) -> Optional[Patient]:
    key = (username or "").strip().lower()
    if not key:
        return None
    account = db.execute(select(Account).where(Account.username == key)).scalar_one_or_none()
    if account is None or account.account_kind != AccountKind.patient:
        return None
    return db.execute(select(Patient).where(Patient.account_id == account.id)).scalar_one_or_none()


def _row_to_dict(row: CarePlan) -> Dict:
    return {
        **(row.body or {}),
        "id": str(row.id),
        "condition": row.condition,
        "status": row.status,
        "clinical_context": row.clinical_context,
        "validation": row.validation,
        "gp_prep_summary": row.gp_prep_summary,
        "created_at": _iso(row.created_at),
        "updated_at": _iso(row.updated_at),
    }


def _find_plan(
    db,
    patient: Optional[Patient],
    plan_id: str,
    *,
    for_update: bool = False,
) -> Optional[CarePlan]:
    if patient is None or not plan_id:
        return None
    try:
        parsed_id = uuid.UUID(str(plan_id))
    except ValueError:
        return None
    statement = select(CarePlan).where(
        CarePlan.patient_id == patient.id,
        CarePlan.id == parsed_id,
    )
    if for_update:
        statement = statement.with_for_update()
    return db.execute(statement).scalar_one_or_none()


class SqlCarePlanStore:
    @staticmethod
    def list_plans(username: str) -> List[Dict]:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(CarePlan).where(CarePlan.patient_id == patient.id).order_by(CarePlan.created_at.asc())
            ).scalars().all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def get_plan(username: str, plan_id: str) -> Optional[Dict]:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            row = _find_plan(db, patient, plan_id)
            return _row_to_dict(row) if row else None

    @staticmethod
    def save_plan(username: str, plan: Dict) -> Dict:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            if patient is None:
                raise ValueError(f"No patient record for username={username!r}; cannot save a care plan.")

            plan = dict(plan)
            body = {k: v for k, v in plan.items() if k not in _PROMOTED_FIELDS}
            body = _stamp_ids(body)

            row = _find_plan(db, patient, plan.get("id")) if plan.get("id") else None
            if row is None:
                row = CarePlan(id=uuid.uuid4(), patient_id=patient.id)
                db.add(row)

            row.condition = plan.get("condition", row.condition if row.condition else "")
            row.status = plan.get("status", row.status if row.status else "active")
            row.body = body
            row.clinical_context = plan.get("clinical_context", row.clinical_context or {})
            row.validation = plan.get("validation", row.validation or {})
            if "gp_prep_summary" in plan:
                row.gp_prep_summary = plan.get("gp_prep_summary")

            db.flush()
            result = _row_to_dict(row)
            db.commit()
            return result

    @staticmethod
    def delete_plan(username: str, plan_id: str) -> bool:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            row = _find_plan(db, patient, plan_id)
            if row is None:
                return False
            db.delete(row)
            db.commit()
            return True

    @staticmethod
    def toggle_task(username: str, plan_id: str, task_id: str, done: bool) -> Optional[Dict]:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            # The entire plan is stored in one JSON document. Lock the row so
            # two rapid checkbox requests cannot overwrite each other's task.
            row = _find_plan(db, patient, plan_id, for_update=True)
            if row is None:
                return None

            # JSON values are nested; a shallow copy can evade SQLAlchemy's
            # dirty tracking when only completed_dates changes.
            body = deepcopy(row.body or {})
            today = _today()
            for task in body.get("daily_tasks", []) + body.get("weekly_tasks", []):
                if task.get("id") == task_id:
                    completed: List[str] = task.setdefault("completed_dates", [])
                    if done and today not in completed:
                        completed.append(today)
                    elif not done and today in completed:
                        completed.remove(today)
                    break
            row.body = body
            db.flush()
            result = _row_to_dict(row)
            db.commit()
            return result

    @staticmethod
    def add_after_visit_note(username: str, plan_id: str, note: str) -> Optional[Dict]:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            row = _find_plan(db, patient, plan_id)
            if row is None:
                return None
            body = dict(row.body or {})
            body.setdefault("after_visit_notes", []).append({"text": note, "date": _today()})
            row.body = body
            db.flush()
            result = _row_to_dict(row)
            db.commit()
            return result

    @staticmethod
    def set_gp_prep(username: str, plan_id: str, summary: str) -> Optional[Dict]:
        session_factory = get_session_factory()
        with session_factory() as db:
            patient = _get_patient(db, username)
            row = _find_plan(db, patient, plan_id)
            if row is None:
                return None
            row.gp_prep_summary = summary
            db.flush()
            result = _row_to_dict(row)
            db.commit()
            return result
