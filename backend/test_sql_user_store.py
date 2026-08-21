"""Integration tests for backend/repositories/sql_user_store.py and
sql_care_plan_store.py against a real Postgres instance with migrations
applied. Skips entirely if DATABASE_URL isn't set or unreachable -- see
backend/test_auth_dependencies.py for the same pattern and rationale.

These exercise SqlUserStore/SqlCarePlanStore directly (not through the
DATA_BACKEND dispatch in user_store.py/care_plan_store.py), matching how
backend/test_auth_dependencies.py tests backend/auth/dependencies.py
directly -- the dispatch installer itself is covered separately (see the
DATA_BACKEND assertions in this repo's test run for user_store.py/
care_plan_store.py import-time behavior).
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy import text

from backend.db import get_session_factory
from backend.repositories.sql_care_plan_store import SqlCarePlanStore
from backend.repositories.sql_user_store import SqlUserStore


def _db_available() -> bool:
    if not os.getenv("DATABASE_URL"):
        return False
    try:
        with get_session_factory()() as session:
            session.execute(text("SELECT 1"))
        return True
    except OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="requires a live Postgres (DATABASE_URL) with migrations applied"
)


def _unique_username(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def test_create_user_and_authenticate_round_trip():
    username = _unique_username("create-auth")
    ok = SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com", role="Individual")
    assert ok is True
    assert SqlUserStore.authenticate(username, "correct horse battery staple") is True
    assert SqlUserStore.authenticate(username, "wrong password") is False


def test_create_user_rejects_duplicate_username():
    username = _unique_username("dup")
    assert SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com") is True
    assert SqlUserStore.create_user(username, "another password 123", email=f"{username}-2@example.com") is False


def test_create_user_clinician_role_has_no_patient_scoped_data():
    username = _unique_username("clinician")
    ok = SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com", role="Doctor / Physician")
    assert ok is True
    # No Patient row -> patient-scoped reads behave like "not found", not an error.
    assert SqlUserStore.get_medications(username) == []
    assert SqlUserStore.get_chat_history(username) == []


def test_profile_round_trip():
    username = _unique_username("profile")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com", display_name="Jane Doe")
    profile = SqlUserStore.get_user_profile(username)
    assert profile["display_name"] == "Jane Doe"
    assert profile["email"] == f"{username}@example.com"

    updated = SqlUserStore.update_profile(username, {"display_name": "Jane Q. Doe", "care_context": "Diabetes management"})
    assert updated is True
    profile = SqlUserStore.get_user_profile(username)
    assert profile["display_name"] == "Jane Q. Doe"
    assert profile["care_context"] == "Diabetes management"


def test_medications_crud_round_trip():
    username = _unique_username("meds")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")

    saved = SqlUserStore.save_medication(username, {"name": "Metformin", "dose": "500mg", "schedule": "BD"})
    assert saved is not None
    assert saved["name"] == "Metformin"

    meds = SqlUserStore.get_medications(username)
    assert len(meds) == 1
    assert meds[0]["dose"] == "500mg"

    # Saving again with the same name upserts rather than duplicating (legacy dedup-by-name semantics).
    SqlUserStore.save_medication(username, {"name": "metformin", "dose": "1000mg"})
    meds = SqlUserStore.get_medications(username)
    assert len(meds) == 1
    assert meds[0]["dose"] == "1000mg"

    assert SqlUserStore.delete_medication(username, meds[0]["medication_id"]) is True
    assert SqlUserStore.get_medications(username) == []


def test_conditions_allergies_vitals_round_trip():
    username = _unique_username("clinical")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")

    condition = SqlUserStore.save_condition(username, {"name": "Asthma", "status": "active"})
    allergy = SqlUserStore.save_allergy(username, {"name": "Penicillin", "severity": "severe", "allergy_type": "drug"})
    vitals = SqlUserStore.save_vitals_entry(username, {"type": "blood_pressure", "value": "120/80", "unit": "mmHg", "recorded_on": "2026-01-01"})

    assert condition["status"] == "active"
    assert allergy["confirmed"] is True
    assert vitals["value"] == "120/80"

    assert len(SqlUserStore.get_conditions(username)) == 1
    assert len(SqlUserStore.get_allergies(username)) == 1
    assert len(SqlUserStore.get_vitals(username)) == 1

    assert SqlUserStore.delete_condition(username, condition["condition_id"]) is True
    assert SqlUserStore.delete_allergy(username, allergy["allergy_id"]) is True
    assert SqlUserStore.delete_vitals_entry(username, vitals["vitals_id"]) is True


def test_chat_history_round_trip():
    username = _unique_username("chat")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")

    SqlUserStore.append_chat(username, {"role": "user", "content": "hello", "sources": []})
    SqlUserStore.append_chat(username, {"role": "assistant", "content": "hi there", "trace_id": "trace-1", "metadata": {}})

    history = SqlUserStore.get_chat_history(username)
    assert len(history) == 2
    assert history[0]["content"] == "hello"
    assert history[1]["trace_id"] == "trace-1"

    SqlUserStore.clear_chat_history(username)
    assert SqlUserStore.get_chat_history(username) == []


def test_feedback_round_trip():
    username = _unique_username("feedback")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")
    SqlUserStore.append_chat(username, {"role": "assistant", "content": "answer", "trace_id": "trace-xyz"})

    ok = SqlUserStore.mark_response_feedback(username, "trace-xyz", "", "up", True)
    assert ok is True
    feedback = SqlUserStore.get_response_feedback(username, "trace-xyz")
    assert feedback["rating"] == "up"


def test_longitudinal_memory_round_trip():
    username = _unique_username("memory")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")

    SqlUserStore.save_longitudinal_memory(username, "Patient has well-controlled T2DM.", source="care_plan")
    memory = SqlUserStore.get_longitudinal_memory(username)
    assert memory["summary"] == "Patient has well-controlled T2DM."
    assert memory["source"] == "care_plan"


def test_triage_summary_round_trip():
    username = _unique_username("triage")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")

    SqlUserStore.save_triage_summary(username, {"question": "chest pain?", "urgency_level": "urgent", "next_step": "ED now"})
    latest = SqlUserStore.get_latest_triage_summary(username)
    assert latest["urgency_level"] == "urgent"


def test_self_activity_log_records_events():
    username = _unique_username("activity")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")
    SqlUserStore.update_last_login(username)

    events = [entry["event"] for entry in SqlUserStore.get_audit(username)]
    assert "account_created" in events
    assert "login" in events


def test_care_plan_lifecycle():
    username = _unique_username("careplan")
    SqlUserStore.create_user(username, "correct horse battery staple", email=f"{username}@example.com")

    plan = SqlCarePlanStore.save_plan(username, {
        "condition": "Type 2 Diabetes",
        "status": "active",
        "goals": [{"text": "Walk 30 min/day"}],
        "daily_tasks": [
            {"id": "t1", "text": "Check blood sugar"},
            {"id": "t2", "text": "Take a short walk"},
        ],
    })
    assert plan["condition"] == "Type 2 Diabetes"
    assert plan["goals"][0]["id"]  # _stamp_ids assigns an id

    fetched = SqlCarePlanStore.get_plan(username, plan["id"])
    assert fetched["condition"] == "Type 2 Diabetes"

    toggled = SqlCarePlanStore.toggle_task(username, plan["id"], "t1", True)
    assert toggled["daily_tasks"][0]["completed_dates"]

    SqlCarePlanStore.toggle_task(username, plan["id"], "t2", True)
    with_both_tasks_done = SqlCarePlanStore.get_plan(username, plan["id"])
    assert {
        task["id"]
        for task in with_both_tasks_done["daily_tasks"]
        if task.get("completed_dates")
    } == {"t1", "t2"}

    with_note = SqlCarePlanStore.add_after_visit_note(username, plan["id"], "Discussed insulin options")
    assert with_note["after_visit_notes"][0]["text"] == "Discussed insulin options"

    with_gp_prep = SqlCarePlanStore.set_gp_prep(username, plan["id"], "Ask about A1C trend")
    assert with_gp_prep["gp_prep_summary"] == "Ask about A1C trend"

    assert len(SqlCarePlanStore.list_plans(username)) == 1
    assert SqlCarePlanStore.delete_plan(username, plan["id"]) is True
    assert SqlCarePlanStore.list_plans(username) == []
