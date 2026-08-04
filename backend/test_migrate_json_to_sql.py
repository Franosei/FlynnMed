import uuid
from datetime import date

import pytest

import backend.user_store as user_store_module
from backend.models.account import AccountKind
from backend.models.patient import Patient
from backend.scripts.migrate_json_to_sql import (
    _account_kind_for,
    _build_account,
    _build_patient,
    _email_verified,
    _migrate_account_activity_log,
    _migrate_allergies,
    _migrate_chat_messages,
    _migrate_conditions,
    _migrate_interaction_traces,
    _migrate_medications,
    _migrate_triage_summaries,
    _migrate_vitals,
    _parse_date,
    _parse_dt,
)
from backend.user_store import UserStore


def test_migration_source_defaults_to_local_json(monkeypatch):
    expected = {"local-user": {"profile": {}}}

    class FakeLocalBackend:
        def list_all_users(self):
            return expected

    monkeypatch.setattr(user_store_module, "_LocalJSONUserBackend", FakeLocalBackend)

    assert UserStore.list_all_users_for_migration() == expected


def test_migration_source_can_read_legacy_postgres(monkeypatch):
    expected = {"railway-user": {"profile": {}}}
    received_urls = []

    class FakePostgresBackend:
        def __init__(self, database_url):
            received_urls.append(database_url)

        def list_all_users(self):
            return expected

    monkeypatch.setenv("DATABASE_URL", "postgresql://example.invalid/flynnmed")
    monkeypatch.setattr(user_store_module, "_PostgresUserBackend", FakePostgresBackend)

    assert UserStore.list_all_users_for_migration("legacy-postgres") == expected
    assert received_urls == ["postgresql://example.invalid/flynnmed"]


def test_migration_source_rejects_unknown_value():
    with pytest.raises(ValueError, match="Migration source"):
        UserStore.list_all_users_for_migration("unknown")


def test_parse_dt_handles_iso_and_z_suffix():
    assert _parse_dt("2026-01-02T03:04:05+00:00").year == 2026
    assert _parse_dt("2026-01-02T03:04:05Z").tzinfo is not None
    assert _parse_dt("") is None
    assert _parse_dt(None) is None
    assert _parse_dt("not-a-date") is None


def test_parse_date_handles_yyyy_mm_dd_and_junk():
    assert _parse_date("1990-05-01") == date(1990, 5, 1)
    assert _parse_date("") is None
    assert _parse_date("garbage") is None


def test_account_kind_for_patient_vs_clinician():
    assert _account_kind_for({"role": "Individual"}) == AccountKind.patient
    assert _account_kind_for({"role": "Doctor / Physician"}) == AccountKind.clinician
    assert _account_kind_for({"role": "Nurse"}) == AccountKind.clinician
    assert _account_kind_for({}) == AccountKind.patient


def test_email_verified_treats_none_as_verified_legacy_default():
    assert _email_verified({}) is True
    assert _email_verified({"email_verified": None}) is True
    assert _email_verified({"email_verified": True}) is True
    assert _email_verified({"email_verified": False}) is False


def test_build_account_carries_legacy_password_hash_verbatim():
    record = {
        "display_name": "Jane Doe",
        "password_hash": "abc123",
        "salt": "s4lt",
        "profile": {"email": "jane@example.com", "role": "Individual"},
    }
    account = _build_account("jane", record)
    assert account.username == "jane"
    assert account.password_hash == "abc123"
    assert account.password_salt == "s4lt"
    assert account.password_algo == "pbkdf2_sha256_legacy"
    assert account.account_kind == AccountKind.patient
    assert account.legacy_username == "jane"


def test_build_patient_generates_fresh_mrn():
    account_id = uuid.uuid4()

    class _FakeAccount:
        id = account_id

    record = {"profile": {"date_of_birth": "1990-01-01", "biological_sex": "Female"}}
    patient = _build_patient(_FakeAccount(), record)
    assert patient.account_id == account_id
    assert patient.patient_id.startswith("FM-")
    assert patient.date_of_birth == date(1990, 1, 1)


def _fake_patient() -> Patient:
    return Patient(id=uuid.uuid4())


def test_migrate_medications_maps_fields():
    patient = _fake_patient()
    record = {
        "medications": [
            {"name": "Metformin", "dose": "500mg", "schedule": "BD", "reason": "T2DM", "notes": "x"}
        ]
    }
    migrated = _migrate_medications(patient, record)
    assert len(migrated) == 1
    assert migrated[0].name == "Metformin"
    assert migrated[0].patient_id == patient.id


def test_migrate_conditions_and_allergies_map_fields():
    patient = _fake_patient()
    record = {
        "conditions": [{"name": "Asthma", "status": "active"}],
        "allergies": [{"name": "Penicillin", "severity": "severe", "allergy_type": "drug"}],
    }
    conditions = _migrate_conditions(patient, record)
    allergies = _migrate_allergies(patient, record)
    assert conditions[0].name == "Asthma"
    assert conditions[0].status == "active"
    assert allergies[0].name == "Penicillin"
    assert allergies[0].confirmed is True


def test_migrate_vitals_maps_fields():
    patient = _fake_patient()
    record = {"vitals": [{"type": "blood_pressure", "value": "120/80", "unit": "mmHg", "recorded_on": "2026-01-01"}]}
    vitals = _migrate_vitals(patient, record)
    assert vitals[0].type == "blood_pressure"
    assert vitals[0].value == "120/80"


def test_migrate_triage_summaries_maps_fields():
    patient = _fake_patient()
    record = {
        "triage_summaries": [
            {
                "question": "chest pain?",
                "urgency_level": "urgent",
                "next_step": "ED now",
                "what_to_monitor": ["breathing"],
                "escalation_triggers": ["worsening pain"],
                "trace_id": "trace-9",
            }
        ]
    }
    summaries = _migrate_triage_summaries(patient, record)
    assert len(summaries) == 1
    assert summaries[0].urgency_level == "urgent"
    assert summaries[0].what_to_monitor == ["breathing"]
    assert summaries[0].trace_id == "trace-9"
    assert summaries[0].patient_id == patient.id


def test_migrate_interaction_traces_promotes_trace_id_rest_into_payload():
    patient = _fake_patient()
    record = {
        "traces": [
            {
                "trace_id": "trace-42",
                "created_at": "2026-01-02T03:04:05Z",
                "question": "what is this rash?",
                "sources": [{"title": "doc"}],
            }
        ]
    }
    traces = _migrate_interaction_traces(patient, record)
    assert len(traces) == 1
    assert traces[0].trace_id == "trace-42"
    assert traces[0].payload == {"question": "what is this rash?", "sources": [{"title": "doc"}]}
    assert traces[0].patient_id == patient.id


def test_migrate_account_activity_log_maps_legacy_audit_list():
    account_id = uuid.uuid4()

    class _FakeAccount:
        id = account_id

    record = {
        "audit": [
            {"time": "2026-01-01T00:00:00Z", "event": "login", "details": "User logged in"},
            {
                "time": "2026-01-02T00:00:00Z",
                "event": "medication_saved",
                "details": "Saved medication: Metformin",
                "trace_id": None,
                "metadata": {"medication_id": "med-1"},
            },
        ]
    }
    entries = _migrate_account_activity_log(_FakeAccount(), record)
    assert len(entries) == 2
    assert entries[0].event == "login"
    assert entries[0].account_id == account_id
    assert entries[1].event_metadata == {"medication_id": "med-1"}


def test_build_patient_carries_last_video_and_trial_search_cache():
    account_id = uuid.uuid4()

    class _FakeAccount:
        id = account_id

    record = {
        "profile": {"last_video_generated_at": "2026-01-01T00:00:00Z"},
        "last_trial_search": {"location": "Boston", "trials": []},
    }
    patient = _build_patient(_FakeAccount(), record)
    assert patient.last_video_generated_at is not None
    assert patient.last_trial_search == {"location": "Boston", "trials": []}


def test_migrate_chat_messages_preserves_metadata_and_sources():
    patient = _fake_patient()
    record = {
        "conversation": [
            {
                "role": "user",
                "content": "hello",
                "timestamp": "2026-01-02T03:04:05Z",
                "sources": [{"title": "doc"}],
                "trace_id": "trace-1",
                "metadata": {"foo": "bar"},
            }
        ]
    }
    messages = _migrate_chat_messages(patient, record)
    assert len(messages) == 1
    assert messages[0].content == "hello"
    assert messages[0].sources == [{"title": "doc"}]
    assert messages[0].message_metadata == {"foo": "bar"}
    assert messages[0].timestamp.tzinfo is not None
