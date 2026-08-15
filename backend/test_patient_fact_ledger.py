"""
Evidence Ledger Phase 3 tests: PatientFact persistence and dedup. Follows
test_clinician_access.py's/test_evidence_ledger.py's exact conventions
(skipif-gated on a live Postgres with migrations applied, rollback-isolated
db_session fixture, real Account/Patient rows for the FK).
"""
from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.patient import Patient
from backend.mrn import generate_mrn
from backend.patient_fact_ledger import (
    persist_patient_fact,
    persist_patient_facts_for_bundle,
)


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
    not _db_available(),
    reason="requires a live Postgres (DATABASE_URL) with migrations applied",
)


@pytest.fixture()
def db_session():
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _account(db_session) -> Account:
    username = f"patient-fact-{uuid.uuid4().hex[:8]}"
    account = Account(
        id=uuid.uuid4(),
        username=username,
        email=f"{username}@example.com",
        display_name="Test Patient",
        password_hash="x",
        password_algo="argon2id",
        account_kind=AccountKind.patient,
        role_label="Patient / Individual",
        clinical_role="Patient / Individual",
        organization="",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    return account


def _patient(db_session) -> Patient:
    account = _account(db_session)
    patient = Patient(
        id=uuid.uuid4(),
        account_id=account.id,
        patient_id=generate_mrn(),
        biological_sex="",
    )
    db_session.add(patient)
    db_session.flush()
    return patient


def test_persist_patient_fact_dedupes_unchanged_fact(db_session):
    patient = _patient(db_session)

    first = persist_patient_fact(
        db_session, patient.id,
        category="medication", label="Metformin", value="500mg twice daily",
    )
    second = persist_patient_fact(
        db_session, patient.id,
        category="medication", label="Metformin", value="500mg twice daily",
    )

    assert first is not None
    assert second is not None
    assert first.id == second.id


def test_persist_patient_fact_creates_new_row_for_changed_value(db_session):
    patient = _patient(db_session)

    original = persist_patient_fact(
        db_session, patient.id,
        category="medication", label="Metformin", value="500mg twice daily",
    )
    updated = persist_patient_fact(
        db_session, patient.id,
        category="medication", label="Metformin", value="1000mg twice daily",
    )

    assert original is not None
    assert updated is not None
    assert original.id != updated.id
    # Immutability: the original row's value is untouched by the later edit.
    assert original.value == "500mg twice daily"
    assert updated.value == "1000mg twice daily"


def test_persist_patient_fact_links_to_patient_and_source_record(db_session):
    patient = _patient(db_session)
    source_id = uuid.uuid4()

    fact = persist_patient_fact(
        db_session, patient.id,
        category="condition", label="Type 2 diabetes", value="active",
        source_record_type="condition", source_record_id=source_id,
    )

    assert fact is not None
    assert fact.patient_id == patient.id
    assert fact.source_record_type == "condition"
    assert fact.source_record_id == source_id


def test_persist_patient_fact_maps_unconfirmed_allergy_to_suspected(db_session):
    patient = _patient(db_session)

    fact = persist_patient_fact(
        db_session, patient.id,
        category="allergy", label="Penicillin", value="Rash", status="suspected",
    )

    assert fact is not None
    assert fact.status == "suspected"


def test_persist_patient_fact_coerces_unrecognised_category_status_source(db_session):
    patient = _patient(db_session)

    fact = persist_patient_fact(
        db_session, patient.id,
        category="not_a_real_category", label="Something", value="x",
        status="extremely_confirmed", source="made_up_source",
    )

    assert fact is not None
    assert fact.category == "unknown"
    assert fact.status == "unknown"
    assert fact.source == "structured_patient_record"


def test_persist_patient_fact_returns_none_without_label(db_session):
    patient = _patient(db_session)

    assert persist_patient_fact(db_session, patient.id, category="condition", label="", value="x") is None


def test_persist_patient_facts_for_bundle_no_op_without_patient_id():
    """No-op when patient_id is falsy (e.g. a clinician's patient-agnostic
    evidence question, with no target patient in view) -- must not raise."""
    persist_patient_facts_for_bundle(
        None,
        medications=[{"name": "Metformin", "medication_id": str(uuid.uuid4())}],
    )
    persist_patient_facts_for_bundle("", medications=[])


def test_persist_patient_facts_for_bundle_persists_medication_and_allergy():
    """
    Integration test through the real short-lived-session write path (this
    one does commit -- matches this codebase's existing precedent of
    account/patient test fixtures creating real rows rather than being
    rollback-isolated, since persist_patient_facts_for_bundle manages its
    own session by design).
    """
    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        db.commit()
        patient_id = str(patient.id)

    medication_id = str(uuid.uuid4())
    allergy_id = str(uuid.uuid4())
    persist_patient_facts_for_bundle(
        patient_id,
        medications=[
            {
                "medication_id": medication_id,
                "name": "Metformin",
                "dose": "500mg",
                "schedule": "twice daily",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        allergies=[
            {
                "allergy_id": allergy_id,
                "name": "Penicillin",
                "reaction": "Rash",
                "confirmed": False,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    )

    with session_factory() as db:
        rows = db.execute(
            text("SELECT category, label, value, status, source_record_type, source_record_id "
                 "FROM patient_facts WHERE patient_id = :pid ORDER BY category"),
            {"pid": patient_id},
        ).mappings().all()

    by_category = {r["category"]: r for r in rows}
    assert by_category["medication"]["label"] == "Metformin"
    assert by_category["medication"]["value"] == "500mg twice daily"
    assert str(by_category["medication"]["source_record_id"]) == medication_id
    assert by_category["allergy"]["label"] == "Penicillin"
    assert by_category["allergy"]["status"] == "suspected"
    assert str(by_category["allergy"]["source_record_id"]) == allergy_id


# ---------------------------------------------------------------------------
# Evidence Ledger v2: explicit supersession + retraction (#6)
# ---------------------------------------------------------------------------

def test_persist_patient_fact_sets_previous_fact_id_on_edit(db_session):
    patient = _patient(db_session)

    original = persist_patient_fact(
        db_session, patient.id,
        category="medication", label="Metformin", value="500mg twice daily",
    )
    updated = persist_patient_fact(
        db_session, patient.id,
        category="medication", label="Metformin", value="1000mg twice daily",
    )

    assert updated.previous_fact_id == original.id
    assert original.previous_fact_id is None


def test_persist_patient_fact_chains_through_multiple_edits(db_session):
    patient = _patient(db_session)

    v1 = persist_patient_fact(
        db_session, patient.id, category="medication", label="Metformin", value="500mg",
    )
    v2 = persist_patient_fact(
        db_session, patient.id, category="medication", label="Metformin", value="1000mg",
    )
    v3 = persist_patient_fact(
        db_session, patient.id, category="medication", label="Metformin", value="1500mg",
    )

    assert v2.previous_fact_id == v1.id
    assert v3.previous_fact_id == v2.id


def test_retract_missing_facts_marks_removed_medication_retracted():
    from backend.patient_fact_ledger import _retract_missing_facts

    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        original = persist_patient_fact(
            db, patient.id, category="medication", label="Amoxicillin", value="500mg",
        )
        db.commit()
        patient_id = patient.id
        original_id = original.id

    with session_factory() as db:
        _retract_missing_facts(db, patient_id, "medication", [])  # no longer in the live list
        db.commit()

    with session_factory() as db:
        rows = db.execute(
            text(
                "SELECT status, previous_fact_id FROM patient_facts "
                "WHERE patient_id = :pid AND category = 'medication' ORDER BY created_at"
            ),
            {"pid": str(patient_id)},
        ).mappings().all()

    assert rows[0]["status"] == "confirmed"
    assert rows[-1]["status"] == "retracted"
    assert str(rows[-1]["previous_fact_id"]) == str(original_id)


def test_retract_missing_facts_is_idempotent_across_repeated_calls():
    """build_safety_reviews-style callers may run this on every poll -- a
    fact already retracted must not be retracted again."""
    from backend.patient_fact_ledger import _retract_missing_facts

    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        persist_patient_fact(db, patient.id, category="allergy", label="Penicillin", value="Rash")
        db.commit()
        patient_id = patient.id

    with session_factory() as db:
        _retract_missing_facts(db, patient_id, "allergy", [])
        db.commit()
    with session_factory() as db:
        _retract_missing_facts(db, patient_id, "allergy", [])
        db.commit()

    with session_factory() as db:
        rows = db.execute(
            text("SELECT status FROM patient_facts WHERE patient_id = :pid AND category = 'allergy'"),
            {"pid": str(patient_id)},
        ).mappings().all()

    assert sum(1 for r in rows if r["status"] == "retracted") == 1


def test_retract_missing_facts_keeps_fact_present_in_live_list():
    from backend.patient_fact_ledger import _retract_missing_facts

    session_factory = get_session_factory()
    with session_factory() as db:
        patient = _patient(db)
        persist_patient_fact(db, patient.id, category="condition", label="Hypertension", value="active")
        db.commit()
        patient_id = patient.id

    with session_factory() as db:
        _retract_missing_facts(db, patient_id, "condition", ["Hypertension"])
        db.commit()

    with session_factory() as db:
        rows = db.execute(
            text("SELECT status FROM patient_facts WHERE patient_id = :pid AND category = 'condition'"),
            {"pid": str(patient_id)},
        ).mappings().all()

    assert all(r["status"] != "retracted" for r in rows)
