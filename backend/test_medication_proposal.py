"""Integration coverage for the clinician medication-proposal workflow.
Follows test_previsit_summary.py's exact conventions (skipif-gated on a live
Postgres, db_session fixture, account/patient factories) -- reusing those
factories directly rather than re-deriving them.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

import backend.api as api
from backend.clinician_access import decide_access_request, request_patient_access
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.patient import Patient, ProposedMedication
from backend.mrn import generate_mrn


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


def _account(db_session, kind: AccountKind, prefix: str) -> Account:
    username = f"{prefix}-{uuid.uuid4().hex[:8]}"
    account = Account(
        id=uuid.uuid4(),
        username=username,
        email=f"{username}@example.com",
        display_name=prefix.title(),
        password_hash="x",
        password_algo="argon2id",
        account_kind=kind,
        role_label=("Doctor / Physician" if kind == AccountKind.clinician else "Patient / Individual"),
        clinical_role=("Doctor / Physician" if kind == AccountKind.clinician else "Patient / Individual"),
        organization="Test Clinic" if kind == AccountKind.clinician else "",
        is_active=True,
    )
    db_session.add(account)
    db_session.flush()
    return account


def _patient(db_session, account: Account) -> Patient:
    patient = Patient(
        id=uuid.uuid4(),
        account_id=account.id,
        patient_id=generate_mrn(),
        biological_sex="",
    )
    db_session.add(patient)
    db_session.flush()
    return patient


def _grant_active_access(db_session, clinician: Account, patient: Patient, include_chat_history=False):
    requested = request_patient_access(
        db_session,
        clinician.username,
        patient.patient_id,
        "Medication review",
        include_chat_history=include_chat_history,
    )
    decide_access_request(db_session, patient.account.username, requested["grant_id"], approve=True)
    return requested["grant_id"]


_NO_FLAGS = {"allergy_flags": [], "interaction_flags": [], "unresolved_medications": [], "checked_at": "now"}
_WITH_ALLERGY_FLAG = {
    "allergy_flags": [{"allergy_name": "Penicillin", "match_type": "exact_name", "severity": "severe", "summary": "conflict"}],
    "interaction_flags": [],
    "unresolved_medications": [],
    "checked_at": "now",
}


@pytest.fixture(autouse=True)
def _mock_generation():
    # generate_medication_proposal makes real retrieval + LLM calls -- mocked
    # so these tests are fast and don't depend on network/API-key
    # availability, consistent with how test_previsit_summary.py mocks
    # generate_previsit_chart_summary.
    with patch.object(
        api,
        "generate_medication_proposal",
        return_value={
            "status": "ok",
            "clinical_situation_text": "Moderate seasonal allergies, no response to antihistamines.",
            "candidate_medication_name": "fluticasone",
            "candidate_dose_frequency": "50mcg once daily",
            "rationale_text": "Fluticasone nasal spray is recommended [S1].",
            "citations": [{"source_id": "S1", "title": "NICE CKS"}],
            "trace_id": "trace-test",
            "safety_check": _NO_FLAGS,
        },
    ):
        yield


class _FakeReleasePayload:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_no_grant_denies_generate_edit_and_release(db_session):
    # These call the FastAPI endpoint functions directly, not the underlying
    # require_active_previsit_access -- the endpoints catch AccessWorkflowError
    # and convert it to HTTPException(403) (see api.py's _access_error), so
    # that's what propagates here, not AccessWorkflowError itself.
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)

    with pytest.raises(HTTPException, match="No active access") as exc_info:
        api.generate_medication_proposal_endpoint(
            patient.patient_id,
            _FakeReleasePayload(clinical_situation="x"),
            username=clinician.username,
            db=db_session,
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException, match="No active access") as exc_info:
        api.save_medication_proposal_draft(
            patient.patient_id,
            _FakeReleasePayload(
                clinical_situation_text="x",
                candidate_medication_name="y",
                candidate_dose_frequency="z",
                rationale_text="r",
                citations=[],
            ),
            username=clinician.username,
            db=db_session,
        )
    assert exc_info.value.status_code == 403

    with pytest.raises(HTTPException, match="No active access") as exc_info:
        api.release_medication_proposal(
            patient.patient_id,
            _FakeReleasePayload(
                clinical_situation_text="x",
                candidate_medication_name="y",
                candidate_dose_frequency="z",
                rationale_text="r",
                citations=[],
                override_reason="",
            ),
            username=clinician.username,
            db=db_session,
        )
    assert exc_info.value.status_code == 403


def test_generate_creates_draft_with_no_flags(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _grant_active_access(db_session, clinician, patient)

    result = api.generate_medication_proposal_endpoint(
        patient.patient_id,
        _FakeReleasePayload(clinical_situation="Moderate seasonal allergies, no response to antihistamines."),
        username=clinician.username,
        db=db_session,
    )

    assert result["status"] == "draft"
    assert result["generation_trigger"] == "ai_generated"
    assert result["candidate_medication_name"] == "fluticasone"
    assert result["safety_check"]["allergy_flags"] == []
    assert result["authored_by_display_name"] == clinician.display_name


def test_release_without_override_reason_fails_when_flags_present(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _grant_active_access(db_session, clinician, patient)

    with patch.object(api, "recheck_candidate_safety", return_value=_WITH_ALLERGY_FLAG):
        with pytest.raises(Exception) as exc_info:
            api.release_medication_proposal(
                patient.patient_id,
                _FakeReleasePayload(
                    clinical_situation_text="Moderate seasonal allergies.",
                    candidate_medication_name="penicillin-based-nasal-spray",
                    candidate_dose_frequency="once daily",
                    rationale_text="Rationale.",
                    citations=[],
                    override_reason="",
                ),
                username=clinician.username,
                db=db_session,
            )
    assert "override reason" in str(exc_info.value).lower()

    rows = db_session.execute(
        select(ProposedMedication).where(ProposedMedication.patient_id == patient.id)
    ).scalars().all()
    assert not any(r.status == "released" for r in rows)


def test_release_without_override_reason_succeeds_when_no_flags(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _grant_active_access(db_session, clinician, patient)

    with patch.object(api, "recheck_candidate_safety", return_value=_NO_FLAGS):
        result = api.release_medication_proposal(
            patient.patient_id,
            _FakeReleasePayload(
                clinical_situation_text="Moderate seasonal allergies.",
                candidate_medication_name="fluticasone",
                candidate_dose_frequency="50mcg once daily",
                rationale_text="Rationale [S1].",
                citations=[{"source_id": "S1", "title": "NICE CKS"}],
                override_reason="",
            ),
            username=clinician.username,
            db=db_session,
        )

    assert result["status"] == "released"
    assert result["override_reason"] == ""
    assert result["released_by_display_name"] == clinician.display_name
    assert result["released_at"]


def test_release_with_override_reason_persists_it_when_flags_present(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _grant_active_access(db_session, clinician, patient)

    with patch.object(api, "recheck_candidate_safety", return_value=_WITH_ALLERGY_FLAG):
        result = api.release_medication_proposal(
            patient.patient_id,
            _FakeReleasePayload(
                clinical_situation_text="Moderate seasonal allergies.",
                candidate_medication_name="penicillin-based-nasal-spray",
                candidate_dose_frequency="once daily",
                rationale_text="Rationale.",
                citations=[],
                override_reason="Patient has tolerated this class before; clinical judgment applied.",
            ),
            username=clinician.username,
            db=db_session,
        )

    assert result["status"] == "released"
    assert "clinical judgment" in result["override_reason"]
    assert result["safety_check"]["allergy_flags"]


def test_edit_reruns_safety_check_against_new_candidate_not_stale_one(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _grant_active_access(db_session, clinician, patient)

    seen_names = []

    def _fake_recheck(db, patient_arg, candidate_name):
        seen_names.append(candidate_name)
        if candidate_name == "penicillin-based-nasal-spray":
            return _WITH_ALLERGY_FLAG
        return _NO_FLAGS

    with patch.object(api, "recheck_candidate_safety", side_effect=_fake_recheck):
        result = api.save_medication_proposal_draft(
            patient.patient_id,
            _FakeReleasePayload(
                clinical_situation_text="Moderate seasonal allergies.",
                candidate_medication_name="penicillin-based-nasal-spray",
                candidate_dose_frequency="once daily",
                rationale_text="Edited rationale.",
                citations=[],
            ),
            username=clinician.username,
            db=db_session,
        )

    assert seen_names == ["penicillin-based-nasal-spray"]
    assert result["generation_trigger"] == "clinician_edited"
    assert result["safety_check"]["allergy_flags"]


def test_patient_read_returns_only_released_rows(db_session):
    clinician = _account(db_session, AccountKind.clinician, "doctor")
    patient_account = _account(db_session, AccountKind.patient, "patient")
    patient = _patient(db_session, patient_account)
    _grant_active_access(db_session, clinician, patient)

    api.generate_medication_proposal_endpoint(
        patient.patient_id,
        _FakeReleasePayload(clinical_situation="Moderate seasonal allergies."),
        username=clinician.username,
        db=db_session,
    )
    with patch.object(api, "recheck_candidate_safety", return_value=_NO_FLAGS):
        api.release_medication_proposal(
            patient.patient_id,
            _FakeReleasePayload(
                clinical_situation_text="Moderate seasonal allergies.",
                candidate_medication_name="fluticasone",
                candidate_dose_frequency="50mcg once daily",
                rationale_text="Rationale [S1].",
                citations=[{"source_id": "S1", "title": "NICE CKS"}],
                override_reason="",
            ),
            username=clinician.username,
            db=db_session,
        )

    result = api.list_my_medication_proposals(username=patient_account.username, db=db_session)
    assert len(result["proposals"]) == 1
    assert result["proposals"][0]["status"] == "released"
    assert result["proposals"][0]["candidate_medication_name"] == "fluticasone"
