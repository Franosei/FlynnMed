"""
Consent-checked loader that shapes a patient's SQL-stored clinical data into
the same dict shapes UserStore.get_*/SqlUserStore already return, so the
existing chat/RAG pipeline (backend/rag_system.py's _prepare_answer_bundle,
via its target_patient_data parameter) can answer questions about a patient
who is NOT the currently-authenticated account, without any changes to
patient_history.py or ClinicalOrchestrator -- both already just consume
whatever dicts/lists they're handed.

This exists because UserStore/SqlUserStore are strictly "the calling
account's own Patient row" by design (see sql_user_store.py's module
docstring) -- there is no method there that fetches a DIFFERENT patient's
records by patient_id/MRN. That capability only lives here, reusing
clinician_access.py's consent-grant validation rather than duplicating it.
"""
from __future__ import annotations

from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.patient import (
    Allergy,
    CarePlan,
    ClinicalNote,
    Condition,
    Medication,
    Patient,
    SymptomLog,
    TriageSummary,
    Upload,
    VitalsEntry,
)
from backend.repositories.sql_user_store import (
    _allergy_to_dict,
    _condition_to_dict,
    _iso,
    _medication_to_dict,
    _symptom_log_to_dict,
    _triage_summary_to_dict,
    _vitals_to_dict,
)
from backend.relationship_engine import derive_relationships, merge_relationships


def _document_summaries_for(db: Session, patient: Patient) -> List[Dict]:
    rows = db.execute(select(Upload).where(Upload.patient_id == patient.id)).scalars().all()
    return [
        {
            "file": u.file_name,
            "summary": u.document_summary.summary,
            "stored_path": u.stored_path,
            "updated_at": _iso(u.document_summary.updated_at),
        }
        for u in rows
        if u.document_summary is not None
    ]


def _user_profile_for(patient: Patient) -> Dict:
    account = patient.account
    return {
        "display_name": account.display_name,
        "email": account.email,
        "care_context": account.care_context,
        "role": account.role_label,
        "clinical_role": account.clinical_role,
        "organization": account.organization,
        "follow_up_preferences": account.follow_up_preferences,
        "terms_version": account.terms_version,
        "terms_role": account.terms_role,
        "terms_accepted_at": _iso(account.terms_accepted_at) or "",
        "privacy_accepted_at": _iso(account.privacy_accepted_at) or "",
        "last_video_generated_at": _iso(patient.last_video_generated_at) or "",
        "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else "",
        "biological_sex": patient.biological_sex,
        "dob_recorded_at": _iso(patient.dob_recorded_at) or "",
        "created_at": _iso(account.created_at),
        "last_login": _iso(account.last_login_at),
        "active_conversation_id": None,
    }


def load_patient_data_bundle(db: Session, patient: Patient) -> Dict:
    """
    Returns a dict matching exactly what _prepare_answer_bundle's
    target_patient_data expects (user_profile, medications, symptom_logs,
    triage_summaries, allergies, conditions, vitals, document_summaries,
    longitudinal_memory_base), plus care_plans/clinical_notes -- extra keys
    _prepare_answer_bundle doesn't read (harmless) but that
    generate_previsit_chart_summary uses for a fuller chart-review draft.
    Ordering mirrors SqlUserStore's own get_* methods for behavioral
    fidelity, including SqlUserStore.get_vitals' default limit=50
    (symptom_logs/triage_summaries are fetched unlimited, matching how
    _prepare_answer_bundle itself calls those two with limit=None).
    """
    medications = db.execute(
        select(Medication)
        .where(Medication.patient_id == patient.id)
        .order_by(Medication.name.asc(), Medication.updated_at.asc())
    ).scalars().all()

    conditions = db.execute(
        select(Condition).where(Condition.patient_id == patient.id)
    ).scalars().all()
    conditions = sorted(conditions, key=lambda item: (item.status != "active", item.name.lower()))

    allergies = db.execute(
        select(Allergy).where(Allergy.patient_id == patient.id).order_by(Allergy.name.asc())
    ).scalars().all()

    vitals = db.execute(
        select(VitalsEntry)
        .where(VitalsEntry.patient_id == patient.id)
        .order_by(VitalsEntry.recorded_on.desc(), VitalsEntry.created_at.desc())
        .limit(50)
    ).scalars().all()

    symptom_logs = db.execute(
        select(SymptomLog)
        .where(SymptomLog.patient_id == patient.id)
        .order_by(SymptomLog.logged_for.desc(), SymptomLog.created_at.desc())
    ).scalars().all()

    triage_summaries = db.execute(
        select(TriageSummary)
        .where(TriageSummary.patient_id == patient.id)
        .order_by(TriageSummary.created_at.desc())
    ).scalars().all()

    care_plans = db.execute(
        select(CarePlan)
        .where(CarePlan.patient_id == patient.id)
        .order_by(CarePlan.created_at.desc())
    ).scalars().all()

    clinical_notes = db.execute(
        select(ClinicalNote)
        .where(ClinicalNote.patient_id == patient.id)
        .order_by(ClinicalNote.created_at.desc())
        .limit(10)
    ).scalars().all()

    medication_dicts = [_medication_to_dict(m) for m in medications]
    symptom_dicts = [_symptom_log_to_dict(s) for s in symptom_logs]
    triage_dicts = [_triage_summary_to_dict(t) for t in triage_summaries]
    allergy_dicts = [_allergy_to_dict(a) for a in allergies]
    condition_dicts = [_condition_to_dict(c) for c in conditions]
    vital_dicts = [_vitals_to_dict(v) for v in vitals]
    care_plan_dicts = [
        {
            **(cp.body or {}),
            "condition": cp.condition,
            "status": cp.status,
            "title": (cp.body or {}).get("title", cp.condition),
            "gp_prep_summary": cp.gp_prep_summary or "",
            "created_at": _iso(cp.created_at),
        }
        for cp in care_plans
    ]
    clinical_note_dicts = [
        {
            "subjective": cn.subjective,
            "objective": cn.objective,
            "assessment": cn.assessment,
            "plan": cn.plan,
            "urgency_level": cn.urgency_level,
            "created_at": _iso(cn.created_at),
        }
        for cn in clinical_notes
    ]
    clinical_relationships = merge_relationships(
        (patient.longitudinal_memory or {}).get("clinical_relationships", []) or [],
        derive_relationships(
            medications=medication_dicts,
            allergies=allergy_dicts,
            conditions=condition_dicts,
            symptom_logs=symptom_dicts,
            vitals=vital_dicts,
            triage_summaries=triage_dicts,
            care_plans=care_plan_dicts,
            clinical_notes=clinical_note_dicts,
        ),
    )

    return {
        "user_profile": _user_profile_for(patient),
        "medications": medication_dicts,
        "symptom_logs": symptom_dicts,
        "triage_summaries": triage_dicts,
        "allergies": allergy_dicts,
        "conditions": condition_dicts,
        "vitals": vital_dicts,
        "document_summaries": _document_summaries_for(db, patient),
        "longitudinal_memory_base": (patient.longitudinal_memory or {}).get("summary", ""),
        "clinical_relationships": clinical_relationships,
        "care_plans": care_plan_dicts,
        "clinical_notes": clinical_note_dicts,
    }
