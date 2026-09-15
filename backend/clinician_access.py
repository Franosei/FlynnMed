"""Consent-based clinician access to patient records."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.account import Account, AccountKind
from backend.models.audit import AuditAction, AuditLogEntry, AuditOutcome
from backend.models.consent import ConsentGrant, ConsentScope, ConsentStatus
from backend.models.patient import (
    Allergy,
    CarePlan,
    ChatMessage,
    ClinicalNote,
    Condition,
    Medication,
    Patient,
    PreVisitSummary,
    ProposedMedication,
    SymptomLog,
    TriageSummary,
    VitalsEntry,
)
from backend.relationship_engine import derive_relationships, merge_relationships


class AccessWorkflowError(Exception):
    """Safe, user-facing access-workflow failure."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _account(db: Session, username: str) -> Account:
    account = db.execute(
        select(Account).where(Account.username == username.strip().lower())
    ).scalar_one_or_none()
    if account is None or not account.is_active:
        raise AccessWorkflowError("This account is not available.")
    return account


def _patient_for_account(db: Session, account: Account) -> Patient:
    patient = db.execute(
        select(Patient).where(Patient.account_id == account.id)
    ).scalar_one_or_none()
    if patient is None:
        raise AccessWorkflowError("Patient account required.")
    return patient


def _audit(
    db: Session,
    *,
    actor: Account,
    patient: Patient | None,
    action: AuditAction,
    outcome: AuditOutcome,
    resource_id: str,
    grant: ConsentGrant | None = None,
) -> None:
    db.add(
        AuditLogEntry(
            actor_account_id=actor.id,
            actor_role_at_time=actor.account_kind.value,
            patient_id=patient.id if patient else None,
            action=action,
            resource_type="patient",
            resource_id=resource_id,
            outcome=outcome,
            consent_grant_id=grant.id if grant else None,
        )
    )
    db.flush()


def _expire_if_needed(grant: ConsentGrant) -> None:
    if (
        grant.status == ConsentStatus.active
        and grant.expires_at is not None
        and grant.expires_at <= _now()
    ):
        grant.status = ConsentStatus.expired


def _grant_dict(
    grant: ConsentGrant,
    patient: Patient,
    clinician: Account,
    *,
    disclose_patient_name: bool = True,
) -> dict:
    return {
        "grant_id": str(grant.id),
        "patient_id": patient.patient_id,
        "patient_name": (
            patient.account.display_name
            if disclose_patient_name
            else "Patient awaiting consent"
        ),
        "clinician_name": clinician.display_name,
        "clinician_role": clinician.clinical_role or clinician.role_label,
        "organization": clinician.organization,
        "status": grant.status.value,
        "scopes": list(grant.scope or []),
        "request_reason": grant.request_reason,
        "requested_at": grant.requested_at.isoformat(),
        "decided_at": grant.decided_at.isoformat() if grant.decided_at else "",
        "expires_at": grant.expires_at.isoformat() if grant.expires_at else "",
    }


def access_overview(db: Session, username: str) -> dict:
    account = _account(db, username)
    if account.account_kind == AccountKind.patient:
        patient = _patient_for_account(db, account)
        grants = db.execute(
            select(ConsentGrant)
            .where(ConsentGrant.patient_id == patient.id)
            .order_by(ConsentGrant.requested_at.desc())
        ).scalars()
        items = []
        for grant in grants:
            _expire_if_needed(grant)
            clinician = db.get(Account, grant.clinician_account_id)
            if clinician is not None:
                items.append(_grant_dict(grant, patient, clinician))
        return {
            "account_kind": "patient",
            "patient_id": patient.patient_id,
            "requests": items,
            "active_count": sum(item["status"] == "active" for item in items),
            "pending_count": sum(item["status"] == "pending" for item in items),
        }

    grants = db.execute(
        select(ConsentGrant)
        .where(ConsentGrant.clinician_account_id == account.id)
        .order_by(ConsentGrant.requested_at.desc())
    ).scalars()
    items = []
    active_items_by_patient_id: dict = {}
    for grant in grants:
        _expire_if_needed(grant)
        patient = db.get(Patient, grant.patient_id)
        if patient is not None:
            item = _grant_dict(
                grant,
                patient,
                account,
                disclose_patient_name=grant.status == ConsentStatus.active,
            )
            # Only an active (consented) grant justifies exposing any clinical
            # signal -- a pending request must never carry patient status.
            if grant.status == ConsentStatus.active:
                item["patient_status"] = None
                active_items_by_patient_id[patient.id] = item
            items.append(item)

    if active_items_by_patient_id:
        # One batched query for the whole roster's latest triage record,
        # instead of a per-patient chart fetch from the client -- avoids an
        # N+1 request storm and doesn't log a "chart accessed" audit event
        # for every patient just to render this overview.
        latest_by_patient: dict = {}
        triage_rows = db.execute(
            select(TriageSummary)
            .where(TriageSummary.patient_id.in_(list(active_items_by_patient_id.keys())))
            .order_by(TriageSummary.patient_id, TriageSummary.created_at.desc())
        ).scalars()
        for row in triage_rows:
            latest_by_patient.setdefault(row.patient_id, row)

        for patient_id, item in active_items_by_patient_id.items():
            latest = latest_by_patient.get(patient_id)
            if latest is not None:
                item["patient_status"] = {
                    "urgency_level": latest.urgency_level,
                    "recorded_at": latest.created_at.isoformat(),
                }

    return {
        "account_kind": "clinician",
        "requests": items,
        "active_count": sum(item["status"] == "active" for item in items),
        "pending_count": sum(item["status"] == "pending" for item in items),
    }


def clinician_dashboard(db: Session, username: str) -> dict:
    """Return a compact, consent-scoped caseload view for any clinician role."""
    clinician = _account(db, username)
    if clinician.account_kind != AccountKind.clinician:
        raise AccessWorkflowError("Clinician account required.")

    overview = access_overview(db, username)
    active_grants = [item for item in overview["requests"] if item["status"] == "active"]
    patients = []
    studies = []
    activity = []
    active_plan_count = 0
    review_count = 0
    attention_count = 0
    urgent_levels = {"high", "urgent", "crisis", "emergency"}

    for grant_item in active_grants:
        patient = db.execute(
            select(Patient).where(Patient.patient_id == grant_item["patient_id"])
        ).scalar_one_or_none()
        if patient is None:
            continue
        grant = db.get(ConsentGrant, uuid.UUID(grant_item["grant_id"]))
        if grant is None:
            continue

        # Loading the dashboard is a real patient-summary read and is audited
        # once per patient, without querying chat-history content.
        _audit(
            db,
            actor=clinician,
            patient=patient,
            action=AuditAction.clinician_read_previsit_summary,
            outcome=AuditOutcome.success,
            resource_id=patient.patient_id,
            grant=grant,
        )

        conditions = list(
            db.execute(
                select(Condition)
                .where(Condition.patient_id == patient.id, Condition.status == "active")
                .order_by(Condition.updated_at.desc())
            ).scalars()
        )
        medications = list(
            db.execute(
                select(Medication)
                .where(Medication.patient_id == patient.id)
                .order_by(Medication.updated_at.desc())
            ).scalars()
        )
        latest_triage = db.execute(
            select(TriageSummary)
            .where(TriageSummary.patient_id == patient.id)
            .order_by(TriageSummary.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        plans = list(
            db.execute(
                select(CarePlan).where(CarePlan.patient_id == patient.id, CarePlan.status == "active")
            ).scalars()
        )
        latest_proposal = db.execute(
            select(ProposedMedication)
            .where(ProposedMedication.patient_id == patient.id)
            .order_by(ProposedMedication.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        latest_summary = db.execute(
            select(PreVisitSummary)
            .where(PreVisitSummary.patient_id == patient.id)
            .order_by(PreVisitSummary.created_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        patient_reviews = int(bool(latest_proposal and latest_proposal.status == "draft")) + int(
            bool(latest_summary and latest_summary.status == "draft")
        )
        urgency = (latest_triage.urgency_level if latest_triage else "routine").strip().lower()
        needs_attention = urgency in urgent_levels
        active_plan_count += len(plans)
        review_count += patient_reviews
        attention_count += int(needs_attention)

        patient_studies = (patient.last_trial_search or {}).get("trials", [])
        for study in patient_studies[:2]:
            studies.append(
                {
                    "patient_id": patient.patient_id,
                    "patient_name": patient.account.display_name,
                    "nct_id": study.get("nct_id", ""),
                    "title": study.get("title", "Untitled study"),
                    "status": study.get("status", "Status not listed"),
                    "phase": study.get("phase", ""),
                    "match_score": study.get("match_score"),
                    "url": study.get("url", ""),
                }
            )

        recent_dates = [
            item
            for item in (
                latest_triage.created_at if latest_triage else None,
                conditions[0].updated_at if conditions else None,
                medications[0].updated_at if medications else None,
                latest_proposal.created_at if latest_proposal else None,
                latest_summary.created_at if latest_summary else None,
            )
            if item is not None
        ]
        last_activity = max(recent_dates) if recent_dates else grant.decided_at or grant.requested_at
        memory = patient.longitudinal_memory or {}
        patients.append(
            {
                "patient_id": patient.patient_id,
                "display_name": patient.account.display_name,
                "active_conditions": [item.name for item in conditions[:3]],
                "condition_count": len(conditions),
                "medication_count": len(medications),
                "active_plan_count": len(plans),
                "study_count": len(patient_studies),
                "review_count": patient_reviews,
                "urgency": urgency or "routine",
                "next_step": latest_triage.next_step if latest_triage else "No recent triage action recorded.",
                "summary": str(memory.get("summary") or "No longitudinal summary recorded yet.")[:280],
                "last_activity_at": last_activity.isoformat() if last_activity else "",
                "access_expires_at": grant_item["expires_at"],
            }
        )

        if latest_triage:
            activity.append(
                {
                    "type": "triage",
                    "patient_id": patient.patient_id,
                    "patient_name": patient.account.display_name,
                    "title": f"{latest_triage.urgency_level or 'Routine'} triage update",
                    "detail": latest_triage.next_step,
                    "created_at": latest_triage.created_at.isoformat(),
                }
            )
        if latest_proposal and latest_proposal.status == "draft":
            activity.append(
                {
                    "type": "medication",
                    "patient_id": patient.patient_id,
                    "patient_name": patient.account.display_name,
                    "title": "Medication proposal needs review",
                    "detail": latest_proposal.candidate_medication_name,
                    "created_at": latest_proposal.created_at.isoformat(),
                }
            )

    patients.sort(key=lambda item: item["last_activity_at"], reverse=True)
    patients.sort(key=lambda item: item["urgency"] not in urgent_levels)
    activity.sort(key=lambda item: item["created_at"], reverse=True)
    def _study_score(item: dict) -> float:
        try:
            return float(item["match_score"] or 0)
        except (TypeError, ValueError):
            return 0

    studies.sort(key=_study_score, reverse=True)
    return {
        "clinician": {
            "display_name": clinician.display_name,
            "clinical_role": clinician.clinical_role or clinician.role_label or "Healthcare professional",
            "organization": clinician.organization,
        },
        "metrics": {
            "active_patients": len(patients),
            "pending_requests": overview["pending_count"],
            "needs_attention": attention_count,
            "active_care_plans": active_plan_count,
            "study_matches": sum(item["study_count"] for item in patients),
            "reviews_due": review_count,
        },
        "patients": patients,
        "studies": studies[:6],
        "recent_activity": activity[:8],
    }


def request_patient_access(
    db: Session,
    username: str,
    patient_id: str,
    reason: str,
    include_chat_history: bool = False,
) -> dict:
    clinician = _account(db, username)
    if clinician.account_kind != AccountKind.clinician:
        raise AccessWorkflowError("Clinician account required.")

    mrn = patient_id.strip().upper()
    request_reason = reason.strip()
    if not request_reason:
        raise AccessWorkflowError("A clinical reason is required.")
    patient = db.execute(
        select(Patient).where(Patient.patient_id == mrn)
    ).scalar_one_or_none()
    if patient is None:
        _audit(
            db,
            actor=clinician,
            patient=None,
            action=AuditAction.clinician_access_requested,
            outcome=AuditOutcome.denied,
            resource_id=mrn,
        )
        raise AccessWorkflowError("The MRN could not be used to create an access request.")

    existing = db.execute(
        select(ConsentGrant).where(
            ConsentGrant.patient_id == patient.id,
            ConsentGrant.clinician_account_id == clinician.id,
            ConsentGrant.status.in_([ConsentStatus.pending, ConsentStatus.active]),
        )
    ).scalar_one_or_none()
    if existing is not None:
        _expire_if_needed(existing)
        if existing.status in (ConsentStatus.pending, ConsentStatus.active):
            return _grant_dict(
                existing,
                patient,
                clinician,
                disclose_patient_name=existing.status == ConsentStatus.active,
            )

    scopes = [ConsentScope.previsit_summary.value]
    if include_chat_history:
        scopes.append(ConsentScope.chat_history.value)
    grant = ConsentGrant(
        id=uuid.uuid4(),
        patient_id=patient.id,
        clinician_account_id=clinician.id,
        status=ConsentStatus.pending,
        scope=scopes,
        request_reason=request_reason[:1000],
        requested_at=_now(),
    )
    db.add(grant)
    db.flush()
    _audit(
        db,
        actor=clinician,
        patient=patient,
        action=AuditAction.clinician_access_requested,
        outcome=AuditOutcome.success,
        resource_id=mrn,
        grant=grant,
    )
    return _grant_dict(
        grant, patient, clinician, disclose_patient_name=False
    )


def decide_access_request(
    db: Session,
    username: str,
    grant_id: str,
    approve: bool,
    decision_note: str = "",
) -> dict:
    account = _account(db, username)
    if account.account_kind != AccountKind.patient:
        raise AccessWorkflowError("Patient account required.")
    patient = _patient_for_account(db, account)
    try:
        parsed_id = uuid.UUID(grant_id)
    except ValueError as exc:
        raise AccessWorkflowError("Access request not found.") from exc
    grant = db.execute(
        select(ConsentGrant).where(
            ConsentGrant.id == parsed_id,
            ConsentGrant.patient_id == patient.id,
            ConsentGrant.status == ConsentStatus.pending,
        )
    ).scalar_one_or_none()
    if grant is None:
        raise AccessWorkflowError("Access request not found.")

    grant.status = ConsentStatus.active if approve else ConsentStatus.denied
    grant.decision_note = decision_note.strip()[:1000]
    grant.decided_at = _now()
    grant.expires_at = _now() + timedelta(days=90) if approve else None
    clinician = db.get(Account, grant.clinician_account_id)
    _audit(
        db,
        actor=account,
        patient=patient,
        action=(
            AuditAction.clinician_access_granted
            if approve
            else AuditAction.clinician_access_denied
        ),
        outcome=AuditOutcome.success,
        resource_id=patient.patient_id,
        grant=grant,
    )
    return _grant_dict(grant, patient, clinician)


def revoke_access(db: Session, username: str, grant_id: str) -> dict:
    actor = _account(db, username)
    try:
        parsed_id = uuid.UUID(grant_id)
    except ValueError as exc:
        raise AccessWorkflowError("Access grant not found.") from exc
    grant = db.get(ConsentGrant, parsed_id)
    if grant is None or grant.status not in (
        ConsentStatus.pending,
        ConsentStatus.active,
    ):
        raise AccessWorkflowError("Access grant not found.")
    patient = db.get(Patient, grant.patient_id)
    owns_patient = patient is not None and patient.account_id == actor.id
    owns_request = grant.clinician_account_id == actor.id
    if not (owns_patient or owns_request):
        raise AccessWorkflowError("Access grant not found.")

    grant.status = ConsentStatus.revoked
    grant.revoked_at = _now()
    grant.revoked_by_account_id = actor.id
    _audit(
        db,
        actor=actor,
        patient=patient,
        action=AuditAction.clinician_access_revoked,
        outcome=AuditOutcome.success,
        resource_id=patient.patient_id if patient else "",
        grant=grant,
    )
    clinician = db.get(Account, grant.clinician_account_id)
    return _grant_dict(grant, patient, clinician)


def _iso(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def _row(item: Any, fields: tuple[str, ...]) -> dict:
    data = {field: getattr(item, field) for field in fields}
    data["id"] = str(item.id)
    data["created_at"] = _iso(getattr(item, "created_at", None))
    return data


def require_active_previsit_access(
    db: Session, username: str, patient_id: str, *, action: AuditAction
) -> tuple[Account, Patient, ConsentGrant]:
    """
    Validates that `username` is an active clinician holding a live, active
    consent grant that covers the previsit_summary scope for the patient
    identified by `patient_id` (MRN). Shared by every read/write path that
    needs this same gate -- the plain read-only chart, AI summary
    drafting/regeneration, the patient-scoped chat, draft edits, and release
    -- each passing its own AuditAction so the audit log distinguishes what
    kind of access it was, without duplicating the validation logic itself.
    Raises AccessWorkflowError (denial is audited before raising) if invalid.
    """
    clinician = _account(db, username)
    if clinician.account_kind != AccountKind.clinician:
        raise AccessWorkflowError("Clinician account required.")
    patient = db.execute(
        select(Patient).where(Patient.patient_id == patient_id.strip().upper())
    ).scalar_one_or_none()
    grant = None
    if patient is not None:
        grant = db.execute(
            select(ConsentGrant).where(
                ConsentGrant.patient_id == patient.id,
                ConsentGrant.clinician_account_id == clinician.id,
                ConsentGrant.status == ConsentStatus.active,
            )
        ).scalar_one_or_none()
        if grant is not None:
            _expire_if_needed(grant)
    valid = bool(
        patient
        and grant
        and (grant.expires_at is None or grant.expires_at > _now())
        and ConsentScope.previsit_summary.value in (grant.scope or [])
    )
    if not valid:
        _audit(
            db,
            actor=clinician,
            patient=patient,
            action=action,
            outcome=AuditOutcome.denied,
            resource_id=patient_id,
            grant=grant,
        )
        raise AccessWorkflowError("No active access grant for this patient.")

    _audit(
        db,
        actor=clinician,
        patient=patient,
        action=action,
        outcome=AuditOutcome.success,
        resource_id=patient.patient_id,
        grant=grant,
    )
    return clinician, patient, grant


def authorized_patient_summary(
    db: Session, username: str, patient_id: str
) -> dict:
    clinician, patient, grant = require_active_previsit_access(
        db, username, patient_id, action=AuditAction.clinician_read_previsit_summary
    )
    chat_allowed = ConsentScope.chat_history.value in (grant.scope or [])
    summary = {
        "patient": {
            "patient_id": patient.patient_id,
            "display_name": patient.account.display_name,
            "date_of_birth": patient.date_of_birth.isoformat()
            if patient.date_of_birth
            else "",
            "biological_sex": patient.biological_sex,
        },
        "grant": _grant_dict(grant, patient, clinician),
        "conditions": [
            _row(item, ("name", "status", "recorded_on", "notes"))
            for item in db.execute(
                select(Condition)
                .where(Condition.patient_id == patient.id)
                .order_by(Condition.created_at.desc())
            ).scalars()
        ],
        "medications": [
            _row(item, ("name", "dose", "schedule", "reason", "started_on", "notes"))
            for item in db.execute(
                select(Medication)
                .where(Medication.patient_id == patient.id)
                .order_by(Medication.created_at.desc())
            ).scalars()
        ],
        "allergies": [
            _row(item, ("name", "reaction", "severity", "allergy_type", "confirmed", "notes"))
            for item in db.execute(
                select(Allergy)
                .where(Allergy.patient_id == patient.id)
                .order_by(Allergy.created_at.desc())
            ).scalars()
        ],
        "vitals": [
            _row(item, ("type", "value", "unit", "recorded_on", "notes"))
            for item in db.execute(
                select(VitalsEntry)
                .where(VitalsEntry.patient_id == patient.id)
                .order_by(VitalsEntry.created_at.desc())
                .limit(20)
            ).scalars()
        ],
        "symptoms": [
            _row(item, ("symptom", "logged_for", "severity", "triggers", "notes"))
            for item in db.execute(
                select(SymptomLog)
                .where(SymptomLog.patient_id == patient.id)
                .order_by(SymptomLog.created_at.desc())
                .limit(20)
            ).scalars()
        ],
        "triage": [
            _row(
                item,
                (
                    "question",
                    "urgency_level",
                    "next_step",
                    "what_to_monitor",
                    "rationale",
                ),
            )
            for item in db.execute(
                select(TriageSummary)
                .where(TriageSummary.patient_id == patient.id)
                .order_by(TriageSummary.created_at.desc())
                .limit(10)
            ).scalars()
        ],
        "care_plans": [
            {
                **(item.body or {}),
                **_row(item, ("condition", "status")),
                "title": (item.body or {}).get("title", item.condition),
                "gp_prep_summary": item.gp_prep_summary or "",
            }
            for item in db.execute(
                select(CarePlan)
                .where(CarePlan.patient_id == patient.id)
                .order_by(CarePlan.created_at.desc())
            ).scalars()
        ],
        "clinical_notes": [
            _row(
                item,
                (
                    "subjective",
                    "objective",
                    "assessment",
                    "plan",
                    "urgency_level",
                ),
            )
            for item in db.execute(
                select(ClinicalNote)
                .where(ClinicalNote.patient_id == patient.id)
                .order_by(ClinicalNote.created_at.desc())
                .limit(10)
            ).scalars()
        ],
        "chat_history": (
            [
                _row(item, ("role", "content", "timestamp"))
                for item in db.execute(
                    select(ChatMessage)
                    .where(ChatMessage.patient_id == patient.id)
                    .order_by(ChatMessage.timestamp.desc())
                    .limit(30)
                ).scalars()
            ]
            if chat_allowed
            else []
        ),
        "chat_history_authorized": chat_allowed,
        # All rows -- draft and released, across every clinician who has ever
        # authored one for this patient. This is what makes past summaries
        # (and the reasoning behind them) visible to any future clinician
        # granted access, not just the original author -- continuity of care,
        # gated by the same previsit_summary scope already checked above, no
        # separate scope/grant type needed.
        "previsit_summaries": [
            {
                **_row(
                    item,
                    (
                        "status",
                        "generation_trigger",
                        "summary_text",
                        "authored_by_display_name",
                        "authored_by_clinical_role",
                        "authored_by_organization",
                        "released_by_display_name",
                        "released_by_clinical_role",
                    ),
                ),
                "released_at": _iso(item.released_at),
            }
            for item in db.execute(
                select(PreVisitSummary)
                .where(PreVisitSummary.patient_id == patient.id)
                .order_by(PreVisitSummary.created_at.desc())
            ).scalars()
        ],
        # Same continuity-of-care rationale as previsit_summaries above: all
        # rows, draft and released, across every clinician -- gated by the
        # same previsit_summary scope, no separate scope needed.
        "proposed_medications": [
            {
                **_row(
                    item,
                    (
                        "status",
                        "generation_trigger",
                        "clinical_situation_text",
                        "candidate_medication_name",
                        "candidate_dose_frequency",
                        "rationale_text",
                        "citations",
                        "safety_check",
                        "override_reason",
                        "authored_by_display_name",
                        "authored_by_clinical_role",
                        "authored_by_organization",
                        "released_by_display_name",
                        "released_by_clinical_role",
                    ),
                ),
                "released_at": _iso(item.released_at),
            }
            for item in db.execute(
                select(ProposedMedication)
                .where(ProposedMedication.patient_id == patient.id)
                .order_by(ProposedMedication.created_at.desc())
            ).scalars()
        ],
    }
    summary["clinical_relationships"] = merge_relationships(
        (patient.longitudinal_memory or {}).get("clinical_relationships", []) or [],
        derive_relationships(
            medications=summary["medications"],
            allergies=summary["allergies"],
            conditions=summary["conditions"],
            symptom_logs=summary["symptoms"],
            vitals=summary["vitals"],
            triage_summaries=summary["triage"],
            care_plans=summary["care_plans"],
            clinical_notes=summary["clinical_notes"],
        ),
    )
    return summary
