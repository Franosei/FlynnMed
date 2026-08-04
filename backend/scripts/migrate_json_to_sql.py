"""One-time, operator-run migration: legacy JSON-blob/Postgres-blob user
store (backend/user_store.py, backend/care_plan_store.py) -> the relational
schema (backend/models/*).

Not an HTTP endpoint -- run manually, from a shell with DATABASE_URL pointed
at the target Postgres instance and (if migrating away from the JSON-file
backend) the old DATABASE_URL unset so UserStore reads users.json.

    python -m backend.scripts.migrate_json_to_sql --dry-run
    python -m backend.scripts.migrate_json_to_sql
    python -m backend.scripts.migrate_json_to_sql --verify

Idempotent: each migrated Account keeps a `legacy_username` marker; reruns
skip usernames that already have a matching Account. Never deletes
users.json / data/care_plans/*.json -- both remain on disk as a rollback
artifact.

Password hashes are carried over verbatim (`password_algo=
"pbkdf2_sha256_legacy"`) rather than forcing a mass reset; backend/auth/
passwords.py (PR4) upgrades an account to argon2 the next time that user
logs in successfully.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.care_plan_store import CarePlanStore
from backend.db import get_session_factory
from backend.mrn import generate_mrn
from backend.models import (
    Account,
    AccountActivityLog,
    AccountKind,
    Allergy,
    CarePlan,
    ChatMessage,
    ClinicalNote,
    Condition,
    DocumentSummary,
    InteractionTrace,
    Medication,
    Patient,
    SymptomLog,
    TriageSummary,
    Upload,
    VitalsEntry,
)
from backend.product_config import is_clinician_role
from backend.user_store import UserStore

_CARE_PLAN_TOP_LEVEL_FIELDS = {
    "id",
    "condition",
    "status",
    "clinical_context",
    "validation",
    "gp_prep_summary",
    "created_at",
    "updated_at",
}


def _parse_dt(value: object) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def _parse_date(value: object) -> Optional[date]:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def _account_kind_for(profile: Dict) -> AccountKind:
    role_label = profile.get("role") or profile.get("clinical_role") or ""
    return AccountKind.clinician if is_clinician_role(role_label) else AccountKind.patient


def _email_verified(record: Dict) -> bool:
    # Mirrors UserStore.is_email_verified: missing/None predates the field
    # and is treated as verified so migrated accounts aren't locked out.
    verified = record.get("email_verified")
    return verified is None or verified is True


def _build_account(username: str, record: Dict) -> Account:
    profile = record.get("profile", {})
    return Account(
        id=uuid4(),
        username=username,
        email=(profile.get("email") or f"{username}@unknown.invalid").strip().lower(),
        display_name=record.get("display_name") or username,
        password_hash=record.get("password_hash", ""),
        password_salt=record.get("salt"),
        password_algo="pbkdf2_sha256_legacy",
        account_kind=_account_kind_for(profile),
        role_label=profile.get("role", ""),
        clinical_role=profile.get("clinical_role", ""),
        organization=profile.get("organization", ""),
        care_context=profile.get("care_context") or "Personal health guidance",
        follow_up_preferences=profile.get("follow_up_preferences", ""),
        email_verified=_email_verified(record),
        is_active=True,
        terms_version=profile.get("terms_version", ""),
        terms_role=profile.get("terms_role", ""),
        terms_accepted_at=_parse_dt(profile.get("terms_accepted_at")),
        privacy_accepted_at=_parse_dt(profile.get("privacy_accepted_at")),
        last_login_at=_parse_dt(record.get("last_login")),
        legacy_username=username,
        created_at=_parse_dt(record.get("created_at")) or datetime.now(timezone.utc),
    )


def _build_patient(account: Account, record: Dict) -> Patient:
    profile = record.get("profile", {})
    return Patient(
        id=uuid4(),
        account_id=account.id,
        patient_id=generate_mrn(),
        date_of_birth=_parse_date(profile.get("date_of_birth")),
        biological_sex=profile.get("biological_sex", "") or "",
        dob_recorded_at=_parse_dt(profile.get("dob_recorded_at")),
        longitudinal_memory=record.get("longitudinal_memory", {}) or {},
        last_video_generated_at=_parse_dt(profile.get("last_video_generated_at")),
        last_trial_search=record.get("last_trial_search"),
    )


def _migrate_medications(patient: Patient, record: Dict) -> list[Medication]:
    return [
        Medication(
            id=uuid4(),
            patient_id=patient.id,
            name=m.get("name", ""),
            dose=m.get("dose", ""),
            schedule=m.get("schedule", ""),
            reason=m.get("reason", ""),
            started_on=m.get("started_on", ""),
            notes=m.get("notes", ""),
            created_at=_parse_dt(m.get("created_at")) or datetime.now(timezone.utc),
        )
        for m in record.get("medications", [])
    ]


def _migrate_conditions(patient: Patient, record: Dict) -> list[Condition]:
    return [
        Condition(
            id=uuid4(),
            patient_id=patient.id,
            name=c.get("name", ""),
            status=c.get("status", "unknown"),
            recorded_on=c.get("recorded_on", ""),
            notes=c.get("notes", ""),
            created_at=_parse_dt(c.get("created_at")) or datetime.now(timezone.utc),
        )
        for c in record.get("conditions", [])
    ]


def _migrate_allergies(patient: Patient, record: Dict) -> list[Allergy]:
    return [
        Allergy(
            id=uuid4(),
            patient_id=patient.id,
            name=a.get("name", ""),
            reaction=a.get("reaction", ""),
            severity=a.get("severity", "unknown"),
            allergy_type=a.get("allergy_type", "other"),
            confirmed=bool(a.get("confirmed", True)),
            notes=a.get("notes", ""),
            created_at=_parse_dt(a.get("created_at")) or datetime.now(timezone.utc),
        )
        for a in record.get("allergies", [])
    ]


def _migrate_vitals(patient: Patient, record: Dict) -> list[VitalsEntry]:
    return [
        VitalsEntry(
            id=uuid4(),
            patient_id=patient.id,
            recorded_on=v.get("recorded_on", ""),
            type=v.get("type", ""),
            value=v.get("value", ""),
            unit=v.get("unit", ""),
            notes=v.get("notes", ""),
            created_at=_parse_dt(v.get("created_at")) or datetime.now(timezone.utc),
        )
        for v in record.get("vitals", [])
    ]


def _migrate_symptom_logs(patient: Patient, record: Dict) -> list[SymptomLog]:
    return [
        SymptomLog(
            id=uuid4(),
            patient_id=patient.id,
            symptom=s.get("symptom", ""),
            logged_for=s.get("logged_for", ""),
            severity=int(s.get("severity", 0) or 0),
            triggers=s.get("triggers", ""),
            notes=s.get("notes", ""),
            created_at=_parse_dt(s.get("created_at")) or datetime.now(timezone.utc),
        )
        for s in record.get("symptom_logs", [])
    ]


def _migrate_triage_summaries(patient: Patient, record: Dict) -> list[TriageSummary]:
    return [
        TriageSummary(
            id=uuid4(),
            patient_id=patient.id,
            question=t.get("question", ""),
            urgency_level=t.get("urgency_level", ""),
            next_step=t.get("next_step", ""),
            what_to_monitor=t.get("what_to_monitor", []) or [],
            rationale=t.get("rationale", ""),
            pathway_label=t.get("pathway_label", ""),
            decision_summary=t.get("decision_summary", ""),
            immediate_actions=t.get("immediate_actions", []) or [],
            escalation_triggers=t.get("escalation_triggers", []) or [],
            communication_points=t.get("communication_points", []) or [],
            rule_hits=t.get("rule_hits", []) or [],
            guideline_references=t.get("guideline_references", []) or [],
            logic_version=t.get("logic_version", ""),
            trace_id=t.get("trace_id"),
            created_at=_parse_dt(t.get("created_at")) or datetime.now(timezone.utc),
        )
        for t in record.get("triage_summaries", [])
    ]


def _migrate_interaction_traces(patient: Patient, record: Dict) -> list[InteractionTrace]:
    traces = []
    for t in record.get("traces", []):
        payload = {k: v for k, v in t.items() if k not in ("trace_id", "created_at")}
        traces.append(
            InteractionTrace(
                id=uuid4(),
                patient_id=patient.id,
                trace_id=t.get("trace_id", ""),
                payload=payload,
                created_at=_parse_dt(t.get("created_at")) or datetime.now(timezone.utc),
            )
        )
    return traces


def _migrate_account_activity_log(account: Account, record: Dict) -> list[AccountActivityLog]:
    entries = []
    for a in record.get("audit", []):
        entries.append(
            AccountActivityLog(
                account_id=account.id,
                event=a.get("event", ""),
                details=a.get("details", ""),
                trace_id=a.get("trace_id"),
                event_metadata=a.get("metadata", {}) or {},
                created_at=_parse_dt(a.get("time")) or datetime.now(timezone.utc),
            )
        )
    return entries


def _migrate_chat_messages(patient: Patient, record: Dict) -> list[ChatMessage]:
    messages = []
    for m in record.get("conversation", []):
        timestamp = _parse_dt(m.get("timestamp")) or datetime.now(timezone.utc)
        metadata = dict(m.get("metadata", {}) or {})
        messages.append(
            ChatMessage(
                id=uuid4(),
                patient_id=patient.id,
                role=m.get("role", "user"),
                content=m.get("content", ""),
                timestamp=timestamp,
                sources=m.get("sources", []) or [],
                trace_id=m.get("trace_id"),
                message_metadata=metadata,
                created_at=timestamp,
            )
        )
    return messages


def _migrate_clinical_notes(patient: Patient, record: Dict) -> list[ClinicalNote]:
    notes = []
    for n in record.get("clinical_notes", []):
        created_at = _parse_dt(n.get("created_at")) or datetime.now(timezone.utc)
        notes.append(
            ClinicalNote(
                id=uuid4(),
                patient_id=patient.id,
                subjective=n.get("subjective", ""),
                objective=n.get("objective", ""),
                assessment=n.get("assessment", ""),
                plan=n.get("plan", ""),
                urgency_level=n.get("urgency_level", ""),
                requires_gp_visit=bool(n.get("requires_gp_visit", False)),
                gp_visit_reason=n.get("gp_visit_reason", ""),
                email_sent=bool(n.get("email_sent", False)),
                email_sent_at=_parse_dt(n.get("email_sent_at")),
                created_at=created_at,
            )
        )
    return notes


def _migrate_uploads(patient: Patient, record: Dict) -> list[Upload]:
    summaries_by_file = {s.get("file"): s for s in record.get("doc_summaries", [])}
    uploads = []
    for u in record.get("uploads", []):
        file_name = u.get("file", "")
        upload = Upload(
            id=uuid4(),
            patient_id=patient.id,
            file_name=file_name,
            stored_path=u.get("stored_path", ""),
            content_hash=u.get("content_hash", ""),
            summary_available=bool(u.get("summary_available", False)),
            created_at=_parse_dt(u.get("uploaded_at")) or datetime.now(timezone.utc),
        )
        summary = summaries_by_file.get(file_name)
        if summary:
            upload.document_summary = DocumentSummary(
                id=uuid4(),
                summary=summary.get("summary", ""),
                created_at=_parse_dt(summary.get("updated_at")) or datetime.now(timezone.utc),
            )
        uploads.append(upload)
    return uploads


def _migrate_care_plans(patient: Patient, username: str) -> list[CarePlan]:
    plans = []
    for plan in CarePlanStore.list_plans(username):
        body = {k: v for k, v in plan.items() if k not in _CARE_PLAN_TOP_LEVEL_FIELDS}
        plans.append(
            CarePlan(
                id=uuid4(),
                patient_id=patient.id,
                condition=plan.get("condition", ""),
                status=plan.get("status", "active"),
                body=body,
                clinical_context=plan.get("clinical_context", {}) or {},
                validation=plan.get("validation", {}) or {},
                gp_prep_summary=plan.get("gp_prep_summary"),
                created_at=_parse_dt(plan.get("created_at")) or datetime.now(timezone.utc),
            )
        )
    return plans


def migrate_user(session: Session, username: str, record: Dict, dry_run: bool) -> str:
    existing = session.execute(
        select(Account).where(Account.legacy_username == username)
    ).scalar_one_or_none()
    if existing is not None:
        return "skipped (already migrated)"

    account = _build_account(username, record)
    session.add(account)
    session.add_all(_migrate_account_activity_log(account, record))

    if account.account_kind == AccountKind.patient:
        patient = _build_patient(account, record)
        session.add(patient)
        session.add_all(_migrate_medications(patient, record))
        session.add_all(_migrate_conditions(patient, record))
        session.add_all(_migrate_allergies(patient, record))
        session.add_all(_migrate_vitals(patient, record))
        session.add_all(_migrate_symptom_logs(patient, record))
        session.add_all(_migrate_chat_messages(patient, record))
        session.add_all(_migrate_clinical_notes(patient, record))
        session.add_all(_migrate_uploads(patient, record))
        session.add_all(_migrate_care_plans(patient, username))
        session.add_all(_migrate_triage_summaries(patient, record))
        session.add_all(_migrate_interaction_traces(patient, record))

    if dry_run:
        session.flush()  # surfaces integrity errors without committing
    return "migrated"


def run_migration(dry_run: bool, source: str = "json") -> Dict[str, int]:
    legacy_users = UserStore.list_all_users_for_migration(source=source)
    session_factory = get_session_factory()
    counts = {"migrated": 0, "skipped": 0}

    with session_factory() as session:
        for username, record in legacy_users.items():
            outcome = migrate_user(session, username, record, dry_run)
            counts["migrated" if outcome == "migrated" else "skipped"] += 1
            print(f"  {username}: {outcome}")

        if dry_run:
            print("Dry run -- rolling back, no changes committed.")
            session.rollback()
        else:
            session.commit()

    return counts


def verify_migration(source: str = "json") -> bool:
    legacy_users = UserStore.list_all_users_for_migration(source=source)
    session_factory = get_session_factory()
    ok = True

    with session_factory() as session:
        migrated_count = session.execute(select(Account)).scalars().all()
        migrated_by_username = {a.legacy_username: a for a in migrated_count if a.legacy_username}

        for username, record in legacy_users.items():
            account = migrated_by_username.get(username)
            if account is None:
                print(f"MISSING: {username} has no migrated Account")
                ok = False
                continue

            if account.account_kind == AccountKind.patient:
                patient = session.execute(
                    select(Patient).where(Patient.account_id == account.id)
                ).scalar_one_or_none()
                if patient is None:
                    print(f"MISMATCH: {username} is patient-kind but has no Patient row")
                    ok = False
                    continue

                expected_chat = len(record.get("conversation", []))
                actual_chat = session.execute(
                    select(ChatMessage).where(ChatMessage.patient_id == patient.id)
                ).scalars().all()
                if len(actual_chat) != expected_chat:
                    print(
                        f"MISMATCH: {username} chat_messages count "
                        f"{len(actual_chat)} != expected {expected_chat}"
                    )
                    ok = False

                expected_meds = len(record.get("medications", []))
                actual_meds = session.execute(
                    select(Medication).where(Medication.patient_id == patient.id)
                ).scalars().all()
                if len(actual_meds) != expected_meds:
                    print(
                        f"MISMATCH: {username} medications count "
                        f"{len(actual_meds)} != expected {expected_meds}"
                    )
                    ok = False

    print("Verification " + ("PASSED" if ok else "FAILED"))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Run without committing changes.")
    parser.add_argument("--verify", action="store_true", help="Verify a prior migration instead of running one.")
    parser.add_argument(
        "--source",
        choices=("json", "legacy-postgres"),
        default="json",
        help="Legacy account source. Railway's former blob store uses legacy-postgres.",
    )
    args = parser.parse_args()

    if args.verify:
        return 0 if verify_migration(source=args.source) else 1

    counts = run_migration(dry_run=args.dry_run, source=args.source)
    print(f"Done. migrated={counts['migrated']} skipped={counts['skipped']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
