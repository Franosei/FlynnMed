"""Create the public FlynnMed demo patient and clinician accounts.

The seed is intentionally idempotent: it fills in missing demo rows and
restores the documented demo-account identity/credentials without deleting
records created while somebody is exploring the application.

All people and clinical details in this module are fictional test data.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.passwords import hash_password, verify_password
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
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
    SymptomLog,
    TriageSummary,
    VitalsEntry,
)
from backend.product_config import TERMS_VERSION

# Match the web process and migration scripts when this module is run directly.
load_dotenv()


PATIENT_USERNAME = "demo.patient.jane"
PATIENT_PASSWORD = "DemoPatient!2026"
PATIENT_MRN = "FM-CKTD-724Z"
CLINICIAN_USERNAME = "demo.dr.omar"
CLINICIAN_PASSWORD = "DemoClinician!2026"
MICHAEL_USERNAME = "demo.patient.michael"
MICHAEL_PASSWORD = "DemoMichael!2026"
MICHAEL_MRN = "FM-H8DJ-10M1"
AISHA_USERNAME = "demo.patient.aisha"
AISHA_PASSWORD = "DemoAisha!2026"
AISHA_MRN = "FM-YE5X-1AMA"

_DEMO_NAMESPACE = uuid.UUID("d9fb5316-f09f-4d5f-91ba-c037b16f75ad")


def _id(name: str) -> uuid.UUID:
    return uuid.uuid5(_DEMO_NAMESPACE, name)


def _utc(year: int, month: int, day: int, hour: int = 9) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


def _account(
    db: Session,
    *,
    username: str,
    password: str,
    email: str,
    display_name: str,
    kind: AccountKind,
    role: str,
    organization: str = "",
    care_context: str = "",
    follow_up_preferences: str = "Email reminders for planned reviews",
) -> Account:
    account = db.execute(select(Account).where(Account.username == username)).scalar_one_or_none()
    if account is None:
        email_owner = db.execute(select(Account).where(Account.email == email)).scalar_one_or_none()
        if email_owner is not None:
            raise RuntimeError(f"Cannot seed {username}: {email} belongs to another account.")
        password_data = hash_password(password)
        account = Account(
            id=_id(f"account:{username}"),
            username=username,
            email=email,
            display_name=display_name,
            password_hash=password_data.hash,
            password_salt=password_data.salt,
            password_algo=password_data.algo,
            account_kind=kind,
            role_label=role,
            clinical_role=role,
            organization=organization,
            care_context=care_context or (
                "Clinical decision support" if kind == AccountKind.clinician else "Personal health guidance"
            ),
            follow_up_preferences=follow_up_preferences,
            email_verified=True,
            is_active=True,
            terms_version=TERMS_VERSION,
            terms_role=role,
            terms_accepted_at=_utc(2026, 8, 1),
            privacy_accepted_at=_utc(2026, 8, 1),
        )
        db.add(account)
        db.flush()
        return account

    if account.account_kind != kind:
        raise RuntimeError(f"Cannot seed {username}: existing account has the wrong account kind.")

    account.display_name = display_name
    account.role_label = role
    account.clinical_role = role
    account.organization = organization
    account.care_context = care_context or account.care_context
    account.follow_up_preferences = follow_up_preferences
    account.email_verified = True
    account.is_active = True
    if not verify_password(password, account.password_hash, account.password_algo, account.password_salt):
        password_data = hash_password(password)
        account.password_hash = password_data.hash
        account.password_salt = password_data.salt
        account.password_algo = password_data.algo
    return account


_NATURAL_KEYS = {
    Condition: ("name",),
    Medication: ("name",),
    Allergy: ("name",),
    VitalsEntry: ("recorded_on", "type"),
    SymptomLog: ("logged_for", "symptom"),
    ChatMessage: ("role", "content"),
    ClinicalNote: ("assessment",),
    TriageSummary: ("question",),
    CarePlan: ("condition",),
    PreVisitSummary: ("status", "summary_text"),
}


def _normal(value: object) -> str:
    return str(value or "").strip().casefold()


def _add_once(db: Session, row) -> None:
    """Add a seed row once, also recognising older demo rows by content."""

    model = type(row)
    seeded = db.get(model, row.id)
    natural_key = _NATURAL_KEYS.get(model, ())
    patient_id = getattr(row, "patient_id", None)
    if natural_key and patient_id is not None:
        candidates = db.scalars(select(model).where(model.patient_id == patient_id)).all()
        equivalent = next(
            (
                candidate
                for candidate in candidates
                if candidate.id != row.id
                and all(
                    _normal(getattr(candidate, field)) == _normal(getattr(row, field))
                    for field in natural_key
                )
            ),
            None,
        )
        if equivalent is not None:
            # A previous ad-hoc version of the demo chart already has this
            # record. Preserve it and remove only our redundant fixed-ID row.
            if seeded is not None:
                db.delete(seeded)
            return
    if seeded is None:
        db.add(row)


def _demo_patient(
    db: Session,
    *,
    slug: str,
    username: str,
    password: str,
    email: str,
    display_name: str,
    mrn: str,
    date_of_birth: date,
    biological_sex: str,
) -> Patient:
    account = _account(
        db,
        username=username,
        password=password,
        email=email,
        display_name=display_name,
        kind=AccountKind.patient,
        role="Patient / Individual",
    )
    patient = db.scalar(select(Patient).where(Patient.account_id == account.id))
    mrn_owner = db.scalar(select(Patient).where(Patient.patient_id == mrn))
    if mrn_owner is not None and (patient is None or mrn_owner.id != patient.id):
        raise RuntimeError(f"Cannot seed {display_name}: MRN {mrn} belongs to another patient.")
    if patient is None:
        patient = Patient(
            id=_id(f"patient:{slug}"),
            account_id=account.id,
            patient_id=mrn,
            date_of_birth=date_of_birth,
            biological_sex=biological_sex,
            dob_recorded_at=_utc(2026, 8, 1),
            longitudinal_memory={},
        )
        db.add(patient)
        db.flush()
    else:
        patient.patient_id = mrn
        patient.date_of_birth = patient.date_of_birth or date_of_birth
        patient.biological_sex = patient.biological_sex or biological_sex
    return patient


def _active_demo_grant(
    db: Session,
    *,
    slug: str,
    patient: Patient,
    clinician: Account,
    reason: str,
) -> ConsentGrant:
    grant_id = _id(f"consent:{slug}:omar")
    grant = db.get(ConsentGrant, grant_id)
    if grant is None:
        grant = db.scalar(
            select(ConsentGrant).where(
                ConsentGrant.patient_id == patient.id,
                ConsentGrant.clinician_account_id == clinician.id,
                ConsentGrant.status.in_([ConsentStatus.pending, ConsentStatus.active]),
            )
        )
    if grant is None:
        grant = ConsentGrant(
            id=grant_id,
            patient_id=patient.id,
            clinician_account_id=clinician.id,
            requested_at=_utc(2026, 8, 1),
        )
        db.add(grant)
    grant.status = ConsentStatus.active
    grant.scope = [ConsentScope.previsit_summary.value, ConsentScope.chat_history.value]
    grant.request_reason = reason
    grant.decision_note = "Approved for the fictional FlynnMed demonstration."
    grant.decided_at = _utc(2026, 8, 1)
    grant.expires_at = None
    grant.revoked_at = None
    grant.revoked_by_account_id = None
    db.flush()
    return grant


def seed_demo_accounts(db: Session) -> dict[str, str | int]:
    """Seed the demo patients, their charts, and Omar's active consent grants."""

    jane_account = _account(
        db,
        username=PATIENT_USERNAME,
        password=PATIENT_PASSWORD,
        email="demo.patient.jane@flynnmed.example",
        display_name="Jane Whitfield",
        kind=AccountKind.patient,
        role="Patient / Individual",
    )
    omar_account = _account(
        db,
        username=CLINICIAN_USERNAME,
        password=CLINICIAN_PASSWORD,
        email="demo.dr.omar@flynnmed.example",
        display_name="Dr. Omar Farouk",
        kind=AccountKind.clinician,
        role="Doctor / Physician",
        organization="FlynnMed Demonstration Clinic",
        care_context=(
            "General practice and longitudinal care: consented chart review, medicines reconciliation, "
            "pre-visit preparation, chronic-condition monitoring, and evidence-based patient education"
        ),
        follow_up_preferences="Prioritise overdue reviews, medication safety, and changes in symptoms or observations",
    )

    patient = db.execute(select(Patient).where(Patient.account_id == jane_account.id)).scalar_one_or_none()
    mrn_owner = db.execute(select(Patient).where(Patient.patient_id == PATIENT_MRN)).scalar_one_or_none()
    if mrn_owner is not None and (patient is None or mrn_owner.id != patient.id):
        raise RuntimeError(f"Cannot seed Jane: MRN {PATIENT_MRN} belongs to another patient.")
    if patient is None:
        patient = Patient(
            id=_id("patient:jane"),
            account_id=jane_account.id,
            patient_id=PATIENT_MRN,
            date_of_birth=date(1987, 4, 16),
            biological_sex="Female",
            dob_recorded_at=_utc(2026, 8, 1),
            longitudinal_memory={},
        )
        db.add(patient)
        db.flush()
    else:
        patient.patient_id = PATIENT_MRN
        patient.date_of_birth = patient.date_of_birth or date(1987, 4, 16)
        patient.biological_sex = patient.biological_sex or "Female"

    # These fixed IDs prevent duplicate chart entries on container restarts.
    records = [
        Condition(
            id=_id("condition:asthma"), patient_id=patient.id, name="Asthma", status="active",
            recorded_on="2014-03-12", notes="Mild intermittent asthma; usually triggered by cold air and pollen.",
        ),
        Condition(
            id=_id("condition:migraine"), patient_id=patient.id, name="Migraine without aura", status="active",
            recorded_on="2019-09-05", notes="Typically one or two episodes per month.",
        ),
        Condition(
            id=_id("condition:iron-deficiency"), patient_id=patient.id, name="Iron-deficiency anaemia", status="resolved",
            recorded_on="2023-06-20", notes="Resolved after oral iron treatment; monitor if fatigue returns.",
        ),
        Medication(
            id=_id("medication:salbutamol"), patient_id=patient.id, name="Salbutamol inhaler", dose="100 micrograms",
            schedule="1-2 puffs when needed", reason="Asthma symptom relief", started_on="2014-03-12",
            notes="Usually needed less than once per week.",
        ),
        Medication(
            id=_id("medication:sumatriptan"), patient_id=patient.id, name="Sumatriptan", dose="50 mg",
            schedule="At migraine onset; may repeat once after 2 hours", reason="Acute migraine", started_on="2021-02-08",
            notes="Do not exceed the prescribed daily dose.",
        ),
        Allergy(
            id=_id("allergy:penicillin"), patient_id=patient.id, name="Penicillin", reaction="Raised itchy rash",
            severity="moderate", allergy_type="drug", confirmed=True, notes="Reaction documented in 2009; no anaphylaxis.",
        ),
        VitalsEntry(
            id=_id("vitals:blood-pressure"), patient_id=patient.id, recorded_on="2026-08-12",
            type="Blood pressure", value="118/76", unit="mmHg", notes="Home reading, seated after five minutes rest.",
        ),
        VitalsEntry(
            id=_id("vitals:heart-rate"), patient_id=patient.id, recorded_on="2026-08-12",
            type="Heart rate", value="72", unit="bpm", notes="Resting.",
        ),
        VitalsEntry(
            id=_id("vitals:weight"), patient_id=patient.id, recorded_on="2026-08-10",
            type="Weight", value="68.4", unit="kg", notes="Morning measurement.",
        ),
        SymptomLog(
            id=_id("symptom:migraine-august"), patient_id=patient.id, symptom="Migraine headache",
            logged_for="2026-08-14", severity=6, triggers="Poor sleep after a late shift",
            notes="Improved after sumatriptan and resting in a dark room.",
        ),
        SymptomLog(
            id=_id("symptom:wheeze-july"), patient_id=patient.id, symptom="Mild wheeze",
            logged_for="2026-07-29", severity=3, triggers="Outdoor exercise during high pollen count",
            notes="Settled after one dose of reliever inhaler.",
        ),
        ChatMessage(
            id=_id("chat:jane-question"), patient_id=patient.id, role="user",
            content="I have had two migraines this month. What should I track before my next review?",
            timestamp=_utc(2026, 8, 14, 18), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("chat:assistant-answer"), patient_id=patient.id, role="assistant",
            content=(
                "Keep a headache diary with the date, duration, severity, possible triggers, medicines taken, "
                "and how well they worked. Seek urgent help for a sudden severe headache or new neurological symptoms."
            ),
            timestamp=_utc(2026, 8, 14, 18), sources=[], trace_id="demo-jane-migraine",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ChatMessage(
            id=_id("chat:jane-asthma-question"), patient_id=patient.id, role="user",
            content="How often is too often to need my blue reliever inhaler?",
            timestamp=_utc(2026, 8, 16, 10), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("chat:jane-asthma-answer"), patient_id=patient.id, role="assistant",
            content=(
                "Needing a reliever more often than usual, waking at night, or limiting activity can mean asthma "
                "control needs review. Follow your asthma action plan and arrange a clinician or asthma-nurse review; "
                "seek urgent help for severe breathlessness, difficulty speaking, or poor response to the inhaler."
            ),
            timestamp=_utc(2026, 8, 16, 10), sources=[], trace_id="demo-jane-asthma",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ChatMessage(
            id=_id("chat:jane-allergy-question"), patient_id=patient.id, role="user",
            content="What information should I tell a new clinician about my penicillin allergy?",
            timestamp=_utc(2026, 8, 17, 13), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("chat:jane-allergy-answer"), patient_id=patient.id, role="assistant",
            content=(
                "Tell them the medicine name, that you developed a raised itchy rash, approximately when it happened, "
                "how soon it appeared after the dose, and that you did not have anaphylaxis. Do not retry penicillin "
                "unless a qualified clinician has reviewed the allergy record and advised it."
            ),
            timestamp=_utc(2026, 8, 17, 13), sources=[], trace_id="demo-jane-allergy",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ChatMessage(
            id=_id("chat:jane-sleep-question"), patient_id=patient.id, role="user",
            content="Could poor sleep be contributing to my migraines, and what should I record?",
            timestamp=_utc(2026, 8, 18, 20), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("chat:jane-sleep-answer"), patient_id=patient.id, role="assistant",
            content=(
                "Sleep disruption can be a migraine trigger for some people. Record bedtime, wake time, sleep quality, "
                "shift patterns, migraine onset, hydration, meals, menstrual timing if relevant, and treatment response. "
                "A diary can help your clinician look for patterns without assuming one factor is the cause."
            ),
            timestamp=_utc(2026, 8, 18, 20), sources=[], trace_id="demo-jane-sleep",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ChatMessage(
            id=_id("chat:jane-exercise-question"), patient_id=patient.id, role="user",
            content="Can I keep exercising when pollen sometimes makes me wheeze?",
            timestamp=_utc(2026, 8, 19, 8), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("chat:jane-exercise-answer"), patient_id=patient.id, role="assistant",
            content=(
                "Regular activity is usually beneficial when asthma is controlled. Check pollen levels, warm up, carry "
                "your prescribed reliever, and follow your action plan. Stop and seek help if wheeze or breathlessness "
                "is severe, rapidly worsening, or does not improve as your plan says."
            ),
            timestamp=_utc(2026, 8, 19, 8), sources=[], trace_id="demo-jane-exercise",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ClinicalNote(
            id=_id("note:annual-review"), patient_id=patient.id,
            subjective="Migraine frequency remains approximately one to two episodes monthly. Asthma symptoms are infrequent.",
            objective="Home BP 118/76 mmHg, resting pulse 72 bpm. No current respiratory distress.",
            assessment="Migraine without aura and mild intermittent asthma, both currently stable.",
            plan="Continue symptom diaries and current prescribed medicines. Review inhaler technique and migraine frequency at routine appointment.",
            urgency_level="routine", requires_gp_visit=False, gp_visit_reason="Routine medication and symptom review.",
            edited_by_account_id=omar_account.id,
        ),
        TriageSummary(
            id=_id("triage:migraine"), patient_id=patient.id,
            question="Two familiar migraines this month without new neurological symptoms.", urgency_level="routine",
            next_step="Continue the headache diary and arrange a routine review if frequency increases.",
            what_to_monitor=["headache frequency", "duration", "new neurological symptoms", "medicine use"],
            rationale="The reported pattern is familiar and no emergency warning signs were recorded.",
            pathway_label="Headache and migraine", decision_summary="Routine self-monitoring with safety-net advice.",
            immediate_actions=["Record this episode in the headache diary", "Use prescribed treatment as directed"],
            escalation_triggers=["sudden severe headache", "weakness", "confusion", "speech or vision change"],
            communication_points=["Discuss increasing frequency or medicine use with a clinician"],
            rule_hits=[], guideline_references=[], logic_version="demo-seed-v1", trace_id="demo-triage-jane",
        ),
        CarePlan(
            id=_id("care-plan:migraine"), patient_id=patient.id, condition="Migraine without aura", status="active",
            body={
                "id": "demo-migraine-plan",
                "title": "Migraine self-management plan",
                "goals": [{"id": "demo-goal-diary", "text": "Identify patterns and reduce disruption from migraines"}],
                "daily_tasks": [{"id": "demo-task-hydration", "text": "Keep regular meals and hydration", "completed_dates": []}],
                "weekly_tasks": [{"id": "demo-task-review", "text": "Review headache diary for patterns", "completed_dates": []}],
                "medication_reminders": [], "lab_reminders": [],
                "escalation_thresholds": [{"id": "demo-escalation", "trigger": "A sudden severe headache or new neurological symptoms", "action": "Seek urgent medical help"}],
                "missed_care_checklist": [], "after_visit_notes": [],
            },
            clinical_context={"source": "fictional demo seed"}, validation={"valid": True},
            gp_prep_summary="Review migraine frequency, response to sumatriptan, and whether preventive treatment is indicated.",
        ),
    ]
    for record in records:
        _add_once(db, record)

    db.flush()
    grant = db.get(ConsentGrant, _id("consent:jane:omar"))
    if grant is None:
        grant = db.execute(
            select(ConsentGrant).where(
                ConsentGrant.patient_id == patient.id,
                ConsentGrant.clinician_account_id == omar_account.id,
                ConsentGrant.status.in_([ConsentStatus.pending, ConsentStatus.active]),
            )
        ).scalar_one_or_none()
    if grant is None:
        grant = ConsentGrant(
            id=_id("consent:jane:omar"), patient_id=patient.id, clinician_account_id=omar_account.id,
            requested_at=_utc(2026, 8, 1),
        )
        db.add(grant)
    grant.status = ConsentStatus.active
    grant.scope = [ConsentScope.previsit_summary.value, ConsentScope.chat_history.value]
    grant.request_reason = "Ongoing review of migraine and asthma management (fictional demo)."
    grant.decision_note = "Approved by Jane for the FlynnMed demonstration."
    grant.decided_at = _utc(2026, 8, 1)
    grant.expires_at = None
    grant.revoked_at = None
    grant.revoked_by_account_id = None
    db.flush()

    _add_once(
        db,
        PreVisitSummary(
            id=_id("previsit-summary:jane:omar"), patient_id=patient.id, status="released",
            generation_trigger="released",
            summary_text=(
                "Fictional demo record: Jane's migraine and mild intermittent asthma are stable. "
                "Review recent headache frequency, response to sumatriptan, inhaler technique, and reliever use. "
                "Penicillin allergy is recorded as a moderate rash."
            ),
            authored_by_account_id=omar_account.id, authored_by_display_name=omar_account.display_name,
            authored_by_clinical_role=omar_account.clinical_role,
            authored_by_organization=omar_account.organization, consent_grant_id=grant.id,
            released_at=_utc(2026, 8, 15), released_by_account_id=omar_account.id,
            released_by_display_name=omar_account.display_name,
            released_by_clinical_role=omar_account.clinical_role,
        ),
    )

    michael = _demo_patient(
        db,
        slug="michael",
        username=MICHAEL_USERNAME,
        password=MICHAEL_PASSWORD,
        email="demo.patient.michael@flynnmed.example",
        display_name="Michael Reed",
        mrn=MICHAEL_MRN,
        date_of_birth=date(1963, 11, 2),
        biological_sex="Male",
    )
    michael_records = [
        Condition(
            id=_id("michael:condition:diabetes"), patient_id=michael.id,
            name="Type 2 diabetes mellitus", status="active", recorded_on="2018-05-14",
            notes="Managed with oral medication, diet, activity, and routine monitoring.",
        ),
        Condition(
            id=_id("michael:condition:hypertension"), patient_id=michael.id,
            name="Hypertension", status="active", recorded_on="2020-01-22",
            notes="Home blood-pressure readings requested before the next review.",
        ),
        Medication(
            id=_id("michael:medication:metformin"), patient_id=michael.id,
            name="Metformin", dose="1,000 mg", schedule="Twice daily with meals",
            reason="Type 2 diabetes", started_on="2018-05-14", notes="Tolerated without current concerns.",
        ),
        Medication(
            id=_id("michael:medication:amlodipine"), patient_id=michael.id,
            name="Amlodipine", dose="5 mg", schedule="Once daily",
            reason="Hypertension", started_on="2020-01-22", notes="Review alongside home BP diary.",
        ),
        Allergy(
            id=_id("michael:allergy:sulfonamide"), patient_id=michael.id,
            name="Sulfonamide antibiotics", reaction="Widespread rash", severity="moderate",
            allergy_type="drug", confirmed=True, notes="No breathing difficulty reported.",
        ),
        VitalsEntry(
            id=_id("michael:vitals:bp"), patient_id=michael.id, recorded_on="2026-08-13",
            type="Blood pressure", value="142/86", unit="mmHg", notes="Seven-day home average.",
        ),
        VitalsEntry(
            id=_id("michael:vitals:hba1c"), patient_id=michael.id, recorded_on="2026-07-30",
            type="HbA1c", value="58", unit="mmol/mol", notes="Most recent routine diabetes blood test.",
        ),
        SymptomLog(
            id=_id("michael:symptom:tingling"), patient_id=michael.id,
            symptom="Intermittent tingling in both feet", logged_for="2026-08-11", severity=3,
            triggers="More noticeable in the evening", notes="No wound, colour change, or weakness reported.",
        ),
        ChatMessage(
            id=_id("michael:chat:question"), patient_id=michael.id, role="user",
            content="What should I bring to my diabetes and blood-pressure review?",
            timestamp=_utc(2026, 8, 13, 17), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("michael:chat:answer"), patient_id=michael.id, role="assistant",
            content=(
                "Bring your current medicines, home blood-pressure readings, glucose records if you use them, and a "
                "list of symptoms such as the foot tingling. Ask about HbA1c, kidney and cholesterol results, foot and "
                "eye checks, and when follow-up is due. Seek prompt care for a foot wound, spreading redness, or sudden weakness."
            ),
            timestamp=_utc(2026, 8, 13, 17), sources=[], trace_id="demo-michael-review",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ClinicalNote(
            id=_id("michael:note:review"), patient_id=michael.id,
            subjective="Home BP average above target and new intermittent bilateral foot tingling.",
            objective="Home BP average 142/86 mmHg; HbA1c 58 mmol/mol.",
            assessment="Type 2 diabetes and hypertension require routine review; assess possible peripheral neuropathy.",
            plan="Medication reconciliation, foot examination, repeat risk-factor review, and agree an individual BP target.",
            urgency_level="routine", requires_gp_visit=True,
            gp_visit_reason="Diabetes review, BP review, and assessment of new foot symptoms.",
            edited_by_account_id=omar_account.id,
        ),
    ]
    for record in michael_records:
        _add_once(db, record)
    _active_demo_grant(
        db,
        slug="michael",
        patient=michael,
        clinician=omar_account,
        reason="Diabetes, blood-pressure, and new foot-symptom review (fictional demo).",
    )

    aisha = _demo_patient(
        db,
        slug="aisha",
        username=AISHA_USERNAME,
        password=AISHA_PASSWORD,
        email="demo.patient.aisha@flynnmed.example",
        display_name="Aisha Khan",
        mrn=AISHA_MRN,
        date_of_birth=date(1996, 6, 24),
        biological_sex="Female",
    )
    aisha_records = [
        Condition(
            id=_id("aisha:condition:hypothyroidism"), patient_id=aisha.id,
            name="Hypothyroidism", status="active", recorded_on="2022-10-03",
            notes="Stable on replacement therapy at the last review.",
        ),
        Condition(
            id=_id("aisha:condition:anxiety"), patient_id=aisha.id,
            name="Generalised anxiety disorder", status="active", recorded_on="2024-02-12",
            notes="Managed with talking therapy, sleep routine, and clinical follow-up.",
        ),
        Medication(
            id=_id("aisha:medication:levothyroxine"), patient_id=aisha.id,
            name="Levothyroxine", dose="75 micrograms", schedule="Once daily before breakfast",
            reason="Hypothyroidism", started_on="2022-10-03", notes="Taken separately from iron or calcium supplements.",
        ),
        Allergy(
            id=_id("aisha:allergy:latex"), patient_id=aisha.id,
            name="Latex", reaction="Contact rash", severity="mild", allergy_type="other",
            confirmed=True, notes="Use latex-free gloves and equipment where possible.",
        ),
        VitalsEntry(
            id=_id("aisha:vitals:pulse"), patient_id=aisha.id, recorded_on="2026-08-09",
            type="Heart rate", value="78", unit="bpm", notes="Resting.",
        ),
        VitalsEntry(
            id=_id("aisha:vitals:weight"), patient_id=aisha.id, recorded_on="2026-08-09",
            type="Weight", value="61.2", unit="kg", notes="Stable compared with the previous month.",
        ),
        SymptomLog(
            id=_id("aisha:symptom:fatigue"), patient_id=aisha.id,
            symptom="Fatigue and poor concentration", logged_for="2026-08-08", severity=5,
            triggers="Several nights of interrupted sleep", notes="No fainting, chest pain, or shortness of breath.",
        ),
        ChatMessage(
            id=_id("aisha:chat:question"), patient_id=aisha.id, role="user",
            content="How can I tell whether my tiredness is from sleep, anxiety, or my thyroid?",
            timestamp=_utc(2026, 8, 9, 19), sources=[], trace_id=None,
            message_metadata={"demo_seed": True},
        ),
        ChatMessage(
            id=_id("aisha:chat:answer"), patient_id=aisha.id, role="assistant",
            content=(
                "Those causes can overlap and symptoms alone may not distinguish them. Track sleep, mood, medicine timing, "
                "energy, menstrual changes, and other symptoms, and arrange a routine review if fatigue persists. A clinician "
                "can review thyroid blood tests and consider other causes; seek urgent help for severe or rapidly worsening symptoms."
            ),
            timestamp=_utc(2026, 8, 9, 19), sources=[], trace_id="demo-aisha-fatigue",
            message_metadata={"demo_seed": True, "educational_only": True},
        ),
        ClinicalNote(
            id=_id("aisha:note:fatigue"), patient_id=aisha.id,
            subjective="One week of fatigue and reduced concentration after interrupted sleep; anxiety also increased.",
            objective="Resting pulse 78 bpm and stable weight. No red-flag symptoms reported.",
            assessment="Non-specific fatigue with sleep and anxiety contributors; thyroid control should be checked if persistent.",
            plan="Review levothyroxine adherence and timing, sleep pattern, mood, and recent thyroid-function results.",
            urgency_level="routine", requires_gp_visit=False,
            gp_visit_reason="Routine review if fatigue persists or worsens.", edited_by_account_id=omar_account.id,
        ),
    ]
    for record in aisha_records:
        _add_once(db, record)
    _active_demo_grant(
        db,
        slug="aisha",
        patient=aisha,
        clinician=omar_account,
        reason="Review of fatigue, thyroid replacement, and wellbeing (fictional demo).",
    )

    db.commit()
    return {
        "patient_username": PATIENT_USERNAME,
        "patient_mrn": PATIENT_MRN,
        "clinician_username": CLINICIAN_USERNAME,
        "consent_status": ConsentStatus.active.value,
        "active_patient_count": 3,
    }


def main() -> None:
    with get_session_factory()() as db:
        result = seed_demo_accounts(db)
    print(
        "Demo accounts ready: "
        f"{result['patient_username']} ({result['patient_mrn']}) and "
        f"{result['clinician_username']} with {result['active_patient_count']} active patient grants."
    )


if __name__ == "__main__":
    main()
