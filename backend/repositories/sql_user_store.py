"""SQL-backed equivalent of backend/user_store.py's UserStore, with an
identical public method surface (same names, same `username`-keyed
signatures, same dict-shaped return values) so backend/user_store.py can
dispatch to this module behind the DATA_BACKEND flag without any caller
(backend/api.py) changing.

Two structural differences from the legacy store, both deliberate:

- IDs returned in dicts (medication_id, allergy_id, ...) are the SQL row's
  UUID as a string, not the legacy `med-xxxxxxxxxxxx`-style short id. Both
  are opaque identifiers to every caller (used only for lookup/delete), so
  this is a safe format change, not a behavior change.
- `account_kind=clinician` accounts have no Patient row (see backend/models/
  account.py's AccountKind docstring) -- every patient-scoped method below
  returns the same "not found" result (None/[]/False) for such accounts
  that it would for a nonexistent user, rather than raising.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.auth.passwords import hash_password, needs_rehash, verify_password
from backend.db import get_session_factory
from backend.models.account import Account, AccountKind
from backend.models.activity import AccountActivityLog
from backend.models.patient import (
    Allergy,
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
from backend.mrn import generate_mrn
from backend.product_config import is_clinician_role
from backend.relationship_engine import RELATION_CLASS

_VALID_SEX_OPTIONS = {"Male", "Female", "Other", "Prefer not to say"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _session() -> Session:
    return get_session_factory()()


def _get_account(db: Session, username: str) -> Optional[Account]:
    key = _normalize_username(username)
    if not key:
        return None
    return db.execute(select(Account).where(Account.username == key)).scalar_one_or_none()


def _get_patient(db: Session, username: str) -> Optional[Patient]:
    account = _get_account(db, username)
    if account is None or account.account_kind != AccountKind.patient:
        return None
    return db.execute(select(Patient).where(Patient.account_id == account.id)).scalar_one_or_none()


def _append_activity(db: Session, account_id: uuid.UUID, event: str, details: str, *, trace_id: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
    db.add(
        AccountActivityLog(
            account_id=account_id,
            event=event,
            details=details,
            trace_id=trace_id,
            event_metadata=metadata or {},
        )
    )


# ── Serializers: ORM row -> legacy-shaped dict ───────────────────────────────


def _medication_to_dict(m: Medication) -> Dict:
    return {
        "medication_id": str(m.id),
        "name": m.name,
        "dose": m.dose,
        "schedule": m.schedule,
        "reason": m.reason,
        "started_on": m.started_on,
        "notes": m.notes,
        "created_at": _iso(m.created_at),
        "updated_at": _iso(m.updated_at),
    }


def _condition_to_dict(c: Condition) -> Dict:
    return {
        "condition_id": str(c.id),
        "name": c.name,
        "status": c.status,
        "recorded_on": c.recorded_on,
        "notes": c.notes,
        "created_at": _iso(c.created_at),
        "updated_at": _iso(c.updated_at),
    }


def _allergy_to_dict(a: Allergy) -> Dict:
    return {
        "allergy_id": str(a.id),
        "name": a.name,
        "reaction": a.reaction,
        "severity": a.severity,
        "allergy_type": a.allergy_type,
        "confirmed": a.confirmed,
        "notes": a.notes,
        "created_at": _iso(a.created_at),
    }


def _vitals_to_dict(v: VitalsEntry) -> Dict:
    return {
        "vitals_id": str(v.id),
        "recorded_on": v.recorded_on,
        "type": v.type,
        "value": v.value,
        "unit": v.unit,
        "notes": v.notes,
        "created_at": _iso(v.created_at),
    }


def _symptom_log_to_dict(s: SymptomLog) -> Dict:
    return {
        "log_id": str(s.id),
        "symptom": s.symptom,
        "logged_for": s.logged_for,
        "severity": s.severity,
        "triggers": s.triggers,
        "notes": s.notes,
        "created_at": _iso(s.created_at),
    }


def _triage_summary_to_dict(t: TriageSummary) -> Dict:
    return {
        "summary_id": str(t.id),
        "question": t.question,
        "urgency_level": t.urgency_level,
        "next_step": t.next_step,
        "what_to_monitor": t.what_to_monitor,
        "rationale": t.rationale,
        "pathway_label": t.pathway_label,
        "decision_summary": t.decision_summary,
        "immediate_actions": t.immediate_actions,
        "escalation_triggers": t.escalation_triggers,
        "communication_points": t.communication_points,
        "rule_hits": t.rule_hits,
        "guideline_references": t.guideline_references,
        "logic_version": t.logic_version,
        "trace_id": t.trace_id,
        "created_at": _iso(t.created_at),
    }


def _chat_message_to_dict(m: ChatMessage) -> Dict:
    return {
        "message_id": str(m.id),
        "role": m.role,
        "content": m.content,
        "timestamp": _iso(m.timestamp),
        "sources": m.sources,
        "trace_id": m.trace_id,
        "metadata": m.message_metadata,
    }


def _clinical_note_to_dict(n: ClinicalNote) -> Dict:
    return {
        "note_id": str(n.id),
        "subjective": n.subjective,
        "objective": n.objective,
        "assessment": n.assessment,
        "plan": n.plan,
        "urgency_level": n.urgency_level,
        "requires_gp_visit": n.requires_gp_visit,
        "gp_visit_reason": n.gp_visit_reason,
        "email_sent": n.email_sent,
        "email_sent_at": _iso(n.email_sent_at),
        "created_at": _iso(n.created_at),
        "updated_at": _iso(n.updated_at),
    }


def _upload_to_dict(u: Upload) -> Dict:
    return {
        "file": u.file_name,
        "uploaded_at": _iso(u.created_at),
        "stored_path": u.stored_path,
        "content_hash": u.content_hash,
        "summary_available": u.summary_available,
    }


def _trace_to_dict(t: InteractionTrace) -> Dict:
    payload = dict(t.payload or {})
    payload.setdefault("question", "")
    payload.setdefault("answer_preview", "")
    payload.setdefault("sources", [])
    return {"trace_id": t.trace_id, "created_at": _iso(t.created_at), **payload}


class SqlUserStore:
    """SQL-backed twin of backend.user_store.UserStore. See module docstring."""

    # ── Account lifecycle ────────────────────────────────────────────────

    @staticmethod
    def create_user(
        username: str,
        password: str,
        display_name: Optional[str] = None,
        email: str = "",
        care_context: str = "Personal health guidance",
        role: str = "Individual",
        clinical_role: str = "",
        organization: str = "",
        terms_version: str = "",
        terms_role: str = "",
        terms_accepted_at: str = "",
        privacy_accepted_at: str = "",
        date_of_birth: str = "",
        biological_sex: str = "",
    ) -> bool:
        key = _normalize_username(username)
        normalized_email = (email or "").strip().lower()
        if not key or len(password) < 8 or "@" not in normalized_email:
            return False

        with _session() as db:
            if _get_account(db, key) is not None:
                return False
            existing_email = db.execute(select(Account).where(Account.email == normalized_email)).scalar_one_or_none()
            if existing_email is not None:
                return False

            hashed = hash_password(password)
            kind = AccountKind.clinician if is_clinician_role(role) else AccountKind.patient
            cleaned_dob = (date_of_birth or "").strip()[:10]
            cleaned_sex = (biological_sex or "").strip()
            if cleaned_sex not in _VALID_SEX_OPTIONS:
                cleaned_sex = ""

            account = Account(
                id=uuid.uuid4(),
                username=key,
                email=normalized_email,
                display_name=(display_name or key).strip() or key,
                password_hash=hashed.hash,
                password_salt=hashed.salt,
                password_algo=hashed.algo,
                account_kind=kind,
                role_label=role.strip() or "Individual",
                clinical_role=clinical_role.strip(),
                organization=organization.strip(),
                care_context=care_context.strip() or "Personal health guidance",
                follow_up_preferences="",
                terms_version=terms_version.strip(),
                terms_role=(terms_role or clinical_role or role).strip(),
                terms_accepted_at=_parse_dt(terms_accepted_at) or _utc_now(),
                privacy_accepted_at=_parse_dt(privacy_accepted_at) or _utc_now(),
            )
            db.add(account)
            db.flush()

            if kind == AccountKind.patient:
                patient = Patient(
                    id=uuid.uuid4(),
                    account_id=account.id,
                    patient_id=generate_mrn(),
                    date_of_birth=_parse_date(cleaned_dob),
                    biological_sex=cleaned_sex,
                    dob_recorded_at=_utc_now() if cleaned_dob else None,
                    longitudinal_memory={},
                )
                db.add(patient)

            _append_activity(db, account.id, "account_created", "Account created", metadata={"terms_version": account.terms_version})
            db.commit()
            return True

    @staticmethod
    def resolve_login_username(identifier: str) -> Optional[str]:
        key = (identifier or "").strip().lower()
        if not key:
            return None
        with _session() as db:
            if _get_account(db, key) is not None:
                return key
            account = db.execute(select(Account).where(Account.email == key)).scalar_one_or_none()
            return account.username if account else None

    @staticmethod
    def authenticate(username: str, password: str) -> bool:
        resolved = SqlUserStore.resolve_login_username(username)
        if not resolved:
            return False
        with _session() as db:
            account = _get_account(db, resolved)
            if account is None:
                return False
            ok = verify_password(password, account.password_hash, account.password_algo, account.password_salt)
            if ok and needs_rehash(account.password_algo):
                upgraded = hash_password(password)
                account.password_hash = upgraded.hash
                account.password_algo = upgraded.algo
                account.password_salt = upgraded.salt
                db.commit()
            return ok

    @staticmethod
    def update_last_login(username: str) -> None:
        with _session() as db:
            account = _get_account(db, username)
            if account is None:
                return
            account.last_login_at = _utc_now()
            _append_activity(db, account.id, "login", "User logged in")
            db.commit()

    @staticmethod
    def is_email_verified(username: str) -> bool:
        with _session() as db:
            account = _get_account(db, username)
            return bool(account and account.email_verified)

    @staticmethod
    def set_email_verified(username: str) -> None:
        with _session() as db:
            account = _get_account(db, username)
            if account is None:
                return
            account.email_verified = True
            _append_activity(db, account.id, "email_verified", "Email address verified")
            db.commit()

    # ── Profile ───────────────────────────────────────────────────────────

    @staticmethod
    def get_user_profile(username: str) -> Dict:
        with _session() as db:
            account = _get_account(db, username)
            if account is None:
                return {}
            patient = _get_patient(db, username)
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
                "last_video_generated_at": _iso(patient.last_video_generated_at) if patient else "" or "",
                "date_of_birth": patient.date_of_birth.isoformat() if patient and patient.date_of_birth else "",
                "biological_sex": patient.biological_sex if patient else "",
                "dob_recorded_at": _iso(patient.dob_recorded_at) if patient else "" or "",
                "created_at": _iso(account.created_at),
                "last_login": _iso(account.last_login_at),
                "active_conversation_id": None,
                "patient_record_id": str(patient.id) if patient else "",
            }

    @staticmethod
    def update_profile(username: str, updates: Dict[str, str]) -> bool:
        allowed_account_keys = {
            "display_name", "email", "care_context", "role", "clinical_role",
            "organization", "follow_up_preferences",
        }
        allowed_patient_keys = {"date_of_birth", "biological_sex", "last_video_generated_at"}

        with _session() as db:
            account = _get_account(db, username)
            if account is None:
                return False

            applied = {}
            for field, value in updates.items():
                if field == "email":
                    normalized_email = (value or "").strip().lower()
                    existing_owner = db.execute(select(Account).where(Account.email == normalized_email)).scalar_one_or_none()
                    if existing_owner and existing_owner.id != account.id:
                        return False
                    account.email = normalized_email
                    applied["email"] = normalized_email
                elif field in allowed_account_keys:
                    setattr(account, field, (value or "").strip())
                    applied[field] = (value or "").strip()
                elif field in allowed_patient_keys:
                    patient = _get_patient(db, username)
                    if patient is None:
                        continue
                    if field == "date_of_birth":
                        patient.date_of_birth = _parse_date((value or "").strip()[:10])
                    elif field == "biological_sex":
                        patient.biological_sex = (value or "").strip()
                    elif field == "last_video_generated_at":
                        patient.last_video_generated_at = _parse_dt(value)
                    applied[field] = value

            _append_activity(db, account.id, "profile_updated", "Profile details updated", metadata=applied)
            db.commit()
            return True

    # ── Symptom logs ──────────────────────────────────────────────────────

    @staticmethod
    def add_symptom_log(username: str, symptom: str, logged_for: str, severity: int, triggers: str = "", notes: str = "") -> Optional[Dict]:
        symptom = (symptom or "").strip()
        logged_for = (logged_for or "").strip()
        if not symptom or not logged_for:
            return None
        try:
            severity_value = max(0, min(10, int(severity)))
        except (TypeError, ValueError):
            severity_value = 0

        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return None
            entry = SymptomLog(
                id=uuid.uuid4(), patient_id=patient.id, symptom=symptom, logged_for=logged_for,
                severity=severity_value, triggers=(triggers or "").strip(), notes=(notes or "").strip(),
            )
            db.add(entry)
            db.flush()
            result = _symptom_log_to_dict(entry)
            db.commit()
            return result

    @staticmethod
    def get_symptom_logs(username: str, limit: Optional[int] = 50) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(SymptomLog).where(SymptomLog.patient_id == patient.id)
                .order_by(SymptomLog.logged_for.desc(), SymptomLog.created_at.desc())
            ).scalars().all()
            results = [_symptom_log_to_dict(r) for r in rows]
            return results[:limit] if limit is not None else results

    @staticmethod
    def delete_symptom_log(username: str, log_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            entry = _find_by_id(db, patient, SymptomLog, log_id)
            if entry is None:
                return False
            db.delete(entry)
            _append_activity(db, patient.account_id, "symptom_deleted", "Removed symptom tracker entry", metadata={"log_id": log_id})
            db.commit()
            return True

    # ── Medications ───────────────────────────────────────────────────────

    @staticmethod
    def save_medication(username: str, medication: Dict) -> Optional[Dict]:
        name = str(medication.get("name") or "").strip()
        if not name:
            return None
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return None
            existing = _find_by_id_or_name(db, patient, Medication, medication.get("medication_id"), name)
            if existing is None:
                existing = Medication(id=uuid.uuid4(), patient_id=patient.id, name=name)
                db.add(existing)
            existing.name = name
            existing.dose = str(medication.get("dose") or "").strip()
            existing.schedule = str(medication.get("schedule") or "").strip()
            existing.reason = str(medication.get("reason") or "").strip()
            existing.started_on = str(medication.get("started_on") or "").strip()
            existing.notes = str(medication.get("notes") or "").strip()
            db.flush()
            result = _medication_to_dict(existing)
            _append_activity(db, patient.account_id, "medication_saved", f"Saved medication: {name}", metadata={"medication_id": result["medication_id"]})
            db.commit()
            return result

    @staticmethod
    def get_medications(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(Medication).where(Medication.patient_id == patient.id)
                .order_by(Medication.name.asc(), Medication.updated_at.asc())
            ).scalars().all()
            return [_medication_to_dict(r) for r in rows]

    @staticmethod
    def delete_medication(username: str, medication_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            entry = _find_by_id(db, patient, Medication, medication_id)
            if entry is None:
                return False
            db.delete(entry)
            _append_activity(db, patient.account_id, "medication_deleted", "Removed medication from list", metadata={"medication_id": medication_id})
            db.commit()
            return True

    # ── Allergies ─────────────────────────────────────────────────────────

    @staticmethod
    def save_allergy(username: str, allergy: Dict) -> Optional[Dict]:
        name = str(allergy.get("name") or "").strip()
        if not name:
            return None
        severity = (allergy.get("severity") or "").strip().lower()
        if severity not in ("mild", "moderate", "severe"):
            severity = "unknown"
        allergy_type = (allergy.get("allergy_type") or "").strip().lower()
        if allergy_type not in ("drug", "food", "environmental", "other"):
            allergy_type = "other"

        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return None
            existing = _find_by_id_or_name(db, patient, Allergy, allergy.get("allergy_id"), name)
            if existing is None:
                existing = Allergy(id=uuid.uuid4(), patient_id=patient.id, name=name)
                db.add(existing)
            existing.name = name
            existing.reaction = str(allergy.get("reaction") or "").strip()
            existing.severity = severity
            existing.allergy_type = allergy_type
            existing.confirmed = bool(allergy.get("confirmed", True))
            existing.notes = str(allergy.get("notes") or "").strip()
            db.flush()
            result = _allergy_to_dict(existing)
            _append_activity(db, patient.account_id, "allergy_saved", f"Saved allergy: {name}", metadata={"allergy_id": result["allergy_id"], "severity": severity})
            db.commit()
            return result

    @staticmethod
    def get_allergies(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(Allergy).where(Allergy.patient_id == patient.id).order_by(Allergy.name.asc())
            ).scalars().all()
            return [_allergy_to_dict(r) for r in rows]

    @staticmethod
    def save_clinical_relationships(
        username: str, relationships: List[Dict]
    ) -> List[Dict]:
        """Store explicit conversation links inside the patient's JSONB memory."""
        allowed_relations = {
            "taken_for", "causes", "triggers", "worsens", "improves",
            "started_after", "allergic_reaction", "associated_with",
            "led_to", "recorded_in", "recommended_for",
        }
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            memory = dict(patient.longitudinal_memory or {})
            stored = list(memory.get("clinical_relationships") or [])
            changed = False
            for item in relationships or []:
                source_name = str(item.get("source_name") or "").strip()
                target_name = str(item.get("target_name") or "").strip()
                relation = str(item.get("relation") or "").strip().lower()
                if not source_name or not target_name or relation not in allowed_relations:
                    continue
                normalized = {
                    "relationship_id": item.get("relationship_id") or f"rel-{uuid.uuid4().hex[:12]}",
                    "source_type": str(item.get("source_type") or "other").strip().lower(),
                    "source_name": source_name,
                    "relation": relation,
                    "relation_class": str(
                        item.get("relation_class") or RELATION_CLASS.get(relation, "association")
                    ).strip(),
                    "target_type": str(item.get("target_type") or "other").strip().lower(),
                    "target_name": target_name,
                    "certainty": (
                        str(item.get("certainty") or "user_reported").strip().lower()
                        if str(item.get("certainty") or "user_reported").strip().lower()
                        in {"documented", "user_reported", "user_suspected", "recorded_association"}
                        else "user_reported"
                    ),
                    "evidence": str(item.get("evidence") or "").strip(),
                    "source": str(item.get("source") or "conversation").strip(),
                    "recorded_at": item.get("recorded_at") or _utc_now().isoformat(),
                }
                key = (
                    normalized["source_type"], normalized["source_name"].lower(),
                    normalized["relation"], normalized["target_type"],
                    normalized["target_name"].lower(),
                )
                existing_index = next(
                    (
                        index for index, existing in enumerate(stored)
                        if (
                            str(existing.get("source_type") or "").lower(),
                            str(existing.get("source_name") or "").lower(),
                            str(existing.get("relation") or "").lower(),
                            str(existing.get("target_type") or "").lower(),
                            str(existing.get("target_name") or "").lower(),
                        ) == key
                    ),
                    None,
                )
                if existing_index is None:
                    stored.append(normalized)
                else:
                    normalized["relationship_id"] = stored[existing_index].get(
                        "relationship_id", normalized["relationship_id"]
                    )
                    stored[existing_index] = normalized
                changed = True
            if changed:
                memory["clinical_relationships"] = stored
                patient.longitudinal_memory = memory
                _append_activity(
                    db,
                    patient.account_id,
                    "clinical_relationships_saved",
                    "Saved user-reported clinical relationships from conversation",
                    metadata={"relationship_count": len(stored)},
                )
                db.commit()
            return stored

    @staticmethod
    def get_clinical_relationships(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            return list(
                dict(patient.longitudinal_memory or {}).get(
                    "clinical_relationships", []
                )
                or []
            )

    @staticmethod
    def delete_clinical_relationship(username: str, relationship_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return False
            memory = dict(patient.longitudinal_memory or {})
            stored = list(memory.get("clinical_relationships") or [])
            kept = [
                item for item in stored
                if str(item.get("relationship_id") or "") != relationship_id
            ]
            if len(kept) == len(stored):
                return False
            memory["clinical_relationships"] = kept
            patient.longitudinal_memory = memory
            _append_activity(
                db,
                patient.account_id,
                "clinical_relationship_deleted",
                "Removed user-reported clinical relationship",
                metadata={"relationship_id": relationship_id},
            )
            db.commit()
            return True

    @staticmethod
    def delete_clinical_relationships_for_entity(
        username: str, entity_type: str, entity_name: str
    ) -> int:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None or not entity_name:
                return 0
            memory = dict(patient.longitudinal_memory or {})
            stored = list(memory.get("clinical_relationships") or [])
            kind = (entity_type or "").strip().lower()
            name = entity_name.strip().lower()
            kept = [
                item
                for item in stored
                if not (
                    (
                        str(item.get("source_type") or "").lower() == kind
                        and str(item.get("source_name") or "").lower() == name
                    )
                    or (
                        str(item.get("target_type") or "").lower() == kind
                        and str(item.get("target_name") or "").lower() == name
                    )
                )
            ]
            removed = len(stored) - len(kept)
            if removed:
                memory["clinical_relationships"] = kept
                patient.longitudinal_memory = memory
                _append_activity(
                    db,
                    patient.account_id,
                    "clinical_relationships_pruned",
                    f"Removed relationships for deleted {kind}: {entity_name}",
                    metadata={
                        "entity_type": kind,
                        "entity_name": entity_name,
                        "removed": removed,
                    },
                )
                db.commit()
            return removed

    @staticmethod
    def delete_allergy(username: str, allergy_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            entry = _find_by_id(db, patient, Allergy, allergy_id)
            if entry is None:
                return False
            db.delete(entry)
            _append_activity(db, patient.account_id, "allergy_deleted", "Removed allergy entry", metadata={"allergy_id": allergy_id})
            db.commit()
            return True

    # ── Conditions ────────────────────────────────────────────────────────

    @staticmethod
    def save_condition(username: str, condition: Dict) -> Optional[Dict]:
        name = str(condition.get("name") or "").strip()
        if not name:
            return None
        status = (condition.get("status") or "").strip().lower()
        if status not in ("active", "past", "resolved", "unknown"):
            status = "unknown"

        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return None
            existing = _find_by_id_or_name(db, patient, Condition, condition.get("condition_id"), name)
            if existing is None:
                existing = Condition(id=uuid.uuid4(), patient_id=patient.id, name=name)
                db.add(existing)
            existing.name = name
            existing.status = status
            existing.recorded_on = str(condition.get("recorded_on") or "").strip()
            existing.notes = str(condition.get("notes") or "").strip()
            db.flush()
            result = _condition_to_dict(existing)
            _append_activity(db, patient.account_id, "condition_saved", f"Saved condition: {name}", metadata={"condition_id": result["condition_id"], "status": status})
            db.commit()
            return result

    @staticmethod
    def get_conditions(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(select(Condition).where(Condition.patient_id == patient.id)).scalars().all()
            rows = sorted(rows, key=lambda item: (item.status != "active", item.name.lower()))
            return [_condition_to_dict(r) for r in rows]

    @staticmethod
    def delete_condition(username: str, condition_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            entry = _find_by_id(db, patient, Condition, condition_id)
            if entry is None:
                return False
            db.delete(entry)
            _append_activity(db, patient.account_id, "condition_deleted", "Removed condition from history", metadata={"condition_id": condition_id})
            db.commit()
            return True

    # ── Vitals ────────────────────────────────────────────────────────────

    @staticmethod
    def save_vitals_entry(username: str, entry: Dict) -> Optional[Dict]:
        vtype = str(entry.get("type") or "").strip()
        value = str(entry.get("value") or "").strip()
        if not vtype or not value:
            return None
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return None
            row = VitalsEntry(
                id=uuid.uuid4(), patient_id=patient.id, recorded_on=str(entry.get("recorded_on") or "").strip(),
                type=vtype, value=value, unit=str(entry.get("unit") or "").strip(), notes=str(entry.get("notes") or "").strip(),
            )
            db.add(row)
            db.flush()
            result = _vitals_to_dict(row)
            _append_activity(db, patient.account_id, "vitals_saved", f"Recorded {vtype}: {value} {row.unit}", metadata={"vitals_id": result["vitals_id"], "type": vtype})
            db.commit()
            return result

    @staticmethod
    def get_vitals(username: str, limit: Optional[int] = 50) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(VitalsEntry).where(VitalsEntry.patient_id == patient.id)
                .order_by(VitalsEntry.recorded_on.desc(), VitalsEntry.created_at.desc())
            ).scalars().all()
            results = [_vitals_to_dict(r) for r in rows]
            return results[:limit] if limit is not None else results

    @staticmethod
    def delete_vitals_entry(username: str, vitals_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            entry = _find_by_id(db, patient, VitalsEntry, vitals_id)
            if entry is None:
                return False
            db.delete(entry)
            _append_activity(db, patient.account_id, "vitals_deleted", "Removed vitals entry", metadata={"vitals_id": vitals_id})
            db.commit()
            return True

    # ── Triage summaries ─────────────────────────────────────────────────

    @staticmethod
    def save_triage_summary(username: str, summary: Dict) -> Optional[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return None
            monitor = summary.get("what_to_monitor", [])
            row = TriageSummary(
                id=uuid.uuid4(), patient_id=patient.id,
                question=str(summary.get("question") or "").strip(),
                urgency_level=str(summary.get("urgency_level") or "").strip(),
                next_step=str(summary.get("next_step") or "").strip(),
                what_to_monitor=[str(i).strip() for i in monitor if str(i).strip()] if isinstance(monitor, list) else [],
                rationale=str(summary.get("rationale") or "").strip(),
                pathway_label=str(summary.get("pathway_label") or "").strip(),
                decision_summary=str(summary.get("decision_summary") or "").strip(),
                immediate_actions=summary.get("immediate_actions", []) or [],
                escalation_triggers=summary.get("escalation_triggers", []) or [],
                communication_points=summary.get("communication_points", []) or [],
                rule_hits=summary.get("rule_hits", []) or [],
                guideline_references=summary.get("guideline_references", []) or [],
                logic_version=str(summary.get("logic_version") or "").strip(),
                trace_id=summary.get("trace_id"),
            )
            db.add(row)
            db.flush()
            result = _triage_summary_to_dict(row)

            # Legacy caps triage_summaries at 25 most recent -- prune oldest.
            all_rows = db.execute(
                select(TriageSummary).where(TriageSummary.patient_id == patient.id).order_by(TriageSummary.created_at.desc())
            ).scalars().all()
            for stale in all_rows[25:]:
                db.delete(stale)

            _append_activity(db, patient.account_id, "triage_saved", f"Saved triage summary: {row.urgency_level} -> {row.next_step}", trace_id=row.trace_id, metadata={"summary_id": result["summary_id"]})
            db.commit()
            return result

    @staticmethod
    def get_triage_summaries(username: str, limit: Optional[int] = 10) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(TriageSummary).where(TriageSummary.patient_id == patient.id).order_by(TriageSummary.created_at.desc())
            ).scalars().all()
            results = [_triage_summary_to_dict(r) for r in rows]
            return results[:limit] if limit is not None else results

    @staticmethod
    def get_latest_triage_summary(username: str) -> Dict:
        summaries = SqlUserStore.get_triage_summaries(username, limit=1)
        return summaries[0] if summaries else {}

    # ── Trial search cache ───────────────────────────────────────────────

    @staticmethod
    def save_trial_search_result(username: str, result: Dict) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            patient.last_trial_search = result
            _append_activity(db, patient.account_id, "trial_search_saved", "Saved trial search result", metadata={
                "location": result.get("location", ""), "trial_count": len(result.get("trials", [])), "searched_at": result.get("searched_at", ""),
            })
            db.commit()

    @staticmethod
    def get_trial_search_result(username: str) -> Optional[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            return patient.last_trial_search if patient else None

    # ── Longitudinal memory ──────────────────────────────────────────────

    @staticmethod
    def get_longitudinal_memory(username: str) -> Dict:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return {"summary": "", "updated_at": None, "source": ""}
            memory = dict(patient.longitudinal_memory or {})
            memory.setdefault("summary", "")
            memory.setdefault("updated_at", None)
            memory.setdefault("source", "")
            return memory

    @staticmethod
    def save_longitudinal_memory(username: str, summary: str, source: str = "conversation", metadata: Optional[Dict] = None) -> None:
        cleaned = (summary or "").strip()
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            memory = dict(patient.longitudinal_memory or {})
            if cleaned == (memory.get("summary") or "").strip():
                return
            memory["summary"] = cleaned
            memory["updated_at"] = _utc_now().isoformat()
            memory["source"] = source
            patient.longitudinal_memory = memory
            _append_activity(db, patient.account_id, "longitudinal_memory_updated", f"Longitudinal memory refreshed from {source}", metadata=metadata or {"summary_length": len(cleaned)})
            db.commit()

    # ── Chat ──────────────────────────────────────────────────────────────

    @staticmethod
    def get_chat_history(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(ChatMessage).where(ChatMessage.patient_id == patient.id).order_by(ChatMessage.timestamp.asc())
            ).scalars().all()
            return [_chat_message_to_dict(r) for r in rows]

    @staticmethod
    def set_chat_history(username: str, history: List[Dict]) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            db.execute(ChatMessage.__table__.delete().where(ChatMessage.patient_id == patient.id))
            for message in history:
                timestamp = _parse_dt(message.get("timestamp")) or _utc_now()
                db.add(ChatMessage(
                    id=uuid.uuid4(), patient_id=patient.id, role=message.get("role", "user"),
                    content=message.get("content", ""), timestamp=timestamp,
                    sources=message.get("sources", []) or [], trace_id=message.get("trace_id"),
                    message_metadata=message.get("metadata", {}) or {},
                ))
            _append_activity(db, patient.account_id, "conversation_replaced", "Conversation history replaced")
            db.commit()

    @staticmethod
    def clear_chat_history(username: str) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            db.execute(ChatMessage.__table__.delete().where(ChatMessage.patient_id == patient.id))
            _append_activity(db, patient.account_id, "conversation_cleared", "Conversation history cleared")
            db.commit()

    @staticmethod
    def append_chat(username: str, message: Dict) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            timestamp = _parse_dt(message.get("timestamp")) or _utc_now()
            row = ChatMessage(
                id=uuid.uuid4(), patient_id=patient.id, role=message.get("role", "user"),
                content=message.get("content", ""), timestamp=timestamp,
                sources=message.get("sources", []) or [], trace_id=message.get("trace_id"),
                message_metadata=message.get("metadata", {}) or {},
            )
            db.add(row)
            db.flush()
            _append_activity(db, patient.account_id, "chat_message", f"{row.role} message stored", trace_id=row.trace_id, metadata={"message_id": str(row.id), "source_count": len(row.sources or [])})
            db.commit()

    @staticmethod
    def get_response_feedback(username: str, trace_id: str, message_id: str = "") -> Optional[Dict]:
        message = _find_assistant_message(SqlUserStore.get_chat_history(username), trace_id, message_id)
        if not message:
            return None
        feedback = (message.get("metadata") or {}).get("feedback")
        return feedback if isinstance(feedback, dict) else None

    @staticmethod
    def get_response_trace(username: str, trace_id: str, message_id: str = "") -> Optional[Dict]:
        message = _find_assistant_message(SqlUserStore.get_chat_history(username), trace_id, message_id)
        if not message:
            return None
        trace = (message.get("metadata") or {}).get("trace")
        return trace if isinstance(trace, dict) else None

    @staticmethod
    def mark_response_feedback(username: str, trace_id: str, message_id: str, rating: str, saved_to_feedback_store: bool) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return False
            rows = db.execute(select(ChatMessage).where(ChatMessage.patient_id == patient.id)).scalars().all()
            for row in rows:
                same_message = bool(message_id) and str(row.id) == message_id
                same_trace = not message_id and bool(trace_id) and row.trace_id == trace_id
                if row.role != "assistant" or not (same_message or same_trace):
                    continue
                metadata = dict(row.message_metadata or {})
                metadata["feedback"] = {"rating": rating, "trace_id": trace_id, "saved_to_feedback_store": saved_to_feedback_store, "updated_at": _utc_now().isoformat()}
                row.message_metadata = metadata
                _append_activity(db, patient.account_id, "response_feedback", "Response feedback recorded", trace_id=trace_id, metadata={"rating": rating, "message_id": str(row.id), "saved_to_feedback_store": saved_to_feedback_store})
                db.commit()
                return True
            return False

    # ── Uploads ───────────────────────────────────────────────────────────

    @staticmethod
    def add_upload(username: str, upload_name: str, stored_path: Optional[str] = None, content_hash: Optional[str] = None) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            existing = db.execute(select(Upload).where(Upload.patient_id == patient.id, Upload.file_name == upload_name)).scalar_one_or_none()
            if existing is None:
                existing = Upload(id=uuid.uuid4(), patient_id=patient.id, file_name=upload_name)
                db.add(existing)
            if stored_path is not None:
                existing.stored_path = stored_path
            if content_hash is not None:
                existing.content_hash = content_hash
            db.flush()
            _append_activity(db, patient.account_id, "upload", f"Uploaded {upload_name}", metadata={"file": upload_name, "stored_path": stored_path or ""})
            db.commit()

    @staticmethod
    def save_document_summary(username: str, filename: str, summary: str, stored_path: Optional[str] = None) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            upload = db.execute(select(Upload).where(Upload.patient_id == patient.id, Upload.file_name == filename)).scalar_one_or_none()
            if upload is None:
                upload = Upload(id=uuid.uuid4(), patient_id=patient.id, file_name=filename)
                db.add(upload)
            upload.summary_available = True
            if stored_path is not None:
                upload.stored_path = stored_path
            db.flush()
            if upload.document_summary is None:
                db.add(DocumentSummary(id=uuid.uuid4(), upload_id=upload.id, summary=summary))
            else:
                upload.document_summary.summary = summary
            _append_activity(db, patient.account_id, "document_indexed", f"Indexed upload {filename}", metadata={"file": filename})
            db.commit()

    @staticmethod
    def get_document_summaries(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(select(Upload).where(Upload.patient_id == patient.id)).scalars().all()
            return [
                {"file": u.file_name, "summary": u.document_summary.summary, "stored_path": u.stored_path, "updated_at": _iso(u.document_summary.updated_at)}
                for u in rows if u.document_summary is not None
            ]

    @staticmethod
    def get_uploads(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(select(Upload).where(Upload.patient_id == patient.id)).scalars().all()
            return [_upload_to_dict(r) for r in rows]

    # ── Interaction traces ───────────────────────────────────────────────

    @staticmethod
    def save_interaction_trace(username: str, trace: Dict) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            payload = {k: v for k, v in trace.items() if k not in ("trace_id", "created_at")}
            payload.setdefault("question", "")
            payload.setdefault("answer_preview", "")
            payload.setdefault("sources", [])
            row = InteractionTrace(id=uuid.uuid4(), patient_id=patient.id, trace_id=trace.get("trace_id", ""), payload=payload)
            db.add(row)
            db.flush()
            _append_activity(db, patient.account_id, "trace_saved", f"Trace saved for question: {str(payload.get('question', ''))[:80]}", trace_id=row.trace_id, metadata={"source_count": len(payload.get("sources", []))})
            db.commit()

    @staticmethod
    def get_interaction_traces(username: str, limit: Optional[int] = 25) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(InteractionTrace).where(InteractionTrace.patient_id == patient.id).order_by(InteractionTrace.created_at.desc())
            ).scalars().all()
            results = [_trace_to_dict(r) for r in rows]
            return results[:limit] if limit is not None else results

    # ── Self-activity log ────────────────────────────────────────────────

    @staticmethod
    def add_audit(username: str, event: str, details: str, trace_id: Optional[str] = None, metadata: Optional[Dict] = None) -> None:
        with _session() as db:
            account = _get_account(db, username)
            if account is None:
                return
            _append_activity(db, account.id, event, details, trace_id=trace_id, metadata=metadata)
            db.commit()

    @staticmethod
    def get_audit(username: str, limit: Optional[int] = 50) -> List[Dict]:
        with _session() as db:
            account = _get_account(db, username)
            if account is None:
                return []
            rows = db.execute(
                select(AccountActivityLog).where(AccountActivityLog.account_id == account.id).order_by(AccountActivityLog.created_at.desc())
            ).scalars().all()
            results = [
                {"time": _iso(r.created_at), "event": r.event, "details": r.details, "trace_id": r.trace_id, "metadata": r.event_metadata}
                for r in rows
            ]
            return results[:limit] if limit is not None else results

    # ── Video generation rate limit ──────────────────────────────────────

    @staticmethod
    def record_video_generated(username: str) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            patient.last_video_generated_at = _utc_now()
            _append_activity(db, patient.account_id, "video_generated", "Sora-2 video generated")
            db.commit()

    @staticmethod
    def get_last_video_generated_at(username: str) -> str:
        with _session() as db:
            patient = _get_patient(db, username)
            return _iso(patient.last_video_generated_at) if patient and patient.last_video_generated_at else ""

    # ── Clinical notes ────────────────────────────────────────────────────

    @staticmethod
    def get_clinical_notes(username: str) -> List[Dict]:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return []
            rows = db.execute(
                select(ClinicalNote).where(ClinicalNote.patient_id == patient.id).order_by(ClinicalNote.created_at.desc())
            ).scalars().all()
            return [_clinical_note_to_dict(r) for r in rows]

    @staticmethod
    def save_clinical_note(username: str, note: Dict) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            if patient is None:
                return
            row = ClinicalNote(
                id=uuid.uuid4(), patient_id=patient.id,
                subjective=note.get("subjective", ""), objective=note.get("objective", ""),
                assessment=note.get("assessment", ""), plan=note.get("plan", ""),
                urgency_level=note.get("urgency_level", ""), requires_gp_visit=bool(note.get("requires_gp_visit", False)),
                gp_visit_reason=note.get("gp_visit_reason", ""),
            )
            db.add(row)
            db.flush()
            _append_activity(db, patient.account_id, "clinical_note_created", f"SOAP note {row.id} generated")
            db.commit()

    @staticmethod
    def update_clinical_note(username: str, note_id: str, updates: Dict) -> Optional[Dict]:
        allowed = {"subjective", "objective", "assessment", "plan", "urgency_level", "requires_gp_visit", "gp_visit_reason"}
        with _session() as db:
            patient = _get_patient(db, username)
            row = _find_by_id(db, patient, ClinicalNote, note_id)
            if row is None:
                return None
            for key, value in updates.items():
                if key in allowed:
                    setattr(row, key, value)
            db.flush()
            result = _clinical_note_to_dict(row)
            _append_activity(db, patient.account_id, "clinical_note_edited", f"Note {note_id} edited by {username}")
            db.commit()
            return result

    @staticmethod
    def delete_clinical_note(username: str, note_id: str) -> bool:
        with _session() as db:
            patient = _get_patient(db, username)
            row = _find_by_id(db, patient, ClinicalNote, note_id)
            if row is None:
                return False
            db.delete(row)
            _append_activity(db, patient.account_id, "clinical_note_deleted", f"Note {note_id} deleted")
            db.commit()
            return True

    @staticmethod
    def mark_note_email_sent(username: str, note_id: str) -> None:
        with _session() as db:
            patient = _get_patient(db, username)
            row = _find_by_id(db, patient, ClinicalNote, note_id)
            if row is None:
                return
            row.email_sent = True
            row.email_sent_at = _utc_now()
            _append_activity(db, patient.account_id, "clinical_note_emailed", f"Note {note_id} sent by email")
            db.commit()

    # ── Export ────────────────────────────────────────────────────────────

    @staticmethod
    def export_user_snapshot(username: str) -> Dict:
        profile = SqlUserStore.get_user_profile(username)
        if not profile:
            return {}
        return {
            "username": _normalize_username(username),
            "profile": profile,
            "conversation": SqlUserStore.get_chat_history(username),
            "medications": SqlUserStore.get_medications(username),
            "allergies": SqlUserStore.get_allergies(username),
            "conditions": SqlUserStore.get_conditions(username),
            "vitals": SqlUserStore.get_vitals(username, limit=None),
            "symptom_logs": SqlUserStore.get_symptom_logs(username, limit=None),
            "clinical_notes": SqlUserStore.get_clinical_notes(username),
            "uploads": SqlUserStore.get_uploads(username),
            "doc_summaries": SqlUserStore.get_document_summaries(username),
            "triage_summaries": SqlUserStore.get_triage_summaries(username, limit=None),
            "longitudinal_memory": SqlUserStore.get_longitudinal_memory(username),
        }


# ── Shared helpers ────────────────────────────────────────────────────────


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


def _find_by_id(db: Session, patient: Optional[Patient], model, row_id: str):
    if patient is None or not row_id:
        return None
    try:
        parsed_id = uuid.UUID(str(row_id))
    except ValueError:
        return None
    return db.execute(select(model).where(model.patient_id == patient.id, model.id == parsed_id)).scalar_one_or_none()


def _find_by_id_or_name(db: Session, patient: Patient, model, row_id: Optional[str], name: str):
    if row_id:
        found = _find_by_id(db, patient, model, row_id)
        if found is not None:
            return found
    return db.execute(
        select(model).where(model.patient_id == patient.id, model.name.ilike(name))
    ).scalar_one_or_none()


def _find_assistant_message(history: List[Dict], trace_id: str, message_id: str) -> Optional[Dict]:
    for message in history:
        same_message = bool(message_id) and message.get("message_id") == message_id
        same_trace = not message_id and bool(trace_id) and message.get("trace_id") == trace_id
        if message.get("role") == "assistant" and (same_message or same_trace):
            return message
    return None
