"""
Evidence Ledger Phase 3 write path: persists a snapshot of the patient's
structured facts (conditions/medications/allergies/vitals/symptoms) at the
time an answer is generated, so a future AnswerClaim (Phase 4) can cite
"which patient fact was used" against a stable row instead of a live,
mutable Condition/Medication/Allergy/VitalsEntry/SymptomLog record.

Same short-lived-session, never-blocks-the-answer discipline as
backend/evidence_ledger.py: a persistence failure must never prevent the
answer already being generated from these same facts.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.db import get_session_factory
from backend.models.patient_fact import CATEGORIES, FACT_SOURCES, FACT_STATUSES, PatientFact


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_for_hash(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _parse_iso(value: Any):
    if not value or not isinstance(value, str):
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_uuid(value: Any) -> Optional[UUID]:
    if not value:
        return None
    try:
        return UUID(str(value))
    except (ValueError, AttributeError):
        return None


def persist_patient_fact(
    db: Session,
    patient_id: UUID,
    *,
    category: str,
    label: str,
    value: str,
    unit: str = "",
    status: str = "confirmed",
    source: str = "structured_patient_record",
    source_record_type: str = "",
    source_record_id: Optional[UUID] = None,
    recorded_at=None,
) -> Optional[PatientFact]:
    """Dedupes on (patient_id, fact_hash): the same fact snapshotted across
    many answers reuses one row; an edited value (e.g. a changed dose)
    produces a new row rather than mutating the old one, so a past
    AnswerClaim keeps pointing at the value that was actually true when
    that answer was generated."""
    label = label.strip()
    if not label:
        return None

    category = category.strip().lower() if category else "unknown"
    if category not in CATEGORIES:
        category = "unknown"
    status = status.strip().lower() if status else "unknown"
    if status not in FACT_STATUSES:
        status = "unknown"
    source = source.strip().lower() if source else "structured_patient_record"
    if source not in FACT_SOURCES:
        source = "structured_patient_record"
    value = (value or "").strip()
    unit = (unit or "").strip()

    fact_hash = _hash_text(
        _normalize_for_hash(category)
        + "|" + _normalize_for_hash(label)
        + "|" + _normalize_for_hash(value)
        + "|" + _normalize_for_hash(unit)
        + "|" + _normalize_for_hash(status)
    )
    existing = db.execute(
        select(PatientFact).where(
            PatientFact.patient_id == patient_id, PatientFact.fact_hash == fact_hash
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    fact = PatientFact(
        patient_id=patient_id,
        category=category[:32],
        label=label[:255],
        value=value[:512],
        unit=unit[:32],
        status=status,
        source=source,
        source_record_type=source_record_type[:32],
        source_record_id=source_record_id,
        recorded_at=recorded_at,
        fact_hash=fact_hash,
    )
    db.add(fact)
    db.flush()
    return fact


def persist_patient_facts_for_bundle(
    patient_id: Any,
    *,
    medications: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    symptom_logs: Optional[List[Dict]] = None,
) -> None:
    """Persists the current full snapshot of the patient's structured
    record. No-op when patient_id is falsy (e.g. a clinician asking a
    patient-agnostic evidence question with no target patient in view).
    Silent per-fact failure -- never raises, since persistence must not
    block the answer already generated from this same data."""
    parsed_patient_id = _parse_uuid(patient_id)
    if parsed_patient_id is None:
        return

    session_factory = get_session_factory()
    with session_factory() as db:
        for med in medications or []:
            try:
                dose_schedule = " ".join(
                    part for part in [med.get("dose", ""), med.get("schedule", "")] if part
                ).strip()
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="medication",
                    label=med.get("name", ""),
                    value=dose_schedule,
                    source_record_type="medication",
                    source_record_id=_parse_uuid(med.get("medication_id")),
                    recorded_at=_parse_iso(med.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] medication persist failed: {exc}")

        for cond in conditions or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="condition",
                    label=cond.get("name", ""),
                    value=cond.get("status", ""),
                    source_record_type="condition",
                    source_record_id=_parse_uuid(cond.get("condition_id")),
                    recorded_at=_parse_iso(cond.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] condition persist failed: {exc}")

        for allergy in allergies or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="allergy",
                    label=allergy.get("name", ""),
                    value=allergy.get("reaction", ""),
                    status="confirmed" if allergy.get("confirmed", True) else "suspected",
                    source_record_type="allergy",
                    source_record_id=_parse_uuid(allergy.get("allergy_id")),
                    recorded_at=_parse_iso(allergy.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] allergy persist failed: {exc}")

        for vital in vitals or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="vital",
                    label=vital.get("type", ""),
                    value=vital.get("value", ""),
                    unit=vital.get("unit", ""),
                    source_record_type="vitals_entry",
                    source_record_id=_parse_uuid(vital.get("vitals_id")),
                    recorded_at=_parse_iso(vital.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] vital persist failed: {exc}")

        for symptom in symptom_logs or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="symptom",
                    label=symptom.get("symptom", ""),
                    value=str(symptom.get("severity", "")),
                    source_record_type="symptom_log",
                    source_record_id=_parse_uuid(symptom.get("log_id")),
                    recorded_at=_parse_iso(symptom.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] symptom persist failed: {exc}")

        db.commit()
