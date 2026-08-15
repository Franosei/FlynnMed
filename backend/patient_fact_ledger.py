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
from typing import Any, Dict, Iterable, List, Optional
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


def _latest_fact_for_label(
    db: Session, patient_id: UUID, category: str, label: str
) -> Optional[PatientFact]:
    """Most recent non-retracted row for this (patient_id, category, label),
    used to link an edit or retraction back to what it replaces. Retracted
    rows are excluded -- a fact that was already retracted and later
    reappears starts a fresh chain rather than un-retracting the old one."""
    return db.execute(
        select(PatientFact)
        .where(
            PatientFact.patient_id == patient_id,
            PatientFact.category == category,
            PatientFact.label.ilike(label),
            PatientFact.status != "retracted",
        )
        .order_by(PatientFact.created_at.desc())
    ).scalars().first()


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
    that answer was generated. The new row's previous_fact_id is set to
    whatever row it's replacing, making the edit an explicit, queryable
    chain instead of something only recoverable by created_at ordering."""
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

    previous_fact = _latest_fact_for_label(db, patient_id, category, label)

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
        previous_fact_id=previous_fact.id if previous_fact is not None else None,
    )
    db.add(fact)
    db.flush()
    return fact


def _retract_missing_facts(
    db: Session, patient_id: UUID, category: str, live_labels: Iterable[str]
) -> None:
    """Marks any previously-persisted, non-retracted fact in this category
    whose label is no longer present in the patient's current live snapshot
    as retracted -- so a removed medication or a disproved allergy leaves an
    explicit terminal row instead of just silently stopping being mentioned.
    Never raises; callers already wrap this in a broad try/except per the
    module's never-blocks-the-answer discipline."""
    live_label_set = {label.strip().lower() for label in live_labels if label and label.strip()}
    # Latest row per label, regardless of status -- if the latest row for a
    # label is already "retracted" (from an earlier call), it must be
    # skipped rather than re-processed; only the ORIGINAL confirmed/suspected
    # row it points back to would otherwise keep matching status != "retracted"
    # forever, since retracting never mutates that original row in place.
    all_rows = db.execute(
        select(PatientFact)
        .where(PatientFact.patient_id == patient_id, PatientFact.category == category)
        .order_by(PatientFact.created_at.desc())
    ).scalars().all()
    latest_by_label: Dict[str, PatientFact] = {}
    for row in all_rows:
        key = row.label.strip().lower()
        if key and key not in latest_by_label:
            latest_by_label[key] = row

    for key, row in latest_by_label.items():
        if row.status == "retracted" or key in live_label_set:
            continue
        retracted = PatientFact(
            patient_id=patient_id,
            category=row.category,
            label=row.label,
            value=row.value,
            unit=row.unit,
            status="retracted",
            source=row.source,
            source_record_type=row.source_record_type,
            source_record_id=row.source_record_id,
            recorded_at=row.recorded_at,
            fact_hash=_hash_text(
                _normalize_for_hash(row.category)
                + "|" + _normalize_for_hash(row.label)
                + "|" + _normalize_for_hash(row.value)
                + "|" + _normalize_for_hash(row.unit)
                + "|retracted|" + str(row.id)
            ),
            previous_fact_id=row.id,
        )
        db.add(retracted)


def persist_patient_facts_for_bundle(
    patient_id: Any,
    *,
    medications: Optional[List[Dict]] = None,
    conditions: Optional[List[Dict]] = None,
    allergies: Optional[List[Dict]] = None,
    vitals: Optional[List[Dict]] = None,
    symptom_logs: Optional[List[Dict]] = None,
    source: str = "structured_patient_record",
) -> None:
    """Persists the current full snapshot of the patient's structured
    record. No-op when patient_id is falsy (e.g. a clinician asking a
    patient-agnostic evidence question with no target patient in view).
    Silent per-fact failure -- never raises, since persistence must not
    block the answer already generated from this same data.

    `source` is applied to every fact persisted this call -- pass
    "document_extracted" when this snapshot was taken during a
    document-analysis chat turn (see backend/rag_system.py), otherwise the
    default "structured_patient_record" (a normal chat turn reading the
    patient's own saved record).

    After persisting each category's current live rows, any previously-seen
    label now absent from that category's live list is marked retracted
    (see _retract_missing_facts) -- this is what surfaces a deleted
    medication or a since-disproved allergy as an explicit event rather than
    a silent gap."""
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
                    source=source,
                    source_record_type="medication",
                    source_record_id=_parse_uuid(med.get("medication_id")),
                    recorded_at=_parse_iso(med.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] medication persist failed: {exc}")
        try:
            _retract_missing_facts(
                db, parsed_patient_id, "medication", (m.get("name", "") for m in medications or [])
            )
        except Exception as exc:
            print(f"[PatientFactLedger] medication retraction pass failed: {exc}")

        for cond in conditions or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="condition",
                    label=cond.get("name", ""),
                    value=cond.get("status", ""),
                    source=source,
                    source_record_type="condition",
                    source_record_id=_parse_uuid(cond.get("condition_id")),
                    recorded_at=_parse_iso(cond.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] condition persist failed: {exc}")
        try:
            _retract_missing_facts(
                db, parsed_patient_id, "condition", (c.get("name", "") for c in conditions or [])
            )
        except Exception as exc:
            print(f"[PatientFactLedger] condition retraction pass failed: {exc}")

        for allergy in allergies or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="allergy",
                    label=allergy.get("name", ""),
                    value=allergy.get("reaction", ""),
                    status="confirmed" if allergy.get("confirmed", True) else "suspected",
                    source=source,
                    source_record_type="allergy",
                    source_record_id=_parse_uuid(allergy.get("allergy_id")),
                    recorded_at=_parse_iso(allergy.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] allergy persist failed: {exc}")
        try:
            _retract_missing_facts(
                db, parsed_patient_id, "allergy", (a.get("name", "") for a in allergies or [])
            )
        except Exception as exc:
            print(f"[PatientFactLedger] allergy retraction pass failed: {exc}")

        for vital in vitals or []:
            try:
                persist_patient_fact(
                    db, parsed_patient_id,
                    category="vital",
                    label=vital.get("type", ""),
                    value=vital.get("value", ""),
                    unit=vital.get("unit", ""),
                    source=source,
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
                    source=source,
                    source_record_type="symptom_log",
                    source_record_id=_parse_uuid(symptom.get("log_id")),
                    recorded_at=_parse_iso(symptom.get("created_at")),
                )
            except Exception as exc:
                print(f"[PatientFactLedger] symptom persist failed: {exc}")

        db.commit()
