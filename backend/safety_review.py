"""Deterministic longitudinal safety checks for the patient record.

This module deliberately implements a small, locked rule set. It does not
diagnose, prescribe, or infer a normal range for tests it does not recognise.
Every displayed clinical claim is paired with the patient facts and guidance
passage used to produce it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional


NICE_HYPERKALAEMIA_URL = "https://www.nice.org.uk/guidance/ta623/chapter/3-Committee-discussion"
NHS_WARFARIN_URL = "https://www.nhs.uk/medicines/warfarin/"
NHS_999_URL = "https://www.swast.nhs.uk/when-to-call-999/"

_EMERGENCY_SYMPTOMS = {
    "chest pain": "Chest pain can require emergency assessment.",
    "difficulty breathing": "Difficulty breathing can require emergency assessment.",
    "cant breathe": "Severe difficulty breathing can require emergency assessment.",
    "cannot breathe": "Severe difficulty breathing can require emergency assessment.",
    "struggling to breathe": "Severe difficulty breathing can require emergency assessment.",
    "severe breathlessness": "Severe breathlessness can require emergency assessment.",
    "unconscious": "Loss of consciousness can require emergency assessment.",
    "loss of consciousness": "Loss of consciousness can require emergency assessment.",
    "seizure": "A seizure that is not stopping can require emergency assessment.",
    "fit": "A fit that is not stopping can require emergency assessment.",
    "severe bleeding": "Severe bleeding can require emergency assessment.",
    "anaphylaxis": "A severe allergic reaction can require emergency assessment.",
}
_WARFARIN_NAMES = {"warfarin"}
_NSAID_NAMES = {"ibuprofen", "naproxen", "diclofenac", "aspirin"}


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _number(value: object) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _active_term(text: str, term: str) -> bool:
    """Match a safety term unless it is directly and plainly negated."""
    for match in re.finditer(rf"\b{re.escape(term)}\b", text):
        prefix = text[max(0, match.start() - 24):match.start()]
        if not re.search(r"\b(?:no|not|without|denies|denied|never)\s+(?:any\s+)?$", prefix):
            return True
    return False


def _recent_date(value: object, max_age_days: int = 2) -> bool:
    try:
        recorded = datetime.strptime(str(value or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return False
    age = (datetime.now(timezone.utc).date() - recorded).days
    return 0 <= age <= max_age_days


def _review_id(rule: str, facts: Iterable[Dict]) -> str:
    identity = "|".join(
        f"{fact.get('record_type')}:{fact.get('record_id')}:{fact.get('value')}"
        for fact in facts
    )
    digest = hashlib.sha256(f"{rule}|{identity}".encode("utf-8")).hexdigest()[:16]
    return f"safety-{digest}"


def _fact(record_type: str, item: Dict, label: str, value: str) -> Dict:
    id_keys = {
        "symptom": "log_id",
        "medicine": "medication_id",
        "allergy": "allergy_id",
        "result": "vitals_id",
    }
    return {
        "record_type": record_type,
        "record_id": str(item.get(id_keys[record_type]) or ""),
        "label": label,
        "value": value,
        "recorded_on": str(
            item.get("recorded_on")
            or item.get("logged_for")
            or item.get("started_on")
            or item.get("created_at")
            or ""
        ),
    }


def _base_review(
    *,
    rule: str,
    priority: str,
    category: str,
    what_changed: str,
    why_it_matters: str,
    proposed_action: str,
    uncertainty: str,
    facts: List[Dict],
    evidence: List[Dict],
) -> Dict:
    return {
        "review_id": _review_id(rule, facts),
        "rule_id": rule,
        "priority": priority,
        "category": category,
        "status": "detected",
        "what_changed": what_changed,
        "why_it_matters": why_it_matters,
        "uncertainty": uncertainty,
        "proposed_action": proposed_action,
        "patient_facts": facts,
        "evidence": evidence,
        "approver": "A qualified clinician",
        "outcome": {
            "action_happened": None,
            "patient_improved": None,
            "note": "",
            "updated_at": "",
        },
        "writeback": {
            "status": "not_configured",
            "message": "Health-record write-back is disabled until a clinician approves the action and SMART-on-FHIR is connected.",
        },
    }


def _potassium_reviews(vitals: List[Dict]) -> List[Dict]:
    candidates = [
        item for item in vitals
        if _normalise(item.get("type")) in {"potassium", "serum potassium", "k"}
        and _number(item.get("value")) is not None
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda item: (str(item.get("recorded_on", "")), str(item.get("created_at", ""))), reverse=True)
    latest = candidates[0]
    value = _number(latest.get("value"))
    if value is None or value < 6.0:
        return []
    unit = str(latest.get("unit") or "mmol/L")
    facts = [_fact("result", latest, "Latest potassium result", f"{latest.get('value')} {unit}".strip())]
    if len(candidates) > 1:
        previous = candidates[1]
        facts.append(_fact("result", previous, "Previous potassium result", f"{previous.get('value')} {previous.get('unit') or unit}".strip()))
    severe = value >= 6.5
    return [_base_review(
        rule="potassium-severe" if severe else "potassium-moderate",
        priority="emergency" if severe else "urgent",
        category="Abnormal result",
        what_changed=(
            f"The latest potassium result is {latest.get('value')} {unit}."
            + (f" The previous recorded result was {candidates[1].get('value')} {candidates[1].get('unit') or unit}." if len(candidates) > 1 else "")
        ),
        why_it_matters=(
            "NICE describes potassium at or above 6.5 mmol/L as severe and says life-threatening acute hyperkalaemia needs emergency hospital treatment."
            if severe
            else "NICE describes 6.0 to 6.4 mmol/L as moderate hyperkalaemia. The result may need confirmation and prompt clinical assessment."
        ),
        proposed_action=(
            "Seek emergency assessment now. Call 999 if you feel unwell, have weakness, palpitations, chest pain, breathing difficulty, or cannot travel safely. Do not change prescribed medicines unless a clinician tells you to."
            if severe
            else "Contact your GP, renal team, or NHS 111 now for same-day advice and confirmation of the result. Do not change prescribed medicines unless a clinician tells you to."
        ),
        uncertainty="A single potassium result can be falsely high, so a clinician must interpret and usually confirm it. This review cannot assess the sample quality or an ECG.",
        facts=facts,
        evidence=[{
            "claim": "Potassium severity and need for emergency treatment" if severe else "Potassium severity",
            "source_title": "NICE: Patiromer for treating hyperkalaemia, committee discussion",
            "source_url": NICE_HYPERKALAEMIA_URL,
            "passage": (
                "Severe (6.5 mmol/litre and above). Life-threatening acute hyperkalaemia needs emergency treatment in hospital."
                if severe
                else "Moderate (6.0 mmol/litre to 6.4 mmol/litre)."
            ),
        }],
    )]


def _emergency_symptom_reviews(symptoms: List[Dict]) -> List[Dict]:
    if not symptoms:
        return []
    latest_date = max(str(item.get("logged_for") or "") for item in symptoms)
    if not _recent_date(latest_date):
        return []
    reviews = []
    for item in symptoms:
        if latest_date and str(item.get("logged_for") or "") != latest_date:
            continue
        text = _normalise(f"{item.get('symptom', '')} {item.get('notes', '')}")
        match = next((term for term in _EMERGENCY_SYMPTOMS if _active_term(text, term)), None)
        if not match:
            continue
        facts = [_fact("symptom", item, "Latest symptom entry", str(item.get("symptom") or "Symptom"))]
        reviews.append(_base_review(
            rule=f"emergency-symptom-{_normalise(match).replace(' ', '-')}",
            priority="emergency",
            category="Emergency symptom",
            what_changed=f"The latest symptom entry records {item.get('symptom') or match}.",
            why_it_matters=_EMERGENCY_SYMPTOMS[match],
            proposed_action="Call 999 now. Do not wait for a FlynnMed or clinician reply. If possible, ask someone to stay with you and gather your medicines.",
            uncertainty="A saved symptom entry cannot show how you look or feel now. Emergency services must assess the situation.",
            facts=facts,
            evidence=[{
                "claim": "This symptom can need emergency help",
                "source_title": "South Western Ambulance Service: When to call 999",
                "source_url": NHS_999_URL,
                "passage": "Life-threatening emergencies include loss of consciousness, chest pain, breathing difficulties, severe bleeding and severe allergic reactions.",
            }],
        ))
    return reviews


def _medicine_reviews(medications: List[Dict], allergies: List[Dict]) -> List[Dict]:
    reviews: List[Dict] = []
    active_medicines = {str(item.get("name") or "").strip().lower(): item for item in medications}
    for allergy in allergies:
        allergy_name = str(allergy.get("name") or "").strip().lower()
        medicine = active_medicines.get(allergy_name)
        if not medicine:
            continue
        facts = [
            _fact("medicine", medicine, "Saved medicine", str(medicine.get("name") or "")),
            _fact("allergy", allergy, "Saved allergy", str(allergy.get("name") or "")),
        ]
        reviews.append(_base_review(
            rule="medicine-allergy-exact-match",
            priority="urgent",
            category="Medicine and allergy conflict",
            what_changed=f"{medicine.get('name')} appears in both the current medicines and allergy lists.",
            why_it_matters="The record contains a direct medicine-allergy conflict that must be checked before another dose is taken.",
            proposed_action="Contact a pharmacist, GP, or NHS 111 now, before the next dose. If you have swelling of the lips, mouth, throat or tongue, or difficulty breathing, call 999 now.",
            uncertainty="Names can refer to an ingredient, brand, intolerance, or past side effect. A clinician or pharmacist must verify the exact medicine and reaction.",
            facts=facts,
            evidence=[{
                "claim": "Breathing difficulty with a severe allergic reaction is an emergency",
                "source_title": "South Western Ambulance Service: When to call 999",
                "source_url": NHS_999_URL,
                "passage": "Life-threatening emergencies include breathing difficulties and severe allergic reactions.",
            }],
        ))

    names = set(active_medicines)
    warfarin = next((active_medicines[name] for name in names & _WARFARIN_NAMES), None)
    for nsaid_name in sorted(names & _NSAID_NAMES):
        if not warfarin:
            break
        nsaid = active_medicines[nsaid_name]
        facts = [
            _fact("medicine", warfarin, "Current anticoagulant", str(warfarin.get("name") or "Warfarin")),
            _fact("medicine", nsaid, "Other current medicine", str(nsaid.get("name") or nsaid_name)),
        ]
        reviews.append(_base_review(
            rule=f"warfarin-{nsaid_name}",
            priority="urgent",
            category="Medicine safety",
            what_changed=f"The record lists warfarin and {nsaid.get('name') or nsaid_name} together.",
            why_it_matters="Warfarin increases bleeding risk, and the combination needs a pharmacist or prescriber to check it.",
            proposed_action="Contact a pharmacist, anticoagulation clinic, GP, or NHS 111 today. Do not stop warfarin on your own. Seek urgent help for blood in vomit, urine or stools, black stools, severe bleeding, or a sudden severe headache.",
            uncertainty="The record does not show whether a clinician intentionally approved this combination, the dose taken, or the latest INR.",
            facts=facts,
            evidence=[{
                "claim": "Warfarin can cause serious bleeding and needs urgent review when bleeding occurs",
                "source_title": "NHS: Warfarin",
                "source_url": NHS_WARFARIN_URL,
                "passage": "The main side effect of warfarin is an increased risk of bleeding.",
            }],
        ))
    return reviews


def build_safety_reviews(
    *,
    vitals: List[Dict],
    symptoms: List[Dict],
    medications: List[Dict],
    allergies: List[Dict],
    conditions: Optional[List[Dict]] = None,
    triage_summaries: Optional[List[Dict]] = None,
    document_summaries: Optional[List[Dict]] = None,
    clinical_relationships: Optional[List[Dict]] = None,
    longitudinal_memory: str = "",
    saved_states: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """Return ordered safety reviews with an auditable full-record context trace.

    Locked rules still trigger only from record types they explicitly support;
    receiving broader context must never turn a free-text summary into a new
    diagnosis or alert.
    """
    reviews = [
        *_emergency_symptom_reviews(symptoms),
        *_potassium_reviews(vitals),
        *_medicine_reviews(medications, allergies),
    ]
    states = saved_states or {}
    context_considered = {
        "conditions": [
            str(item.get("name") or "").strip()
            for item in conditions or []
            if str(item.get("name") or "").strip()
        ][:20],
        "triage_record_count": len(triage_summaries or []),
        "document_summary_count": len(document_summaries or []),
        "clinical_relationship_count": len(clinical_relationships or []),
        "longitudinal_summary_available": bool((longitudinal_memory or "").strip()),
    }
    for review in reviews:
        review["context_considered"] = context_considered
        state = states.get(review["review_id"], {})
        if state:
            review["status"] = state.get("status", review["status"])
            review["outcome"].update(state.get("outcome") or {})
    rank = {"emergency": 0, "urgent": 1, "review": 2}
    return sorted(reviews, key=lambda item: (rank.get(item["priority"], 9), item["category"], item["review_id"]))


def update_review_state(current: Optional[Dict], payload: Dict) -> Dict:
    """Validate patient-controlled workflow fields without granting approval."""
    state = dict(current or {})
    requested_status = str(payload.get("status") or state.get("status") or "detected")
    if requested_status not in {"detected", "patient_confirmed", "follow_up_recorded"}:
        raise ValueError("Patients can confirm a proposal or record follow-up, but only a clinician can approve it.")
    outcome = dict(state.get("outcome") or {})
    for key in ("action_happened", "patient_improved"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{key} must be true, false, or omitted.")
        if value is not None:
            outcome[key] = value
    if "note" in payload:
        outcome["note"] = str(payload.get("note") or "").strip()[:500]
    if requested_status == "follow_up_recorded" and outcome.get("action_happened") is None:
        raise ValueError("Record whether the proposed action happened before saving follow-up.")
    outcome["updated_at"] = datetime.now(timezone.utc).isoformat()
    return {"status": requested_status, "outcome": outcome}
