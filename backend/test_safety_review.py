from datetime import datetime, timezone

import pytest

from backend.safety_review import build_safety_reviews, update_review_state


def _build(*, vitals=None, symptoms=None, medications=None, allergies=None, states=None):
    return build_safety_reviews(
        vitals=vitals or [],
        symptoms=symptoms or [],
        medications=medications or [],
        allergies=allergies or [],
        saved_states=states,
    )


def test_severe_potassium_is_an_emergency_with_traceable_change():
    reviews = _build(vitals=[
        {"vitals_id": "new", "type": "Potassium", "value": "6.8", "unit": "mmol/L", "recorded_on": "2026-08-04"},
        {"vitals_id": "old", "type": "potassium", "value": "4.7", "unit": "mmol/L", "recorded_on": "2026-07-20"},
    ])

    assert len(reviews) == 1
    review = reviews[0]
    assert review["priority"] == "emergency"
    assert "6.8" in review["what_changed"]
    assert "4.7" in review["what_changed"]
    assert {fact["record_id"] for fact in review["patient_facts"]} == {"new", "old"}
    assert review["evidence"][0]["source_url"].startswith("https://www.nice.org.uk/")
    assert "Do not change prescribed medicines" in review["proposed_action"]


def test_moderate_potassium_is_urgent_and_labels_confirmation_uncertainty():
    review = _build(vitals=[
        {"vitals_id": "k1", "type": "serum potassium", "value": "6.2 mmol/L", "unit": "", "recorded_on": "2026-08-04"}
    ])[0]

    assert review["priority"] == "urgent"
    assert "same-day" in review["proposed_action"]
    assert "falsely high" in review["uncertainty"]


def test_unknown_result_is_suppressed_instead_of_inventing_a_range():
    reviews = _build(vitals=[
        {"vitals_id": "x", "type": "unfamiliar assay", "value": "999", "unit": "widgets", "recorded_on": "2026-08-04"}
    ])

    assert reviews == []


def test_messy_breathing_language_blocks_other_findings():
    today = datetime.now(timezone.utc).date().isoformat()
    reviews = _build(
        symptoms=[{
            "log_id": "s1", "symptom": "Really struggling to breathe!!!", "severity": 10,
            "logged_for": today, "notes": "getting worse",
        }],
        vitals=[{"vitals_id": "k1", "type": "potassium", "value": "6.1", "unit": "mmol/L", "recorded_on": "2026-08-04"}],
    )

    assert [review["priority"] for review in reviews] == ["emergency", "urgent"]
    assert "Call 999 now" in reviews[0]["proposed_action"]


def test_plainly_negated_emergency_symptom_does_not_trigger():
    today = datetime.now(timezone.utc).date().isoformat()
    reviews = _build(symptoms=[{
        "log_id": "s1", "symptom": "Indigestion", "severity": 3,
        "logged_for": today, "notes": "denies chest pain and no difficulty breathing",
    }])

    assert reviews == []


def test_old_emergency_wording_is_not_presented_as_current():
    reviews = _build(symptoms=[{
        "log_id": "s1", "symptom": "Chest pain", "severity": 8,
        "logged_for": "2020-01-01", "notes": "historic entry",
    }])

    assert reviews == []


def test_medicine_allergy_conflict_uses_exact_saved_facts():
    review = _build(
        medications=[{"medication_id": "m1", "name": "Penicillin", "created_at": "2026-08-04"}],
        allergies=[{"allergy_id": "a1", "name": "penicillin", "reaction": "rash", "created_at": "2025-01-01"}],
    )[0]

    assert review["category"] == "Medicine and allergy conflict"
    assert {fact["record_id"] for fact in review["patient_facts"]} == {"m1", "a1"}
    assert review["approver"] == "A qualified clinician"


def test_safety_review_records_the_complete_context_it_considered():
    reviews = build_safety_reviews(
        vitals=[],
        symptoms=[],
        medications=[{"medication_id": "m1", "name": "Penicillin"}],
        allergies=[{"allergy_id": "a1", "name": "Penicillin"}],
        conditions=[{"name": "Mastitis"}],
        triage_summaries=[{"urgency_level": "routine"}],
        document_summaries=[{"file": "letter.pdf", "summary": "Clinical letter"}],
        clinical_relationships=[{"relation": "taken_for"}],
        longitudinal_memory="Patient longitudinal summary",
    )

    considered = reviews[0]["context_considered"]
    assert considered["conditions"] == ["Mastitis"]
    assert considered["triage_record_count"] == 1
    assert considered["document_summary_count"] == 1
    assert considered["clinical_relationship_count"] == 1
    assert considered["longitudinal_summary_available"] is True


def test_warfarin_and_ibuprofen_never_tells_patient_to_stop_warfarin():
    review = _build(medications=[
        {"medication_id": "m1", "name": "Warfarin"},
        {"medication_id": "m2", "name": "Ibuprofen"},
    ])[0]

    assert review["rule_id"] == "warfarin-ibuprofen"
    assert "Do not stop warfarin on your own" in review["proposed_action"]


def test_saved_patient_confirmation_is_merged_without_clinician_approval():
    first = _build(vitals=[{"vitals_id": "k1", "type": "potassium", "value": "6.1"}])[0]
    state = update_review_state(None, {"status": "patient_confirmed"})
    merged = _build(
        vitals=[{"vitals_id": "k1", "type": "potassium", "value": "6.1"}],
        states={first["review_id"]: state},
    )[0]

    assert merged["status"] == "patient_confirmed"
    assert merged["approver"] == "A qualified clinician"
    assert merged["writeback"]["status"] == "not_configured"


def test_patient_cannot_claim_clinician_approval():
    with pytest.raises(ValueError, match="only a clinician"):
        update_review_state(None, {"status": "clinician_approved"})


def test_follow_up_requires_action_outcome():
    with pytest.raises(ValueError, match="whether the proposed action happened"):
        update_review_state(None, {"status": "follow_up_recorded", "note": "Called the practice"})
