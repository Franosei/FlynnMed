from backend.response_templates import (
    DEFAULT_PERSONA_BLOCK,
    build_crisis_response,
    get_conversation_turn_guidance,
    get_persona_block,
    get_section_headings,
)
from backend.role_router import RoleRouter


def test_default_persona_is_safe_competent_and_not_senior():
    assert "safe and competent clinical information assistant" in DEFAULT_PERSONA_BLOCK
    assert "senior clinical information specialist" not in DEFAULT_PERSONA_BLOCK


def test_doctor_persona_pushes_management_without_senior_claim():
    persona = get_persona_block("doctor").lower()

    assert "initial management" in persona
    assert "not a senior specialist" in persona
    assert "clear route" in persona
    assert "one prioritized working impression" in persona
    assert "do not produce an unranked differential list" in persona
    assert "inline citation" in persona


def test_clinician_headings_lead_with_management_sections():
    assert get_section_headings("doctor")[:4] == [
        "## Working Impression",
        "## Immediate Management",
        "## Investigations / Monitoring",
        "## Escalate Now If",
    ]
    assert get_section_headings("nurse")[1] == "## Immediate Nursing Actions"


def test_patient_headings_include_monitoring_and_urgent_route():
    headings = get_section_headings("patient")

    assert "## What To Monitor" in headings
    assert "## Get Urgent Help If" in headings


def test_other_clinician_has_scope_aware_decision_format():
    persona = get_persona_block("healthcare_professional").lower()
    headings = get_section_headings("healthcare_professional")

    assert "exact regulated profession is not specified" in persona
    assert "locally authorised scope" in persona
    assert "cite the decision and action inline" in persona
    assert headings[0] == "## Prioritized Decision"


def test_crisis_response_is_role_appropriate():
    assert "local emergency number" in build_crisis_response("patient")
    clinical = build_crisis_response("doctor")
    assert "resuscitation pathway" in clinical
    assert "local emergency number" not in clinical
    assert "obstetric" not in clinical.lower()

    maternity = build_crisis_response("midwife")
    assert "maternity" in maternity.lower()
    assert "resuscitation pathway" in maternity


def test_every_role_gets_an_ongoing_conversation_contract():
    roles = [
        "patient",
        "caregiver",
        "doctor",
        "nurse",
        "midwife",
        "physiotherapist",
        "healthcare_professional",
    ]

    for role in roles:
        guidance = get_conversation_turn_guidance(role).lower()
        assert "ongoing conversation" in guidance
        assert "one focused follow-up question" in guidance
        assert "do not ask a batch of questions" in guidance

    assert "peer-to-peer colleague" in get_conversation_turn_guidance("doctor")
    assert "warm everyday language" in get_conversation_turn_guidance("patient")


def test_health_worker_alias_receives_healthcare_professional_conversation():
    role = RoleRouter().resolve("Health Worker")

    assert role.role_key == "healthcare_professional"
    assert "peer-to-peer colleague" in get_conversation_turn_guidance(role.role_key)
