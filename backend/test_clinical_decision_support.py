from backend.clinical_decision_support import ClinicalDecisionSupportEngine
from backend.intent_risk_classifier import IntentClassification
from backend.role_router import RoleRouter
from backend.triage_summary import normalize_triage_output


def _decision_for(
    question: str,
    presentation_hint: str,
    role: str = "nurse",
    vulnerable_flags: list[str] | None = None,
):
    engine = ClinicalDecisionSupportEngine()
    router = RoleRouter()
    intent = IntentClassification(
        intent_category="symptom_triage",
        risk_level="routine",
        pathway_hint="general_triage",
        presentation_hint=presentation_hint,
        vulnerable_flags=vulnerable_flags or [],
    )
    return engine.assess(question, intent, router.resolve(role))


def test_thunderclap_headache_pathway_is_immediate_review():
    decision = _decision_for(
        "I have had a severe headache that came on suddenly an hour ago and it is the worst headache I have ever had.",
        "thunderclap_headache",
    )

    assert decision.pathway_id == "thunderclap_headache"
    assert decision.next_step == "Immediate review"
    assert decision.minimum_risk_level == "urgent"
    assert any("neurological observations" in item for item in decision.immediate_actions)


def test_possible_sepsis_pathway_adds_elderly_flag_and_news2_action():
    decision = _decision_for(
        "A 68-year-old patient has become increasingly confused over two days, has a temperature of 38.9 and is passing very little urine.",
        "possible_sepsis",
        vulnerable_flags=["elderly"],
    )

    assert decision.pathway_id == "possible_sepsis"
    assert decision.next_step == "Immediate review"
    assert "elderly" in decision.vulnerable_flags
    assert any("NEWS2" in item for item in decision.immediate_actions)


def test_possible_sepsis_pathway_cannot_invent_missing_findings_for_mastitis():
    question = (
        "I am 3 weeks postpartum and breastfeeding. My left breast is painful, "
        "swollen and red with a hot wedge-shaped area. I have a temperature of "
        "38.7 C, chills and body aches, and I can feel a firm lump. Could this be "
        "mastitis or a breast abscess?"
    )
    intent = IntentClassification(
        intent_category="maternity",
        risk_level="urgent",
        vulnerable_flags=["postpartum"],
        escalation_required=True,
        escalation_reason="Possible sepsis.",
        pathway_hint="maternity",
        presentation_hint="possible_sepsis",
    )

    decision = ClinicalDecisionSupportEngine().assess(
        question,
        intent,
        RoleRouter().resolve("patient"),
    )

    assert decision.pathway_id == "general_triage"
    assert decision.deterministic_response is False
    assert intent.presentation_hint == "none"
    assert "confusion" not in decision.rationale.lower()
    assert "urine" not in decision.rationale.lower()


def test_assistant_sepsis_claims_cannot_become_evidence_on_the_next_turn():
    question = "I have a fever of 38.7 C, chills and body aches."
    user_case_text = (
        "I am 3 weeks postpartum and have a painful red breast and fever.\n"
        + question
    )
    intent = IntentClassification(
        intent_category="crisis",
        risk_level="crisis",
        vulnerable_flags=["postpartum"],
        escalation_required=True,
        escalation_reason="Confusion, fever and reduced urine output suggest sepsis.",
        crisis_detected=True,
        pathway_hint="general_triage",
        presentation_hint="possible_sepsis",
    )

    decision = ClinicalDecisionSupportEngine().assess(
        question,
        intent,
        RoleRouter().resolve("patient"),
        case_text=user_case_text,
    )

    assert decision.pathway_id == "general_triage"
    assert intent.risk_level == "urgent"
    assert intent.crisis_detected is False
    assert intent.escalation_reason == (
        "The reported acute symptoms warrant prompt clinical assessment."
    )


def test_recurrent_blackout_pathway_is_same_day_review():
    decision = _decision_for(
        "A patient tells me they have been having episodes where everything goes black for a few seconds and they nearly fall. This has happened three times in the past two weeks.",
        "recurrent_blackout",
    )

    assert decision.pathway_id == "recurrent_blackout"
    assert decision.next_step == "Same-day review"
    assert any("12-lead ECG" in item for item in decision.immediate_actions)


def test_thunderclap_backstop_escalates_a_missed_classifier_hint():
    decision = _decision_for(
        "The worst headache I have ever had started suddenly one hour ago.",
        "none",
    )

    assert decision.pathway_id == "thunderclap_headache"
    assert decision.next_step == "Immediate review"


def test_sepsis_backstop_requires_all_grounded_findings():
    complete = _decision_for(
        "They are newly confused, febrile at 39.1 C, and passing very little urine.",
        "none",
    )
    incomplete = _decision_for(
        "They have a fever and chills but no confusion and normal urine output.",
        "none",
    )

    assert complete.pathway_id == "possible_sepsis"
    assert incomplete.pathway_id == "general_triage"


def test_recurrent_blackout_backstop_escalates_a_missed_classifier_hint():
    decision = _decision_for(
        "I have blacked out three times in the last two weeks.",
        "none",
    )

    assert decision.pathway_id == "recurrent_blackout"
    assert decision.next_step == "Same-day review"


def test_pathway_omits_named_guideline_when_retrieval_did_not_verify_it():
    decision = _decision_for(
        "A patient has had three near-blackout episodes in two weeks.",
        "recurrent_blackout",
    )
    answer = decision.render_markdown(
        "nurse",
        sources=[
            {
                "source_id": "S1",
                "title": "Unrelated general advice",
                "url": "https://example.test/unrelated",
                "provider": "example",
                "source_type": "official_guidance",
            }
        ],
    )

    assert "CG109" not in answer
    assert "No named pathway guideline was verified" in answer


def test_pathway_keeps_named_guideline_when_matching_source_was_retrieved():
    decision = _decision_for(
        "A patient has had three near-blackout episodes in two weeks.",
        "recurrent_blackout",
    )
    answer = decision.render_markdown(
        "nurse",
        sources=[
            {
                "source_id": "S1",
                "title": "Transient loss of consciousness ('blackouts') in over 16s",
                "url": "https://www.nice.org.uk/guidance/cg109",
                "provider": "nice",
                "source_type": "official_guidance",
            }
        ],
    )

    assert "CG109" in answer
    assert "[S1]" in answer


def test_chronic_cough_pathway_remains_prompt_without_red_flags():
    decision = _decision_for(
        "I have had a persistent cough for eight weeks. I am a non-smoker, I have not lost weight, and I have no night sweats.",
        "chronic_cough_no_red_flags",
    )

    assert decision.pathway_id == "chronic_cough_no_red_flags"
    assert decision.urgency_level == "Prompt"
    assert decision.next_step == "GP"


def test_normalize_triage_output_preserves_immediate_review_step():
    fallback = {
        "urgency_level": "Urgent",
        "next_step": "111",
        "what_to_monitor": ["Deterioration"],
        "rationale": "Urgent review needed.",
    }
    normalized = normalize_triage_output(
        {
            "urgency_level": "Emergency",
            "next_step": "Immediate review",
            "what_to_monitor": ["NEWS2 change"],
            "rationale": "Deterministic pathway selected.",
        },
        fallback,
    )

    assert normalized["next_step"] == "Immediate review"
    assert normalized["what_to_monitor"] == ["NEWS2 change"]
