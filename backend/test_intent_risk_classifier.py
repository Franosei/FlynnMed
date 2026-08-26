import json
from types import SimpleNamespace


from backend.intent_risk_classifier import IntentRiskClassifier


class _FakeCompletions:
    def __init__(self, payload: dict):
        self._payload = payload
        self.last_messages = None

    def create(self, model, messages, response_format, temperature=None):
        self.last_messages = messages
        content = json.dumps(self._payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self, payload: dict):
        self.chat = SimpleNamespace(completions=_FakeCompletions(payload))


def _classifier_with_response(monkeypatch, payload: dict) -> IntentRiskClassifier:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    classifier = IntentRiskClassifier()
    classifier.client = _FakeClient(payload)
    return classifier


_AMBIGUOUS_PEAK_FLOW_PAYLOAD = {
    "intent_category": "general_info",
    "risk_level": "routine",
    "vulnerable_flags": [],
    "escalation_required": False,
    "escalation_reason": "",
    "pathway_hint": "general_triage",
    "confidence": 0.7,
    "presentation_hint": "none",
    "ambiguous_term_detected": True,
    "ambiguous_term": "peak flow",
    "ambiguity_clarifying_question": "Was your peak flow measured with a breathing device or during a urine flow test?",
    "ambiguity_reply_options": [
        {
            "display": "It was a breathing test",
            "prompt": "My peak flow was measured with a breathing/asthma peak flow meter -- what does my reading mean?",
        },
        {
            "display": "It was a urine flow test",
            "prompt": "My peak flow was measured during a urology urine flow test (uroflowmetry) -- what does my reading mean?",
        },
    ],
}


def test_ambiguous_term_with_no_history_is_flagged_with_reply_options(monkeypatch):
    classifier = _classifier_with_response(monkeypatch, _AMBIGUOUS_PEAK_FLOW_PAYLOAD)

    result = classifier.classify(
        "What is my peak flow level and what does it mean?",
        role_key="patient",
        patient_history=None,
    )

    assert result.ambiguous_term_detected is True
    assert result.ambiguous_term == "peak flow"
    assert result.ambiguity_clarifying_question
    assert len(result.ambiguity_reply_options) == 2
    assert all(o["display"] and o["prompt"] for o in result.ambiguity_reply_options)


def test_patient_history_is_included_in_prompt_for_disambiguation(monkeypatch):
    classifier = _classifier_with_response(monkeypatch, _AMBIGUOUS_PEAK_FLOW_PAYLOAD)
    history = SimpleNamespace(
        as_prompt_block=lambda: (
            "Recent vitals / labs:\n"
            "  Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement): 18 ml/s (2026-07-07)"
        )
    )

    classifier.classify(
        "What is my peak flow level and what does it mean?",
        role_key="patient",
        patient_history=history,
    )

    sent_prompt = classifier.client.chat.completions.last_messages[0]["content"]
    assert "NOT a respiratory measurement" in sent_prompt


def test_malformed_llm_response_never_surfaces_broken_clarification(monkeypatch):
    payload = dict(_AMBIGUOUS_PEAK_FLOW_PAYLOAD)
    payload["ambiguity_clarifying_question"] = ""  # model set the flag but forgot the question
    classifier = _classifier_with_response(monkeypatch, payload)

    result = classifier.classify("What is my peak flow level?", role_key="patient")

    assert result.ambiguous_term_detected is False
    assert result.ambiguity_clarifying_question == ""
    assert result.ambiguity_reply_options == []


def test_ambiguity_never_flagged_when_risk_level_is_urgent(monkeypatch):
    payload = dict(_AMBIGUOUS_PEAK_FLOW_PAYLOAD)
    payload["risk_level"] = "urgent"
    classifier = _classifier_with_response(monkeypatch, payload)

    result = classifier.classify("What is my peak flow level?", role_key="patient")

    assert result.ambiguous_term_detected is False
    assert result.risk_level == "urgent"


def test_recent_turns_continuation_block_is_included_in_prompt(monkeypatch):
    payload = dict(_AMBIGUOUS_PEAK_FLOW_PAYLOAD)
    payload["ambiguous_term_detected"] = False
    classifier = _classifier_with_response(monkeypatch, payload)
    recent_turns = [
        {"role": "user", "content": "What is my peak flow level and what does it mean?"},
        {
            "role": "assistant",
            "content": "Was your peak flow measured with a breathing device or during a urine flow test?",
        },
    ]

    classifier.classify(
        "It was a urine flow test",
        role_key="patient",
        recent_turns=recent_turns,
    )

    sent_prompt = classifier.client.chat.completions.last_messages[0]["content"]
    assert "Recent conversation" in sent_prompt
    assert "urine flow test" in sent_prompt


def test_default_ambiguity_fields_are_off():
    from backend.intent_risk_classifier import IntentClassification

    default = IntentClassification()
    assert default.ambiguous_term_detected is False
    assert default.ambiguous_term == ""
    assert default.ambiguity_clarifying_question == ""
    assert default.ambiguity_reply_options == []
    assert default.clarification_required is False
    assert default.clarification_reason == ""
    assert default.clarifying_questions == []


def test_personalized_medication_question_can_require_specific_intake(monkeypatch):
    payload = {
        **_AMBIGUOUS_PEAK_FLOW_PAYLOAD,
        "intent_category": "medication_query",
        "pathway_hint": "medications",
        "ambiguous_term_detected": False,
        "ambiguity_clarifying_question": "",
        "ambiguity_reply_options": [],
        "clarification_required": True,
        "clarification_reason": "the prescription indication is unknown",
        "clarifying_questions": [
            "What did the prescriber say the flucloxacillin is treating?",
            "Have you taken any doses, and did anything happen afterwards?",
            "Did the prescriber know about your recorded penicillin allergy?",
        ],
    }
    classifier = _classifier_with_response(monkeypatch, payload)

    result = classifier.classify(
        "My doctor prescribed flucloxacillin 500 mg for me?",
        role_key="patient",
    )

    assert result.clarification_required is True
    assert result.clarification_reason == "the prescription indication is unknown"
    assert len(result.clarifying_questions) == 3
    assert result.pathway_hint == "medications"
    sent_prompt = classifier.client.chat.completions.last_messages[0]["content"]
    assert "do not ask for anything already recorded or answered" in sent_prompt


def test_information_clarification_never_delays_urgent_guidance(monkeypatch):
    payload = {
        **_AMBIGUOUS_PEAK_FLOW_PAYLOAD,
        "risk_level": "urgent",
        "ambiguous_term_detected": False,
        "ambiguity_clarifying_question": "",
        "ambiguity_reply_options": [],
        "clarification_required": True,
        "clarification_reason": "details are missing",
        "clarifying_questions": ["What happened?"],
    }
    classifier = _classifier_with_response(monkeypatch, payload)

    result = classifier.classify("I need urgent help", role_key="patient")

    assert result.risk_level == "urgent"
    assert result.clarification_required is False
    assert result.clarifying_questions == []


def test_clinician_guideline_question_does_not_trigger_crisis_prescreen():
    classifier = IntentRiskClassifier.__new__(IntentRiskClassifier)
    question = (
        "I'm an emergency medicine physician seeing more in-hospital adult cardiac arrests. "
        "Walk me through the new BLS and ACLS guideline updates, vasopressor dosing intervals, "
        "and advanced airway research."
    )

    assert classifier._crisis_prescreen(question, role_key="doctor") is False


def test_active_clinician_emergency_still_triggers_crisis_prescreen():
    classifier = IntentRiskClassifier.__new__(IntentRiskClassifier)
    question = "My patient is in cardiac arrest right now and we are doing CPR."

    assert classifier._crisis_prescreen(question, role_key="doctor") is True


def test_postpartum_convulsions_trigger_crisis_prescreen():
    classifier = object.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "My postpartum patient is having repeated convulsions after delivery.",
        role_key="doctor",
    )

    assert result.risk_level == "crisis"
    assert result.crisis_detected is True


def test_skin_and_breathing_allergy_features_trigger_crisis_prescreen():
    classifier = object.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "One hour after eating squid I have itching and heaviness in my breath.",
        role_key="patient",
    )

    assert result.risk_level == "crisis"
    assert result.crisis_detected is True


def test_active_clinician_chest_pain_case_is_at_least_urgent():
    classifier = object.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "Need an ESI note for a 56-year-old patient presenting with chest pain, HR 110.",
        role_key="doctor",
    )

    assert result.risk_level == "urgent"
    assert result.escalation_required is True


def test_painful_dental_swelling_is_at_least_urgent():
    classifier = object.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "I have painful swelling in my jaw by a decayed tooth and it is throbbing.",
        role_key="patient",
    )

    assert result.risk_level == "urgent"


def test_active_cholera_outbreak_without_clean_water_is_urgent():
    classifier = object.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "How do we stop cholera in my village when we have no clean water?",
        role_key="patient",
    )

    assert result.risk_level == "urgent"


def test_explicitly_stable_clinical_note_cannot_become_unqualified_crisis(monkeypatch):
    payload = {
        **_AMBIGUOUS_PEAK_FLOW_PAYLOAD,
        "intent_category": "crisis",
        "risk_level": "crisis",
        "escalation_required": True,
        "escalation_reason": "Drowsiness may indicate deterioration.",
        "presentation_hint": "none",
        "ambiguous_term_detected": False,
        "ambiguity_clarifying_question": "",
        "ambiguity_reply_options": [],
    }
    classifier = _classifier_with_response(monkeypatch, payload)
    note = (
        "A 70-year-old gentleman is more drowsy than usual. He remains clinically stable, "
        "denies chest pain or shortness of breath, and the nurse is about to record his vital signs."
    )

    result = classifier.classify(note, role_key="nurse")

    assert result.risk_level == "urgent"
    assert result.intent_category == "symptom_triage"
    assert result.crisis_detected is False
    assert result.escalation_required is True
    assert "does not establish an active emergency" in result.escalation_reason


def test_explicit_active_emergency_overrides_stability_wording():
    classifier = object.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "The note says stable, but the patient is unresponsive and not breathing right now.",
        role_key="nurse",
    )

    assert result.risk_level == "crisis"
    assert result.crisis_detected is True


def test_witnessed_collapse_triggers_active_emergency_prescreen():
    classifier = object.__new__(IntentRiskClassifier)

    assert classifier._crisis_prescreen(
        "On the dance floor I saw someone collapse.", role_key="patient"
    ) is True


def test_general_education_about_collapse_is_not_an_active_emergency():
    classifier = object.__new__(IntentRiskClassifier)

    assert classifier._crisis_prescreen(
        "What are common reasons someone might collapse?", role_key="patient"
    ) is False


def test_patient_general_emergency_education_is_not_a_personal_crisis():
    classifier = IntentRiskClassifier.__new__(IntentRiskClassifier)

    assert classifier._crisis_prescreen(
        "What are stroke symptoms and how does the FAST test work?",
        role_key="patient",
    ) is False


def test_portuguese_personal_pneumonia_antibiotic_request_is_urgent():
    classifier = IntentRiskClassifier.__new__(IntentRiskClassifier)

    result = classifier.classify(
        "Estou com pneumonia e preciso saber qual antibiotico devo tomar.",
        role_key="patient",
    )

    assert result.risk_level == "urgent"
    assert result.intent_category == "medication_query"
    assert result.escalation_required is True


def test_clinician_pneumonia_guideline_request_is_not_personal_triage():
    classifier = IntentRiskClassifier.__new__(IntentRiskClassifier)

    assert classifier._acute_treatment_prescreen(
        "Review the antibiotic guideline updates for pneumonia treatment.",
        role_key="doctor",
    ) is False
