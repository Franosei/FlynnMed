from backend.clinical_orchestrator import ClinicalOrchestrator
from backend.context_graph import ContextEdge, ContextGraph, format_relationships_for_user
from backend.intent_risk_classifier import IntentClassification
from backend.patient_history import PatientHistoryContext
from backend.policy_engine import PolicyEngine
from backend.relationship_engine import derive_relationships, truncate_at_word_boundary
from backend.role_router import RoleRouter


def _relationship_graph() -> ContextGraph:
    return ContextGraph(
        edges=[
            ContextEdge(
                source_type="triage",
                source_name="Dry-eye symptoms after a long clinical note",
                relation="led_to",
                target_type="care_action",
                target_name="Arrange a routine optometry review",
                relation_class="clinical_decision",
                certainty="documented",
                evidence="Recorded triage decision",
                relevance_score=0.9,
            )
        ]
    )


def _fail_if_internal_prompt_format_is_used(self):
    raise AssertionError("internal ContextEdge.prompt_line() reached user output")


def test_patient_relationship_formatter_never_uses_internal_prompt_line(monkeypatch):
    monkeypatch.setattr(ContextEdge, "prompt_line", _fail_if_internal_prompt_format_is_used)

    text = format_relationships_for_user(_relationship_graph().edges)

    assert "Your record notes that" in text
    assert "Arrange a routine optometry review" in text
    assert "[clinical_decision; documented]" not in text
    assert "Recorded triage decision" not in text
    assert "led_to" not in text


def test_information_clarification_uses_patient_facing_relationship_text(monkeypatch):
    monkeypatch.setattr(ContextEdge, "prompt_line", _fail_if_internal_prompt_format_is_used)
    orchestrator = ClinicalOrchestrator.__new__(ClinicalOrchestrator)
    intent = IntentClassification(
        intent_category="symptom_triage",
        clarification_required=True,
        clarifying_questions=["When did the symptoms begin?"],
    )

    bundle = orchestrator._build_information_clarification_bundle(
        question="What should I do?",
        normalized_user="patient1",
        role_config=RoleRouter().resolve("patient"),
        intent=intent,
        patient_history=PatientHistoryContext(),
        context_graph=_relationship_graph(),
    )
    answer = bundle["payload"]["answer_markdown"]

    assert "Relevant record context:" in answer
    assert "[clinical_decision; documented]" not in answer
    assert "Recorded triage decision" not in answer
    assert "led_to" not in answer


def test_limited_evidence_response_uses_same_patient_facing_formatter(monkeypatch):
    monkeypatch.setattr(ContextEdge, "prompt_line", _fail_if_internal_prompt_format_is_used)
    orchestrator = ClinicalOrchestrator.__new__(ClinicalOrchestrator)
    intent = IntentClassification(intent_category="symptom_triage")

    answer = orchestrator._build_limited_evidence_response(
        question="What should I do?",
        personal_context=[],
        role_config=RoleRouter().resolve("patient"),
        intent=intent,
        patient_history=PatientHistoryContext(),
        question_medications=[],
        context_graph=_relationship_graph(),
    )

    assert "Relevant record context:" in answer
    assert "[clinical_decision; documented]" not in answer
    assert "Recorded triage decision" not in answer
    assert "led_to" not in answer


def test_relationship_text_truncates_at_a_word_boundary():
    text = "A detailed clinical narrative about eye discomfort and dryness"

    truncated = truncate_at_word_boundary(text, 42)

    assert truncated == "A detailed clinical narrative about..."
    assert len(truncated) <= 42


def test_triage_relationship_prefers_the_structured_pathway_label():
    relationships = derive_relationships(
        triage_summaries=[
            {
                "pathway_label": "Eye symptoms",
                "decision_summary": "A very long free-text clinical narrative " * 20,
                "next_step": "Routine optometry review",
            }
        ]
    )

    assert relationships[0]["source_name"] == "Eye symptoms"


def test_triage_relationship_does_not_store_a_free_text_note_as_its_name():
    relationships = derive_relationships(
        triage_summaries=[
            {
                "decision_summary": "A full free-text clinical narrative " * 20,
                "next_step": "Routine optometry review",
            }
        ]
    )

    assert relationships[0]["source_name"] == "Recorded triage concern"


def test_postpartum_maternity_intent_does_not_apply_pregnancy_gate_for_midwife():
    decision = PolicyEngine().gate(
        intent=IntentClassification(
            intent_category="maternity",
            vulnerable_flags=["postpartum"],
        ),
        role_config=RoleRouter().resolve("midwife"),
        question="Six weeks postpartum, is pelvic pressure during squats expected?",
    )

    assert "pregnancy_safety" not in {
        gate.gate_name for gate in decision.gates_applied
    }
    assert "Pregnancy-related question" not in decision.escalation_banner
    assert "postpartum_safety" in {gate.gate_name for gate in decision.gates_applied}


def test_newborn_maternity_intent_does_not_apply_pregnancy_gate():
    decision = PolicyEngine().gate(
        intent=IntentClassification(
            intent_category="maternity",
            vulnerable_flags=["newborn"],
        ),
        role_config=RoleRouter().resolve("patient"),
        question="How often should my newborn feed?",
    )

    assert "pregnancy_safety" not in {
        gate.gate_name for gate in decision.gates_applied
    }
    assert "Pregnancy-related question" not in decision.escalation_banner
    assert "newborn_safety" in {gate.gate_name for gate in decision.gates_applied}


def test_postpartum_medication_caution_uses_postpartum_wording():
    decision = PolicyEngine().gate(
        intent=IntentClassification(
            intent_category="medication_query",
            vulnerable_flags=["postpartum"],
        ),
        role_config=RoleRouter().resolve("patient"),
        question="Can I take this medicine six weeks after giving birth?",
    )

    assert "Postpartum medicine check" in decision.vulnerability_notice
    assert "Pregnancy-related question" not in decision.vulnerability_notice


def test_newborn_medication_caution_uses_newborn_wording():
    decision = PolicyEngine().gate(
        intent=IntentClassification(
            intent_category="medication_query",
            vulnerable_flags=["newborn"],
        ),
        role_config=RoleRouter().resolve("patient"),
        question="Can I give this medicine to my newborn?",
    )

    assert "Newborn medicine check" in decision.vulnerability_notice
    assert "Pregnancy-related question" not in decision.vulnerability_notice


def test_precise_pregnancy_flag_still_applies_pregnancy_gate():
    decision = PolicyEngine().gate(
        intent=IntentClassification(
            intent_category="maternity",
            vulnerable_flags=["pregnancy"],
        ),
        role_config=RoleRouter().resolve("patient"),
        question="I am pregnant and have a routine question.",
    )

    assert "pregnancy_safety" in {gate.gate_name for gate in decision.gates_applied}
    assert "Pregnancy-related question" in decision.escalation_banner


def test_explicit_pregnancy_medication_question_is_a_safe_fallback_trigger():
    decision = PolicyEngine().gate(
        intent=IntentClassification(intent_category="medication_query"),
        role_config=RoleRouter().resolve("patient"),
        question="Can I take ibuprofen while pregnant?",
    )

    assert "pregnancy_safety" in {gate.gate_name for gate in decision.gates_applied}
