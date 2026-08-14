from backend.summarizer import LLMHelper
from backend.task_mode import decide_task_mode
from backend.role_router import RoleRouter


def test_detects_documentation_without_granting_clinical_authority():
    decision = decide_task_mode(
        "Please draft this as a standard SOAP note.",
        chat_history=None,
        authenticated_role_key="patient",
    )

    assert decision.mode == "documentation"
    assert decision.presentation_audience == "professional"
    assert decision.requires_evidence_retrieval is False
    assert "does not change the authenticated role" in decision.prompt_block()


def test_translation_continuity_uses_earlier_user_instruction():
    history = [
        {
            "role": "user",
            "content": "Preciso que traduza os seguintes textos para português de Portugal.",
        },
        {"role": "assistant", "content": "Claro. Envie o texto."},
        {
            "role": "user",
            "content": "Implementation frameworks address clinical protocols.",
        },
        {
            "role": "assistant",
            "content": "Os modelos de implementação abrangem protocolos clínicos.",
        },
    ]

    decision = decide_task_mode(
        "Finally, compile the major systematic reviews and identify a standard framework.",
        chat_history=history,
        authenticated_role_key="patient",
    )

    assert decision.mode == "translation"
    assert decision.requires_evidence_retrieval is False
    assert "Translate only the current user text" in decision.prompt_block()


def test_professional_evidence_depth_does_not_change_patient_authorization():
    history = [
        {
            "role": "user",
            "content": "Share recent clinical trial data on SGLT2 inhibitors.",
        },
        {"role": "assistant", "content": "Summary."},
        {
            "role": "user",
            "content": "Summarize the 2021 ESC guidelines on heart failure.",
        },
        {"role": "assistant", "content": "Summary."},
    ]

    decision = decide_task_mode(
        "Latest advancements in atrial fibrillation treatment.",
        chat_history=history,
        authenticated_role_key="patient",
    )

    assert decision.mode == "professional_evidence_review"
    assert decision.presentation_audience == "professional"
    assert decision.requires_evidence_retrieval is True
    assert "authenticated role, permissions" in decision.prompt_block()
    assert "current formal guidelines" in decision.retrieval_question("AF advances")


def test_simple_patient_question_keeps_default_mode():
    decision = decide_task_mode(
        "Is a small scoop of ice cream likely to worsen mild bloating?",
        chat_history=None,
        authenticated_role_key="patient",
    )

    assert decision.mode == "clinical_answer"
    assert decision.presentation_audience == "patient"
    assert decision.requires_evidence_retrieval is True


def test_translation_prompt_suppresses_clinical_formatting_and_citations():
    helper = object.__new__(LLMHelper)
    captured = {}

    def _capture(messages, model=None):
        captured["messages"] = messages
        return "translated"

    helper._complete_response = _capture
    decision = decide_task_mode(
        "Translate this to Portuguese: Heart failure follow-up.",
        chat_history=None,
        authenticated_role_key="patient",
    )

    result = helper.answer_question(
        question="Translate this to Portuguese: Heart failure follow-up.",
        context="",
        task_mode=decision,
    )

    assert result == "translated"
    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "CONTROLLED TASK MODE: TRANSLATION" in system_prompt
    assert "output constraints override clinical headings" in system_prompt
    assert "do not add citations" in user_prompt
    assert "Available role-appropriate headings" not in user_prompt


def test_postpartum_completeness_contract_is_bounded_by_existing_policy():
    decision = decide_task_mode(
        "Should I seek care for mild postpartum pelvic pressure during squats?",
        chat_history=None,
        authenticated_role_key="patient",
    )

    block = decision.completion_block("maternity", ["postpartum"])

    assert "time since delivery" in block
    assert "pelvic-health physiotherapy" in block
    assert "cannot override deterministic clinical decisions or policy gates" in block


def test_medication_completeness_requires_inputs_without_granting_prescribing_authority():
    decision = decide_task_mode(
        "How much Tylenol should I give my 3-year-old?",
        chat_history=None,
        authenticated_role_key="patient",
    )

    block = decision.completion_block("medication_query", ["paediatric"])

    assert "weight, formulation, strength" in block
    assert "without prescribing" in block
    assert "breathing difficulty, a seizure, or inability to wake" in block
    assert "cannot override deterministic clinical decisions" in block


def test_completion_guidance_is_present_even_when_sources_are_supplied():
    helper = object.__new__(LLMHelper)
    captured = {}

    def _capture(messages, model=None):
        captured["messages"] = messages
        return "answer"

    helper._complete_response = _capture
    decision = decide_task_mode(
        "I feel warm. Is it serious?",
        chat_history=None,
        authenticated_role_key="patient",
    )
    guidance = decision.completion_block("symptom_triage")

    helper.answer_question(
        question="I feel warm. Is it serious?",
        context="context that would otherwise be hidden",
        source_briefings=[
            {
                "source_id": "S1",
                "title": "Fever guidance",
                "snippet": "Check temperature and symptoms.",
            }
        ],
        task_mode=decision,
        response_completion_guidance=guidance,
    )

    user_prompt = captured["messages"][1]["content"]
    assert "Controlled response requirements" in user_prompt
    assert "Identify only missing facts" in user_prompt


def test_clinician_answer_prompt_requires_one_prioritized_cited_decision():
    helper = object.__new__(LLMHelper)
    captured = {}

    def _capture(messages, model=None):
        captured["messages"] = messages
        return "answer"

    helper._complete_response = _capture
    helper.answer_question(
        question="A 50-year-old woman has right-arm numbness and tingling.",
        context="Clinical evidence context",
        role_config=RoleRouter().resolve("doctor"),
        source_briefings=[
            {
                "source_id": "S1",
                "title": "Formal neurological guidance",
                "snippet": "Assess acute focal neurological symptoms urgently.",
                "url": "https://example.test/guidance",
            }
        ],
    )

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "one prioritized provisional impression or must-not-miss syndrome" in system_prompt
    assert "Keep routine clinician answers to about 120 words" in system_prompt
    assert "use no more than three short sections" in user_prompt
    assert "prioritized decision and disposition" in user_prompt


def test_other_clinician_is_not_routed_as_a_doctor():
    role = RoleRouter().resolve("Other Clinician")

    assert role.role_key == "healthcare_professional"
    assert role.display_label == "Other Clinician"


def test_chart_lookup_detects_simple_factual_questions_before_clinician_catchall():
    """
    The chart_lookup check must run BEFORE the authenticated-clinician
    catch-all that otherwise routes every clinician question to
    professional_evidence_review unconditionally -- this is a regression
    guard for exactly that ordering.
    """
    for question in (
        "What was the recent medication?",
        "Does she have any allergies?",
        "What was her last BP reading?",
        "What are her current medications?",
    ):
        decision = decide_task_mode(question, chat_history=None, authenticated_role_key="doctor")
        assert decision.mode == "chart_lookup", question
        assert decision.requires_evidence_retrieval is False
        assert decision.presentation_audience == "professional"


def test_chart_lookup_also_fires_for_a_patient_asking_about_their_own_chart():
    decision = decide_task_mode(
        "What medications am I on?", chat_history=None, authenticated_role_key="patient"
    )
    assert decision.mode == "chart_lookup"
    assert decision.presentation_audience == "patient"


def test_chart_lookup_does_not_fire_for_questions_needing_real_clinical_evidence():
    """
    The safety-critical negative cases: a chart field being mentioned is not
    enough on its own -- any evaluative/advisory/causal-reasoning language
    must fall through to the normal evidence-retrieval path instead. A false
    positive here (skipping retrieval on a question that needed real
    evidence) is worse than the unnecessary search a false negative costs.
    """
    for question in (
        "Is the patient's current dose of metformin appropriate given their eGFR?",
        "Should we adjust her warfarin given this new medication?",
        "What does current guidance say about her blood pressure target?",
        "What is the latest evidence on treating her hypertension?",
        "What is causing her elevated blood pressure?",
        "Why does she have recurrent UTIs?",
        "What treatment would you recommend for her condition?",
    ):
        decision = decide_task_mode(question, chat_history=None, authenticated_role_key="doctor")
        assert decision.mode != "chart_lookup", question


def test_chart_lookup_completion_block_is_empty():
    decision = decide_task_mode(
        "What was the recent medication?", chat_history=None, authenticated_role_key="doctor"
    )
    assert decision.completion_block("administrative") == ""


def test_chart_lookup_prompt_suppresses_citations_but_keeps_clinical_commentary():
    helper = object.__new__(LLMHelper)
    captured = {}

    def _capture(messages, model=None):
        captured["messages"] = messages
        return "answer"

    helper._complete_response = _capture
    decision = decide_task_mode(
        "What was the recent medication?", chat_history=None, authenticated_role_key="doctor"
    )

    helper.answer_question(
        question="What was the recent medication?",
        context="Current medications: Metformin, Warfarin",
        task_mode=decision,
    )

    system_prompt = captured["messages"][0]["content"]
    user_prompt = captured["messages"][1]["content"]
    assert "CONTROLLED TASK MODE: CHART LOOKUP" in system_prompt
    assert "do not add [S#]" in user_prompt
    assert "Available role-appropriate headings" not in user_prompt
    # Unlike transformation mode, chart_lookup does NOT suppress clinical
    # commentary -- only the is_transformation branch says that.
    assert "do not append clinical commentary" not in user_prompt


def test_requests_structured_compilation_fires_for_clinician_flowsheet_request():
    """
    Found via a real HealthBench case: "Go ahead and compile everything into
    a single standardized flowsheet for daily rounding" resolved role
    correctly as doctor, but still only got a partial prose description
    instead of the requested exhaustive structured artifact.
    """
    for question in (
        "Go ahead and compile everything into a single standardized flowsheet for daily rounding",
        "Can you build a checklist for pre-op assessment?",
        "Put together a rounding template for the ward",
        "Please create a tracking sheet for glucose readings",
    ):
        decision = decide_task_mode(question, chat_history=None, authenticated_role_key="doctor")
        assert decision.requests_structured_compilation is True, question


def test_requests_structured_compilation_does_not_fire_for_a_patient():
    decision = decide_task_mode(
        "Can you compile a checklist for my medications?",
        chat_history=None,
        authenticated_role_key="patient",
    )
    assert decision.requests_structured_compilation is False


def test_requests_structured_compilation_does_not_fire_for_ordinary_clinician_questions():
    for question in (
        "What is the latest evidence on treating hypertension?",
        "Should we adjust her warfarin given this new medication?",
        "What was the recent medication?",
    ):
        decision = decide_task_mode(question, chat_history=None, authenticated_role_key="doctor")
        assert decision.requests_structured_compilation is False, question
