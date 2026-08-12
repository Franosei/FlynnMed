import json

from backend.role_router import RoleRouter
from backend.summarizer import LLMHelper


class _FakeFollowUpMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeFollowUpChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeFollowUpMessage(content)


class _FakeFollowUpResponse:
    def __init__(self, payload: dict) -> None:
        self.choices = [_FakeFollowUpChoice(json.dumps(payload))]


class _FakeFollowUpCompletions:
    def __init__(self) -> None:
        self.last_kwargs = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeFollowUpResponse({"questions": [{"display": "stub", "prompt": "stub"}]})


class _FakeFollowUpChat:
    def __init__(self) -> None:
        self.completions = _FakeFollowUpCompletions()


class _FakeFollowUpClient:
    def __init__(self) -> None:
        self.chat = _FakeFollowUpChat()


def _helper_with_fake_client() -> tuple[LLMHelper, _FakeFollowUpCompletions]:
    helper = LLMHelper()
    fake_client = _FakeFollowUpClient()
    helper.client = fake_client
    return helper, fake_client.chat.completions


def test_follow_up_questions_patient_role_keeps_first_person_voice_prompt():
    helper, completions = _helper_with_fake_client()

    helper.generate_follow_up_questions(
        question="What should I take for a UTI?",
        answer="Nitrofurantoin is first-line for uncomplicated UTI.",
        role_key="patient",
        is_patient_scoped=False,
    )

    system_prompt = completions.last_kwargs["messages"][0]["content"]
    user_prompt = completions.last_kwargs["messages"][1]["content"]
    assert "patient's voice" in system_prompt
    assert "I also have a fever" in system_prompt
    assert "Patient health record" in user_prompt


def test_follow_up_questions_clinician_general_uses_research_style_and_omits_own_record():
    helper, completions = _helper_with_fake_client()

    helper.generate_follow_up_questions(
        question="First-line antibiotic for uncomplicated UTI?",
        answer="Nitrofurantoin is first-line, working via bacterial enzyme inhibition.",
        role_key="doctor",
        is_patient_scoped=False,
    )

    system_prompt = completions.last_kwargs["messages"][0]["content"]
    user_prompt = completions.last_kwargs["messages"][1]["content"]
    assert "NOT about a specific patient" in system_prompt
    assert "CONFIRM as true about themselves" not in system_prompt
    # The acting clinician's own health-record data must not leak into a
    # patient-agnostic research question.
    assert "Patient health record" not in user_prompt
    assert "Patient profile" not in user_prompt


def test_follow_up_questions_clinician_patient_scoped_uses_third_person_clinical_actions():
    helper, completions = _helper_with_fake_client()

    helper.generate_follow_up_questions(
        question="What antibiotic should we use for her UTI?",
        answer="Nitrofurantoin is appropriate given her records.",
        role_key="doctor",
        is_patient_scoped=True,
        patient_context="Allergies: Penicillin (severe)\nMedications: Warfarin",
    )

    system_prompt = completions.last_kwargs["messages"][0]["content"]
    user_prompt = completions.last_kwargs["messages"][1]["content"]
    assert "SPECIFIC patient's chart" in system_prompt
    assert "third person" in system_prompt
    assert "CONFIRM as true about themselves" not in system_prompt
    assert "Patient health record" in user_prompt
    assert "Penicillin" in user_prompt


def test_follow_up_questions_caregiver_role_stays_in_patient_voice_bucket():
    """
    Caregiver is deliberately excluded from the clinician tuple (matching the
    frontend's isClinicianRole allowlist) -- confirms it still gets the
    first-person patient-voice prompt, not a clinician one.
    """
    helper, completions = _helper_with_fake_client()

    helper.generate_follow_up_questions(
        question="What should my mother take for a UTI?",
        answer="Nitrofurantoin is first-line for uncomplicated UTI.",
        role_key="caregiver",
        is_patient_scoped=False,
    )

    system_prompt = completions.last_kwargs["messages"][0]["content"]
    assert "patient's voice" in system_prompt


def test_answer_question_uses_third_person_voice_for_clinician_patient_scoped_chart():
    """
    Regression test: a clinician asking about a specific patient's chart (e.g.
    "what was the recent medication" in the per-patient chart-lookup chat) was
    getting patient-voice "your medications include..." answers, because
    nothing told the model this was a clinician reading about someone else's
    chart -- role_config alone ("doctor") doesn't carry that signal.
    """
    helper, completions = _helper_with_fake_client()
    doctor_role = RoleRouter().resolve("doctor")

    helper.answer_question(
        question="What was the recent medication?",
        context="Patient: Jane Whitfield. Medications: Metformin 500mg.",
        role_config=doctor_role,
        is_patient_scoped=True,
    )

    system_prompt = completions.last_kwargs["messages"][0]["content"]
    assert "THIRD PERSON" in system_prompt
    assert "your blood pressure appears elevated" not in system_prompt


def test_answer_question_keeps_second_person_voice_for_clinicians_own_evidence_chat():
    """
    Companion test: a clinician's general (not patient-scoped) Evidence Review
    question must NOT get the third-person patient-chart voice instruction.
    """
    helper, completions = _helper_with_fake_client()
    doctor_role = RoleRouter().resolve("doctor")

    helper.answer_question(
        question="What is the first-line antibiotic for uncomplicated UTI?",
        context="",
        role_config=doctor_role,
        is_patient_scoped=False,
    )

    system_prompt = completions.last_kwargs["messages"][0]["content"]
    assert "THIRD PERSON" not in system_prompt
    assert "your blood pressure appears elevated" in system_prompt


def test_answer_generation():
    helper = LLMHelper()
    question = "Is dexamethasone safe for elderly patients?"
    sources = [
        {
            "source_id": "S1",
            "title": "Example corticosteroid safety study",
            "journal": "Example Journal",
            "year": "2024",
            "section": "Discussion",
            "snippet": "Older adults may require closer monitoring because adverse effects can be more frequent in frail populations.",
        }
    ]

    response = helper.answer_question(
        question=question,
        context="",
        source_briefings=sources,
        stream=False,
    )
    print(response)


_UNSUPPORTED_CLAIM_SOURCES = [
    {
        "source_id": "S1",
        "title": "Migraine self-care guidance",
        "snippet": "Rest in a dark, quiet room and stay hydrated during a migraine attack.",
    }
]
# The mechanism claim below is not present in any source and is specific
# enough that a reader would expect it to be evidence-backed.
_UNSUPPORTED_CLAIM_ANSWER = (
    "## Likely Explanation\n"
    "Migraines are caused by a 23% drop in serotonin binding at 5-HT2B receptors, "
    "which directly triggers the aura phase in 90% of cases.\n\n"
    "## What To Do Now\n"
    "Rest in a dark, quiet room and stay hydrated."
)


def test_check_claim_source_alignment_flags_unsupported_specific_claim():
    helper = LLMHelper()

    claims = helper.check_claim_source_alignment(
        answer_markdown=_UNSUPPORTED_CLAIM_ANSWER,
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
    )

    assert isinstance(claims, list)
    assert claims, "expected at least one extracted claim"
    for claim in claims:
        assert set(claim.keys()) == {"claim", "status", "requires_evidence", "source_ids"}
        assert claim["status"] in ("supported", "general_knowledge")

    unsupported = [
        c for c in claims if c["status"] == "general_knowledge" and c["requires_evidence"]
    ]
    assert unsupported, (
        "expected the fabricated serotonin/5-HT2B statistic to be flagged as "
        f"unsupported and evidence-requiring; got {claims}"
    )


def test_apply_claim_corrections_attributes_unsupported_claim_as_general_knowledge():
    helper = LLMHelper()
    # Synthetic finding, as if check_claim_source_alignment had flagged it --
    # kept independent of that test so each test makes its own bounded set of
    # live calls rather than chaining through another test function.
    unsupported_claims = [
        {
            "claim": "Migraines are caused by a 23% drop in serotonin binding at 5-HT2B receptors, which directly triggers the aura phase in 90% of cases.",
            "status": "general_knowledge",
            "requires_evidence": True,
            "source_ids": [],
        }
    ]

    revised = helper.apply_claim_corrections(
        answer_markdown=_UNSUPPORTED_CLAIM_ANSWER,
        unsupported_claims=unsupported_claims,
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
    )

    assert revised
    assert "## What To Do Now" in revised, "unrelated sections must survive the rewrite"
    # A cosmetic hedge swap ("may" for "is") isn't enough -- the specific
    # fabricated 23%/5-HT2B/90% figures must actually be gone, not just
    # softened in place (this is the exact failure mode found in the
    # generate-only evaluation: 77.7% of "corrected" claims still contained
    # the original wording verbatim).
    assert "23%" not in revised
    assert "5-HT2B" not in revised
    print(revised)


def test_apply_claim_corrections_inserts_missing_citation_for_supported_claim():
    helper = LLMHelper()
    answer = (
        "## Likely Explanation\n"
        "Resting in a dark, quiet room and staying well hydrated is commonly "
        "recommended during a migraine attack.\n\n"
        "## What To Do Now\n"
        "Track your triggers in a diary."
    )
    # Synthetic finding, as if check_claim_source_alignment had confirmed this
    # claim IS backed by S1 but the model never added the marker.
    uncited_supported_claims = [
        {
            "claim": "Resting in a dark, quiet room and staying well hydrated is commonly recommended during a migraine attack.",
            "status": "supported",
            "requires_evidence": True,
            "source_ids": ["S1"],
        }
    ]

    revised = helper.apply_claim_corrections(
        answer_markdown=answer,
        unsupported_claims=[],
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
        uncited_supported_claims=uncited_supported_claims,
    )

    assert revised
    assert "[S1]" in revised, "the confirmed citation must be inserted into the text"
    assert "## What To Do Now" in revised, "unrelated sections must survive the correction"
    print(revised)


def test_apply_claim_corrections_is_noop_without_flagged_claims():
    helper = LLMHelper()
    revised = helper.apply_claim_corrections(
        answer_markdown=_UNSUPPORTED_CLAIM_ANSWER,
        unsupported_claims=[],
        source_briefings=_UNSUPPORTED_CLAIM_SOURCES,
    )
    assert revised == _UNSUPPORTED_CLAIM_ANSWER


if __name__ == "__main__":
    test_answer_generation()
    test_check_claim_source_alignment_flags_unsupported_specific_claim()
    test_apply_claim_corrections_attributes_unsupported_claim_as_general_knowledge()
    test_apply_claim_corrections_inserts_missing_citation_for_supported_claim()
    test_apply_claim_corrections_is_noop_without_flagged_claims()
