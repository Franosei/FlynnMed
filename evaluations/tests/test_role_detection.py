from evaluations.models import ConversationTurn, EvalCase
from evaluations.role_detection import (
    detect_stated_role,
    eval_account_username,
    resolve_case_role,
)


def _case(*user_texts: str, tags=None) -> EvalCase:
    conversation = [ConversationTurn(role="user", content=text) for text in user_texts]
    return EvalCase(
        case_id="case-1",
        source_dataset="healthbench",
        conversation=conversation,
        rubrics=[],
        tags=tags or [],
    )


def test_detects_doctor_self_identification():
    case = _case("I'm an emergency medicine physician. What are the new ACLS updates?")
    assert detect_stated_role(case) == "doctor"


def test_detects_doctor_with_curly_apostrophe():
    """Real HealthBench text uses a curly/smart apostrophe (U+2019, "I'm")
    rather than a straight one ("I'm") -- verified against an actual case
    (0b8f1d60-2081-4562-98f7-b6a976fe1c6d) whose text is literally
    "I’m a 39 year old female. I’m an emergency medicine physician...".
    A regex written only for the straight apostrophe silently fails on this
    and everything else in the dataset that uses smart quotes."""
    case = _case(
        "I’m a 39 year old female. I’m an emergency medicine "
        "physician. What are the new ACLS updates?"
    )
    assert detect_stated_role(case) == "doctor"


def test_detects_doctor_variants():
    for phrasing in [
        "I am a doctor and I have a question.",
        "I'm a GP looking for the latest guidance.",
        "I am a general practitioner working in a busy clinic.",
    ]:
        assert detect_stated_role(_case(phrasing)) == "doctor", phrasing


def test_detects_informal_doctor_self_identification_from_healthbench():
    case = _case(
        "I have a septic shock patient in the ICU. I'm a critical care doc "
        "and I want to review the evidence."
    )

    assert detect_stated_role(case) == "doctor"


def test_detects_nurse():
    case = _case("I'm a nurse on a busy ward, what's the correct dose?")
    assert detect_stated_role(case) == "nurse"


def test_detects_midwife():
    case = _case("I am a midwife caring for a postpartum patient.")
    assert detect_stated_role(case) == "midwife"


def test_detects_physiotherapist():
    case = _case("I'm a physiotherapist treating a patient with lower back pain.")
    assert detect_stated_role(case) == "physiotherapist"
    case2 = _case("I am a physical therapist and need guidance on this case.")
    assert detect_stated_role(case2) == "physiotherapist"


def test_defaults_to_patient_with_no_self_identification():
    case = _case("I have a mild headache, what should I do?")
    assert detect_stated_role(case) == "patient"


def test_healthbench_professional_audience_tag_routes_to_clinician():
    case = _case(
        "What is the recommended blood pressure target for octogenarians?",
        tags=["physician_agreed_category:health-professional"],
    )

    resolution = resolve_case_role(case)

    assert resolution.role == "healthcare_professional"
    assert "audience tag" in resolution.reason
    assert resolution.confidence == 0.95


def test_healthbench_non_professional_tag_keeps_patient_mode():
    case = _case(
        "What are the current guidelines for treating my symptoms?",
        tags=["physician_agreed_category:not-health-professional"],
    )

    assert resolve_case_role(case).role == "patient"


def test_clinician_managed_patient_language_routes_to_clinician():
    case = _case("My patient started methotrexate. How often should I check LFTs?")

    assert resolve_case_role(case).role == "healthcare_professional"


def test_nursing_context_routes_to_nurse():
    case = _case(
        "My patient on the psych ward has chest tightness. Act like a senior nurse."
    )

    assert resolve_case_role(case).role == "nurse"


def test_multiple_professional_language_signals_route_to_clinician():
    case = _case(
        "What is the latest consensus on perioperative beta-blocker management?"
    )

    assert resolve_case_role(case).role == "healthcare_professional"


def test_personal_clinical_language_remains_patient_mode():
    case = _case("My eGFR is 35. Should I change my dose?")

    assert resolve_case_role(case).role == "patient"


def test_does_not_misdetect_third_party_mentions():
    """A patient saying 'my doctor told me...' must not be misdetected as the
    patient themselves being a doctor -- this is exactly the false-positive
    risk the conservative first-person-only pattern guards against."""
    case = _case(
        "My doctor told me to take ibuprofen, but I'm not sure if that's safe."
    )
    assert detect_stated_role(case) == "patient"


def test_only_scans_user_turns_not_assistant_turns():
    conversation = [
        ConversationTurn(role="user", content="What should I do for a headache?"),
        ConversationTurn(
            role="assistant", content="I am a doctor, so trust my advice."
        ),
    ]
    case = EvalCase(
        case_id="case-1",
        source_dataset="healthbench",
        conversation=conversation,
        rubrics=[],
        tags=[],
    )
    assert detect_stated_role(case) == "patient"


def test_recognizes_pt_abbreviation_as_clinician_managed_patient_language():
    """
    Regression test: found via a real HealthBench case
    ("pt can't take doxycycline. any other options?") that fell through to
    patient-mode role resolution -- the existing clinician-context patterns
    only recognized the spelled-out word "patient" ("patient has", "patient
    is", ...), not the extremely common clinical-note abbreviation "pt".
    """
    case = _case("pt can't take doxycycline. any other options?")
    assert resolve_case_role(case).role == "healthcare_professional"

    case2 = _case("pt has a fever and cough, what should we start?")
    assert resolve_case_role(case2).role == "healthcare_professional"


def test_pt_abbreviation_does_not_misdetect_physical_therapy_mentions():
    """
    Companion test: "pt" is also the common abbreviation for physical
    therapy, so the pattern must require a clinical-verb follow-up
    ("pt can't/has/is...") rather than matching the bare token anywhere.
    """
    case = _case("Should I be referred for PT and some rehab exercises?")
    assert detect_stated_role(case) == "patient"

    case2 = _case("My PT/INR came back elevated, is that a problem?")
    assert detect_stated_role(case2) == "patient"


def test_eval_account_username_is_namespaced_and_deterministic():
    username = eval_account_username("doctor", "abc-123-def")
    assert username.startswith("eval-harness-doctor-")
    assert eval_account_username("doctor", "abc-123-def") == username


def test_eval_account_username_strips_unsafe_characters():
    username = eval_account_username("doctor", "abc/123 def!!")
    assert "/" not in username
    assert " " not in username
    assert "!" not in username
