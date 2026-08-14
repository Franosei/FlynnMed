import json
from types import SimpleNamespace

import backend.clinical_trials as clinical_trials
from backend.clinical_trials import (
    TrialSearchProfile,
    _exclude_not_relevant,
    _llm_batch_condition_match,
    build_trial_search_profile,
)


def test_trial_profile_contains_all_patient_record_families():
    profile = build_trial_search_profile(
        profile={},
        memory={"summary": "Combined longitudinal summary"},
        symptom_logs=[{"symptom": "Breast pain"}],
        medications=[{"name": "Flucloxacillin", "reason": "Mastitis"}],
        allergies=[{"name": "Penicillin", "reaction": "Hives", "severity": "severe"}],
        conditions=[{"name": "Mastitis"}],
        vitals=[{"type": "temperature", "value": "38", "unit": "C"}],
        triage_summaries=[{"escalation_triggers": ["Worsening fever"]}],
        document_summaries=[{"summary": "Discharge letter summary"}],
        clinical_relationships=[{
            "source_name": "Flucloxacillin", "relation": "taken_for",
            "target_name": "Mastitis", "certainty": "recorded_association",
        }],
        clinical_notes=[{"assessment": "Mastitis", "plan": "Clinical review"}],
        care_plans=[{"condition": "Mastitis", "status": "active"}],
        safety_reviews=[{"category": "Medicine safety", "what_changed": "Allergy conflict"}],
        conversation_summary="Earlier breastfeeding discussion",
    )

    for expected in (
        "Combined longitudinal summary", "Breast pain", "Flucloxacillin",
        "Penicillin", "Discharge letter summary", "taken for", "Clinical review",
        "Existing care plan", "Allergy conflict", "breastfeeding discussion",
    ):
        assert expected.lower() in profile.raw_context.lower()


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps({"results": []})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeClient:
    def __init__(self, api_key=None):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_batch_condition_match_prompt_instructs_flagging_ambiguous_terms(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = _FakeClient()
    monkeypatch.setattr(clinical_trials, "OpenAI", lambda **kw: fake_client)

    profile = TrialSearchProfile(
        conditions=["peak flow"],
        symptoms=[],
        medications=[],
        age=45,
        biological_sex="female",
        raw_context="Patient mentioned peak flow of 18.",
    )
    trial_stubs = [
        {
            "index": 0,
            "title": "A Study",
            "conditions": "Asthma",
            "summary": "A trial about asthma.",
            "eligibility": "Adults with asthma.",
        }
    ]

    _llm_batch_condition_match(trial_stubs, profile)

    assert fake_client.chat.completions.calls, "expected an LLM call"
    sent_prompt = fake_client.chat.completions.calls[0]["messages"][0]["content"]
    assert "ambiguous" in sent_prompt.lower()
    assert "reasoning" in sent_prompt


def test_exclude_not_relevant_drops_only_not_relevant_candidates():
    candidates = [
        {"nct_id": "NCT1", "match_level": "high"},
        {"nct_id": "NCT2", "match_level": "not_relevant"},
        {"nct_id": "NCT3", "match_level": "low"},
        {"nct_id": "NCT4"},  # missing match_level -- must not be dropped
    ]

    result = _exclude_not_relevant(candidates)

    assert [c["nct_id"] for c in result] == ["NCT1", "NCT3", "NCT4"]


def test_exclude_not_relevant_drops_specialty_mismatch_regardless_of_match_level():
    """
    A trial the LLM flagged specialty_mismatch=True must be dropped even if it was
    scored "medium"/"high" rather than "not_relevant" -- e.g. a trial that looked
    strong because it matched the wrong meaning of an ambiguous patient-profile
    term. Without this, only the softer "score conservatively" prompt instruction
    would apply, and the mismatched trial could still surface to the patient.
    """
    candidates = [
        {"nct_id": "NCT1", "match_level": "high", "specialty_mismatch": False},
        {"nct_id": "NCT2", "match_level": "medium", "specialty_mismatch": True},
        {"nct_id": "NCT3", "match_level": "low"},  # missing key -- must not be dropped
    ]

    result = _exclude_not_relevant(candidates)

    assert [c["nct_id"] for c in result] == ["NCT1", "NCT3"]


def test_batch_condition_match_parses_specialty_mismatch_from_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class _MismatchCompletions:
        def create(self, **kwargs):
            content = json.dumps({
                "results": [{
                    "index": 0,
                    "alignment_score": 35,
                    "match_level": "medium",
                    "aligned_conditions": [],
                    "exclusion_risks": [],
                    "reasoning": "Trial concerns renal clearance; patient's clearance reading is a surgical margin measurement.",
                    "specialty_mismatch": True,
                }]
            })
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_MismatchCompletions()))
    monkeypatch.setattr(clinical_trials, "OpenAI", lambda **kw: fake_client)

    profile = TrialSearchProfile(
        conditions=["clearance"], symptoms=[], medications=[], age=45,
        biological_sex="female", raw_context="Patient mentioned a clearance of 45.",
    )
    trial_stubs = [{
        "index": 0, "title": "A Study", "conditions": "CKD",
        "summary": "A trial about renal clearance.", "eligibility": "Adults with CKD.",
    }]

    result = _llm_batch_condition_match(trial_stubs, profile)

    assert result[0]["specialty_mismatch"] is True
