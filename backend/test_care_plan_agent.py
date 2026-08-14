import json
from types import SimpleNamespace

import backend.care_plan_agent as care_plan_agent
from backend.care_plan_agent import CarePlanAgent
from backend.evidence_schema import ArticleEvidence


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("response_format"):
            content = json.dumps(
                {
                    "condition": "Type 2 Diabetes",
                    "title": "Type 2 Diabetes Care Plan",
                    "goals": [],
                    "daily_tasks": [],
                    "weekly_tasks": [],
                    "medication_reminders": [],
                    "lab_reminders": [],
                    "escalation_thresholds": [],
                    "lifestyle": {},
                    "missed_care_checklist": [],
                    "evidence_summary": "test",
                    "safety_notes": "test",
                }
            )
        else:
            content = ""
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content, tool_calls=None)
                )
            ]
        )


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _build_agent(monkeypatch) -> CarePlanAgent:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = CarePlanAgent()
    agent._client = _FakeClient()
    agent._guidance = SimpleNamespace(search=lambda *a, **kw: [])
    agent._pubmed = SimpleNamespace(search_article_records=lambda *a, **kw: [])
    return agent


def test_generate_system_prompt_instructs_flagging_ambiguous_data(monkeypatch):
    agent = _build_agent(monkeypatch)

    agent.generate(
        condition="Type 2 Diabetes",
        user_context={"profile": {}, "medications": [], "conditions": []},
    )

    calls = agent._client.chat.completions.calls
    assert calls, "expected at least one LLM call"
    system_prompt = calls[0]["messages"][0]["content"]
    assert "ambiguous" in system_prompt.lower()
    assert "safety_notes" in system_prompt


def test_generate_receives_all_patient_record_families(monkeypatch):
    agent = _build_agent(monkeypatch)

    agent.generate(
        condition="Mastitis",
        user_context={
            "profile": {},
            "medications": [{"name": "Flucloxacillin"}],
            "conditions": [{"name": "Mastitis"}],
            "allergies": [{"name": "Penicillin"}],
            "symptom_logs": [{"symptom": "Breast pain"}],
            "document_summaries": [{"summary": "Discharge letter"}],
            "clinical_relationships": [{"source_name": "Medicine", "relation": "taken_for", "target_name": "Mastitis"}],
            "clinical_notes": [{"assessment": "Mastitis", "plan": "Review"}],
            "existing_care_plans": [{"condition": "Mastitis"}],
            "safety_reviews": [{"category": "Medicine and allergy conflict"}],
            "longitudinal_memory": "Longitudinal summary",
            "conversation_summary": "Whole chat summary",
            "previous_five_chat": [{"role": "user", "content": "Previous exact message"}],
        },
    )

    system_prompt = agent._client.chat.completions.calls[0]["messages"][0]["content"]
    for expected in (
        "Breast pain", "Discharge letter", "taken_for", "clinical_notes",
        "Medicine and allergy conflict", "Longitudinal summary", "Whole chat summary",
        "Previous exact message",
    ):
        assert expected in system_prompt


def test_generate_gp_prep_instructs_flagging_ambiguous_data(monkeypatch):
    agent = _build_agent(monkeypatch)
    plan = {
        "condition": "Type 2 Diabetes",
        "goals": [],
        "medication_reminders": [],
        "lab_reminders": [],
        "escalation_thresholds": [],
        "missed_care_checklist": [],
        "after_visit_notes": [],
    }

    agent.generate_gp_prep(plan, user_context={"profile": {}})

    calls = agent._client.chat.completions.calls
    sent_prompt = calls[-1]["messages"][0]["content"]
    assert "ambiguous" in sent_prompt.lower()
    assert "Symptoms or concerns to mention" in sent_prompt


def test_nhs_search_excludes_specialty_mismatched_sources(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._extraction_context = {
        "question": "Care plan for chronic kidney disease",
        "patient_summary": "test patient",
        "medications": [],
        "conditions": [],
    }
    agent._guidance = SimpleNamespace(
        search=lambda queries, per_source_limit=2: [
            {"title": "Respiratory peak flow guidance", "snippet": "about breathing", "url": "http://a"},
            {"title": "Kidney disease guidance", "snippet": "about kidneys", "url": "http://b"},
        ]
    )

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        if source["title"] == "Respiratory peak flow guidance":
            return ArticleEvidence(
                source_id=source["source_id"], title=source["title"],
                answers_question=False, alignment_confidence=0.0,
            )
        return ArticleEvidence(
            source_id=source["source_id"], title=source["title"],
            answers_question=True, alignment_confidence=0.7,
        )

    monkeypatch.setattr(care_plan_agent, "_extract_one_article", fake_extract)

    result = agent._nhs("kidney disease guidance")

    assert "Kidney disease guidance" in result
    assert "Respiratory peak flow guidance" not in result


def test_pubmed_search_excludes_specialty_mismatched_sources(monkeypatch):
    agent = _build_agent(monkeypatch)
    agent._extraction_context = {
        "question": "Care plan for chronic kidney disease",
        "patient_summary": "test patient",
        "medications": [],
        "conditions": [],
    }
    agent._pubmed = SimpleNamespace(
        search_article_records=lambda query, n: [
            {"title": "Unrelated respiratory study", "abstract": "about lungs", "year": "2020", "journal": "J"},
            {"title": "Relevant nephrology study", "abstract": "about kidneys", "year": "2021", "journal": "J"},
        ]
    )

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        if source["title"] == "Unrelated respiratory study":
            return ArticleEvidence(
                source_id=source["source_id"], title=source["title"],
                answers_question=False, alignment_confidence=0.0,
            )
        return ArticleEvidence(
            source_id=source["source_id"], title=source["title"],
            answers_question=True, alignment_confidence=0.8,
        )

    monkeypatch.setattr(care_plan_agent, "_extract_one_article", fake_extract)

    result = agent._pubmed_search("kidney disease")

    assert "Relevant nephrology study" in result
    assert "Unrelated respiratory study" not in result


def test_filter_relevant_excludes_specialty_mismatch_regardless_of_confidence(monkeypatch):
    """A source the extractor confirms is a specialty mismatch must be dropped even
    if it otherwise scored as high-confidence and answering the question -- the
    hard specialty_mismatch signal must not be silently ignored (this was the bug:
    a urology 'peak flow' reading pulled in respiratory/asthma evidence)."""
    agent = _build_agent(monkeypatch)
    agent._extraction_context = {
        "question": "Care plan for Asthma",
        "patient_summary": "Recent vitals: Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement): 18 mL/s",
        "medications": [],
        "conditions": [],
    }
    agent._guidance = SimpleNamespace(
        search=lambda queries, per_source_limit=2: [
            {"title": "Respiratory peak flow guidance", "snippet": "about breathing", "url": "http://a"},
        ]
    )

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        return ArticleEvidence(
            source_id=source["source_id"], title=source["title"],
            answers_question=True, alignment_confidence=0.9,
            specialty_mismatch=True,
            specialty_mismatch_reason="Discusses respiratory PEFR, patient's reading is urology Qmax.",
        )

    monkeypatch.setattr(care_plan_agent, "_extract_one_article", fake_extract)

    result = agent._nhs("peak flow guidance")

    assert "No relevant NHS/NICE results" in result


def test_generate_includes_disambiguated_vitals_in_prompt(monkeypatch):
    """CarePlanAgent.generate() must thread the patient's real vitals through
    build_patient_history_context so the synthesis LLM sees the disambiguated
    label (e.g. urology Qmax, not a bare 'peak_flow' key it would default to
    reading as respiratory)."""
    agent = _build_agent(monkeypatch)

    agent.generate(
        condition="Asthma",
        user_context={
            "profile": {},
            "medications": [],
            "conditions": [],
            "vitals": [
                {"type": "peak_urinary_flow_rate", "value": "18", "unit": "mL/s", "recorded_on": "2026-06-01"}
            ],
        },
    )

    calls = agent._client.chat.completions.calls
    system_prompt = calls[0]["messages"][0]["content"]
    assert "qmax" in system_prompt.lower()
    assert "urology" in system_prompt.lower()

    # The same disambiguated context must also reach the evidence-mismatch filter,
    # not just the synthesis prompt.
    assert "qmax" in agent._extraction_context["patient_summary"].lower()
