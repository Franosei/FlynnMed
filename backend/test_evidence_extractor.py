import json
from types import SimpleNamespace

import backend.evidence_extractor as evidence_extractor
from backend.evidence_extractor import _extract_one_article, build_evidence_dossier
from backend.evidence_schema import ArticleEvidence


class _FakeCompletions:
    def __init__(self, payload: dict):
        self.calls = []
        self._payload = payload

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(self._payload)))
            ]
        )


class _FakeLLM:
    def __init__(self, payload: dict):
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(payload)))


def test_extract_one_article_parses_specialty_mismatch_from_llm_response():
    llm = _FakeLLM(
        {
            "answers_question": False,
            "patient_aligned_facts": [],
            "alignment_confidence": 0.0,
            "specialty_mismatch": True,
            "specialty_mismatch_reason": "Discusses respiratory peak flow, not the patient's urology reading.",
            "patient_relevant_summary": "Different meaning -- does not apply.",
        }
    )
    source = {"snippet": "Peak flow guidance for asthma patients.", "title": "Respiratory Peak Flow", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="What does my peak flow of 18 mean?",
        patient_summary="Recent vitals: Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement): 18 ml/s",
        medications=[],
        conditions=[],
    )

    assert result.specialty_mismatch is True
    assert "respiratory" in result.specialty_mismatch_reason.lower()


def test_extract_one_article_prompt_instructs_specialty_mismatch_detection():
    llm = _FakeLLM(
        {
            "answers_question": False,
            "patient_aligned_facts": [],
            "alignment_confidence": 0.0,
            "patient_relevant_summary": "Different meaning -- does not apply.",
        }
    )
    source = {"snippet": "Peak flow guidance for asthma patients.", "title": "Respiratory Peak Flow", "source_id": "S1"}

    _extract_one_article(
        llm=llm,
        source=source,
        question="What does my peak flow of 18 mean?",
        patient_summary="Recent vitals: Peak urinary flow rate / Qmax (urology, NOT a respiratory measurement): 18 ml/s",
        medications=[],
        conditions=[],
    )

    sent_prompt = llm.client.chat.completions.calls[0]["messages"][0]["content"]
    assert "SPECIALTY/MEANING MISMATCH" in sent_prompt
    assert "different clinical meaning" in sent_prompt.lower()


def test_extract_one_article_prefers_detail_snippet_over_shorter_snippet():
    """Evidence Ledger v2 (#3): detail_snippet (the richer fetched excerpt)
    must be used over the shorter search-result snippet when both are
    present -- previously `or` precedence picked snippet first."""
    llm = _FakeLLM(
        {
            "answers_question": True,
            "patient_aligned_facts": [],
            "alignment_confidence": 0.5,
            "patient_relevant_summary": "x",
        }
    )
    source = {
        "snippet": "Short search-result blurb.",
        "detail_snippet": "The full fetched paragraph with much richer clinical detail than the blurb.",
        "title": "Guidance page",
        "source_id": "S1",
    }

    _extract_one_article(
        llm=llm, source=source, question="q", patient_summary="p", medications=[], conditions=[],
    )

    sent_prompt = llm.client.chat.completions.calls[0]["messages"][0]["content"]
    assert "richer clinical detail" in sent_prompt
    assert "Short search-result blurb." not in sent_prompt


def test_extract_one_article_keeps_a_verbatim_extracted_passage():
    snippet = "Flucloxacillin can potentiate warfarin's anticoagulant effect in some patients."
    llm = _FakeLLM(
        {
            "answers_question": True,
            "question_facts": ["Flucloxacillin can affect warfarin."],
            "patient_aligned_facts": [],
            "extracted_passages": [
                "Flucloxacillin can potentiate warfarin's anticoagulant effect"
            ],
            "alignment_confidence": 0.9,
        }
    )
    source = {"snippet": snippet, "title": "Flucloxacillin guidance", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="Can I take flucloxacillin with warfarin?",
        patient_summary="On warfarin.",
        medications=["Warfarin"],
        conditions=[],
    )

    assert result.extracted_passages == [
        "Flucloxacillin can potentiate warfarin's anticoagulant effect"
    ]


def test_extract_one_article_recovers_a_quote_with_minor_formatting_differences():
    """
    Canonicalization: a claimed passage that differs from the source only by
    whitespace or a smart-vs-straight quote (formatting drift a model
    introduces while "copying") must still be accepted -- and recovered as
    the CLEAN source text, not the model's messy variant. This is what turns
    the strict verbatim gate from "usually finds nothing" into "reliably
    finds the real passage", without ever accepting a genuine paraphrase
    (see the companion drop test below).
    """
    snippet = "Flucloxacillin can potentiate warfarin's anticoagulant effect in some patients."
    llm = _FakeLLM(
        {
            "answers_question": True,
            "question_facts": ["Flucloxacillin can affect warfarin."],
            "patient_aligned_facts": [],
            "extracted_passages": [
                "Flucloxacillin can potentiate warfarin’s anticoagulant effect"
            ],
            "alignment_confidence": 0.9,
        }
    )
    source = {"snippet": snippet, "title": "Flucloxacillin guidance", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="Can I take flucloxacillin with warfarin?",
        patient_summary="On warfarin.",
        medications=["Warfarin"],
        conditions=[],
    )

    assert len(result.extracted_passages) == 1
    # Recovered text is genuine source text (straight quote), not the
    # model's smart-quote variant.
    assert "'" in result.extracted_passages[0]
    assert "’" not in result.extracted_passages[0]


def test_extract_one_article_drops_a_non_verbatim_extracted_passage():
    """
    Regression/quality-gate test: a claimed passage that is not an exact
    substring of the article text must be dropped, not kept in paraphrased
    form -- excluded rather than passed through as unverified text.
    """
    snippet = "Flucloxacillin can potentiate warfarin's anticoagulant effect in some patients."
    llm = _FakeLLM(
        {
            "answers_question": True,
            "question_facts": ["Flucloxacillin can affect warfarin."],
            "patient_aligned_facts": [],
            # Paraphrased, NOT a verbatim substring of snippet.
            "extracted_passages": ["Flucloxacillin increases warfarin's effect"],
            "alignment_confidence": 0.9,
        }
    )
    source = {"snippet": snippet, "title": "Flucloxacillin guidance", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="Can I take flucloxacillin with warfarin?",
        patient_summary="On warfarin.",
        medications=["Warfarin"],
        conditions=[],
    )

    assert result.extracted_passages == []


def test_extract_one_article_keeps_a_verified_structured_claim():
    snippet = (
        "In a randomised controlled trial, Drug X reduced relapse rate compared to "
        "placebo in adults with condition Y over 12 months."
    )
    llm = _FakeLLM(
        {
            "answers_question": True,
            "question_facts": ["Drug X reduced relapse rate vs placebo."],
            "patient_aligned_facts": [],
            "structured_claims": [
                {
                    "claim_text": "Drug X reduced relapse rate compared to placebo.",
                    "population": "Adults with condition Y",
                    "intervention": "Drug X",
                    "comparator": "Placebo",
                    "outcome": "Relapse rate at 12 months",
                    "study_design": "rct",
                    "certainty": "moderate",
                    "exact_quote": "Drug X reduced relapse rate compared to placebo",
                }
            ],
            "alignment_confidence": 0.8,
        }
    )
    source = {"snippet": snippet, "title": "Drug X trial", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="How effective is Drug X compared to placebo?",
        patient_summary="Adult with condition Y.",
        medications=[],
        conditions=["Condition Y"],
    )

    assert len(result.structured_claims) == 1
    claim = result.structured_claims[0]
    assert claim.study_design == "rct"
    assert claim.certainty == "moderate"
    assert claim.exact_quote in snippet


def test_extract_one_article_coerces_unrecognised_study_design_and_certainty():
    snippet = "Drug X reduced relapse rate compared to placebo in adults with condition Y."
    llm = _FakeLLM(
        {
            "answers_question": True,
            "structured_claims": [
                {
                    "claim_text": "Drug X reduced relapse rate compared to placebo.",
                    "study_design": "some_made_up_design",
                    "certainty": "super_duper_high",
                    "exact_quote": "Drug X reduced relapse rate compared to placebo",
                }
            ],
            "alignment_confidence": 0.8,
        }
    )
    source = {"snippet": snippet, "title": "Drug X trial", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="How effective is Drug X compared to placebo?",
        patient_summary="Adult with condition Y.",
        medications=[],
        conditions=[],
    )

    assert len(result.structured_claims) == 1
    assert result.structured_claims[0].study_design == "unknown"
    assert result.structured_claims[0].certainty == "unknown"


def test_extract_one_article_drops_structured_claim_with_unverifiable_quote():
    """A structured claim whose exact_quote can't be matched back to the
    article text must be dropped entirely, not kept with an unverified
    quote -- same discipline as extracted_passages."""
    snippet = "Drug X reduced relapse rate compared to placebo in adults with condition Y."
    llm = _FakeLLM(
        {
            "answers_question": True,
            "structured_claims": [
                {
                    "claim_text": "Drug X cured the condition entirely.",
                    "study_design": "rct",
                    "certainty": "high",
                    # Not present anywhere in the source snippet.
                    "exact_quote": "Drug X cured every single patient completely",
                }
            ],
            "alignment_confidence": 0.8,
        }
    )
    source = {"snippet": snippet, "title": "Drug X trial", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="How effective is Drug X compared to placebo?",
        patient_summary="Adult with condition Y.",
        medications=[],
        conditions=[],
    )

    assert result.structured_claims == []


def test_extract_one_article_empty_structured_claims_for_non_comparative_source():
    """A source with no structured_claims in the LLM response (e.g. plain
    dosing instructions with no comparator) must produce an empty list, not
    a forced/invented claim."""
    snippet = "Take one tablet of flucloxacillin four times a day with water."
    llm = _FakeLLM(
        {
            "answers_question": True,
            "question_facts": ["Take one tablet four times a day."],
            "alignment_confidence": 0.7,
        }
    )
    source = {"snippet": snippet, "title": "Flucloxacillin dosing", "source_id": "S1"}

    result = _extract_one_article(
        llm=llm,
        source=source,
        question="How do I take flucloxacillin?",
        patient_summary="On flucloxacillin.",
        medications=["Flucloxacillin"],
        conditions=[],
    )

    assert result.structured_claims == []


def test_build_evidence_dossier_excludes_confirmed_mismatched_sources(monkeypatch):
    mismatched = ArticleEvidence(
        source_id="S1",
        title="Respiratory peak flow guidance",
        evidence_tier=1,
        tier_label="Tier 1",
        answers_question=False,
        alignment_confidence=0.0,
        patient_relevant_summary="This concerns a different measurement and does not apply.",
    )
    matched = ArticleEvidence(
        source_id="S2",
        title="Uroflowmetry guidance",
        evidence_tier=1,
        tier_label="Tier 1",
        answers_question=True,
        alignment_confidence=0.6,
        patient_relevant_summary="Directly relevant to the patient's urology reading.",
    )
    canned = {"S1": mismatched, "S2": matched}

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        return canned[source["source_id"]]

    monkeypatch.setattr(evidence_extractor, "_extract_one_article", fake_extract)

    dossier = build_evidence_dossier(
        llm=object(),
        sources=[{"source_id": "S1", "title": "x"}, {"source_id": "S2", "title": "y"}],
        question="What does my peak flow of 18 mean?",
        user_profile={},
    )

    assert [a.source_id for a in dossier.articles] == ["S2"]
    assert dossier.excluded_source_ids == ["S1"]
    assert "excluded" in dossier.extraction_notes.lower()
    assert "different" in dossier.extraction_notes.lower()


def test_build_evidence_dossier_excludes_explicit_specialty_mismatch_regardless_of_confidence(monkeypatch):
    """
    A source the extractor explicitly flags specialty_mismatch=True must be excluded even
    if it scored a middling alignment_confidence and answers_question=True -- the explicit
    flag is a hard signal, not something inferred from the confidence threshold alone.

    This only holds when there's an actual patient profile on record for the article to
    conflict with (a concrete urology reading here) -- see the companion test below for the
    no-profile case, where specialty_mismatch must instead be ignored.
    """
    mismatched_but_confident = ArticleEvidence(
        source_id="S1",
        title="Respiratory peak flow guidance",
        evidence_tier=1,
        tier_label="Tier 1",
        answers_question=True,
        alignment_confidence=0.55,
        specialty_mismatch=True,
        specialty_mismatch_reason="Discusses respiratory peak flow, not this patient's urology reading.",
        patient_relevant_summary="Concerns a different measurement.",
    )

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        return mismatched_but_confident

    monkeypatch.setattr(evidence_extractor, "_extract_one_article", fake_extract)

    dossier = build_evidence_dossier(
        llm=object(),
        sources=[{"source_id": "S1", "title": "x"}],
        question="What does my peak flow of 18 mean?",
        user_profile={"date_of_birth": "1980-01-01", "biological_sex": "male"},
    )

    assert dossier.articles == []
    assert dossier.excluded_source_ids == ["S1"]


def test_build_evidence_dossier_ignores_specialty_mismatch_with_no_patient_profile_on_record(monkeypatch):
    """
    specialty_mismatch means "conflicts with a concrete fact in the patient's own profile" --
    with no profile recorded (the common case for a professional evidence-review question with
    no specific patient in view, e.g. a clinician asking about guidelines in the abstract),
    there is nothing for an article to conflict with. Honoring the flag anyway was a real bug:
    an aux-model misfire on this signal silently zeroed out every retrieved source, which then
    also disabled the downstream claim-alignment gate (guarded on combined_sources being
    non-empty). The flag must be ignored -- not treated as a hard exclusion -- whenever there's
    no profile on record, regardless of what the extractor returned.
    """
    mismatched_with_no_profile = ArticleEvidence(
        source_id="S1",
        title="2023 ACLS Guidelines for Adult Cardiac Arrest",
        evidence_tier=1,
        tier_label="Tier 1",
        answers_question=True,
        alignment_confidence=0.55,
        specialty_mismatch=True,
        specialty_mismatch_reason="Extractor misfire -- no actual conflicting profile fact exists.",
        patient_relevant_summary="On-topic guideline content.",
    )

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        return mismatched_with_no_profile

    monkeypatch.setattr(evidence_extractor, "_extract_one_article", fake_extract)

    dossier = build_evidence_dossier(
        llm=object(),
        sources=[{"source_id": "S1", "title": "x"}],
        question="Walk me through the current ACLS guidelines for adult cardiac arrest.",
        user_profile={},
    )

    assert [a.source_id for a in dossier.articles] == ["S1"]
    assert dossier.excluded_source_ids == []


def test_build_evidence_dossier_keeps_low_but_nonzero_general_context(monkeypatch):
    general_background = ArticleEvidence(
        source_id="S1",
        title="General wellbeing article",
        evidence_tier=3,
        tier_label="Tier 3",
        answers_question=False,
        alignment_confidence=0.2,
        patient_relevant_summary="Broad background only.",
    )

    def fake_extract(llm, source, question, patient_summary, medications, conditions):
        return general_background

    monkeypatch.setattr(evidence_extractor, "_extract_one_article", fake_extract)

    dossier = build_evidence_dossier(
        llm=object(),
        sources=[{"source_id": "S1", "title": "x"}],
        question="What does my peak flow of 18 mean?",
        user_profile={},
    )

    assert [a.source_id for a in dossier.articles] == ["S1"]
    assert "general context" in dossier.extraction_notes.lower()
