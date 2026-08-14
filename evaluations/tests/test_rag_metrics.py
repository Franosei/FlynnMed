import json
from types import SimpleNamespace

from evaluations.config import EvalConfig
from evaluations.models import (
    ClaimAssessment,
    ClaimSourceEvidence,
    ConversationTurn,
    EvalCase,
    PipelineResponse,
    RubricItem,
)
from evaluations import rag_metrics


def _case(ideal_completion="Gold answer"):
    return EvalCase(
        case_id="case-1",
        source_dataset="healthbench",
        conversation=[ConversationTurn(role="user", content="What should I do?")],
        rubrics=[RubricItem(criterion="Helpful", points=1)],
        ideal_completion=ideal_completion,
    )


def _response():
    return PipelineResponse(
        case_id="case-1",
        answer_markdown="Use the relevant advice [S1].",
        answer_text="Use the relevant advice.",
        sources=[
            {
                "source_id": "S1",
                "title": "Direct guidance",
                "snippet": "Relevant advice",
            },
            {
                "source_id": "S2",
                "title": "Distractor",
                "snippet": "Unrelated material",
            },
        ],
        consistency_answers=["Use the relevant advice."],
    )


def _metric_payload():
    return {
        name: {
            "score": 0.8,
            "applicable": True,
            "explanation": "graded",
            "findings": [],
        }
        for name in rag_metrics._METRIC_NAMES
    }


def _claim_payload(answer="Use the relevant advice [S1].", cited=True):
    return {
        "claims": [
            {
                "claim_id": "C1",
                "claim": "Use the relevant advice.",
                "answer_quote": answer,
                "material": True,
                "kind": "recommendation",
                "requires_evidence": True,
                "supported_by_conversation": False,
                "conversation_evidence": "",
                "citation_ids": ["S1"] if cited else [],
                "source_evidence": [
                    {
                        "source_id": "S1",
                        "support_score": 0.95,
                        "entails": True,
                        "source_quote": "Relevant advice",
                        "explanation": "Direct support.",
                    }
                ]
                if cited
                else [],
                "explanation": "Material recommendation.",
            }
        ]
    }


def test_json_completion_falls_back_and_records_actual_model(monkeypatch):
    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "gpt-5.6-luna":
            error = Exception("luna unavailable")
            error.status_code = 401
            raise error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps({"ok": True}))
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )
    monkeypatch.setattr(rag_metrics, "OpenAI", lambda **kwargs: client)
    monkeypatch.setattr(rag_metrics, "call_with_retry", lambda fn, max_retries=5: fn())

    payload = rag_metrics._json_completion(
        "test",
        EvalConfig(
            rag_metrics_model="gpt-5.6-luna",
            evaluator_fallback_model="gpt-5.4-mini",
        ),
    )

    assert payload == {"ok": True}
    assert payload.model == "gpt-5.4-mini"
    assert [call["model"] for call in calls] == [
        "gpt-5.6-luna",
        "gpt-5.4-mini",
    ]


def test_three_stage_metrics_persist_split_and_validated_claims(monkeypatch):
    calls = []

    def _fake_completion(prompt, config):
        calls.append(prompt)
        if len(calls) == 1:
            return {
                "documents": [
                    {
                        "source_id": "S1",
                        "rank": 1,
                        "relevance_score": 0.95,
                        "relevant": False,
                        "explanation": "direct",
                    },
                    {
                        "source_id": "S2",
                        "rank": 2,
                        "relevance_score": 0.1,
                        "relevant": True,
                        "explanation": "noise",
                    },
                ]
            }
        if len(calls) == 2:
            return _claim_payload()
        return _metric_payload()

    monkeypatch.setattr(rag_metrics, "_json_completion", _fake_completion)

    result = rag_metrics.grade_rag_metrics(_case(), _response(), EvalConfig())

    assert result.relevant_source_ids == ["S1"]
    assert result.irrelevant_source_ids == ["S2"]
    assert result.context_relevance.score == 0.5
    assert result.context_precision_ranking.score == 1.0
    assert result.answer_correctness.applicable is True
    assert result.consistency.applicable is True
    assert result.citation_accuracy.score == 0.95
    assert result.citation_completeness.score == 1.0
    assert result.claim_assessments[0].answer_quote_validated is True
    assert "Source records" in calls[1]
    assert "Validated claim audit" in calls[2]


def test_metrics_mark_missing_inputs_not_applicable(monkeypatch):
    response = PipelineResponse(
        case_id="case-1",
        answer_markdown="General answer.",
        answer_text="General answer.",
    )
    calls = []

    def _fake_completion(prompt, config):
        calls.append(prompt)
        if len(calls) == 1:
            return _claim_payload(answer="General answer.", cited=False)
        return _metric_payload()

    monkeypatch.setattr(rag_metrics, "_json_completion", _fake_completion)

    result = rag_metrics.grade_rag_metrics(
        _case(ideal_completion=None), response, EvalConfig()
    )

    assert result.faithfulness.applicable is True
    assert result.faithfulness.score == 0.0
    assert result.context_recall.applicable is False
    assert result.noise_sensitivity.applicable is False
    assert result.answer_correctness.applicable is False
    assert result.citation_accuracy.applicable is False
    assert result.consistency.applicable is False


def test_citation_accuracy_does_not_penalize_a_different_uncited_claim():
    records = rag_metrics._source_records(_response().sources)
    claims = [
        ClaimAssessment(
            claim_id="C1",
            claim="Relevant advice is useful.",
            answer_quote="Relevant advice [S1]",
            answer_quote_validated=True,
            citation_ids=["S1"],
            source_evidence=[
                ClaimSourceEvidence(
                    source_id="S1",
                    support_score=1.0,
                    entails=True,
                    source_quote="Relevant advice",
                    quote_validated=True,
                )
            ],
        ),
        ClaimAssessment(
            claim_id="C2",
            claim="A separate clinical claim.",
            answer_quote="A separate clinical claim.",
            answer_quote_validated=True,
        ),
    ]

    pairs, accuracy, completeness = rag_metrics._citation_metrics(
        claims, "Relevant advice [S1]. A separate clinical claim.", records
    )

    assert len(pairs) == 1
    assert accuracy.score == 1.0
    assert accuracy.sample_size == 1
    assert completeness.score == 0.5
    assert completeness.sample_size == 2


def test_citation_accuracy_rejects_marker_linked_to_wrong_url():
    records = rag_metrics._source_records(_response().sources)
    claim = ClaimAssessment(
        claim_id="C1",
        claim="Relevant advice is useful.",
        answer_quote="Relevant advice [S1]",
        citation_ids=["S1"],
        source_evidence=[
            ClaimSourceEvidence(
                source_id="S1",
                support_score=1.0,
                entails=True,
                source_quote="Relevant advice",
                quote_validated=True,
            )
        ],
    )

    pairs, accuracy, _ = rag_metrics._citation_metrics(
        [claim], "Relevant advice [S1](https://wrong.example/article).", records
    )

    assert pairs[0].target_matches_source is False
    assert accuracy.score == 0.0


def test_claim_validation_rejects_uncovered_answer_and_marker_mismatch():
    case = _case()
    records = rag_metrics._source_records(_response().sources)
    claims = [
        ClaimAssessment(
            claim_id="C1",
            claim="Use the relevant advice.",
            answer_quote="Use the relevant advice [S1].",
            citation_ids=[],
        )
    ]

    _, errors, warnings = rag_metrics._validate_claims(
        claims,
        case,
        "Use the relevant advice [S1]. Monitor for worsening symptoms today.",
        records,
    )

    assert errors == []
    assert any("citation ids" in warning for warning in warnings)
    assert any("uncovered answer unit" in warning for warning in warnings)


def test_claim_validation_trusts_declared_citation_outside_the_narrow_quote():
    """
    Regression test: a citation marker almost always sits at the end of a
    sentence, after the claim content the judge quotes -- e.g. the answer is
    "Fatigue may be complex and may not have a single definitive explanation
    [S3][S4]." and the judge's answer_quote for the claim is just "Fatigue
    may be complex and may not have a single definitive explanation" (no
    trailing markers). The judge's own declared citation_ids=["S3","S4"]
    (it read the whole answer) must not be discarded just because those
    markers fall outside the narrow quoted substring -- found via a real
    50-case evaluation run where this drove citation_accuracy/completeness to
    ~0 across nearly every case regardless of whether citations were correct.
    """
    case = _case()
    answer = (
        "Fatigue may be complex and may not have a single definitive "
        "explanation [S3][S4]."
    )
    claims = [
        ClaimAssessment(
            claim_id="C1",
            claim="Fatigue may be complex and may not have a single definitive explanation.",
            answer_quote="Fatigue may be complex and may not have a single definitive explanation",
            citation_ids=["S3", "S4"],
        )
    ]

    validated, errors, warnings = rag_metrics._validate_claims(
        claims, case, answer, []
    )

    assert errors == []
    assert validated[0].citation_ids == ["S3", "S4"]


def test_claim_validation_drops_a_hallucinated_citation_id():
    """
    Companion test: a declared citation_id that doesn't correspond to any
    marker actually rendered anywhere in the answer must still be dropped --
    the fix above must not become "trust the judge unconditionally."
    """
    case = _case()
    answer = "Fatigue may have several causes [S3]."
    claims = [
        ClaimAssessment(
            claim_id="C1",
            claim="Fatigue may have several causes.",
            answer_quote="Fatigue may have several causes",
            citation_ids=["S3", "S9"],
        )
    ]

    validated, errors, warnings = rag_metrics._validate_claims(
        claims, case, answer, []
    )

    assert errors == []
    assert validated[0].citation_ids == ["S3"]
    assert any("S9" in warning and "dropped" in warning for warning in warnings)


def test_claim_validation_canonicalizes_markdown_without_inventing_text():
    claims = [
        ClaimAssessment(
            claim_id="C1",
            claim="Mirtazapine may be considered.",
            answer_quote="Mirtazapine is often considered a good option.",
        )
    ]
    answer = "For this situation, **mirtazapine** is often considered a good option."

    validated, errors, warnings = rag_metrics._validate_claims(
        claims, _case(), answer, []
    )

    assert errors == []
    assert validated[0].answer_quote in answer
    assert validated[0].answer_quote_validated is True
    assert any("canonicalized" in warning for warning in warnings)


def test_private_gold_answer_overrides_dataset_ideal(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "answer": "Clinician reviewed answer",
                "reviewer": "clinical-panel-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rag_metrics._load_gold_answers.cache_clear()

    answer, provenance = rag_metrics.resolve_gold_answer(
        _case(), EvalConfig(gold_answers_path=path)
    )

    assert answer == "Clinician reviewed answer"
    assert provenance == "clinical-panel-v1"


def test_ranking_penalizes_relevant_document_below_noise():
    assessments = [
        rag_metrics.DocumentRelevanceAssessment(
            source_id="S1", rank=1, relevance_score=0.1, relevant=False
        ),
        rag_metrics.DocumentRelevanceAssessment(
            source_id="S2", rank=2, relevance_score=0.9, relevant=True
        ),
    ]

    score = rag_metrics._ranking_score(assessments)

    assert score.applicable is True
    assert 0.0 < score.score < 1.0
