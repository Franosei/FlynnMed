import json

from evaluations.config import EvalConfig
from evaluations.models import (
    AdjudicationDecision,
    CaseResult,
    ConversationTurn,
    DeterministicFindings,
    EvalCase,
    GradingResult,
    MetricScore,
    PipelineResponse,
    RAGMetricsResult,
)
from evaluations.reporting import (
    REPORT_LABEL,
    _sanitized_case_entry,
    build_report_summary,
    write_report,
)

_UNIQUE_CONVERSATION_MARKER = (
    "a very specific patient sentence that must never leak into the sanitised summary"
)


def _rag_metrics(score=0.8, error=None):
    applicable = MetricScore(score=score, applicable=True, explanation="ok")
    unavailable = MetricScore(score=None, applicable=False, explanation="n/a")
    return RAGMetricsResult(
        case_id="case-1",
        judge_model="judge",
        relevant_source_ids=["S1"],
        irrelevant_source_ids=["S2"],
        faithfulness=applicable,
        context_relevance=applicable,
        noise_sensitivity=applicable,
        context_recall=applicable,
        answer_correctness=unavailable,
        calibration=applicable,
        contradiction_handling=unavailable,
        citation_accuracy=applicable,
        context_precision_ranking=applicable,
        clinical_harmlessness=applicable,
        consistency=unavailable,
        evaluation_error=error,
    )


def _case_result(case_id="case-1", rag_metrics=None, tags=None):
    case = EvalCase(
        case_id=case_id,
        source_dataset="healthbench",
        conversation=[
            ConversationTurn(role="user", content=_UNIQUE_CONVERSATION_MARKER)
        ],
        rubrics=[],
        tags=tags if tags is not None else ["theme:test"],
    )
    return CaseResult(
        case=case,
        pipeline_response=PipelineResponse(
            case_id=case_id,
            answer_markdown=_UNIQUE_CONVERSATION_MARKER + " answer",
            answer_text=_UNIQUE_CONVERSATION_MARKER + " answer",
            trace={"risk_level": "routine"},
        ),
        rag_metrics=rag_metrics,
    )


def _with_healthbench(result: CaseResult) -> CaseResult:
    grade = GradingResult(
        case_id=result.case.case_id,
        grader_model="luna",
        rubric_results=[],
        clinical_correctness_score=0.8,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="Rubric-bound grade.",
        expected_urgency_level="routine",
    )
    result.adjudication = AdjudicationDecision(
        case_id=result.case.case_id,
        triggered=False,
        luna_grade=grade,
        final_grade=grade,
    )
    result.deterministic = DeterministicFindings(
        case_id=result.case.case_id,
        expected_urgency_level="routine",
        actual_urgency_level="routine",
        under_triage=False,
        severe_under_triage=False,
        over_triage=False,
        expected_actual_match=True,
        crisis_gate_expected=False,
        crisis_gate_activated=False,
        medication_or_allergy_fabrication=False,
        patient_record_fabrication=False,
        cross_patient_leakage=False,
        citations_present=None,
        citations_supported=None,
        safe_refusal_violated=False,
        clarification_expected=False,
        clarification_observed=False,
        prompt_injection_detected=False,
        prompt_injection_resisted=None,
        deterministic_pass=True,
    )
    result.weighted_score = 0.8
    result.overall_pass = True
    return result


def test_report_supports_rag_metrics_when_healthbench_grade_is_missing():
    result = _case_result(rag_metrics=_rag_metrics())

    summary = build_report_summary(
        [result], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.total_cases == 1
    assert summary.rag_metric_aggregates["faithfulness"].average_score == 0.8
    assert summary.rag_metric_aggregates["faithfulness"].applicable_cases == 1
    assert summary.rag_metric_aggregates["answer_correctness"].average_score is None
    assert summary.relevant_document_count == 1
    assert summary.irrelevant_document_count == 1
    assert summary.healthbench_graded_cases == 0
    assert summary.pass_rate is None
    assert summary.weighted_healthbench_score is None
    assert summary.adjudication_rate is None


def test_sanitized_entry_distinguishes_triggered_from_completed_adjudication():
    result = _with_healthbench(_case_result())
    result.adjudication.triggered = True
    result.adjudication.adjudication_skipped = True

    entry = _sanitized_case_entry(result)

    assert entry["healthbench"]["adjudication_triggered"] is True
    assert entry["healthbench"]["adjudication_completed"] is False
    assert entry["healthbench"]["adjudication_skipped"] is True


def test_sanitized_entry_marks_adjudication_completed_when_terra_grade_present():
    result = _with_healthbench(_case_result())
    terra_grade = result.adjudication.luna_grade.model_copy(
        update={"grader_model": "terra"}
    )
    result.adjudication.triggered = True
    result.adjudication.adjudication_skipped = False
    result.adjudication.terra_grade = terra_grade
    result.adjudication.final_grade = terra_grade

    entry = _sanitized_case_entry(result)

    assert entry["healthbench"]["adjudication_triggered"] is True
    assert entry["healthbench"]["adjudication_completed"] is True


def test_build_report_summary_breaks_healthbench_scoring_down_by_tag():
    case_1 = _with_healthbench(
        _case_result(case_id="case-1", tags=["theme:communication"])
    )
    case_1.weighted_score = 1.0
    case_1.overall_pass = True

    case_2 = _with_healthbench(
        _case_result(case_id="case-2", tags=["theme:communication", "theme:hedging"])
    )
    case_2.weighted_score = 0.0
    case_2.overall_pass = False
    case_2.deterministic.under_triage = True

    case_3 = _with_healthbench(
        _case_result(case_id="case-3", tags=["theme:emergency_referrals"])
    )
    case_3.weighted_score = 0.6
    case_3.overall_pass = True

    summary = build_report_summary(
        [case_1, case_2, case_3], EvalConfig(), dataset_version="healthbench"
    )

    assert set(summary.by_tag) == {
        "theme:communication",
        "theme:hedging",
        "theme:emergency_referrals",
    }

    communication = summary.by_tag["theme:communication"]
    assert communication.case_count == 2
    assert communication.healthbench_graded_cases == 2
    assert communication.pass_rate == 0.5
    assert communication.weighted_healthbench_score == 0.5
    assert communication.under_triage_rate == 0.5

    hedging = summary.by_tag["theme:hedging"]
    assert hedging.case_count == 1
    assert hedging.pass_rate == 0.0
    assert hedging.under_triage_rate == 1.0

    emergency = summary.by_tag["theme:emergency_referrals"]
    assert emergency.case_count == 1
    assert emergency.pass_rate == 1.0
    assert emergency.weighted_healthbench_score == 0.6
    assert emergency.under_triage_rate == 0.0


def test_build_report_summary_handles_empty_list():
    summary = build_report_summary([], EvalConfig(), dataset_version="healthbench")
    assert summary.total_cases == 0
    assert summary.notes


def test_report_combines_healthbench_and_tier_metrics(tmp_path):
    result = _with_healthbench(_case_result(rag_metrics=_rag_metrics()))
    result.pipeline_response.resolved_role = "nurse"
    result.pipeline_response.role_resolution_reason = (
        "explicit clinical role in user text"
    )
    result.pipeline_response.role_resolution_confidence = 1.0
    result.pipeline_response.sources = [
        {"source_id": "S1", "url": "https://www.nhs.uk/conditions/example/"},
        {"source_id": "S2", "url": ""},
    ]
    result.deterministic.citation_count = 1
    result.deterministic.resolved_citation_count = 1
    summary = build_report_summary(
        [result], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.healthbench_graded_cases == 1
    assert summary.pass_rate == 1.0
    assert summary.weighted_healthbench_score == 0.8
    assert summary.rag_metric_aggregates["faithfulness"].average_score == 0.8
    assert summary.role_counts == {"nurse": 1}
    assert summary.related_link_url_coverage == 0.5
    assert summary.rendered_citation_resolution_rate == 1.0

    _, json_path, markdown_path = write_report(
        [result], EvalConfig(output_path=tmp_path), "healthbench", "combined-run"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["cases"][0]["healthbench"]["weighted_score"] == 0.8
    assert payload["cases"][0]["resolved_role"] == "nurse"
    assert payload["cases"][0]["role_resolution_confidence"] == 1.0
    assert payload["cases"][0]["healthbench"]["link_integrity"] == {
        "related_links": 2,
        "valid_related_links": 1,
        "rendered_citations": 1,
        "resolved_rendered_citations": 1,
    }
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Weighted HealthBench score: 0.800" in markdown
    assert "Tiered RAG quality metrics" in markdown
    assert "PROVISIONAL - insufficient sample" in markdown
    assert "Related source URLs: 1/2 valid (50.0%)" in markdown
    assert "Rendered citation targets: 1/1 resolved" in markdown


def test_metric_aggregate_reports_item_denominator_and_sufficiency():
    result = _case_result(rag_metrics=_rag_metrics())
    result.rag_metrics.faithfulness.sample_size = 7

    summary = build_report_summary(
        [result],
        EvalConfig(minimum_reliable_sample_size=5),
        dataset_version="healthbench",
    )

    aggregate = summary.rag_metric_aggregates["faithfulness"]
    assert aggregate.assessment_count == 7
    assert aggregate.status == "sufficient"


def test_historical_report_can_preserve_original_prompt_version():
    summary = build_report_summary(
        [_case_result(rag_metrics=_rag_metrics())],
        EvalConfig(),
        dataset_version="healthbench",
        prompt_version="healthbench-rubric-v1",
        run_date="2026-07-13T06:30:27+00:00",
    )
    assert summary.prompt_version == "healthbench-rubric-v1"
    assert summary.run_date == "2026-07-13T06:30:27+00:00"


def test_write_report_creates_raw_and_sanitised_files_for_partial_historical_row(
    tmp_path,
):
    result = _case_result(rag_metrics=_rag_metrics())
    raw_path, summary_json_path, summary_md_path = write_report(
        [result],
        EvalConfig(output_path=tmp_path),
        dataset_version="healthbench",
        run_id="test-run",
    )

    assert _UNIQUE_CONVERSATION_MARKER in raw_path.read_text(encoding="utf-8")
    summary_content = summary_json_path.read_text(encoding="utf-8")
    assert _UNIQUE_CONVERSATION_MARKER not in summary_content
    payload = json.loads(summary_content)
    assert payload["label"] == REPORT_LABEL
    assert "healthbench" not in payload["cases"][0]

    markdown = summary_md_path.read_text(encoding="utf-8")
    assert "Tier 1" in markdown
    assert "Noise robustness" in markdown
    assert "Pass rate" not in markdown
    assert "No completed HealthBench rubric grades" in markdown


def test_metric_error_requires_human_review_and_is_excluded(tmp_path):
    result = _case_result(
        rag_metrics=_rag_metrics(score=0.8, error="temporary judge failure")
    )
    summary = build_report_summary(
        [result], EvalConfig(), dataset_version="healthbench"
    )
    assert result.requires_human_review() is True
    assert summary.rag_metrics_error_count == 1

    _, _, report_path = write_report(
        [result], EvalConfig(output_path=tmp_path), "healthbench", "error-run"
    )
    report = report_path.read_text(encoding="utf-8")
    assert "metric judge error" in report


def _moderation_blocked_result(case_id="case-mod", category="toxicity") -> CaseResult:
    result = _case_result(case_id=case_id)
    result.pipeline_response.trace = {
        "trace_id": "trace-mod",
        "moderation_category": category,
    }
    return result


def _insufficient_evidence_result(case_id="case-limited") -> CaseResult:
    result = _case_result(case_id=case_id)
    result.pipeline_response.trace = {"trace_id": "trace-limited"}
    return result


def test_moderation_block_is_counted_and_categorised():
    blocked = _moderation_blocked_result(category="toxicity")
    normal = _with_healthbench(_case_result(case_id="case-normal"))

    summary = build_report_summary(
        [blocked, normal], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.moderation_block_count == 1
    assert summary.moderation_block_rate == 0.5
    assert summary.moderation_block_categories == {"toxicity": 1}
    # A moderation block preempts HealthBench grading entirely, so it must
    # not be counted in the HealthBench denominator.
    assert summary.healthbench_graded_cases == 1


def test_insufficient_evidence_is_counted():
    limited = _insufficient_evidence_result()
    normal = _with_healthbench(_case_result(case_id="case-normal"))

    summary = build_report_summary(
        [limited, normal], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.insufficient_evidence_count == 1
    assert summary.insufficient_evidence_rate == 0.5
    assert summary.healthbench_graded_cases == 1


def test_moderation_and_insufficient_evidence_absent_when_no_cases_trip_them():
    summary = build_report_summary(
        [_with_healthbench(_case_result())], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.moderation_block_count == 0
    assert summary.moderation_block_rate == 0.0
    assert summary.moderation_block_categories == {}
    assert summary.insufficient_evidence_count == 0
    assert summary.insufficient_evidence_rate == 0.0


def test_sanitized_entry_flags_moderation_block_and_insufficient_evidence():
    blocked_entry = _sanitized_case_entry(
        _moderation_blocked_result(category="obscene")
    )
    assert blocked_entry["moderation_blocked"] is True
    assert blocked_entry["moderation_category"] == "obscene"
    assert blocked_entry["insufficient_evidence"] is False

    limited_entry = _sanitized_case_entry(_insufficient_evidence_result())
    assert limited_entry["moderation_blocked"] is False
    assert limited_entry["insufficient_evidence"] is True

    normal_entry = _sanitized_case_entry(_with_healthbench(_case_result()))
    assert normal_entry["moderation_blocked"] is False
    assert normal_entry["moderation_category"] is None
    assert normal_entry["insufficient_evidence"] is False


def test_response_time_is_aggregated_from_pipeline_response_duration():
    fast = _with_healthbench(_case_result(case_id="case-fast"))
    fast.pipeline_response.duration_seconds = 10.0
    mid = _with_healthbench(_case_result(case_id="case-mid"))
    mid.pipeline_response.duration_seconds = 20.0
    slow = _with_healthbench(_case_result(case_id="case-slow"))
    slow.pipeline_response.duration_seconds = 30.0

    summary = build_report_summary(
        [fast, mid, slow], EvalConfig(), dataset_version="healthbench"
    )

    assert summary.average_duration_seconds == 20.0
    assert summary.median_duration_seconds == 20.0
    assert summary.max_duration_seconds == 30.0


def test_response_time_is_none_when_no_case_recorded_a_duration():
    summary = build_report_summary(
        [_with_healthbench(_case_result())], EvalConfig(), dataset_version="healthbench"
    )
    assert summary.average_duration_seconds is None


def test_markdown_report_includes_short_circuit_section(tmp_path):
    blocked = _moderation_blocked_result(category="toxicity")
    limited = _insufficient_evidence_result()
    normal = _with_healthbench(_case_result(case_id="case-normal"))

    _, _, report_path = write_report(
        [blocked, limited, normal],
        EvalConfig(output_path=tmp_path),
        "healthbench",
        "short-circuit-run",
    )
    markdown = report_path.read_text(encoding="utf-8")
    assert "Pipeline short-circuit rates" in markdown
    assert "Moderation blocks: 1/3 (33.3%)" in markdown
    assert "`toxicity`=1" in markdown
    assert "Insufficient-evidence refusals: 1/3 (33.3%)" in markdown
