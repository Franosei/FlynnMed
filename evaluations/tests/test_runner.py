import argparse
import json
import random

import pytest

from evaluations import runner
from evaluations.config import EvalConfig
from evaluations.models import (
    AdjudicationDecision,
    CaseResult,
    ConversationTurn,
    DeterministicFindings,
    EvalCase,
    GradingResult,
    PipelineResponse,
    MetricScore,
    RAGMetricsResult,
    RubricItem,
    RubricResult,
)


def _valid_case(case_id="case-1"):
    return EvalCase(
        case_id=case_id,
        source_dataset="healthbench",
        conversation=[ConversationTurn(role="user", content="I have a headache.")],
        rubrics=[RubricItem(criterion="Advises rest.", points=5, tags=[])],
        tags=[],
    )


def _invalid_case(case_id="bad-case"):
    return EvalCase(
        case_id=case_id,
        source_dataset="healthbench",
        conversation=[
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello"),
        ],
        rubrics=[],
        tags=[],
    )


def test_dry_run_reports_validation_errors_without_calling_models(capsys):
    cases = [_valid_case("case-1"), _valid_case("case-2"), _invalid_case("bad-case")]

    runner.dry_run(cases, EvalConfig())

    captured = capsys.readouterr()
    assert "3 cases loaded, 1 failed validation" in captured.out
    assert "bad-case" in captured.err
    assert "no models were called" in captured.out


def _minimal_case_result(case_id: str) -> CaseResult:
    grade = GradingResult(
        case_id=case_id,
        grader_model="luna",
        rubric_results=[],
        clinical_correctness_score=0.9,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="fine",
        expected_urgency_level="routine",
    )
    findings = DeterministicFindings(
        case_id=case_id,
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
        failure_reasons=[],
    )
    metric = MetricScore(score=0.9, applicable=True, explanation="fine")
    unavailable = MetricScore(score=None, applicable=False, explanation="n/a")
    return CaseResult(
        case=_valid_case(case_id),
        pipeline_response=PipelineResponse(
            case_id=case_id,
            answer_markdown="ans",
            answer_text="ans",
            trace={"risk_level": "routine"},
        ),
        adjudication=AdjudicationDecision(
            case_id=case_id, triggered=False, luna_grade=grade, final_grade=grade
        ),
        deterministic=findings,
        weighted_score=1.0,
        overall_pass=True,
        rag_metrics=RAGMetricsResult(
            case_id=case_id,
            judge_model="judge",
            faithfulness=metric,
            context_relevance=metric,
            noise_sensitivity=unavailable,
            context_recall=metric,
            answer_correctness=metric,
            calibration=metric,
            contradiction_handling=unavailable,
            citation_accuracy=unavailable,
            context_precision_ranking=metric,
            clinical_harmlessness=metric,
            consistency=unavailable,
        ),
    )


def test_load_completed_case_ids_returns_empty_set_for_missing_file(tmp_path):
    assert runner._load_completed_case_ids(tmp_path / "does_not_exist.jsonl") == set()


def test_regrade_healthbench_preserves_answer_and_uses_checkpoint(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "raw" / "saved-run"
    source_dir.mkdir(parents=True)
    saved = _minimal_case_result("case-1")
    (source_dir / "cases.jsonl").write_text(
        saved.model_dump_json() + "\n", encoding="utf-8"
    )
    grade = GradingResult(
        case_id="case-1",
        grader_model="gpt-5.6-sol",
        rubric_results=[
            RubricResult(
                criterion="Advises rest.",
                points=5,
                met=True,
                explanation="The answer advises rest.",
                answer_evidence="ans",
            )
        ],
        clinical_correctness_score=0.9,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="Meets the rubric.",
        expected_urgency_level="routine",
    )
    calls = []
    monkeypatch.setattr(
        runner,
        "grade_with_primary",
        lambda *args: calls.append(args) or grade,
    )
    config = EvalConfig(
        output_path=tmp_path,
        primary_grader_model="gpt-5.6-sol",
        adjudicator_model="gpt-5.6-sol",
    )
    args = argparse.Namespace(run_id="saved-run")

    runner._regrade_saved_healthbench("healthbench", args, config)

    report = (
        tmp_path
        / "reports"
            / "saved-run_healthbench-rubric-v4_gpt_5_6_sol_summary.json"
    )
    assert report.exists()
    assert len(calls) == 1
    regraded = json.loads(report.read_text(encoding="utf-8"))
    assert regraded["summary"]["primary_grader_model"] == "gpt-5.6-sol"

    monkeypatch.setattr(
        runner,
        "grade_with_primary",
        lambda *args: pytest.fail("completed checkpoint must be reused"),
    )
    runner._regrade_saved_healthbench("healthbench", args, config)


def test_regrade_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as exc_info:
        runner.main(
            [
                "--dataset",
                "healthbench",
                "--run-id",
                "saved-run",
                "--regrade-healthbench",
                "--regrade-rag",
            ]
        )

    assert exc_info.value.code == 2


def test_load_completed_case_ids_reads_existing_results(tmp_path):
    raw_path = tmp_path / "cases.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(_minimal_case_result("case-1").model_dump_json() + "\n")
        fh.write(_minimal_case_result("case-2").model_dump_json() + "\n")

    assert runner._load_completed_case_ids(raw_path) == {"case-1", "case-2"}


def test_load_completed_case_ids_skips_malformed_lines(tmp_path):
    raw_path = tmp_path / "cases.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        fh.write(_minimal_case_result("case-1").model_dump_json() + "\n")
        fh.write("not valid json\n")

    assert runner._load_completed_case_ids(raw_path) == {"case-1"}


def test_load_completed_case_ids_ignores_legacy_results_without_new_metrics(tmp_path):
    raw_path = tmp_path / "cases.jsonl"
    payload = _minimal_case_result("legacy-case").model_dump()
    payload["rag_metrics"] = None
    raw_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert runner._load_completed_case_ids(raw_path) == set()


def test_load_completed_case_ids_ignores_rag_only_results_without_healthbench(tmp_path):
    raw_path = tmp_path / "cases.jsonl"
    payload = _minimal_case_result("rag-only-case").model_dump()
    payload["adjudication"] = None
    payload["deterministic"] = None
    payload["weighted_score"] = None
    payload["overall_pass"] = None
    raw_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert runner._load_completed_case_ids(raw_path) == set()


def test_case_manifest_reproduces_prior_report_order(tmp_path):
    manifest = tmp_path / "summary.json"
    manifest.write_text(
        json.dumps({"cases": [{"case_id": "case-2"}, {"case_id": "case-1"}]}),
        encoding="utf-8",
    )

    selected = runner._select_manifest_cases(
        [_valid_case("case-1"), _valid_case("case-2"), _valid_case("case-3")],
        manifest,
    )

    assert [case.case_id for case in selected] == ["case-2", "case-1"]


def test_case_manifest_can_select_only_locked_partition(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "cases": [
                    {"case_id": "case-1", "partition": "development"},
                    {"case_id": "case-2", "partition": "locked_test"},
                    {"case_id": "case-3", "partition": "locked_test"},
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = runner._select_manifest_cases(
        [_valid_case("case-1"), _valid_case("case-2"), _valid_case("case-3")],
        manifest,
        partition="locked_test",
    )

    assert [case.case_id for case in selected] == ["case-2", "case-3"]


def test_case_manifest_rejects_missing_case(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(["case-404"]), encoding="utf-8")

    with pytest.raises(ValueError, match="absent from this dataset"):
        runner._select_manifest_cases([_valid_case("case-1")], manifest)


def test_run_dataset_applies_sample_limit_and_resume(tmp_path, monkeypatch):
    all_cases = [_valid_case(f"case-{i}") for i in range(5)]
    monkeypatch.setattr(
        runner, "_prepare_cases", lambda dataset_name, force_download: all_cases
    )

    class _FakeRagEngine:
        pass

    monkeypatch.setattr(
        "evaluations.pipeline.build_rag_engine", lambda config: _FakeRagEngine()
    )

    evaluated_ids = []

    def _fake_evaluate_case(case, rag_engine, config):
        evaluated_ids.append(case.case_id)
        return _minimal_case_result(case.case_id)

    monkeypatch.setattr(runner, "evaluate_case", _fake_evaluate_case)

    config = EvalConfig(output_path=tmp_path, sample_limit=3)
    args = argparse.Namespace(
        dry_run=False,
        resume=False,
        run_id="fixed-run",
        force_download=False,
        batch=False,
    )

    runner.run_dataset("healthbench", args, config)

    # Only the first 3 (sample limit) were evaluated.
    assert evaluated_ids == ["case-0", "case-1", "case-2"]

    summary_path = tmp_path / "reports" / "fixed-run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["total_cases"] == 3

    # Now resume: same run id, sample bumped to 5 -- only the 2 new cases should run.
    evaluated_ids.clear()
    config2 = EvalConfig(output_path=tmp_path, sample_limit=5)
    args2 = argparse.Namespace(
        dry_run=False,
        resume=True,
        run_id="fixed-run",
        force_download=False,
        batch=False,
    )
    runner.run_dataset("healthbench", args2, config2)

    assert evaluated_ids == ["case-3", "case-4"]

    summary_path_2 = tmp_path / "reports" / "fixed-run_summary.json"
    summary2 = json.loads(summary_path_2.read_text(encoding="utf-8"))
    assert summary2["summary"]["total_cases"] == 5


def test_run_dataset_dry_run_never_builds_rag_engine(tmp_path, monkeypatch):
    all_cases = [_valid_case("case-1")]
    monkeypatch.setattr(
        runner, "_prepare_cases", lambda dataset_name, force_download: all_cases
    )

    def _should_not_be_called(config):
        raise AssertionError("build_rag_engine must not be called in --dry-run")

    monkeypatch.setattr("evaluations.pipeline.build_rag_engine", _should_not_be_called)

    config = EvalConfig(output_path=tmp_path)
    args = argparse.Namespace(
        dry_run=True, resume=False, run_id=None, force_download=False, batch=False
    )

    runner.run_dataset("healthbench", args, config)  # should not raise


def test_run_dataset_randomizes_before_sampling(tmp_path, monkeypatch):
    all_cases = [_valid_case(f"case-{i}") for i in range(8)]
    monkeypatch.setattr(
        runner, "_prepare_cases", lambda dataset_name, force_download: list(all_cases)
    )
    observed = []
    monkeypatch.setattr(
        runner,
        "dry_run",
        lambda cases, config: observed.extend(case.case_id for case in cases),
    )

    expected = list(all_cases)
    random.Random(20260713).shuffle(expected)

    config = EvalConfig(output_path=tmp_path, sample_limit=3)
    args = argparse.Namespace(
        dry_run=True,
        resume=False,
        run_id=None,
        force_download=False,
        batch=False,
        random_seed=20260713,
    )

    runner.run_dataset("healthbench", args, config)

    assert observed == [case.case_id for case in expected[:3]]


def test_run_dataset_uses_exact_manifest_instead_of_sample_limit(tmp_path, monkeypatch):
    all_cases = [_valid_case(f"case-{i}") for i in range(5)]
    monkeypatch.setattr(
        runner, "_prepare_cases", lambda dataset_name, force_download: all_cases
    )
    observed = []
    monkeypatch.setattr(
        runner,
        "dry_run",
        lambda cases, config: observed.extend(case.case_id for case in cases),
    )
    manifest = tmp_path / "summary.json"
    manifest.write_text(
        json.dumps({"cases": [{"case_id": "case-4"}, {"case_id": "case-1"}]}),
        encoding="utf-8",
    )
    config = EvalConfig(output_path=tmp_path, sample_limit=1)
    args = argparse.Namespace(
        dry_run=True,
        resume=False,
        run_id=None,
        force_download=False,
        random_seed=None,
        case_manifest=manifest,
    )

    runner.run_dataset("healthbench", args, config)

    assert observed == ["case-4", "case-1"]


def test_consistency_repeats_are_opt_in(monkeypatch):
    primary = PipelineResponse(
        case_id="case-1", answer_markdown="primary", answer_text="primary"
    )
    answers = iter(
        [
            PipelineResponse(
                case_id="case-1", answer_markdown="repeat 1", answer_text="repeat 1"
            ),
            PipelineResponse(
                case_id="case-1", answer_markdown="repeat 2", answer_text="repeat 2"
            ),
        ]
    )
    monkeypatch.setattr(runner, "run_case_pipeline", lambda *args: next(answers))

    result = runner._add_consistency_repeats(
        _valid_case(), primary, object(), EvalConfig(consistency_repeats=2)
    )

    assert result.consistency_answers == ["repeat 1", "repeat 2"]


def _grade(model, points, met, harm="none", correctness=0.9):
    case = _valid_case()
    return GradingResult(
        case_id=case.case_id,
        grader_model=model,
        rubric_results=[
            RubricResult(
                criterion="Advises rest.",
                points=points,
                met=met,
                answer_evidence="Rest",
            )
        ],
        clinical_correctness_score=correctness,
        triage_appropriateness="appropriate",
        potential_harm_level=harm,
        confidence=0.9,
        explanation="test grade",
        expected_urgency_level="routine",
    )


def test_more_conservative_grade_prefers_worse_harm_level_regardless_of_score():
    """
    Regression test: unconditionally trusting whichever grader ran second
    let a lenient grader silently overrule a stricter one that flagged real
    harm -- found via a real 50-case run where the adjudicator (a smaller,
    weaker model) never once agreed with the primary grader's only
    severe-harm finding, and its lenient grade became "final" every time.
    """
    case = _valid_case()
    strict = _grade("gpt-5.6-luna", points=5, met=False, harm="severe")
    lenient = _grade("gpt-4o-mini", points=5, met=True, harm="none")

    assert runner._more_conservative_grade(strict, lenient, case) is strict
    assert runner._more_conservative_grade(lenient, strict, case) is strict


def test_more_conservative_grade_prefers_lower_score_on_harm_tie():
    case = _valid_case()
    stricter = _grade("gpt-5.6-luna", points=5, met=False, harm="low")
    lenient = _grade("gpt-4o-mini", points=5, met=True, harm="low")

    assert runner._more_conservative_grade(stricter, lenient, case) is stricter
    assert runner._more_conservative_grade(lenient, stricter, case) is stricter


def test_finalize_healthbench_result_prefers_stricter_grade_on_successful_adjudication(
    monkeypatch,
):
    """
    Full integration test through finalize_healthbench_result: a successful
    adjudication call must not unconditionally replace the primary grade --
    the stricter of the two (here, the primary grader's) must win as
    final_grade, both graders must still be recorded for the audit trail.
    """
    case = _valid_case()
    response = PipelineResponse(
        case_id=case.case_id,
        answer_markdown="Rest.",
        answer_text="Rest.",
        trace={"risk_level": "routine", "crisis_detected": False},
    )
    strict_primary = _grade("gpt-5.6-luna", points=5, met=False, harm="moderate")
    lenient_adjudicator = _grade("gpt-4o-mini", points=5, met=True, harm="none")

    monkeypatch.setattr(
        runner, "should_adjudicate", lambda *args: (True, ["flagged_for_review"])
    )
    monkeypatch.setattr(
        runner, "grade_with_terra", lambda *args: lenient_adjudicator
    )

    config = EvalConfig(
        primary_grader_model="gpt-5.6-luna", adjudicator_model="gpt-4o-mini"
    )
    result = runner.finalize_healthbench_result(case, response, strict_primary, config)

    assert result.adjudication.final_grade is strict_primary
    assert result.adjudication.terra_grade is lenient_adjudicator
    assert result.adjudication.luna_grade is strict_primary
    assert result.weighted_score == 0.0


def test_finalize_healthbench_score_uses_dataset_points_not_grader_score(monkeypatch):
    case = _valid_case()
    response = PipelineResponse(
        case_id=case.case_id,
        answer_markdown="Rest.",
        answer_text="Rest.",
        trace={"risk_level": "routine", "crisis_detected": False},
    )
    grade = GradingResult(
        case_id=case.case_id,
        grader_model="luna",
        rubric_results=[
            RubricResult(
                criterion="Advises rest.",
                points=5,
                met=True,
                explanation="The answer advises rest.",
                answer_evidence="Rest",
            )
        ],
        # Deliberately different: this field must not become the weighted score.
        clinical_correctness_score=0.1,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="Rubric evidence controls the HealthBench score.",
        expected_urgency_level="routine",
    )
    monkeypatch.setattr(runner, "should_adjudicate", lambda *args: (False, []))

    result = runner.finalize_healthbench_result(
        case,
        response,
        grade,
        EvalConfig(adjudicator_model="gpt-4o-mini"),
    )

    assert result.weighted_score == 1.0
    assert result.adjudication.final_grade is grade
    assert result.overall_pass is True


def test_failed_secondary_adjudication_preserves_primary_grade(monkeypatch):
    case = _valid_case()
    response = PipelineResponse(
        case_id=case.case_id,
        answer_markdown="Rest.",
        answer_text="Rest.",
        trace={"risk_level": "routine", "crisis_detected": False},
    )
    grade = GradingResult(
        case_id=case.case_id,
        grader_model="gpt-5.6-luna",
        rubric_results=[
            RubricResult(
                criterion="Advises rest.",
                points=5,
                met=True,
                answer_evidence="Rest",
            )
        ],
        clinical_correctness_score=0.9,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="Primary grade is valid.",
        expected_urgency_level="routine",
    )
    monkeypatch.setattr(
        runner, "should_adjudicate", lambda *args: (True, ["high_stakes_category"])
    )
    monkeypatch.setattr(
        runner,
        "grade_with_terra",
        lambda *args: (_ for _ in ()).throw(ValueError("invalid evidence")),
    )

    config = EvalConfig(
        primary_grader_model="gpt-5.6-luna",
        adjudicator_model="gpt-5.6-terra",
    )
    result = runner.finalize_healthbench_result(case, response, grade, config)

    assert result.weighted_score == 1.0
    assert result.adjudication.final_grade is grade
    assert result.adjudication.terra_grade is None
    assert "adjudicator_failed" in result.adjudication.trigger_reasons
    assert "invalid evidence" in result.adjudication.adjudication_error


def test_same_model_secondary_adjudication_is_skipped(monkeypatch):
    case = _valid_case()
    response = PipelineResponse(
        case_id=case.case_id,
        answer_markdown="Rest.",
        answer_text="Rest.",
        trace={"risk_level": "routine", "crisis_detected": False},
    )
    grade = GradingResult(
        case_id=case.case_id,
        grader_model="gpt-5.4-mini",
        rubric_results=[
            RubricResult(
                criterion="Advises rest.",
                points=5,
                met=True,
                answer_evidence="Rest",
            )
        ],
        clinical_correctness_score=0.9,
        triage_appropriateness="appropriate",
        potential_harm_level="none",
        confidence=0.9,
        explanation="Primary grade is valid.",
        expected_urgency_level="routine",
    )
    monkeypatch.setattr(
        runner, "should_adjudicate", lambda *args: (True, ["high_stakes_category"])
    )
    monkeypatch.setattr(
        runner,
        "grade_with_terra",
        lambda *args: pytest.fail("same model must not be called twice"),
    )

    config = EvalConfig(
        primary_grader_model="gpt-5.4-mini",
        adjudicator_model="gpt-5.4-mini",
    )
    result = runner.finalize_healthbench_result(case, response, grade, config)

    assert result.adjudication.triggered is True
    assert result.adjudication.adjudication_skipped is True
    assert result.adjudication.terra_grade is None
    assert "same_model_adjudication_skipped" in result.adjudication.trigger_reasons


def test_main_stops_before_dataset_when_evaluator_access_fails(monkeypatch, capsys):
    from evaluations import grading

    monkeypatch.setattr(runner, "load_config", lambda: EvalConfig())
    monkeypatch.setattr(
        grading,
        "validate_evaluator_access",
        lambda config, **kwargs: (_ for _ in ()).throw(
            grading.EvaluatorAccessError("HTTP 401; set EVAL_API_KEY")
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_dataset",
        lambda *args: pytest.fail("dataset must not start after failed preflight"),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.main(["--dataset", "healthbench", "--sample", "10"])

    assert exc_info.value.code == 2
    assert "HTTP 401" in capsys.readouterr().err


def test_main_reports_missing_evaluation_key_without_traceback(monkeypatch, capsys):
    from evaluations import grading

    monkeypatch.setattr(runner, "load_config", lambda: EvalConfig())
    monkeypatch.setattr(
        grading,
        "validate_evaluator_access",
        lambda config, **kwargs: (_ for _ in ()).throw(
            ValueError(
                "Set EVAL_API_KEY or OPENAI_API_KEY in the current shell or .env."
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_dataset",
        lambda *args: pytest.fail("dataset must not start without credentials"),
    )

    with pytest.raises(SystemExit) as exc_info:
        runner.main(["--dataset", "healthbench", "--sample", "10"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "evaluator access check FAILED" in captured.err
    assert "EVAL_API_KEY" in captured.err
    assert "Traceback" not in captured.err


def test_rejects_matching_primary_and_adjudicator_models():
    config = EvalConfig(
        primary_grader_model="gpt-5.4-mini", adjudicator_model="gpt-5.4-mini"
    )

    with pytest.raises(ValueError, match="independent second opinion"):
        runner.require_independent_adjudicator(config)


def test_accepts_distinct_primary_and_adjudicator_models():
    config = EvalConfig(
        primary_grader_model="gpt-5.4-mini", adjudicator_model="gpt-4o-mini"
    )

    runner.require_independent_adjudicator(config)


def test_unadjudicated_moderate_harm_cannot_pass(monkeypatch):
    case = _valid_case()
    response = PipelineResponse(
        case_id=case.case_id,
        answer_markdown="Rest.",
        answer_text="Rest.",
        trace={"risk_level": "routine", "crisis_detected": False},
    )
    grade = _grade("gpt-5.6-luna", points=5, met=True, harm="moderate")
    monkeypatch.setattr(
        runner,
        "grade_with_terra",
        lambda *args: (_ for _ in ()).throw(RuntimeError("second grader unavailable")),
    )

    result = runner.finalize_healthbench_result(
        case,
        response,
        grade,
        EvalConfig(
            primary_grader_model="gpt-5.6-luna",
            adjudicator_model="gpt-4o-mini",
            max_retries=1,
        ),
    )

    assert result.overall_pass is False
    assert "mandatory_harm_second_opinion_missing" in result.adjudication.trigger_reasons
