import json

import pytest

from evaluations.release_pipeline import (
    audit_results,
    build_blinded_review_package,
    build_clinical_review_package,
    build_scorecard,
    create_manifest,
    merge_evaluation_checkpoints,
    stage_failure_record,
    verify_manifest,
    summarize_clinical_reviews,
)


def _record(case_id="case-1", category="emergent", risk="routine", model="candidate"):
    return {
        "case": {
            "case_id": case_id,
            "source_dataset": "test",
            "conversation": [{"role": "user", "content": "test prompt"}],
            "rubrics": [],
            "tags": [f"physician_agreed_category:{category}"],
        },
        "pipeline_response": {
            "answer_markdown": "answer",
            "duration_seconds": 2.0,
            "resolved_role": "patient",
            "sources": [{"source_id": "S1"}],
            "trace": {
                "trace_id": "trace-1",
                "model": model,
                "risk_level": risk,
                "intent_category": "symptom_triage",
                "retrieval_mode": "agentic_multi_source",
                "task_mode": "clinical_answer",
                "claim_alignment": [],
            },
        },
        "adjudication": None,
        "deterministic": None,
        "rag_metrics": None,
    }


def test_manifest_seals_content_and_does_not_call_old_data_untouched(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    manifest = create_manifest(path, exposure="previously_evaluated")

    assert manifest["untouched_test_valid"] is False
    assert manifest["case_count"] == 1
    assert verify_manifest(path, manifest) == []
    changed = _record()
    changed["case"]["tags"].append("changed")
    path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    assert "source_file_hash_changed" in verify_manifest(path, manifest)
    assert "case_content_changed:case-1" in verify_manifest(path, manifest)


def test_stage_failure_taxonomy_identifies_emergency_miss_and_missing_evals():
    row = stage_failure_record(_record(), expected_generator_model="candidate")

    assert "physician_emergent_not_recognised" in row["failure_codes"]
    assert "severe_under_triage" in row["failure_codes"]
    assert "healthbench_grading_not_run" in row["failure_codes"]
    assert "rag_metrics_not_run" in row["failure_codes"]


def test_scorecard_fails_closed_when_metrics_or_untouched_test_are_missing():
    _, audit = audit_results([_record()])
    scorecard = build_scorecard(
        audit,
        {"untouched_test_valid": False},
        clinician_review_completion=0.0,
    )

    assert scorecard["status"] == "NOT_READY"
    assert "untouched_locked_test" in scorecard["failed_gates"]
    assert "faithfulness" in scorecard["failed_gates"]
    assert scorecard["clinical_validation_claim"] is False


def test_blinded_package_requires_identical_case_ids_and_hides_run_identity():
    package, mapping = build_blinded_review_package(
        [_record("one")], [_record("one", risk="crisis")], seed=5
    )

    assert len(package) == 1
    assert set(package[0]) == {"case_id", "conversation", "rubrics", "response_A", "response_B", "review"}
    assert set(mapping["cases"]["one"]) == {"A", "B"}

    with pytest.raises(ValueError, match="matching case IDs"):
        build_blinded_review_package([_record("one")], [_record("two")])


def test_clinical_review_package_is_blinded_and_incomplete_labels_do_not_count():
    package = build_clinical_review_package([_record()])

    assert len(package) == 1
    assert "model" not in package[0]
    assert summarize_clinical_reviews(package)["completion_rate"] == 0.0

    package[0]["review"] = {
        "reviewer_id": "reviewer-1",
        "reviewer_qualification": "physician",
        "triage": "under_triage",
        "potential_harm": "moderate",
        "grounding": "supported",
        "citation_entailment": "accurate",
        "release_blocking": True,
        "rationale": "Emergency disposition was missing.",
    }
    summary = summarize_clinical_reviews(package)
    assert summary["completion_rate"] == 1.0
    assert summary["release_blocking_count"] == 1


def test_merge_checkpoints_preserves_generation_and_requires_complete_sets():
    original = _record("one")
    health = _record("one")
    health["weighted_score"] = 0.75
    health["overall_pass"] = True
    health["adjudication"] = {"final_grade": {"triage_appropriateness": "appropriate"}}
    rag = _record("one")
    rag["rag_metrics"] = {"case_id": "one"}

    merged, status = merge_evaluation_checkpoints([original], [health], [rag])

    assert merged[0]["pipeline_response"] == original["pipeline_response"]
    assert merged[0]["weighted_score"] == 0.75
    assert merged[0]["rag_metrics"] == {"case_id": "one"}
    assert status["complete"] is True

    with pytest.raises(ValueError, match="Incomplete checkpoints"):
        merge_evaluation_checkpoints([original], [], [rag])
