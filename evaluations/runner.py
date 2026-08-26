"""CLI entry point for the evaluation harness.

    python -m evaluations.runner --dataset healthbench --dry-run
    python -m evaluations.runner --dataset healthbench_hard --sample 8
    python -m evaluations.runner --dataset healthbench_consensus --resume --run-id my-run
    python -m evaluations.runner --dataset all

See evaluations/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from evaluations.config import (
    DATASET_URLS,
    DOWNLOADED_DIR,
    HEALTHBENCH_GRADING_PROMPT_VERSION,
    NORMALIZED_DIR,
    RAG_METRICS_PROMPT_VERSION,
    EvalConfig,
    load_config,
)
from evaluations.datasets.adapter import load_cases, normalize_dataset, normalized_path
from evaluations.datasets.download import dataset_path, download_dataset
from evaluations.deterministic_metrics import compute_deterministic_findings
from evaluations.grading import (
    agreement_between,
    grade_with_primary,
    grade_with_terra,
    should_adjudicate,
)
from evaluations.models import AdjudicationDecision, CaseResult, EvalCase, GradingResult
from evaluations.reporting import write_report
from evaluations.retry import call_with_retry


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def require_independent_adjudicator(config: EvalConfig) -> None:
    """Reject graded runs that cannot provide an independent second opinion."""
    if config.primary_grader_model == config.adjudicator_model:
        raise ValueError(
            "EVAL_PRIMARY_GRADER_MODEL and EVAL_ADJUDICATOR_MODEL must be different "
            "to provide an independent second opinion for safety-relevant cases."
        )


def _prepare_cases(dataset_name: str, force_download: bool) -> List[EvalCase]:
    raw_path = dataset_path(dataset_name, dest_dir=DOWNLOADED_DIR)
    if force_download or not raw_path.exists():
        raw_path = download_dataset(
            dataset_name, dest_dir=DOWNLOADED_DIR, force=force_download
        )

    norm_path = normalized_path(dataset_name, dest_dir=NORMALIZED_DIR)
    normalize_dataset(raw_path, norm_path, source_dataset=dataset_name)
    return load_cases(norm_path)


def _load_case_manifest(path: Path, partition: str | None = None) -> List[str]:
    """Load ordered case IDs from a report JSON or a plain JSON list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(
            "Case manifest must contain a JSON list or a report 'cases' list."
        )
    if partition:
        entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("partition") == partition
        ]
    case_ids = [
        str(entry.get("case_id") if isinstance(entry, dict) else entry).strip()
        for entry in entries
    ]
    if not case_ids or any(not case_id or case_id == "None" for case_id in case_ids):
        raise ValueError("Case manifest contains an empty case ID.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Case manifest contains duplicate case IDs.")
    return case_ids


def _select_manifest_cases(
    cases: List[EvalCase], manifest_path: Path, partition: str | None = None
) -> List[EvalCase]:
    case_ids = _load_case_manifest(manifest_path, partition=partition)
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(
            f"Case manifest contains {len(missing)} IDs absent from this dataset: "
            + ", ".join(missing[:5])
        )
    return [by_id[case_id] for case_id in case_ids]


def _load_completed_case_ids(raw_path: Path, generate_only: bool = False) -> Set[str]:
    if not raw_path.exists():
        return set()
    completed = set()
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                # A generate-only run never populates the grading fields --
                # "completed" here just means the case has a real generated
                # answer on record, not that it was graded.
                done = (
                    bool(payload.get("pipeline_response"))
                    if generate_only
                    else (
                        payload.get("rag_metrics")
                        and payload.get("adjudication")
                        and payload.get("deterministic")
                        and payload.get("weighted_score") is not None
                    )
                )
                if done:
                    completed.add(payload["case"]["case_id"])
            except Exception:
                continue
    return completed


def dry_run(cases: List[EvalCase], config: EvalConfig) -> None:
    """Validates data and prompts without calling any model. Also reports the
    role each case would be routed as (see resolve_case_user/role_detection.py)
    without actually creating any eval account -- purely informational."""
    from evaluations.role_detection import resolve_case_role

    errors = 0
    role_counts: dict = {}
    reason_counts: dict = {}
    for case in cases:
        resolution = resolve_case_role(case)
        role_counts[resolution.role] = role_counts.get(resolution.role, 0) + 1
        reason_counts[resolution.reason] = reason_counts.get(resolution.reason, 0) + 1
        try:
            case.last_user_turn()
        except Exception as exc:
            errors += 1
            print(
                f"[dry-run] case {case.case_id} failed validation: {exc}",
                file=sys.stderr,
            )

    print(f"[dry-run] {len(cases)} cases loaded, {errors} failed validation.")
    print(f"[dry-run] detected roles: {role_counts}")
    print(f"[dry-run] role reasons: {reason_counts}")
    print(
        f"[dry-run] generator_model={config.generator_model} "
        f"primary_grader_model={config.primary_grader_model} "
        f"adjudicator_model={config.adjudicator_model} "
        f"rag_metrics_model={config.rag_metrics_model}"
    )
    print("[dry-run] no models were called.")


def resolve_case_user(case: EvalCase):
    """Resolve an audited case role and its isolated evaluation account."""
    from evaluations.pipeline import ensure_eval_account
    from evaluations.role_detection import resolve_case_role

    resolution = resolve_case_role(case)
    return resolution, ensure_eval_account(resolution.role, case.case_id)


def run_case_pipeline(case: EvalCase, rag_engine, config: EvalConfig):
    from evaluations.pipeline import run_case

    resolution, user = resolve_case_user(case)
    return call_with_retry(
        lambda: run_case(
            rag_engine,
            case,
            user=user,
            role=resolution.role,
            role_reason=resolution.reason,
            role_confidence=resolution.confidence,
        ),
        max_retries=config.max_retries,
    )


def _add_consistency_repeats(
    case: EvalCase, pipeline_response, rag_engine, config: EvalConfig
):
    """Periodically repeat the exact production call when explicitly enabled."""

    for _ in range(config.consistency_repeats):
        repeated = run_case_pipeline(case, rag_engine, config)
        pipeline_response.consistency_answers.append(repeated.answer_text)
    return pipeline_response


def _attach_rag_metrics(case_result: CaseResult, config: EvalConfig) -> CaseResult:
    from evaluations.rag_metrics import grade_rag_metrics, unavailable_rag_metrics

    try:
        case_result.rag_metrics = call_with_retry(
            lambda: grade_rag_metrics(
                case_result.case, case_result.pipeline_response, config
            ),
            max_retries=config.max_retries,
        )
    except Exception as exc:  # metric failure must not discard the core grade
        case_result.rag_metrics = unavailable_rag_metrics(
            case_result.case.case_id, config, exc
        )
    return case_result


_HARM_SEVERITY_RANK = {"none": 0, "low": 1, "moderate": 2, "severe": 3}


def _more_conservative_grade(
    a: GradingResult, b: GradingResult, case: EvalCase
) -> GradingResult:
    """Pick whichever grade is stricter, rather than always trusting whichever
    grader ran second. For a safety-critical clinical eval, a lenient grader
    must never silently mask a real deficiency the stricter grader caught --
    found via a real 50-case run where unconditionally trusting the
    adjudicator (a smaller, weaker model than the primary grader) turned a
    17.9%-pooled primary-grader score into a misleadingly optimistic 40.5%,
    and dropped the only case either grader flagged as severe harm.
    Precedence: worse potential_harm_level wins outright; on a harm-level
    tie, the lower (stricter) weighted_score wins.
    """
    harm_a = _HARM_SEVERITY_RANK.get(a.potential_harm_level, 0)
    harm_b = _HARM_SEVERITY_RANK.get(b.potential_harm_level, 0)
    if harm_a != harm_b:
        return a if harm_a > harm_b else b
    return a if a.weighted_score(case) <= b.weighted_score(case) else b


def finalize_healthbench_result(
    case: EvalCase, pipeline_response, luna_grade, config: EvalConfig
) -> CaseResult:
    """Apply deterministic checks and independent secondary adjudication.

    Both graders receive the same captured FlynnMed answer and physician-authored
    rubrics. The adjudicator never receives the primary output. The final weighted score is
    computed locally from exact rubric points, never supplied by either model.
    """
    preliminary = compute_deterministic_findings(case, pipeline_response, luna_grade)
    triggered, reasons = should_adjudicate(
        case, pipeline_response, luna_grade, preliminary, config
    )

    terra_grade = None
    agreement = None
    adjudication_skipped = False
    adjudication_error = None
    final_grade = luna_grade
    final_findings = preliminary
    if triggered:
        if config.adjudicator_model == luna_grade.grader_model:
            adjudication_skipped = True
            reasons.append("same_model_adjudication_skipped")
        else:
            try:
                terra_grade = call_with_retry(
                    lambda: grade_with_terra(case, pipeline_response, config),
                    max_retries=config.max_retries,
                )
            except Exception as exc:
                adjudication_error = f"{type(exc).__name__}: {exc}"
                reasons.append("adjudicator_failed")
            else:
                agreement = agreement_between(luna_grade, terra_grade)
                final_grade = _more_conservative_grade(luna_grade, terra_grade, case)
                final_findings = compute_deterministic_findings(
                    case, pipeline_response, final_grade
                )

    mandatory_harm_second_opinion_missing = (
        luna_grade.potential_harm_level in ("moderate", "severe")
        and terra_grade is None
    )
    if mandatory_harm_second_opinion_missing:
        reasons.append("mandatory_harm_second_opinion_missing")

    adjudication = AdjudicationDecision(
        case_id=case.case_id,
        triggered=triggered,
        trigger_reasons=reasons,
        luna_grade=luna_grade,
        terra_grade=terra_grade,
        agreement=agreement,
        adjudication_skipped=adjudication_skipped,
        adjudication_error=adjudication_error,
        final_grade=final_grade,
    )
    weighted_score = final_grade.weighted_score(case)
    ai_pass = final_grade.potential_harm_level != "severe" and weighted_score >= 0.5
    if mandatory_harm_second_opinion_missing:
        ai_pass = False
    return CaseResult(
        case=case,
        pipeline_response=pipeline_response,
        adjudication=adjudication,
        deterministic=final_findings,
        weighted_score=weighted_score,
        overall_pass=final_findings.deterministic_pass and ai_pass,
    )


def evaluate_case(case: EvalCase, rag_engine, config: EvalConfig) -> CaseResult:
    pipeline_response = run_case_pipeline(case, rag_engine, config)
    pipeline_response = _add_consistency_repeats(
        case, pipeline_response, rag_engine, config
    )
    luna_grade = call_with_retry(
        lambda: grade_with_primary(case, pipeline_response, config),
        max_retries=config.max_retries,
    )
    result = finalize_healthbench_result(case, pipeline_response, luna_grade, config)
    return _attach_rag_metrics(result, config)


def generate_case_only(case: EvalCase, rag_engine, config: EvalConfig) -> CaseResult:
    """
    Generation-only path: produces the real pipeline_response -- answer,
    retrieved sources with their evidence-ranking scores/tiers, full trace,
    role resolution -- with NO LLM grading of any kind (no rubric scoring,
    no adjudication, no RAG-metrics judge). Used when grading will happen on
    a separate platform. The grading fields on CaseResult stay None, which
    is a shape evaluations/models.py's CaseResult already supports (see its
    "historical raw JSONL readable" comment) rather than a new one.
    """
    pipeline_response = run_case_pipeline(case, rag_engine, config)
    pipeline_response = _add_consistency_repeats(
        case, pipeline_response, rag_engine, config
    )
    return CaseResult(case=case, pipeline_response=pipeline_response)


def run_dataset(
    dataset_name: str, args: argparse.Namespace, config: EvalConfig
) -> None:
    if getattr(args, "regrade_healthbench", False):
        _regrade_saved_healthbench(dataset_name, args, config)
        return
    if getattr(args, "regrade_rag", False):
        _regrade_saved_rag_metrics(dataset_name, args, config)
        return
    cases = _prepare_cases(dataset_name, force_download=args.force_download)

    case_manifest = getattr(args, "case_manifest", None)
    random_seed = getattr(args, "random_seed", None)
    if case_manifest:
        case_partition = getattr(args, "case_partition", None)
        cases = _select_manifest_cases(
            cases, Path(case_manifest), partition=case_partition
        )
        suffix = f" partition={case_partition}" if case_partition else ""
        print(
            f"[{dataset_name}] selected {len(cases)} cases from "
            f"{case_manifest}{suffix}."
        )
    elif random_seed is not None:
        random.Random(random_seed).shuffle(cases)
        print(f"[{dataset_name}] randomized case order with seed {random_seed}.")

    if config.sample_limit is not None and not case_manifest:
        cases = cases[: config.sample_limit]

    if args.dry_run:
        dry_run(cases, config)
        return

    generate_only = getattr(args, "generate_only", False)
    run_id = args.run_id or f"{dataset_name}_{_utc_timestamp()}"
    raw_path = Path(config.output_path) / "raw" / run_id / "cases.jsonl"

    completed_ids: Set[str] = set()
    if args.resume:
        completed_ids = _load_completed_case_ids(raw_path, generate_only=generate_only)
        if completed_ids:
            print(
                f"[resume] {len(completed_ids)} cases already completed for run '{run_id}', skipping them."
            )

    remaining = [c for c in cases if c.case_id not in completed_ids]
    if not remaining:
        print(
            f"[{dataset_name}] nothing to do -- all {len(cases)} cases already completed."
        )
        return

    from evaluations.pipeline import build_rag_engine, configure_evaluation_storage

    storage_root = raw_path.parent / "runtime"
    configure_evaluation_storage(storage_root)
    print(f"[{dataset_name}] isolated evaluation accounts under {storage_root}")
    rag_engine = build_rag_engine(config)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    results: List[CaseResult] = []
    if completed_ids:
        with open(raw_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    result = CaseResult.model_validate_json(line)
                    done = (
                        bool(result.pipeline_response)
                        if generate_only
                        else (
                            result.rag_metrics
                            and result.adjudication
                            and result.deterministic
                            and result.weighted_score is not None
                        )
                    )
                    if done:
                        results.append(result)

    with open(raw_path, "a", encoding="utf-8") as append_fh:
        _run_synchronous(
            dataset_name, remaining, rag_engine, config, results, append_fh,
            generate_only=generate_only,
        )

    if generate_only:
        print(
            f"[{dataset_name}] generation-only run complete -- "
            f"{len(results)} case(s) with real answers/sources/ranking metadata, no grading applied."
        )
        print(f"[{dataset_name}] wrote raw results to {raw_path}")
        print(
            f"[{dataset_name}] no report was generated -- rag_metrics/adjudication/weighted_score "
            "are intentionally None; grading happens on a separate platform."
        )
        return

    _, summary_json_path, summary_md_path = write_report(
        results,
        config,
        dataset_version=dataset_name,
        run_id=run_id,
    )
    print(f"[{dataset_name}] wrote raw results to {raw_path}")
    print(f"[{dataset_name}] wrote report to {summary_json_path} and {summary_md_path}")


def _regrade_saved_rag_metrics(
    dataset_name: str, args: argparse.Namespace, config: EvalConfig
) -> None:
    """Re-run only RAG metrics against immutable saved answers and sources."""
    if not args.run_id:
        raise ValueError("--regrade-rag requires --run-id")
    raw_path = Path(config.output_path) / "raw" / args.run_id / "cases.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Saved run not found: {raw_path}")

    saved: List[CaseResult] = []
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                saved.append(CaseResult.model_validate_json(line))
    case_ids = [result.case.case_id for result in saved]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Saved run contains duplicate case ids; refusing to re-grade.")

    checkpoint_name = f"rag_regrade_{RAG_METRICS_PROMPT_VERSION}.jsonl"
    checkpoint_path = raw_path.parent / checkpoint_name
    completed: dict[str, CaseResult] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                result = CaseResult.model_validate_json(line)
                if result.rag_metrics and not result.rag_metrics.evaluation_error:
                    completed[result.case.case_id] = result
        if completed:
            print(
                f"[{dataset_name}] loaded {len(completed)} completed RAG re-grades "
                f"from {checkpoint_path.name}"
            )

    regraded_by_id = dict(completed)
    pending = [result for result in saved if result.case.case_id not in completed]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_mode = "a" if checkpoint_path.exists() else "w"
    checkpoint_fh = open(checkpoint_path, checkpoint_mode, encoding="utf-8")
    try:
        print(
            f"[{dataset_name}] RAG re-grading {len(pending)} saved answers with "
            f"{config.rag_regrade_workers} worker(s)"
        )
        with ThreadPoolExecutor(max_workers=config.rag_regrade_workers) as executor:
            future_to_result = {
                executor.submit(_attach_rag_metrics, result, config): result
                for result in pending
            }
            completed_now = 0
            for future in as_completed(future_to_result):
                original = future_to_result[future]
                regraded = future.result()
                case_id = original.case.case_id
                regraded_by_id[case_id] = regraded
                completed_now += 1
                print(
                    f"[{dataset_name}] RAG re-grade "
                    f"({len(completed) + completed_now}/{len(saved)}) {case_id}"
                )
                if regraded.rag_metrics and not regraded.rag_metrics.evaluation_error:
                    checkpoint_fh.write(regraded.model_dump_json() + "\n")
                    checkpoint_fh.flush()
    finally:
        checkpoint_fh.close()

    regraded = [regraded_by_id[result.case.case_id] for result in saved]

    _, summary_json_path, summary_md_path = write_report(
        regraded,
        config,
        dataset_version=dataset_name,
        run_id=args.run_id,
    )
    print(f"[{dataset_name}] preserved generation and re-graded {len(regraded)} cases")
    print(f"[{dataset_name}] wrote report to {summary_json_path} and {summary_md_path}")


def _safe_model_slug(model: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in model)


def _regrade_saved_healthbench(
    dataset_name: str, args: argparse.Namespace, config: EvalConfig
) -> None:
    """Re-grade saved answers without calling FlynnMed's generation pipeline."""
    if not args.run_id:
        raise ValueError("--regrade-healthbench requires --run-id")
    raw_path = Path(config.output_path) / "raw" / args.run_id / "cases.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Saved run not found: {raw_path}")

    saved: List[CaseResult] = []
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                saved.append(CaseResult.model_validate_json(line))
    case_ids = [result.case.case_id for result in saved]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Saved run contains duplicate case ids; refusing to re-grade.")

    model_slug = _safe_model_slug(config.primary_grader_model)
    regrade_id = (
        f"{args.run_id}_{HEALTHBENCH_GRADING_PROMPT_VERSION}_{model_slug}"
    )
    checkpoint_path = raw_path.parent / f"{regrade_id}_checkpoint.jsonl"
    completed: dict[str, CaseResult] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                if line.strip():
                    try:
                        result = CaseResult.model_validate_json(line)
                    except Exception:
                        print(
                            f"[{dataset_name}] ignoring interrupted checkpoint "
                            f"record at line {line_number}",
                            file=sys.stderr,
                        )
                        continue
                    if (
                        result.adjudication.luna_grade.grader_model
                        != config.primary_grader_model
                    ):
                        continue
                    completed[result.case.case_id] = result
        if completed:
            print(
                f"[{dataset_name}] loaded {len(completed)} completed HealthBench "
                f"re-grades from {checkpoint_path.name}"
            )

    regraded_by_id = dict(completed)
    pending = [
        (index, result)
        for index, result in enumerate(saved, start=1)
        if result.case.case_id not in completed
    ]

    def regrade_one(saved_result: CaseResult) -> CaseResult:
        grade = call_with_retry(
            lambda: grade_with_primary(
                saved_result.case, saved_result.pipeline_response, config
            ),
            max_retries=config.max_retries,
        )
        regraded_result = finalize_healthbench_result(
            saved_result.case, saved_result.pipeline_response, grade, config
        )
        regraded_result.rag_metrics = saved_result.rag_metrics
        return regraded_result

    print(
        f"[{dataset_name}] re-grading {len(pending)} saved answers with "
        f"{config.regrade_workers} worker(s)"
    )
    # Rewrite only validated, current-model records before resuming. This
    # removes a partial final line left by process termination and discards
    # fallback-model grades from a model-specific checkpoint.
    with open(checkpoint_path, "w", encoding="utf-8") as checkpoint_fh:
        for result in completed.values():
            checkpoint_fh.write(result.model_dump_json() + "\n")
        checkpoint_fh.flush()
        with ThreadPoolExecutor(max_workers=config.regrade_workers) as executor:
            futures = {
                executor.submit(regrade_one, saved_result): (index, saved_result)
                for index, saved_result in pending
            }
            completed_this_run = 0
            for future in as_completed(futures):
                index, saved_result = futures[future]
                regraded = future.result()
                case_id = saved_result.case.case_id
                regraded_by_id[case_id] = regraded
                checkpoint_fh.write(regraded.model_dump_json() + "\n")
                checkpoint_fh.flush()
                completed_this_run += 1
                print(
                    f"[{dataset_name}] HealthBench re-grade "
                    f"{len(completed) + completed_this_run}/{len(saved)} "
                    f"(source position {index}) {case_id}"
                )

    regraded = [regraded_by_id[result.case.case_id] for result in saved]
    _, summary_json_path, summary_md_path = write_report(
        regraded,
        config,
        dataset_version=dataset_name,
        run_id=regrade_id,
    )
    print(f"[{dataset_name}] preserved generation and re-graded {len(regraded)} cases")
    print(f"[{dataset_name}] wrote report to {summary_json_path} and {summary_md_path}")


def _run_synchronous(
    dataset_name: str,
    remaining: List[EvalCase],
    rag_engine,
    config: EvalConfig,
    results: List[CaseResult],
    append_fh,
    generate_only: bool = False,
) -> None:
    for index, case in enumerate(remaining, start=1):
        print(f"[{dataset_name}] ({index}/{len(remaining)}) {case.case_id}")
        try:
            case_result = (
                generate_case_only(case, rag_engine, config)
                if generate_only
                else evaluate_case(case, rag_engine, config)
            )
        except Exception as exc:
            print(
                f"[{dataset_name}] case {case.case_id} FAILED: {exc}", file=sys.stderr
            )
            traceback.print_exc()
            continue
        results.append(case_result)
        append_fh.write(case_result.model_dump_json() + "\n")
        append_fh.flush()


def main(argv: Optional[List[str]] = None) -> None:
    # Root-cause fix, not a per-callsite patch: on Windows, sys.stdout/stderr
    # default to the console codepage (cp1252), not UTF-8. Any print() of
    # LLM-generated or dataset text containing a character outside that range
    # (an en-dash, curly quote, etc.) raises UnicodeEncodeError -- and in
    # backend/pubmed_search.py and backend/clinical_orchestrator.py, that
    # crash was getting caught by a broad `except Exception` and silently
    # discarding already-successful retrieval results. Found via a real 50-case
    # run where this happened on ~46% of cases before the pubmed_search.py fix,
    # and still 4/50 afterward from a second print site in the agentic loop.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Run the FlynnMed HealthBench evaluation harness."
    )
    parser.add_argument(
        "--dataset", choices=[*DATASET_URLS.keys(), "all"], required=True
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and prompts without calling any model.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run the first N cases (overrides EVAL_SAMPLE_LIMIT).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Shuffle cases reproducibly before applying --sample.",
    )
    parser.add_argument(
        "--case-manifest",
        type=Path,
        default=None,
        help="Use the exact ordered case IDs from a prior summary JSON or JSON list.",
    )
    parser.add_argument(
        "--case-partition",
        help="Select only entries with this partition from --case-manifest.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run, skipping already-completed cases.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (required to --resume a specific run).",
    )
    parser.add_argument(
        "--regrade-rag",
        action="store_true",
        help="Re-grade only RAG metrics for a saved --run-id without regenerating answers.",
    )
    parser.add_argument(
        "--regrade-healthbench",
        action="store_true",
        help=(
            "Re-grade saved answers against HealthBench rubrics for a --run-id "
            "without regenerating answers."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the dataset even if a local copy exists.",
    )
    parser.add_argument(
        "--consistency-repeats",
        type=int,
        default=None,
        help="Additional identical production calls per case for periodic consistency scoring.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help=(
            "Produce real generated answers, sources, evidence-ranking metadata, and traces "
            "with NO LLM grading (no rubric scoring, no adjudication, no RAG-metrics judge). "
            "rag_metrics/adjudication/deterministic/weighted_score stay None on every case. "
            "Use when grading will happen on a separate platform."
        ),
    )
    args = parser.parse_args(argv)

    if args.case_manifest and (args.sample is not None or args.random_seed is not None):
        parser.error(
            "--case-manifest cannot be combined with --sample or --random-seed"
        )
    if args.case_partition and not args.case_manifest:
        parser.error("--case-partition requires --case-manifest")
    if args.generate_only and (args.regrade_rag or args.regrade_healthbench):
        parser.error("--generate-only cannot be combined with --regrade-rag or --regrade-healthbench")
    if args.regrade_rag and args.regrade_healthbench:
        parser.error("--regrade-rag and --regrade-healthbench are mutually exclusive")

    config = load_config()
    if args.sample is not None:
        config.sample_limit = args.sample
    if args.consistency_repeats is not None:
        config.consistency_repeats = max(0, args.consistency_repeats)

    if not args.generate_only:
        try:
            require_independent_adjudicator(config)
        except ValueError as exc:
            parser.error(str(exc))

    if not args.dry_run:
        from evaluations.grading import EvaluatorAccessError, validate_evaluator_access

        # A generate-only run calls no grading model at all -- check access
        # to the generator model instead of primary/adjudicator/rag_metrics,
        # which is what the default (models_to_check=None) would check.
        print("[runner] checking access to configured evaluator models...")
        try:
            models_to_check = None
            if args.generate_only:
                models_to_check = [config.generator_model]
            elif args.regrade_healthbench:
                models_to_check = [
                    config.primary_grader_model,
                    config.adjudicator_model,
                ]
            elif args.regrade_rag:
                models_to_check = [config.rag_metrics_model]
            validate_evaluator_access(config, models_to_check=models_to_check)
        except (EvaluatorAccessError, ValueError) as exc:
            print(f"[runner] evaluator access check FAILED: {exc}", file=sys.stderr)
            raise SystemExit(2) from None
        print("[runner] evaluator model access confirmed.")

    dataset_names = (
        list(DATASET_URLS.keys()) if args.dataset == "all" else [args.dataset]
    )
    started = time.perf_counter()
    for dataset_name in dataset_names:
        run_dataset(dataset_name, args, config)
    print(f"[runner] done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
