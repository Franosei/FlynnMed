"""Builds the raw results file plus a sanitised JSON + Markdown summary from
a completed evaluation run.

Raw results (full conversations, full answer text, rubric evidence, source splits,
metric explanations)
are written under `evaluations/results/raw/` -- gitignored, local only. The
sanitised summary/report only ever contains case identifiers, tags, scores,
and flags -- never full conversation or answer text. HealthBench and RAG
results share one report with an explicit list of cases needing qualified
clinician review.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlsplit

from evaluations.config import (
    HEALTHBENCH_GRADING_PROMPT_VERSION,
    RAG_METRICS_PROMPT_VERSION,
    EvalConfig,
)
from evaluations.models import CaseResult, MetricAggregate, ReportSummary, TagAggregate

REPORT_LABEL = "Automated HealthBench and RAG evaluation -- not clinical validation"

_RAG_METRIC_TIERS = {
    "Tier 1 - Core": (
        ("faithfulness", "Faithfulness / groundedness"),
        ("context_relevance", "Context relevance (precision)"),
        ("noise_sensitivity", "Noise robustness (1 = no contamination)"),
        ("context_recall", "Context recall (coverage)"),
        ("answer_correctness", "Answer correctness vs. gold"),
        ("calibration", "Calibration / appropriate hedging"),
    ),
    "Tier 2 - Important": (
        ("contradiction_handling", "Contradiction / conflict handling"),
        ("citation_accuracy", "Citation accuracy (attached claim only)"),
        ("citation_completeness", "Citation completeness (material-claim coverage)"),
        ("context_precision_ranking", "Context precision (ranking nDCG)"),
    ),
    "Tier 3 - Periodic safety monitoring": (
        ("clinical_harmlessness", "Clinical harmlessness"),
        ("consistency", "Consistency / reproducibility"),
    ),
}


def _is_valid_related_link(value: object) -> bool:
    try:
        parts = urlsplit(str(value or "").strip())
    except ValueError:
        return False
    return parts.scheme.lower() in {"http", "https"} and bool(parts.netloc)


def _aggregate_rag_metrics(
    case_results: List[CaseResult],
    minimum_reliable_sample_size: int = 5,
) -> dict[str, MetricAggregate]:
    aggregates: dict[str, MetricAggregate] = {}
    total = len(case_results)
    for metrics in _RAG_METRIC_TIERS.values():
        for name, _ in metrics:
            scores = []
            assessment_count = 0
            for case_result in case_results:
                rag_metrics = case_result.rag_metrics
                if not rag_metrics:
                    continue
                metric = getattr(rag_metrics, name)
                if metric.applicable and metric.score is not None:
                    scores.append(metric.score)
                    assessment_count += metric.sample_size or 1
            status = "not_applicable"
            if scores:
                status = (
                    "sufficient"
                    if assessment_count >= minimum_reliable_sample_size
                    else "insufficient_sample"
                )
            aggregates[name] = MetricAggregate(
                average_score=sum(scores) / len(scores) if scores else None,
                applicable_cases=len(scores),
                total_cases=total,
                assessment_count=assessment_count,
                status=status,
            )
    return aggregates


def _healthbench_graded(case_results: List[CaseResult]) -> List[CaseResult]:
    return [
        cr
        for cr in case_results
        if cr.adjudication
        and cr.deterministic
        and cr.weighted_score is not None
        and cr.overall_pass is not None
    ]


def _aggregate_by_tag(case_results: List[CaseResult]) -> dict[str, TagAggregate]:
    """Breaks HealthBench scoring down per tag (theme:*, physician_agreed_category:*,
    etc). A case can carry several tags, so it contributes to each of its tags'
    buckets -- these do not sum to the overall totals."""
    tags = sorted({tag for cr in case_results for tag in cr.case.tags})
    aggregates: dict[str, TagAggregate] = {}
    for tag in tags:
        tagged = [cr for cr in case_results if tag in cr.case.tags]
        graded = _healthbench_graded(tagged)
        graded_count = len(graded)
        aggregates[tag] = TagAggregate(
            case_count=len(tagged),
            healthbench_graded_cases=graded_count,
            pass_rate=(
                sum(1 for cr in graded if cr.overall_pass) / graded_count
                if graded_count
                else None
            ),
            weighted_healthbench_score=(
                sum(cr.weighted_score for cr in graded) / graded_count
                if graded_count
                else None
            ),
            under_triage_rate=(
                sum(1 for cr in graded if cr.deterministic.under_triage) / graded_count
                if graded_count
                else None
            ),
            severe_under_triage_rate=(
                sum(1 for cr in graded if cr.deterministic.severe_under_triage)
                / graded_count
                if graded_count
                else None
            ),
        )
    return aggregates


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pipeline_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def build_report_summary(
    case_results: List[CaseResult],
    config: EvalConfig,
    dataset_version: str,
    prompt_version: str | None = None,
    run_date: str | None = None,
) -> ReportSummary:
    effective_prompt_version = prompt_version or HEALTHBENCH_GRADING_PROMPT_VERSION
    effective_run_date = run_date or _utc_now()
    total = len(case_results)
    if total == 0:
        return ReportSummary(
            dataset_version=dataset_version,
            pipeline_version=pipeline_version(),
            prompt_version=effective_prompt_version,
            rag_metrics_prompt_version=RAG_METRICS_PROMPT_VERSION,
            generator_model=config.generator_model,
            primary_grader_model=config.primary_grader_model,
            adjudicator_model=config.adjudicator_model,
            rag_metrics_model=config.rag_metrics_model,
            run_date=effective_run_date,
            total_cases=0,
            cases_requiring_human_review=[],
            notes=["No cases were evaluated in this run."],
        )
    rag_metric_aggregates = _aggregate_rag_metrics(
        case_results, config.minimum_reliable_sample_size
    )
    healthbench_results = _healthbench_graded(case_results)
    graded_count = len(healthbench_results)
    pass_rate = (
        sum(1 for cr in healthbench_results if cr.overall_pass) / graded_count
        if graded_count
        else None
    )
    weighted_healthbench_score = (
        sum(cr.weighted_score for cr in healthbench_results) / graded_count
        if graded_count
        else None
    )
    under_triage_rate = (
        sum(1 for cr in healthbench_results if cr.deterministic.under_triage)
        / graded_count
        if graded_count
        else None
    )
    severe_under_triage_rate = (
        sum(1 for cr in healthbench_results if cr.deterministic.severe_under_triage)
        / graded_count
        if graded_count
        else None
    )
    emergency_expected = [
        cr
        for cr in healthbench_results
        if "physician_agreed_category:emergent" in cr.case.tags
    ]
    emergencies_recognised = sum(
        1 for cr in emergency_expected if cr.deterministic.crisis_gate_activated
    )
    emergency_sensitivity = (
        emergencies_recognised / len(emergency_expected)
        if emergency_expected
        else None
    )
    conditionally_emergent_cases = sum(
        "physician_agreed_category:conditionally-emergent" in cr.case.tags
        for cr in healthbench_results
    )
    physician_labelled_non_emergent_cases = sum(
        "physician_agreed_category:non-emergent" in cr.case.tags
        for cr in healthbench_results
    )
    adjudication_rate = (
        sum(1 for cr in healthbench_results if cr.adjudication.terra_grade is not None)
        / graded_count
        if graded_count
        else None
    )
    disagreement_count = sum(
        1 for cr in healthbench_results if cr.adjudication.agreement is False
    )
    relevant_document_count = sum(
        len(cr.rag_metrics.relevant_source_ids) for cr in case_results if cr.rag_metrics
    )
    irrelevant_document_count = sum(
        len(cr.rag_metrics.irrelevant_source_ids)
        for cr in case_results
        if cr.rag_metrics
    )
    rag_metrics_error_count = sum(
        1 for cr in case_results if cr.rag_metrics and cr.rag_metrics.evaluation_error
    )
    claim_audit_warning_case_count = sum(
        1
        for cr in case_results
        if cr.rag_metrics and cr.rag_metrics.claim_audit_warnings
    )
    claim_audit_warning_count = sum(
        len(cr.rag_metrics.claim_audit_warnings)
        for cr in case_results
        if cr.rag_metrics
    )
    unmapped_claim_count = sum(
        1
        for cr in case_results
        if cr.rag_metrics
        for claim in cr.rag_metrics.claim_assessments
        if claim.answer_quote_validated is not True
    )
    role_counts = dict(
        sorted(
            Counter(cr.pipeline_response.resolved_role for cr in case_results).items()
        )
    )
    related_sources = [
        source for cr in case_results for source in (cr.pipeline_response.sources or [])
    ]
    valid_related_link_count = sum(
        _is_valid_related_link(source.get("url")) for source in related_sources
    )
    related_link_count = len(related_sources)
    related_link_url_coverage = (
        valid_related_link_count / related_link_count if related_link_count else None
    )
    deterministic_results = [
        cr.deterministic for cr in case_results if cr.deterministic
    ]
    rendered_citation_count = sum(
        result.citation_count for result in deterministic_results
    )
    resolved_rendered_citation_count = sum(
        result.resolved_citation_count for result in deterministic_results
    )
    rendered_citation_resolution_rate = (
        resolved_rendered_citation_count / rendered_citation_count
        if rendered_citation_count
        else None
    )

    review_cases = [
        cr.case.case_id for cr in case_results if cr.requires_human_review()
    ]
    by_tag = _aggregate_by_tag(case_results)
    notes = [
        "Overall under-triage rates remain model-judge audit proxies. Emergency "
        "recognition sensitivity uses only HealthBench's physician-agreed emergent "
        "category; conditionally emergent cases are reported separately."
    ]
    if config.primary_grader_model == config.adjudicator_model:
        notes.append(
            "Primary and adjudicator models are identical, so triggered secondary "
            "adjudication was skipped and the run has no independent cross-check."
        )

    return ReportSummary(
        dataset_version=dataset_version,
        pipeline_version=pipeline_version(),
        prompt_version=effective_prompt_version,
        rag_metrics_prompt_version=RAG_METRICS_PROMPT_VERSION,
        generator_model=config.generator_model,
        primary_grader_model=config.primary_grader_model,
        adjudicator_model=config.adjudicator_model,
        rag_metrics_model=config.rag_metrics_model,
        run_date=effective_run_date,
        total_cases=total,
        healthbench_graded_cases=graded_count,
        pass_rate=pass_rate,
        weighted_healthbench_score=weighted_healthbench_score,
        under_triage_rate=under_triage_rate,
        severe_under_triage_rate=severe_under_triage_rate,
        emergency_sensitivity=emergency_sensitivity,
        physician_labelled_emergency_cases=len(emergency_expected),
        physician_labelled_emergencies_recognised=emergencies_recognised,
        conditionally_emergent_cases=conditionally_emergent_cases,
        physician_labelled_non_emergent_cases=(
            physician_labelled_non_emergent_cases
        ),
        adjudication_rate=adjudication_rate,
        disagreement_count=disagreement_count,
        rag_metric_aggregates=rag_metric_aggregates,
        relevant_document_count=relevant_document_count,
        irrelevant_document_count=irrelevant_document_count,
        rag_metrics_error_count=rag_metrics_error_count,
        claim_audit_warning_case_count=claim_audit_warning_case_count,
        claim_audit_warning_count=claim_audit_warning_count,
        unmapped_claim_count=unmapped_claim_count,
        role_counts=role_counts,
        related_link_count=related_link_count,
        valid_related_link_count=valid_related_link_count,
        related_link_url_coverage=related_link_url_coverage,
        rendered_citation_count=rendered_citation_count,
        resolved_rendered_citation_count=resolved_rendered_citation_count,
        rendered_citation_resolution_rate=rendered_citation_resolution_rate,
        cases_requiring_human_review=review_cases,
        by_tag=by_tag,
        notes=notes,
    )


def _sanitized_case_entry(case_result: CaseResult) -> dict:
    entry = {
        "case_id": case_result.case.case_id,
        "source_dataset": case_result.case.source_dataset,
        "tags": case_result.case.tags,
        "resolved_role": case_result.pipeline_response.resolved_role,
        "role_resolution_reason": case_result.pipeline_response.role_resolution_reason,
        "role_resolution_confidence": case_result.pipeline_response.role_resolution_confidence,
        "requires_human_review": case_result.requires_human_review(),
    }
    if (
        case_result.adjudication
        and case_result.deterministic
        and case_result.weighted_score is not None
        and case_result.overall_pass is not None
    ):
        entry["healthbench"] = {
            "weighted_score": round(case_result.weighted_score, 4),
            "overall_pass": case_result.overall_pass,
            "adjudication_triggered": case_result.adjudication.triggered,
            "adjudication_completed": (
                case_result.adjudication.triggered
                and not case_result.adjudication.adjudication_skipped
                and case_result.adjudication.adjudication_error is None
            ),
            "trigger_reasons": case_result.adjudication.trigger_reasons,
            "agreement": case_result.adjudication.agreement,
            "adjudication_skipped": case_result.adjudication.adjudication_skipped,
            "adjudication_error": bool(case_result.adjudication.adjudication_error),
            "deterministic_pass": case_result.deterministic.deterministic_pass,
            "failure_reasons": case_result.deterministic.failure_reasons,
            "expected_urgency_level": case_result.deterministic.expected_urgency_level,
            "actual_urgency_level": case_result.deterministic.actual_urgency_level,
            "potential_harm_level": case_result.adjudication.final_grade.potential_harm_level,
            "grader_confidence": case_result.adjudication.final_grade.confidence,
            "unvalidated_rubric_evidence_count": sum(
                1
                for result in case_result.adjudication.final_grade.rubric_results
                if result.answer_evidence_validated is False
            ),
            "link_integrity": {
                "related_links": len(case_result.pipeline_response.sources),
                "valid_related_links": sum(
                    _is_valid_related_link(source.get("url"))
                    for source in case_result.pipeline_response.sources
                ),
                "rendered_citations": case_result.deterministic.citation_count,
                "resolved_rendered_citations": (
                    case_result.deterministic.resolved_citation_count
                ),
            },
        }
    if case_result.rag_metrics:
        entry["rag_metrics"] = {
            name: {
                "score": getattr(case_result.rag_metrics, name).score,
                "applicable": getattr(case_result.rag_metrics, name).applicable,
            }
            for metrics in _RAG_METRIC_TIERS.values()
            for name, _ in metrics
        }
        entry["retrieved_document_split"] = {
            "relevant": len(case_result.rag_metrics.relevant_source_ids),
            "irrelevant": len(case_result.rag_metrics.irrelevant_source_ids),
        }
        entry["claim_audit"] = {
            "material_claims": sum(
                claim.material for claim in case_result.rag_metrics.claim_assessments
            ),
            "citation_pairs": len(case_result.rag_metrics.citation_assessments),
            "error": bool(case_result.rag_metrics.claim_audit_error),
            "warning_count": len(case_result.rag_metrics.claim_audit_warnings),
        }
        entry["gold_answer_provenance"] = case_result.rag_metrics.gold_answer_provenance
        entry["rag_metrics_error"] = bool(case_result.rag_metrics.evaluation_error)
    return entry


def _rag_metric_lines(summary: ReportSummary) -> list[str]:
    lines = ["## Tiered RAG quality metrics", ""]
    if not summary.rag_metric_aggregates:
        return lines + ["No completed RAG metric results were available.", ""]
    lines.extend(
        [
            f"Document split: {summary.relevant_document_count} relevant, "
            f"{summary.irrelevant_document_count} irrelevant/distractor. "
            "Relevance is judged from stored excerpts before dependent metrics.",
            "",
            "Link integrity is reported separately from clinical citation grounding. "
            "A working URL does not prove that its excerpt supports a claim.",
            (
                f"- Related source URLs: {summary.valid_related_link_count}/"
                f"{summary.related_link_count} valid "
                f"({summary.related_link_url_coverage:.1%})"
                if summary.related_link_url_coverage is not None
                else "- Related source URLs: n/a (no retrieved sources)"
            ),
            (
                f"- Rendered citation targets: {summary.resolved_rendered_citation_count}/"
                f"{summary.rendered_citation_count} resolved to their stored source "
                f"({summary.rendered_citation_resolution_rate:.1%})"
                if summary.rendered_citation_resolution_rate is not None
                else "- Rendered citation targets: n/a (no rendered citations)"
            ),
            "",
        ]
    )
    for tier, metrics in _RAG_METRIC_TIERS.items():
        lines.extend([f"### {tier}", ""])
        for name, label in metrics:
            aggregate = summary.rag_metric_aggregates.get(name)
            if aggregate and aggregate.average_score is not None:
                qualification = (
                    "Sufficient sample"
                    if aggregate.status == "sufficient"
                    else "PROVISIONAL - insufficient sample"
                )
                lines.append(
                    f"- {label}: {aggregate.average_score:.3f} "
                    f"({aggregate.applicable_cases}/{aggregate.total_cases} applicable cases; "
                    f"{aggregate.assessment_count} assessed items; {qualification})"
                )
            else:
                lines.append(f"- {label}: n/a (no applicable cases)")
        lines.append("")
    if summary.rag_metrics_error_count:
        lines.extend(
            [
                f"RAG metric judge errors: {summary.rag_metrics_error_count} case(s). "
                "These cases are excluded from metric denominators, not scored zero.",
                "",
            ]
        )
    if summary.claim_audit_warning_case_count:
        lines.extend(
            [
                f"Claim-audit warnings: {summary.claim_audit_warning_count} across "
                f"{summary.claim_audit_warning_case_count} case(s). "
                f"Unmapped claims excluded from claim-level denominators: "
                f"{summary.unmapped_claim_count}.",
                "",
            ]
        )
    return lines


def _healthbench_lines(summary: ReportSummary) -> list[str]:
    lines = ["## HealthBench rubric scoring", ""]
    if not summary.healthbench_graded_cases:
        return lines + ["No completed HealthBench rubric grades were available.", ""]
    lines.extend(
        [
            "The weighted score is computed locally from the physician-authored rubric "
            "points. The graders classify each exact rubric against the captured FlynnMed "
            "answer; they do not generate or compare against their own answer.",
            "",
            f"- Graded cases: {summary.healthbench_graded_cases}/{summary.total_cases}",
            f"- Pass rate: {summary.pass_rate:.1%}",
            f"- Weighted HealthBench score: {summary.weighted_healthbench_score:.3f}",
            "Overall triage rates below use model-judge urgency inference and remain "
            "audit proxies. Emergency recognition uses the dataset's physician-agreed "
            "emergent category only.",
            "",
            f"- Judge-inferred under-triage rate: {summary.under_triage_rate:.1%}",
            f"- Judge-inferred severe under-triage rate: {summary.severe_under_triage_rate:.1%}",
            (
                f"- Physician-labelled emergency recognition sensitivity: "
                f"{summary.emergency_sensitivity:.1%} "
                f"({summary.physician_labelled_emergencies_recognised}/"
                f"{summary.physician_labelled_emergency_cases})"
                if summary.emergency_sensitivity is not None
                else "- Physician-labelled emergency recognition sensitivity: n/a"
            ),
            f"- Conditionally emergent cases: {summary.conditionally_emergent_cases}",
            f"- Physician-labelled non-emergent cases: "
            f"{summary.physician_labelled_non_emergent_cases}",
            f"- Secondary adjudication rate: {summary.adjudication_rate:.1%}",
            f"- Primary/adjudicator disagreements: {summary.disagreement_count}",
            "",
        ]
    )
    return lines


def _by_tag_lines(summary: ReportSummary) -> list[str]:
    lines = ["## HealthBench scoring by tag", ""]
    graded_tags = {
        tag: agg for tag, agg in summary.by_tag.items() if agg.healthbench_graded_cases
    }
    if not graded_tags:
        return lines + ["No tagged cases had a completed HealthBench grade.", ""]
    lines.append(
        "A case can carry more than one tag, so rows do not sum to the overall totals."
    )
    lines.append("")
    lines.append(
        "| Tag | Graded cases | Pass rate | Weighted score | Judge-inferred under-triage | Judge-inferred severe under-triage |"
    )
    lines.append("|---|---|---|---|---|---|")
    for tag, agg in sorted(graded_tags.items()):
        lines.append(
            f"| `{tag}` | {agg.healthbench_graded_cases}/{agg.case_count} | "
            f"{agg.pass_rate:.1%} | {agg.weighted_healthbench_score:.3f} | "
            f"{agg.under_triage_rate:.1%} | {agg.severe_under_triage_rate:.1%} |"
        )
    lines.append("")
    return lines


def _markdown_report(summary: ReportSummary, case_results: List[CaseResult]) -> str:
    lines = [
        f"# {REPORT_LABEL}",
        "",
        "This report is produced by an automated benchmark harness. It is **not** a",
        "clinical validation of FlynnMed and must not be represented as one. Severe,",
        "uncertain, or disputed cases are listed below and require review by a",
        "qualified clinician before any conclusion is drawn from them.",
        "",
        "## Run metadata",
        "",
        f"- Dataset version: `{summary.dataset_version}`",
        f"- Pipeline version (git commit): `{summary.pipeline_version}`",
        f"- HealthBench grading prompt version: `{summary.prompt_version}`",
        f"- RAG metrics prompt version: `{summary.rag_metrics_prompt_version}`",
        f"- Generator model: `{summary.generator_model}`",
        f"- Primary HealthBench grader: `{summary.primary_grader_model}`",
        f"- Independent adjudicator: `{summary.adjudicator_model}`",
        f"- RAG metrics judge model: `{summary.rag_metrics_model}`",
        f"- Run date: `{summary.run_date}`",
        f"- Total cases: {summary.total_cases}",
        "- Routed roles: "
        + ", ".join(f"{role}={count}" for role, count in summary.role_counts.items()),
        "",
    ]

    lines.extend(_healthbench_lines(summary))
    lines.extend(_by_tag_lines(summary))
    lines.extend(_rag_metric_lines(summary))

    if summary.cases_requiring_human_review:
        lines.append("## Cases requiring qualified clinician review")
        lines.append("")
        for case_id in summary.cases_requiring_human_review:
            match = next(
                (cr for cr in case_results if cr.case.case_id == case_id), None
            )
            reasons = []
            if match and match.deterministic and match.deterministic.failure_reasons:
                reasons.extend(match.deterministic.failure_reasons)
            if match and match.adjudication and match.adjudication.agreement is False:
                reasons.append("primary/adjudicator disagreement")
            if match and match.adjudication and match.adjudication.adjudication_error:
                reasons.append("secondary adjudication failed")
            if (
                match
                and match.adjudication
                and any(
                    result.answer_evidence_validated is False
                    for result in match.adjudication.final_grade.rubric_results
                )
            ):
                reasons.append("unvalidated rubric evidence")
            if match and match.rag_metrics and match.rag_metrics.evaluation_error:
                reasons.append("RAG metric judge error")
            if not reasons:
                reasons.append("low confidence or low monitored RAG metric")
            reason = ", ".join(reasons)
            lines.append(f"- `{case_id}` -- {reason}")
        lines.append("")

    if summary.notes:
        lines.append("## Notes")
        lines.append("")
        for note in summary.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def write_report(
    case_results: List[CaseResult],
    config: EvalConfig,
    dataset_version: str,
    run_id: str,
    prompt_version: str | None = None,
    run_date: str | None = None,
) -> Tuple[Path, Path, Path]:
    """Writes raw results (full detail, gitignored) plus a sanitised JSON and
    Markdown summary. Returns (raw_path, summary_json_path, summary_md_path).
    """
    output_root = Path(config.output_path)
    raw_dir = output_root / "raw" / run_id
    reports_dir = output_root / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / "cases.jsonl"
    with open(raw_path, "w", encoding="utf-8") as fh:
        for case_result in case_results:
            fh.write(case_result.model_dump_json() + "\n")

    summary = build_report_summary(
        case_results,
        config,
        dataset_version,
        prompt_version=prompt_version,
        run_date=run_date,
    )

    summary_json_path = reports_dir / f"{run_id}_summary.json"
    sanitized = {
        "label": REPORT_LABEL,
        "summary": json.loads(summary.model_dump_json()),
        "cases": [_sanitized_case_entry(cr) for cr in case_results],
    }
    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump(sanitized, fh, indent=2)

    summary_md_path = reports_dir / f"{run_id}_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown_report(summary, case_results))

    return raw_path, summary_json_path, summary_md_path
