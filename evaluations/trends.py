"""Cross-run trend tracking over the evaluation harness's saved reports.

Every `evaluations.runner` run writes an isolated snapshot to
`evaluations/results/reports/<run_id>_summary.json` (see reporting.py's
write_report) with no comparison to prior runs. This module is purely
read-only over those existing JSON files -- no API calls, no re-running the
production pipeline -- so it is safe and free to run after any eval run to
see how headline quality metrics moved.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evaluations.config import EvalConfig, load_config

# (metric_key, label, path-of-keys into the summary dict). Includes both
# HealthBench headline scores and the moderation-block / insufficient-
# evidence rates (see reporting.py's build_report_summary), plus the Tier 1
# RAG metrics most useful for spotting a retrieval or grounding regression.
_TREND_METRICS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("pass_rate", "HealthBench pass rate", ("pass_rate",)),
    (
        "weighted_healthbench_score",
        "Weighted HealthBench score",
        ("weighted_healthbench_score",),
    ),
    (
        "severe_under_triage_rate",
        "Severe under-triage rate",
        ("severe_under_triage_rate",),
    ),
    ("moderation_block_rate", "Moderation block rate", ("moderation_block_rate",)),
    (
        "insufficient_evidence_rate",
        "Insufficient-evidence rate",
        ("insufficient_evidence_rate",),
    ),
    (
        "faithfulness",
        "Faithfulness",
        ("rag_metric_aggregates", "faithfulness", "average_score"),
    ),
    (
        "context_relevance",
        "Context relevance",
        ("rag_metric_aggregates", "context_relevance", "average_score"),
    ),
    (
        "answer_correctness",
        "Answer correctness",
        ("rag_metric_aggregates", "answer_correctness", "average_score"),
    ),
    (
        "calibration",
        "Calibration",
        ("rag_metric_aggregates", "calibration", "average_score"),
    ),
    (
        "median_duration_seconds",
        "Median response time (s)",
        ("median_duration_seconds",),
    ),
    ("p95_duration_seconds", "p95 response time (s)", ("p95_duration_seconds",)),
)


@dataclass
class RunSnapshot:
    run_id: str
    run_date: str
    dataset_version: str
    pipeline_version: str
    total_cases: int
    metrics: dict[str, Optional[float]]


def _walk(summary: dict, path: tuple[str, ...]) -> Optional[float]:
    current: object = summary
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, (int, float)) else None


def load_snapshots(
    reports_dir: Path, dataset: Optional[str] = None
) -> list[RunSnapshot]:
    """Reads every `*_summary.json` in reports_dir, skipping malformed files
    and empty runs (total_cases == 0, e.g. a --dry-run artifact). Sorted
    chronologically by run_date so trend deltas read left-to-right in time."""
    snapshots: list[RunSnapshot] = []
    for path in sorted(reports_dir.glob("*_summary.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = raw.get("summary") or {}
        if not summary or not summary.get("total_cases"):
            continue
        dataset_version = str(summary.get("dataset_version", ""))
        if dataset is not None and dataset_version != dataset:
            continue
        run_id = path.name[: -len("_summary.json")]
        snapshots.append(
            RunSnapshot(
                run_id=run_id,
                run_date=str(summary.get("run_date", "")),
                dataset_version=dataset_version,
                pipeline_version=str(summary.get("pipeline_version", "unknown")),
                total_cases=int(summary.get("total_cases", 0)),
                metrics={key: _walk(summary, path) for key, _, path in _TREND_METRICS},
            )
        )
    snapshots.sort(key=lambda snap: snap.run_date)
    return snapshots


def _delta_cell(previous: Optional[float], current: Optional[float]) -> str:
    if previous is None or current is None:
        return "n/a"
    diff = current - previous
    if abs(diff) < 1e-9:
        return "–"
    arrow = "▲" if diff > 0 else "▼"
    return f"{arrow} {diff:+.3f}"


def render_markdown(snapshots: list[RunSnapshot]) -> str:
    lines = [
        "# FlynnMed evaluation trend report",
        "",
        "Automatically generated from `evaluations/results/reports/*_summary.json`. "
        "Read-only -- this does not re-run the pipeline or call any model, and only "
        "reflects whatever eval runs already exist locally.",
        "",
    ]
    if not snapshots:
        lines.append(
            "No completed evaluation reports were found. Run "
            "`py -m evaluations.runner --dataset healthbench --sample 10` first."
        )
        return "\n".join(lines) + "\n"

    header = ["Run", "Date", "Dataset", "Commit", "Cases"] + [
        label for _, label, _ in _TREND_METRICS
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    previous: Optional[RunSnapshot] = None
    for snap in snapshots:
        commit = (
            snap.pipeline_version[:8]
            if snap.pipeline_version and snap.pipeline_version != "unknown"
            else "unknown"
        )
        row = [
            snap.run_id,
            snap.run_date[:10] if snap.run_date else "unknown",
            snap.dataset_version,
            commit,
            str(snap.total_cases),
        ]
        for key, _, _ in _TREND_METRICS:
            value = snap.metrics.get(key)
            cell = f"{value:.3f}" if value is not None else "n/a"
            if previous is not None:
                cell += f" ({_delta_cell(previous.metrics.get(key), value)})"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
        previous = snap

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize how evaluation quality metrics moved across saved "
            "evaluations.runner reports. Read-only -- makes no model calls."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Only include runs with this exact dataset_version (e.g. healthbench).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only include the N most recent runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the markdown trend report (default: <output_path>/trends.md).",
    )
    args = parser.parse_args(argv)

    config: EvalConfig = load_config()
    reports_dir = Path(config.output_path) / "reports"
    snapshots = load_snapshots(reports_dir, dataset=args.dataset)
    if args.limit:
        snapshots = snapshots[-args.limit :]

    report = render_markdown(snapshots)
    output_path = args.output or (Path(config.output_path) / "trends.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"[trends] {len(snapshots)} run(s) summarized -> {output_path}")


if __name__ == "__main__":
    main()
