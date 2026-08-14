import json
from pathlib import Path

from evaluations.trends import load_snapshots, render_markdown

_LABEL = "Automated HealthBench and RAG evaluation -- not clinical validation"


def _write_summary(
    reports_dir: Path,
    run_id: str,
    run_date: str,
    dataset_version: str = "healthbench",
    pipeline_version: str = "abcdef1234567890",
    total_cases: int = 10,
    pass_rate: float | None = 0.8,
    moderation_block_rate: float | None = 0.0,
    faithfulness: float | None = 0.9,
    median_duration_seconds: float | None = 25.0,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_version": dataset_version,
        "pipeline_version": pipeline_version,
        "run_date": run_date,
        "total_cases": total_cases,
        "pass_rate": pass_rate,
        "weighted_healthbench_score": pass_rate,
        "severe_under_triage_rate": 0.0,
        "moderation_block_rate": moderation_block_rate,
        "insufficient_evidence_rate": 0.0,
        "median_duration_seconds": median_duration_seconds,
        "p95_duration_seconds": median_duration_seconds,
        "rag_metric_aggregates": {
            "faithfulness": {"average_score": faithfulness},
            "context_relevance": {"average_score": 0.85},
            "answer_correctness": {"average_score": None},
            "calibration": {"average_score": 0.7},
        },
    }
    payload = {"label": _LABEL, "summary": summary, "cases": []}
    (reports_dir / f"{run_id}_summary.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_load_snapshots_sorts_chronologically_by_run_date(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_summary(reports_dir, "run-b", "2026-08-10T00:00:00+00:00", pass_rate=0.9)
    _write_summary(reports_dir, "run-a", "2026-08-01T00:00:00+00:00", pass_rate=0.7)

    snapshots = load_snapshots(reports_dir)

    assert [snap.run_id for snap in snapshots] == ["run-a", "run-b"]
    assert snapshots[0].metrics["pass_rate"] == 0.7
    assert snapshots[1].metrics["pass_rate"] == 0.9


def test_load_snapshots_skips_empty_and_malformed_files(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    # A --dry-run/zero-case artifact must not appear as a trend point.
    _write_summary(reports_dir, "empty-run", "2026-08-01T00:00:00+00:00", total_cases=0)
    (reports_dir / "corrupt_summary.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    _write_summary(reports_dir, "real-run", "2026-08-02T00:00:00+00:00")

    snapshots = load_snapshots(reports_dir)

    assert [snap.run_id for snap in snapshots] == ["real-run"]


def test_load_snapshots_filters_by_exact_dataset_version(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_summary(
        reports_dir,
        "hb-run",
        "2026-08-01T00:00:00+00:00",
        dataset_version="healthbench",
    )
    _write_summary(
        reports_dir,
        "hb-hard-run",
        "2026-08-02T00:00:00+00:00",
        dataset_version="healthbench_hard",
    )

    snapshots = load_snapshots(reports_dir, dataset="healthbench")

    # Exact match only -- "healthbench_hard" must not match a "healthbench" filter.
    assert [snap.run_id for snap in snapshots] == ["hb-run"]


def test_render_markdown_shows_delta_between_consecutive_runs(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_summary(reports_dir, "run-a", "2026-08-01T00:00:00+00:00", pass_rate=0.7)
    _write_summary(reports_dir, "run-b", "2026-08-10T00:00:00+00:00", pass_rate=0.9)
    snapshots = load_snapshots(reports_dir)

    markdown = render_markdown(snapshots)

    assert "run-a" in markdown and "run-b" in markdown
    assert "0.700" in markdown
    # Second row must show an upward delta vs. the first run's 0.700.
    assert "0.900 (▲ +0.200)" in markdown


def test_render_markdown_marks_missing_metric_as_na(tmp_path):
    reports_dir = tmp_path / "reports"
    _write_summary(reports_dir, "run-a", "2026-08-01T00:00:00+00:00", pass_rate=None)
    snapshots = load_snapshots(reports_dir)

    markdown = render_markdown(snapshots)

    assert "n/a" in markdown


def test_render_markdown_handles_no_runs():
    markdown = render_markdown([])
    assert "No completed evaluation reports were found" in markdown
