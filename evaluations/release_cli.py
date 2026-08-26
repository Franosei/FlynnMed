"""Command-line interface for release_pipeline.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluations.release_pipeline import (
    audit_results,
    build_evidence_label_queue,
    build_blinded_review_package,
    build_clinical_review_package,
    build_scorecard,
    code_snapshot,
    compare_runs,
    create_manifest,
    iter_jsonl,
    merge_evaluation_checkpoints,
    render_model_card,
    summarize_shadow_events,
    summarize_clinical_reviews,
    verify_manifest,
    write_json,
    write_jsonl,
)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlynnMed pilot release governance")
    commands = parser.add_subparsers(dest="command", required=True)

    manifest_parser = commands.add_parser("manifest")
    manifest_parser.add_argument("--input", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--seed", default="flynnmed-2026-pilot-v1")
    manifest_parser.add_argument("--exposure", choices=("previously_evaluated", "unseen"), default="previously_evaluated")
    manifest_parser.add_argument("--generator-model")

    verify_parser = commands.add_parser("verify-manifest")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)

    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--input", type=Path, required=True)
    audit_parser.add_argument("--manifest", type=Path)
    audit_parser.add_argument("--partition")
    audit_parser.add_argument("--generator-model")
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.add_argument("--clinician-review-completion", type=float, default=0.0)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--run", action="append", required=True, help="NAME=path/to/cases.jsonl")
    compare_parser.add_argument("--output", type=Path, required=True)

    blind_parser = commands.add_parser("blind-review")
    blind_parser.add_argument("--run-a", type=Path, required=True)
    blind_parser.add_argument("--run-b", type=Path, required=True)
    blind_parser.add_argument("--output", type=Path, required=True)
    blind_parser.add_argument("--mapping-output", type=Path, required=True)
    blind_parser.add_argument("--seed", type=int, default=96845348303)

    clinical_review_parser = commands.add_parser("clinical-review")
    clinical_review_parser.add_argument("--input", type=Path, required=True)
    clinical_review_parser.add_argument("--output", type=Path, required=True)
    clinical_review_parser.add_argument("--all-cases", action="store_true")

    review_summary_parser = commands.add_parser("review-summary")
    review_summary_parser.add_argument("--input", type=Path, required=True)
    review_summary_parser.add_argument("--output", type=Path, required=True)

    shadow_parser = commands.add_parser("shadow-summary")
    shadow_parser.add_argument("--input", type=Path, required=True)
    shadow_parser.add_argument("--output", type=Path, required=True)

    snapshot_parser = commands.add_parser("snapshot")
    snapshot_parser.add_argument("--root", type=Path, default=Path.cwd())
    snapshot_parser.add_argument("--output", type=Path, required=True)

    merge_parser = commands.add_parser("merge-checkpoints")
    merge_parser.add_argument("--generation", type=Path, required=True)
    merge_parser.add_argument("--healthbench", type=Path, required=True)
    merge_parser.add_argument("--rag", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--status-output", type=Path, required=True)
    merge_parser.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "manifest":
        write_json(args.output, create_manifest(args.input, seed=args.seed, exposure=args.exposure, expected_generator_model=args.generator_model))
    elif args.command == "verify-manifest":
        errors = verify_manifest(args.input, _load_json(args.manifest))
        if errors:
            print("\n".join(errors))
            return 1
        print("Manifest verification passed.")
    elif args.command == "audit":
        manifest = _load_json(args.manifest) if args.manifest else None
        model = args.generator_model or (manifest or {}).get("expected_generator_model")
        rows, summary = audit_results(list(iter_jsonl(args.input)), expected_generator_model=model, partition=args.partition, manifest=manifest)
        selected_records = list(iter_jsonl(args.input))
        if args.partition and manifest:
            selected_ids = {
                item["case_id"]
                for item in manifest.get("cases", [])
                if item.get("partition") == args.partition
            }
            selected_records = [record for record in selected_records if (record.get("case") or {}).get("case_id") in selected_ids]
        document_labels, claim_labels = build_evidence_label_queue(selected_records)
        scorecard = build_scorecard(summary, manifest, clinician_review_completion=args.clinician_review_completion)
        snapshot = code_snapshot(Path.cwd())
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output_dir / "stage_failures.jsonl", rows)
        write_jsonl(
            args.output_dir / "priority_triage_review.jsonl",
            (
                row
                for row in rows
                if {"severe_under_triage", "physician_emergent_not_recognised", "unclear_triage"}
                & set(row["failure_codes"])
            ),
        )
        write_jsonl(args.output_dir / "retrieval_relevance_labels.jsonl", document_labels)
        write_jsonl(args.output_dir / "citation_entailment_labels.jsonl", claim_labels)
        write_json(args.output_dir / "stage_metrics.json", summary)
        write_json(args.output_dir / "subgroup_metrics.json", {"subgroups": summary["subgroups"], "by_tag": summary["by_tag"]})
        write_json(args.output_dir / "release_scorecard.json", scorecard)
        write_json(args.output_dir / "code_snapshot.json", snapshot)
        (args.output_dir / "MODEL_CARD.md").write_text(render_model_card(snapshot, manifest or {}, summary, scorecard), encoding="utf-8")
        print(f"{scorecard['status']}: {len(scorecard['failed_gates'])} release gate(s) failed.")
    elif args.command == "compare":
        runs = {}
        for spec in args.run:
            name, separator, raw_path = spec.partition("=")
            if not separator or not name or not raw_path:
                raise ValueError("Each --run must be NAME=path.")
            runs[name] = list(iter_jsonl(Path(raw_path)))
        write_json(args.output, compare_runs(runs))
    elif args.command == "blind-review":
        package, mapping = build_blinded_review_package(list(iter_jsonl(args.run_a)), list(iter_jsonl(args.run_b)), seed=args.seed)
        write_jsonl(args.output, package)
        write_json(args.mapping_output, mapping)
    elif args.command == "clinical-review":
        package = build_clinical_review_package(
            list(iter_jsonl(args.input)), priority_only=not args.all_cases
        )
        write_jsonl(args.output, package)
    elif args.command == "review-summary":
        write_json(args.output, summarize_clinical_reviews(list(iter_jsonl(args.input))))
    elif args.command == "shadow-summary":
        write_json(args.output, summarize_shadow_events(list(iter_jsonl(args.input))))
    elif args.command == "snapshot":
        write_json(args.output, code_snapshot(args.root))
    elif args.command == "merge-checkpoints":
        merged, status = merge_evaluation_checkpoints(
            list(iter_jsonl(args.generation)),
            list(iter_jsonl(args.healthbench)),
            list(iter_jsonl(args.rag)),
            require_complete=not args.allow_partial,
        )
        write_jsonl(args.output, merged)
        write_json(args.status_output, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
