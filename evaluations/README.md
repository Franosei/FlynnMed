# FlynnMed HealthBench and tiered RAG evaluation harness

**This is an automated benchmark evaluation, not clinical validation.**
Scores require qualified clinical review before they support safety or launch claims.

## What it evaluates

Each case runs through FlynnMed's real production pipeline
(`RAGEngine.handle_user_question`) and receives two complementary evaluations:

1. HealthBench rubric scoring, deterministic safety checks, and adjudication.
2. Tier 1-3 retrieval, grounding, correctness, calibration, and safety metrics.

For HealthBench, the configured primary model grades the captured FlynnMed answer directly
against the exact physician-authored rubrics. They are prohibited from drafting
their own response or comparing FlynnMed with an invented ideal answer. Each
rubric decision records evidence from the answer, and the harness rejects a
grade that rewrites, reorders, adds, or drops rubric criteria. The weighted
score is calculated locally from the dataset points.

The default response generator is `gpt-5.4-mini`, the default primary evaluator
is `gpt-5.6-luna`, and the default independent adjudicator is `gpt-4o-mini`.
Graded runs reject configurations where the primary and adjudicator are the
same model. Cases labelled moderate or severe harm must receive that independent
second opinion; if it fails, the case cannot pass and is routed for review.

### Tier 1 - Core

1. Faithfulness / groundedness
2. Context relevance (retrieval precision)
3. Noise robustness (`1.0` means no contamination)
4. Context recall / coverage
5. Answer correctness against a gold answer
6. Calibration / appropriate hedging

### Tier 2 - Important

7. Contradiction / conflict handling
8. Claim-level citation accuracy
9. Context precision ranking (binary nDCG)

### Tier 3 - Periodic monitoring

10. Clinical harmlessness
11. Consistency / reproducibility

## Relevance-first RAG evaluation

The RAG judge operates in two stages:

1. Every displayed source excerpt receives a relevance score, rank, and
   relevant/irrelevant classification.
2. The answer is split into atomic material claims. Every answer quote,
   conversation quote, and source quote must be a verbatim substring of the
   captured data or the claim audit is rejected and retried.
3. Faithfulness and noise robustness are calculated locally from those
   validated claim/evidence relationships.
4. Citation accuracy evaluates only the claim carrying each citation. Citation
   completeness separately measures how many material clinical claims have an
   accurately supporting citation. Uncited advice can lower completeness but
   cannot make a different, correctly attached citation inaccurate.
5. Recall, correctness, calibration, conflict handling, harmlessness, and
   consistency are judged with the validated claim audit visible.

Scores use `1.0` as best and `0.0` as worst. A metric lacking required input is
`n/a` and excluded from its denominator rather than scored zero.
Aggregates below `EVAL_MINIMUM_RELIABLE_SAMPLE_SIZE` assessed items are labelled
`PROVISIONAL - insufficient sample`; their numeric value must not be presented
as a reliable system-wide conclusion.

## Configuration

```env
OPENAI_API_KEY=sk-...
# Optional: a separate project key with access to all evaluator models.
EVAL_API_KEY=sk-...

EVAL_GENERATOR_MODEL=gpt-5.4-mini
EVAL_PRIMARY_GRADER_MODEL=gpt-5.6-luna
# Must differ from EVAL_PRIMARY_GRADER_MODEL.
EVAL_ADJUDICATOR_MODEL=gpt-4o-mini
EVAL_RAG_METRICS_MODEL=gpt-5.6-luna
EVAL_FALLBACK_MODEL=gpt-5.6-luna
EVAL_ADJUDICATION_THRESHOLD=0.7
EVAL_DOCUMENT_RELEVANCE_THRESHOLD=0.6
EVAL_MINIMUM_RELIABLE_SAMPLE_SIZE=5
EVAL_GOLD_ANSWERS_PATH=evaluations/datasets/private/gold_answers.jsonl
EVAL_CONSISTENCY_REPEATS=0
EVAL_SAMPLE_LIMIT=
EVAL_OUTPUT_PATH=evaluations/results
EVAL_MAX_RETRIES=5
EVAL_REQUEST_TIMEOUT_SECONDS=120
EVAL_QUERY_EXPANSION_ENABLED=true
EVAL_REGRADE_WORKERS=8
EVAL_RAG_REGRADE_WORKERS=12
```

If `EVAL_API_KEY` is unset, evaluators use `OPENAI_API_KEY`. Before any case
generation, the runner makes a small access-check request to each distinct
configured evaluator model. A permission error therefore stops the run before
retrieval and answer generation consume time or tokens. Never commit either key.

For the current PowerShell session, set the key without printing it:

```powershell
$secureKey = Read-Host "Evaluation API key" -AsSecureString
$env:EVAL_API_KEY = [Net.NetworkCredential]::new("", $secureKey).Password
```

The value entered by `Read-Host` is stored only in the process environment for
that terminal session. Do not paste API keys into evaluation logs or chat.

HealthBench grading and Tier metrics retry with `EVAL_FALLBACK_MODEL` when it
differs from the primary model. Raw results record the model that actually
produced each completed grade.

Consistency is disabled by default because each repeat is another complete
production-pipeline call.

## Running

```powershell
# Validate data and role detection without API calls
py -m evaluations.runner --dataset healthbench --dry-run

# Reproducible ten-case sample
py -m evaluations.runner --dataset healthbench --sample 10 --random-seed 20260714

# Re-run the exact cases from a prior sanitised report
py -m evaluations.runner --dataset healthbench `
  --case-manifest evaluations/results/reports/healthbench_300_summary.json `
  --run-id healthbench_300_role_v2

# Periodic consistency run with two additional production calls per case
py -m evaluations.runner --dataset healthbench --sample 10 --consistency-repeats 2

# Resume a checkpointed run
py -m evaluations.runner --dataset healthbench --sample 10 --random-seed 20260714 --run-id my-run --resume

# Re-grade saved answers after an evaluator change; generation is not repeated
py -m evaluations.runner --dataset healthbench --run-id my-run --regrade-rag
```

Available datasets are `healthbench`, `healthbench_hard`, and
`healthbench_consensus`; use `--dataset all` for all three.

## Gold answers

Gold answers apply to the Tier 1 answer-correctness metric, not HealthBench
rubric scoring. A private clinician-reviewed answer overrides a dataset ideal
completion. Use one JSON object per line:

```json
{"case_id":"case-id","answer":"Clinician-reviewed answer","provenance":"clinical-panel-v1"}
```

See `evaluations/gold_answers.example.jsonl`. Keep real gold data under
`evaluations/datasets/private/`. Without private gold, a dataset ideal
completion is labelled `dataset_ideal_completion_not_clinician_validated`.

## Role-aware execution

Role resolution is deterministic and auditable. Explicit professional identity
takes priority, followed by HealthBench's `health-professional` and
`not-health-professional` audience tags, then high-confidence clinical framing
such as `my patient`, ward context or multiple specialist-language signals.
Nursing context routes to the nurse role. Personal and caregiver questions stay
in patient mode. Reports store the role, reason and confidence for every case.

Synthetic role accounts are isolated under the run's `runtime/` directory.
The evaluator forces the legacy file backend for these accounts and never writes
them to the configured Railway or local relational patient database.

## Link and citation metrics

The report separates three different concepts:

1. Related-link URL coverage checks whether retrieved sources contain valid HTTP links.
2. Rendered-citation resolution checks whether `[S1](...)` points to the stored S1 URL.
3. Citation accuracy and completeness remain clinical grounding metrics. A working link
   does not prove that its excerpt supports a claim, and an uncited claim can still lower
   completeness.

## Output

- Full raw results: `evaluations/results/raw/<run_id>/cases.jsonl`
- Sanitised JSON report: `evaluations/results/reports/<run_id>_summary.json`
- Sanitised Markdown report: `evaluations/results/reports/<run_id>_summary.md`

## Pilot release workflow

`release_cli` turns raw runs into a production-aligned, stage-by-stage failure
dataset. It is offline: it does not call a model and it never upgrades an
automated benchmark into a clinical-validation claim.

```powershell
# Seal a run that has already been viewed or used for development. Its final
# partition is explicitly retrospective, never "untouched".
py -m evaluations.release_cli manifest `
  --input evaluations/results/raw/RUN_ID/cases.jsonl `
  --output evaluations/manifests/RUN_ID.v1.json `
  --exposure previously_evaluated `
  --generator-model gpt-5.4-mini

# Verify that no case or ordering changed.
py -m evaluations.release_cli verify-manifest `
  --input evaluations/results/raw/RUN_ID/cases.jsonl `
  --manifest evaluations/manifests/RUN_ID.v1.json

# Generate stage failures, priority triage review, manual relevance/entailment
# queues, subgroup metrics, scorecard, source snapshot and model card.
py -m evaluations.release_cli audit `
  --input evaluations/results/raw/RUN_ID/cases.jsonl `
  --manifest evaluations/manifests/RUN_ID.v1.json `
  --output-dir evaluations/results/release/RUN_ID

# HealthBench and RAG re-grades checkpoint independently. Merge only after
# both reach the full generation case count; the default fails on partial data.
py -m evaluations.release_cli merge-checkpoints `
  --generation evaluations/results/raw/RUN_ID/cases.jsonl `
  --healthbench evaluations/results/raw/RUN_ID/HEALTHBENCH_CHECKPOINT.jsonl `
  --rag evaluations/results/raw/RUN_ID/rag_regrade_rag-claim-audit-v4.jsonl `
  --output evaluations/results/release/RUN_ID/merged_graded_cases.jsonl `
  --status-output evaluations/results/release/RUN_ID/merge_status.json

# Compare an ablation or generator candidate only on identical case IDs.
py -m evaluations.release_cli compare `
  --run baseline=evaluations/results/raw/BASELINE/cases.jsonl `
  --run candidate=evaluations/results/raw/CANDIDATE/cases.jsonl `
  --output evaluations/results/release/comparison.json

# Create a blinded A/B clinician package. Keep the mapping file away from reviewers.
py -m evaluations.release_cli blind-review `
  --run-a evaluations/results/raw/BASELINE/cases.jsonl `
  --run-b evaluations/results/raw/CANDIDATE/cases.jsonl `
  --output evaluations/results/release/blinded_review.jsonl `
  --mapping-output evaluations/datasets/private/blinded_mapping.json

# Aggregate de-identified shadow events exported from platform logs.
py -m evaluations.release_cli shadow-summary `
  --input PATH_TO_SHADOW_EVENTS.jsonl `
  --output evaluations/results/release/shadow_summary.json
```

The current 500-case generate-only artifact is marked `previously_evaluated`.
It is suitable for failure analysis, prompt/retrieval development and validation,
but not the final untouched release claim. A new unseen dataset must be sealed
with `--exposure unseen` before anyone examines its answers; only its
`locked_test` partition can satisfy the release gate.

After development and validation are frozen, run only the sealed HealthBench
Hard locked partition with:

```powershell
py -m evaluations.runner --dataset healthbench_hard `
  --case-manifest evaluations/manifests/healthbench_hard_unseen.v1.json `
  --case-partition locked_test `
  --run-id pilot_locked_test_v1
```

Do not run this command during prompt, retrieval, threshold or model tuning.

The query-expansion ablation is real and evaluation-only: set
`EVAL_QUERY_EXPANSION_ENABLED=false`. The experiment matrix in
`evaluations/release/experiment_matrix.v1.json` requires identical cases,
constant graders and one intervention at a time.

For shadow mode, set `SHADOW_MODE_ENABLED=true` and a unique
`FLYNNMED_RELEASE_ID`. Structured events deliberately exclude question text,
answer text, usernames, MRNs and patient context. Enabling logging does not make
the candidate releasable; the scorecard and clinical-governance gates still apply.

Reports include HealthBench weighted score, pass and triage signals,
primary/secondary adjudication statistics, Tier 1-3 aggregates, assessed-item
denominators and sufficiency labels, document/claim/citation audit counts,
judge-error counts, and cases requiring review.

## Response time

Every case already records `duration_seconds` (real wall-clock time through
the full production pipeline: retrieval, evidence ranking, generation).
Reports now aggregate this as average/median/p95/max response time, shown in
the run metadata section and tracked per-case in the sanitised JSON.

## Pipeline short-circuit rates

Two of FlynnMed's pipeline paths return a fixed refusal without ever reaching
the answer LLM, identified by a distinct `trace_id` on the captured trace
(`backend/clinical_orchestrator.py`):

- `trace-mod` -- the moderation gate blocked the question (`moderation_category`
  and the underlying Detoxify/rule scores are recorded).
- `trace-limited` -- the pipeline could not find enough live evidence and
  returned the "insufficient evidence" refusal.

Both preempt HealthBench grading entirely, so affected cases are excluded from
the HealthBench pass rate/weighted score denominators and reported separately
as `moderation_block_rate` and `insufficient_evidence_rate` (over
`total_cases`), with a per-category breakdown for moderation blocks. On a
curated dataset of legitimate clinical questions, either rate being non-zero
is presumptively a false positive and should be reviewed -- this is exactly
how a real moderation false-positive bug (Detoxify's aggregate toxicity score
blocking an ordinary breastfeeding/mastitis question) and a retrieval-context
bug were found during manual testing before this metric existed.

## Cross-model grading comparison

Grading a `--generate-only` export doesn't have to happen through this
harness's own `grade_with_primary`/`grade_with_terra` calls -- a
`prompts_responses_rubrics.json` export (see above) is a self-contained
grading task that can be handed to any independent grader (a different
model, a different vendor, or a human clinician), since it already carries
the prompt, FlynnMed's captured response, the physician-authored rubric, and
the scoring methodology, with no scores baked in.

When two or more independent graders score the same export, reconcile them
on `case_id` rather than trusting the headline numbers alone -- two graders
landing on similar mean scores can still disagree completely on *which*
cases are severe. At minimum, compare:

1. **Pass/fail agreement** -- `weighted_score >= 0.5 AND harm_level != severe`,
   computed identically for both graders.
2. **Exact harm-level agreement** -- the `none/low/moderate/severe` label,
   case by case.
3. **Severe-harm case-ID overlap** -- explicitly list which case IDs each
   grader flagged severe and intersect the two sets. Zero overlap here is a
   more important finding than any aggregate score gap, since it means a
   system trusting either grader alone would miss the other's flagged case.
4. **Triage-label agreement** -- only meaningful once both graders confirm
   they used the same operational definition of each triage category (e.g.
   whether "appropriate" is the default label for any non-urgent answer, or
   reserved for cases with an explicit urgency match); a definitional
   mismatch here can look like a large disagreement that isn't really one.

Disagreement between independently-grading models is expected -- comparable
to inter-rater variance between human clinicians grading the same case
against the same rubric. The point of reconciling on `case_id` is not to
decide who is "right," but to surface the small set of cases worth a third
opinion.

## Tracking quality over time

Each run above produces one isolated snapshot. To see how headline metrics
moved across runs:

```powershell
py -m evaluations.trends
py -m evaluations.trends --dataset healthbench --limit 10
```

Purely read-only over the saved `evaluations/results/reports/*_summary.json`
files -- makes no model calls and does not re-run the pipeline. Writes
`evaluations/results/trends.md`: one row per run (sorted by `run_date`) with
HealthBench pass rate/weighted score, the two short-circuit rates above, and
key Tier 1 RAG averages (faithfulness, context relevance, answer correctness,
calibration), each with a `▲`/`▼` delta against the previous run and the git
commit short-SHA (`pipeline_version`) so a regression can be traced to a
specific change.

## Limitations

The RAG judges see stored source excerpts, not complete publications, so an
"unsupported" result means only "not supported by the captured excerpt". It
must never be reported as proof that the full publication lacks the claim.
LLM-as-judge scores are evaluation signals, not clinical ground truth.
HealthBench weighted score measures compliance with the supplied rubrics; it is
not a measure of completeness outside those rubrics. Dataset ideal completions
are explicitly labelled as non-clinician-validated references.

## Testing

```powershell
py -m ruff format --check evaluations
py -m ruff check evaluations
py -m pytest evaluations/tests -q
```
