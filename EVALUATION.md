# FlynnMed evaluation solution

FlynnMed uses a production-aligned evaluation system to test clinical response
quality, evidence grounding, safety behaviour and release readiness. It runs the
real application pipeline rather than grading isolated prompts. The results are
engineering evidence, not clinical validation or regulatory approval.

## How the solution works

```text
Versioned evaluation case
  -> production role detection and RAG pipeline
  -> captured answer, sources, trace and timing
  -> deterministic safety and citation checks
  -> HealthBench rubric grading
  -> claim-level RAG and calibration grading
  -> independent adjudication for higher-risk disagreements
  -> sanitised report, review queues and fail-closed release gates
```

The solution combines complementary controls:

1. **Production-path execution** sends each case through
   `RAGEngine.handle_user_question`, including retrieval, evidence ranking,
   policy checks and role-aware response generation.
2. **Deterministic checks** detect missed crisis escalation, unsafe refusal
   behaviour, unsupported record claims, citation mismatches and prompt
   injection failures without relying on another language model.
3. **HealthBench grading** applies the original physician-authored rubrics to
   the captured answer. Rubric points are calculated locally, and rewritten or
   incomplete rubric criteria are rejected.
4. **Tiered RAG grading** evaluates relevance, groundedness, recall,
   correctness, calibration, contradiction handling, citation accuracy,
   harmlessness and consistency at document and claim level.
5. **Independent adjudication** is required for moderate or severe harm and
   selected high-risk disagreements. A missing required adjudication cannot
   silently pass.
6. **Release controls** combine locked manifests, blinded clinician review,
   safety thresholds, evidence-quality thresholds, latency limits and shadow
   monitoring. Missing evidence fails closed.

## Interpreting the output

- `1.0` is best and `0.0` is worst for tiered quality metrics.
- A metric without the required input is reported as `n/a`; it is not converted
  to a zero or hidden in an aggregate.
- Small samples are explicitly labelled provisional.
- A working citation link does not prove that its source supports the attached
  clinical claim, so link resolution and citation entailment are measured
  separately.
- Automated judge scores identify review priorities. They do not replace
  qualified clinical review.

Generated datasets, raw responses and reports remain local under
`evaluations/results/` and are excluded from version control because they can be
large or contain restricted evaluation material. Versioned manifests, policies,
schemas and critical-presentation cases remain in the repository.

## Current evaluation decision

The current recorded decision is **NOT_READY**. The existing 500-case run is a
retrospective engineering baseline, not an untouched release test. It exposed
emergency-recognition gaps, evidence failures, incomplete external grading and
latency above the candidate threshold. FlynnMed must complete independent
grading and clinician adjudication, then pass a new locked unseen test and
shadow-mode gates before the candidate can be considered ready.

See the [500-case baseline analysis](evaluations/release/BASELINE_ANALYSIS.md)
for the evidence, limitations, implemented controls and outstanding gates.

## Run the evaluation

Install the application dependencies and configure the evaluation environment
described in the [evaluation harness guide](evaluations/README.md). Then run:

```powershell
# Validate configuration and dataset structure without model calls
py -m evaluations.runner --dataset healthbench --dry-run

# Run a reproducible sample through the production pipeline
py -m evaluations.runner --dataset healthbench --sample 10 --random-seed 20260714

# Run deterministic harness tests
py -m pytest evaluations/tests -q
```

Model-backed runs require an evaluation API key and can consume paid model
usage. The harness performs evaluator access checks before generation begins.

## Evaluation documentation

| Document or directory | Purpose |
|---|---|
| [Harness guide](evaluations/README.md) | Configuration, metrics, commands, outputs and limitations |
| [Judge contract](evaluations/JUDGE_PROMPT.md) | Grading constraints and output requirements |
| [Baseline analysis](evaluations/release/BASELINE_ANALYSIS.md) | Current findings and release decision |
| [`evaluations/manifests/`](evaluations/manifests/) | Reproducible case selection and data fingerprints |
| [`evaluations/release/`](evaluations/release/) | Release thresholds, schemas and locked critical cases |
| [`evaluations/tests/`](evaluations/tests/) | Deterministic regression coverage for the evaluation system |
