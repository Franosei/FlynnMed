# Automated HealthBench and RAG evaluation -- not clinical validation

This report is produced by an automated benchmark harness. It is **not** a
clinical validation of FlynnMed and must not be represented as one. Severe,
uncertain, or disputed cases are listed below and require review by a
qualified clinician before any conclusion is drawn from them.

## Run metadata

- Dataset version: `healthbench`
- Pipeline version (git commit): `7b96d2ce6855e195d34450b406d79bbc4b1d9f65`
- HealthBench grading prompt version: `healthbench-rubric-v4`
- RAG metrics prompt version: `rag-claim-audit-v4`
- Generator model: `gpt-5.4-mini`
- Primary HealthBench grader: `gpt-5.6-luna`
- Independent adjudicator: `gpt-4o-mini`
- RAG metrics judge model: `gpt-5.6-luna`
- Run date: `2026-08-25T19:09:20.141845+00:00`
- Total cases: 1
- Routed roles: nurse=1
- Response time: n/a

## HealthBench rubric scoring

The weighted score is computed locally from the physician-authored rubric points. The graders classify each exact rubric against the captured FlynnMed answer; they do not generate or compare against their own answer.

- Graded cases: 1/1
- Pass rate: 100.0%
- Weighted HealthBench score: 0.800
Overall triage rates below use model-judge urgency inference and remain audit proxies. Emergency recognition uses the dataset's physician-agreed emergent category only.

- Judge-inferred under-triage rate: 0.0%
- Judge-inferred severe under-triage rate: 0.0%
- Physician-labelled emergency recognition sensitivity: n/a
- Conditionally emergent cases: 0
- Physician-labelled non-emergent cases: 0
- Secondary adjudication rate: 0.0%
- Primary/adjudicator disagreements: 0

## Pipeline short-circuit rates

On this curated dataset of legitimate clinical questions (not adversarial red-team prompts), a moderation block or an insufficient-evidence refusal is presumptively a false positive and should be reviewed -- both preempt HealthBench grading entirely, so these cases are excluded from the HealthBench section above.

- Moderation blocks: 0/1 (0.0%)
- Insufficient-evidence refusals: 0/1 (0.0%)

## HealthBench scoring by tag

A case can carry more than one tag, so rows do not sum to the overall totals.

| Tag | Graded cases | Pass rate | Weighted score | Judge-inferred under-triage | Judge-inferred severe under-triage |
|---|---|---|---|---|---|
| `theme:test` | 1/1 | 100.0% | 0.800 | 0.0% | 0.0% |

## Tiered RAG quality metrics

Document split: 1 relevant, 1 irrelevant/distractor. Relevance is judged from stored excerpts before dependent metrics.

Link integrity is reported separately from clinical citation grounding. A working URL does not prove that its excerpt supports a claim.
- Related source URLs: 1/2 valid (50.0%)
- Rendered citation targets: 1/1 resolved to their stored source (100.0%)

### Tier 1 - Core

- Faithfulness / groundedness: 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)
- Context relevance (precision): 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)
- Noise robustness (1 = no contamination): 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)
- Context recall (coverage): 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)
- Answer correctness vs. gold: n/a (no applicable cases)
- Calibration / appropriate hedging: 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)

### Tier 2 - Important

- Contradiction / conflict handling: n/a (no applicable cases)
- Citation accuracy (attached claim only): 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)
- Citation completeness (material-claim coverage): n/a (no applicable cases)
- Context precision (ranking nDCG): 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)

### Tier 3 - Periodic safety monitoring

- Clinical harmlessness: 0.800 (1/1 applicable cases; 1 assessed items; PROVISIONAL - insufficient sample)
- Consistency / reproducibility: n/a (no applicable cases)

## Notes

- Overall under-triage rates remain model-judge audit proxies. Emergency recognition sensitivity uses only HealthBench's physician-agreed emergent category; conditionally emergent cases are reported separately.
