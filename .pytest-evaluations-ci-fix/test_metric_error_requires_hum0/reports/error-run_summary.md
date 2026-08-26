# Automated HealthBench and RAG evaluation -- not clinical validation

This report is produced by an automated benchmark harness. It is **not** a
clinical validation of FlynnMed and must not be represented as one. Severe,
uncertain, or disputed cases are listed below and require review by a
qualified clinician before any conclusion is drawn from them.

## Run metadata

- Dataset version: `healthbench`
- Pipeline version (git commit): `c2c3dbc28f3f3a937a8423a63facc3c4d742d5e3`
- HealthBench grading prompt version: `healthbench-rubric-v4`
- RAG metrics prompt version: `rag-claim-audit-v4`
- Generator model: `gpt-5.4-mini`
- Primary HealthBench grader: `gpt-5.6-luna`
- Independent adjudicator: `gpt-4o-mini`
- RAG metrics judge model: `gpt-5.6-luna`
- Run date: `2026-08-26T06:07:55.776211+00:00`
- Total cases: 1
- Routed roles: patient=1
- Response time: n/a

## HealthBench rubric scoring

No completed HealthBench rubric grades were available.

## Pipeline short-circuit rates

On this curated dataset of legitimate clinical questions (not adversarial red-team prompts), a moderation block or an insufficient-evidence refusal is presumptively a false positive and should be reviewed -- both preempt HealthBench grading entirely, so these cases are excluded from the HealthBench section above.

- Moderation blocks: 0/1 (0.0%)
- Insufficient-evidence refusals: 0/1 (0.0%)

## HealthBench scoring by tag

No tagged cases had a completed HealthBench grade.

## Tiered RAG quality metrics

Document split: 1 relevant, 1 irrelevant/distractor. Relevance is judged from stored excerpts before dependent metrics.

Link integrity is reported separately from clinical citation grounding. A working URL does not prove that its excerpt supports a claim.
- Related source URLs: n/a (no retrieved sources)
- Rendered citation targets: n/a (no rendered citations)

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

RAG metric judge errors: 1 case(s). These cases are excluded from metric denominators, not scored zero.

## Cases requiring qualified clinician review

- `case-1` -- RAG metric judge error

## Notes

- Overall under-triage rates remain model-judge audit proxies. Emergency recognition sensitivity uses only HealthBench's physician-agreed emergent category; conditionally emergent cases are reported separately.
