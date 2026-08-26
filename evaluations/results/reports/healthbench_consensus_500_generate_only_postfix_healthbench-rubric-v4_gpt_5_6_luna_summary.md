# Automated HealthBench and RAG evaluation -- not clinical validation

This report is produced by an automated benchmark harness. It is **not** a
clinical validation of FlynnMed and must not be represented as one. Severe,
uncertain, or disputed cases are listed below and require review by a
qualified clinician before any conclusion is drawn from them.

## Run metadata

- Dataset version: `healthbench_consensus`
- Pipeline version (git commit): `7b96d2ce6855e195d34450b406d79bbc4b1d9f65`
- HealthBench grading prompt version: `healthbench-rubric-v4`
- RAG metrics prompt version: `rag-claim-audit-v4`
- Generator model: `gpt-5.4-mini`
- Primary HealthBench grader: `gpt-5.6-luna`
- Independent adjudicator: `gpt-4o-mini`
- RAG metrics judge model: `gpt-5.6-luna`
- Run date: `2026-08-25T19:49:43.732976+00:00`
- Total cases: 500
- Routed roles: doctor=3, healthcare_professional=86, nurse=3, patient=408
- Response time: avg 37.8s, median 41.0s, p95 57.7s, max 92.9s

## HealthBench rubric scoring

The weighted score is computed locally from the physician-authored rubric points. The graders classify each exact rubric against the captured FlynnMed answer; they do not generate or compare against their own answer.

- Graded cases: 500/500
- Pass rate: 72.0%
- Weighted HealthBench score: 0.610
Overall triage rates below use model-judge urgency inference and remain audit proxies. Emergency recognition uses the dataset's physician-agreed emergent category only.

- Judge-inferred under-triage rate: 30.4%
- Judge-inferred severe under-triage rate: 5.8%
- Physician-labelled emergency recognition sensitivity: 88.2% (15/17)
- Conditionally emergent cases: 26
- Physician-labelled non-emergent cases: 23
- Secondary adjudication rate: 86.2%
- Primary/adjudicator disagreements: 340

## Pipeline short-circuit rates

On this curated dataset of legitimate clinical questions (not adversarial red-team prompts), a moderation block or an insufficient-evidence refusal is presumptively a false positive and should be reviewed -- both preempt HealthBench grading entirely, so these cases are excluded from the HealthBench section above.

- Moderation blocks: 1/500 (0.2%)
  - By category: `violence_illegal`=1
- Insufficient-evidence refusals: 5/500 (1.0%)

## HealthBench scoring by tag

A case can carry more than one tag, so rows do not sum to the overall totals.

| Tag | Graded cases | Pass rate | Weighted score | Judge-inferred under-triage | Judge-inferred severe under-triage |
|---|---|---|---|---|---|
| `physician_agreed_category:any-reducible-uncertainty` | 35/35 | 88.6% | 0.752 | 20.0% | 0.0% |
| `physician_agreed_category:conditionally-emergent` | 26/26 | 88.5% | 0.827 | 69.2% | 26.9% |
| `physician_agreed_category:context-does-not-matter` | 42/42 | 97.6% | 0.929 | 16.7% | 0.0% |
| `physician_agreed_category:context-matters-but-unclear` | 29/29 | 96.6% | 0.793 | 44.8% | 10.3% |
| `physician_agreed_category:context-matters-is-clear` | 18/18 | 100.0% | 0.889 | 44.4% | 11.1% |
| `physician_agreed_category:detailed` | 18/18 | 33.3% | 0.167 | 44.4% | 16.7% |
| `physician_agreed_category:emergent` | 17/17 | 64.7% | 0.735 | 76.5% | 5.9% |
| `physician_agreed_category:enough-context` | 34/34 | 76.5% | 0.779 | 20.6% | 5.9% |
| `physician_agreed_category:enough-info-to-complete-task` | 30/30 | 63.3% | 0.500 | 30.0% | 0.0% |
| `physician_agreed_category:health-professional` | 43/43 | 7.0% | 0.035 | 30.2% | 14.0% |
| `physician_agreed_category:no-uncertainty` | 33/33 | 87.9% | 0.798 | 18.2% | 0.0% |
| `physician_agreed_category:non-emergent` | 23/23 | 95.7% | 0.696 | 17.4% | 0.0% |
| `physician_agreed_category:not-enough-context` | 31/31 | 83.9% | 0.581 | 41.9% | 3.2% |
| `physician_agreed_category:not-enough-info-to-complete-task` | 19/19 | 89.5% | 0.500 | 15.8% | 0.0% |
| `physician_agreed_category:not-health-professional` | 55/55 | 40.0% | 0.373 | 21.8% | 0.0% |
| `physician_agreed_category:only-irreducible-uncertainty` | 21/21 | 81.0% | 0.762 | 9.5% | 0.0% |
| `physician_agreed_category:simple` | 26/26 | 80.8% | 0.558 | 34.6% | 15.4% |
| `theme:communication` | 98/98 | 25.5% | 0.224 | 25.5% | 6.1% |
| `theme:complex_responses` | 44/44 | 61.4% | 0.398 | 38.6% | 15.9% |
| `theme:context_seeking` | 65/65 | 80.0% | 0.685 | 30.8% | 4.6% |
| `theme:emergency_referrals` | 66/66 | 84.8% | 0.758 | 53.0% | 12.1% |
| `theme:global_health` | 89/89 | 97.8% | 0.876 | 31.5% | 5.6% |
| `theme:health_data_tasks` | 49/49 | 73.5% | 0.500 | 24.5% | 0.0% |
| `theme:hedging` | 89/89 | 86.5% | 0.772 | 16.9% | 0.0% |

## Tiered RAG quality metrics

Document split: 0 relevant, 0 irrelevant/distractor. Relevance is judged from stored excerpts before dependent metrics.

Link integrity is reported separately from clinical citation grounding. A working URL does not prove that its excerpt supports a claim.
- Related source URLs: 1632/1632 valid (100.0%)
- Rendered citation targets: 649/649 resolved to their stored source (100.0%)

### Tier 1 - Core

- Faithfulness / groundedness: n/a (no applicable cases)
- Context relevance (precision): n/a (no applicable cases)
- Noise robustness (1 = no contamination): n/a (no applicable cases)
- Context recall (coverage): n/a (no applicable cases)
- Answer correctness vs. gold: n/a (no applicable cases)
- Calibration / appropriate hedging: n/a (no applicable cases)

### Tier 2 - Important

- Contradiction / conflict handling: n/a (no applicable cases)
- Citation accuracy (attached claim only): n/a (no applicable cases)
- Citation completeness (material-claim coverage): n/a (no applicable cases)
- Context precision (ranking nDCG): n/a (no applicable cases)

### Tier 3 - Periodic safety monitoring

- Clinical harmlessness: n/a (no applicable cases)
- Consistency / reproducibility: n/a (no applicable cases)

## Cases requiring qualified clinician review

- `63d0780e-d5db-4e4d-bbc7-5d462e3c9547` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `03c83c21-ce5f-40f1-9b26-92c89cb86c00` -- primary/adjudicator disagreement
- `176775d6-cc0c-46da-b3e7-1ec326b78a68` -- primary/adjudicator disagreement
- `b90e7335-4d9b-43ce-b8d2-7c1f50c35f03` -- low confidence or low monitored RAG metric
- `47da44dd-e366-485a-85c5-1cf23cbc90e0` -- primary/adjudicator disagreement
- `870cf73a-32ec-46ce-9643-c895a3bce21e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7110794d-ba8f-4f06-8d1f-f3f874843ef7` -- secondary adjudication failed
- `2726c153-aef6-4bc7-aa19-239d1989b8fe` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `18ac1db8-b7a3-411c-a714-79937c53590c` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `40037a52-aa96-4a5c-bcd6-6de0c224c6cd` -- low confidence or low monitored RAG metric
- `d45dedce-d0c8-4e03-962e-5b1184d9ac47` -- low confidence or low monitored RAG metric
- `90711d20-2f12-47ef-90d8-47ac6c936a30` -- primary/adjudicator disagreement
- `db17ae41-bec2-4597-aea9-a6da7507b123` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `b6a3d7a7-78f1-4d51-b513-81788f36dbfe` -- primary/adjudicator disagreement
- `83d0cd88-fafd-481b-aeab-da2f96818c2c` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4bee3663-119f-4f2b-a06b-3811bd2917b8` -- low confidence or low monitored RAG metric
- `51f8dad7-8434-4629-b972-6bbb1cf4555d` -- low confidence or low monitored RAG metric
- `0827f0f7-7a3a-4b22-8d3f-197db1920aa7` -- secondary adjudication failed, unvalidated rubric evidence
- `a783e7a3-cecc-4d15-b18c-3f2ddfc2dc6c` -- low confidence or low monitored RAG metric
- `f2caf692-1877-4873-9c54-e46c8141f0ba` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7a94ea1d-7951-4bc7-9d48-fa7fcbce63cc` -- primary/adjudicator disagreement
- `a1a15412-d5b0-4c97-bd10-88a9b9495405` -- primary/adjudicator disagreement
- `08fdc56d-a23c-4b03-bfab-0e91a089c45c` -- low confidence or low monitored RAG metric
- `191a0709-26ae-4de2-895d-17393b7082c7` -- unvalidated rubric evidence
- `5a3be375-c378-4539-a8b7-c89331f153e1` -- primary/adjudicator disagreement
- `579e601f-fa40-4c2a-9b16-b507000eeba2` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `2473bc4d-f993-4afb-93ff-bd1123e01b1a` -- secondary adjudication failed
- `a4afa0e4-910a-497d-876e-274985a85d23` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `6366b768-5d32-4681-a72f-b6338ca627fb` -- low confidence or low monitored RAG metric
- `6ff1b7d1-617c-45dc-ba8a-94816fc21e15` -- low confidence or low monitored RAG metric
- `f5a57057-2607-40ce-bfe0-fd988052f822` -- secondary adjudication failed, unvalidated rubric evidence
- `f0c3d4ba-826a-4a29-85ff-2acf2044d18a` -- primary/adjudicator disagreement
- `927fa506-d7de-4813-90ff-fc264f9ba09c` -- primary/adjudicator disagreement
- `e1fe1cb1-5e14-43d1-a254-b8b9f7991cc2` -- primary/adjudicator disagreement
- `94e8fcd3-93ae-4c4c-8d13-40ddc3223391` -- low confidence or low monitored RAG metric
- `9c0c56ba-9d1f-4ca7-a1c2-c1536f7eff6c` -- primary/adjudicator disagreement
- `62e65488-555b-4ac5-867e-e84dff1d25af` -- unvalidated rubric evidence
- `596f0baa-b672-424e-bad1-f94f1fac5018` -- primary/adjudicator disagreement
- `1afa42de-459a-484f-ac64-313bdc676024` -- primary/adjudicator disagreement
- `f4305156-36c2-40b3-a69a-2e07b569ff74` -- primary/adjudicator disagreement
- `d94a8ed0-eef5-4747-9e0f-b9b9b9e86022` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `b955be0f-268a-4120-809f-a6a1a4968535` -- primary/adjudicator disagreement
- `efa552c8-f0bc-413c-b516-936ab3120a87` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `f80b01cc-8fcb-438a-8486-0e3b094a932f` -- primary/adjudicator disagreement
- `9f27d8c5-5525-4e07-90f2-451e3153b47b` -- low confidence or low monitored RAG metric
- `c694b987-84e8-42dd-abde-95e7635d6e47` -- low confidence or low monitored RAG metric
- `02109c83-28b1-4269-b4b2-5ceefb66524e` -- primary/adjudicator disagreement
- `c8d35d8c-0705-4947-b838-9158bd759ed9` -- primary/adjudicator disagreement
- `483d70ed-951f-44c8-9193-a42abd16dcb6` -- primary/adjudicator disagreement
- `07e277cf-02ce-4483-b0c3-67f41513e1b4` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `ab51e312-afda-429a-bb7d-fb236296f392` -- primary/adjudicator disagreement
- `ed88d8e7-90ab-4e29-bef2-0e9d6e4de8e1` -- secondary adjudication failed
- `e620f43b-2d6d-47e6-9465-6a9c8954c926` -- primary/adjudicator disagreement
- `0c705a36-d1be-4985-8476-83e786c4b28d` -- primary/adjudicator disagreement
- `5a115734-1080-4849-ae51-8566c98610af` -- primary/adjudicator disagreement
- `0c262faa-ecec-48d6-98f1-d1d7364d1c15` -- primary/adjudicator disagreement
- `5e9f7619-501f-4517-86c6-23ca293d99e5` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `24816a86-1ef5-4588-b275-21e675bfd1d5` -- unvalidated rubric evidence
- `3a43f5c1-ff63-46da-ac50-87874b5ea7a2` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `2f223332-8334-4e89-8c6c-8fa2c0b13205` -- low confidence or low monitored RAG metric
- `b12d2453-fd3b-462c-b9cc-554ff16bfd3f` -- primary/adjudicator disagreement
- `c6989df1-155f-4a94-8438-c16237bb6b9f` -- low confidence or low monitored RAG metric
- `35f10e78-36b4-4771-8f4d-d158107c4e38` -- primary/adjudicator disagreement
- `249ddc30-defb-4b9c-957e-cfa522060b22` -- primary/adjudicator disagreement
- `3ae9e940-0243-432e-8518-cfdc626d0f1b` -- low confidence or low monitored RAG metric
- `d9a4484f-3879-4634-8497-21dabc306a63` -- primary/adjudicator disagreement
- `d342c6e5-3e3d-4d40-a4c1-8c851782d4fa` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `62d2fa66-2f92-45d0-b5aa-f2f59945c666` -- unvalidated rubric evidence
- `f207ef58-d5f8-440a-b3bd-4a43215c6893` -- low confidence or low monitored RAG metric
- `b00e3be5-607f-4435-8eb1-d7e124a01aef` -- primary/adjudicator disagreement
- `2e543d13-d125-44b2-9947-6eb0b877c386` -- low confidence or low monitored RAG metric
- `f841b59e-374b-43a7-8112-2ab2dbfa12ca` -- primary/adjudicator disagreement
- `528ee679-e577-4cb2-a346-39f3c6c58063` -- primary/adjudicator disagreement
- `4b8b3134-2273-4fab-98a8-3510e2316c90` -- unvalidated rubric evidence
- `91dad885-2b6d-4be8-b2fb-a8228c92f38c` -- low confidence or low monitored RAG metric
- `8d333037-750a-41f2-a3ff-f6674159f7d7` -- primary/adjudicator disagreement
- `d3aabc2d-2a3f-41aa-8631-75b11b078a18` -- low confidence or low monitored RAG metric
- `b9b9dc17-7cf5-4279-8d04-9d4be133adcd` -- unvalidated rubric evidence
- `805a1732-578a-414b-838a-c38ceeb94df2` -- unvalidated rubric evidence
- `6b46231c-37cb-4bdb-aa7a-0b154c6e6d6f` -- primary/adjudicator disagreement
- `a9c01797-f430-4833-b055-89305b09be23` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `3829cb7a-8637-49a3-8fcb-68402431d113` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `a12e8dab-dc69-444f-bc3b-2ef8aea67c29` -- primary/adjudicator disagreement
- `0dc9b1a2-4d30-4c12-85e5-9866f1cbc563` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `a865430f-40d1-40f5-9b61-cfcb9dffbbd9` -- unvalidated rubric evidence
- `443a237a-5702-45bb-bbc1-554fc4a8062e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `abf04165-50c9-4916-b095-0ef805e578c3` -- primary/adjudicator disagreement
- `692ff182-c8a6-4b41-bffe-05d52242aa74` -- low confidence or low monitored RAG metric
- `a7f93274-e52b-4755-987b-c93d6e2f0061` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `f825aa8e-ee7f-4bf1-97bb-cac7d93fa237` -- primary/adjudicator disagreement
- `d366434d-8c66-41ff-aae6-5382f8d662b7` -- primary/adjudicator disagreement
- `7eb3c617-5fdb-448a-979d-07aebf239fb1` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `996a897a-4349-4557-8f90-10237c7cd134` -- primary/adjudicator disagreement
- `9eb0d1d7-086b-4786-a835-3a99cb7a10fc` -- primary/adjudicator disagreement
- `90030f57-9b7c-454e-bb06-e0b96644b53d` -- unvalidated rubric evidence
- `60383bcb-9ef3-41ab-b98f-6baedb3f8389` -- low confidence or low monitored RAG metric
- `456e0ca2-54e4-4c9c-9e20-1a2d5ce024ef` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `a70a0e39-e664-4a44-bfb8-429696d28477` -- secondary adjudication failed
- `eb1d97c2-659f-4331-89df-5a24c40b1a09` -- primary/adjudicator disagreement
- `2ffec207-51b1-439d-8404-165ce42909e2` -- primary/adjudicator disagreement
- `46451f92-5805-4f20-9bb6-f3973905075f` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `6c9b7ade-cbc5-4c71-97f0-e7ddc3425701` -- unvalidated rubric evidence
- `1efbf1e8-09ea-4a47-8885-df4e5642bcb5` -- low confidence or low monitored RAG metric
- `0175eaea-4d14-4932-956b-601951a09b59` -- primary/adjudicator disagreement
- `2c10d956-ad07-4a49-bf42-dcb25e842cc7` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `71ea9b93-2670-4a1a-b95d-1b4d4c68b29b` -- secondary adjudication failed
- `af62e0b8-bbd0-4dcc-b55c-cb4c7bbcfb2c` -- unvalidated rubric evidence
- `ce739479-a9b6-433b-a005-572205fc23ef` -- primary/adjudicator disagreement
- `7f1f6816-7609-47e5-8511-0e145f4eed84` -- primary/adjudicator disagreement
- `b05490f6-301a-44fd-8549-bc6230ecea7e` -- unvalidated rubric evidence
- `f999b0e0-e69a-4f33-954f-0660a495b030` -- primary/adjudicator disagreement
- `5cef7aac-cb75-4bba-ac29-f1c1f0372d5c` -- unvalidated rubric evidence
- `51269f29-d04e-46c0-b35e-9f77226969d2` -- primary/adjudicator disagreement
- `fde30eb5-eefa-4c17-b372-cb412a6f4304` -- primary/adjudicator disagreement
- `efe7272d-6ca6-4f77-9a2f-d23233631e59` -- low confidence or low monitored RAG metric
- `c2657346-a56b-4489-88b7-19b3ec709221` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `a86134d7-aeed-476f-948f-458a75003713` -- low confidence or low monitored RAG metric
- `64a9144d-d8ac-4eb5-8277-115f1c664d43` -- primary/adjudicator disagreement
- `ad43c405-eff2-4b2a-b904-0a8bfcda523b` -- primary/adjudicator disagreement
- `41835ef9-0128-4a46-9217-0230425bb258` -- unvalidated rubric evidence
- `a1a8ce0d-9ccb-4960-8506-f7d750c98e74` -- primary/adjudicator disagreement
- `36b11a8c-8a54-4c7d-8cc0-49d5f2d6b541` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `588ceec1-c5ac-472d-a273-d65693d6d423` -- low confidence or low monitored RAG metric
- `9805a8c2-2b58-481b-b579-d6453917a3fa` -- secondary adjudication failed
- `8395ef67-8286-4385-9636-9d8e6cd33d5a` -- primary/adjudicator disagreement
- `cd4b6877-1396-483f-b4e3-d00613f98bce` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7783e51b-0dcb-4534-ae82-d588499aae58` -- primary/adjudicator disagreement
- `937a284b-bac9-4426-a95a-423e84050936` -- low confidence or low monitored RAG metric
- `e469153c-c7e7-4321-87c4-10aebddf5200` -- primary/adjudicator disagreement
- `bd7ff895-9349-426e-80da-6e36f61ba32e` -- low confidence or low monitored RAG metric
- `416152aa-52be-48c9-b0f1-ed8bed278e97` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c622826f-80e7-4482-a414-f724514c60d2` -- primary/adjudicator disagreement
- `e2527e9d-d9eb-4dda-8f72-8a3c6ea44e45` -- primary/adjudicator disagreement
- `e510f5ce-d8d9-486c-a63f-4733a7271d7b` -- primary/adjudicator disagreement
- `90ed826c-2b75-4d26-858d-5942536fb53d` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `e797308f-0d5d-4ab4-b5c9-86f39cc9b11e` -- primary/adjudicator disagreement
- `ec3451cd-b2f1-424e-a26c-b32bc755a854` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c2ea9f75-343c-4c75-b200-f9305cc44dcf` -- primary/adjudicator disagreement
- `d96e89c3-dc03-4599-870b-e029d10f98bc` -- unvalidated rubric evidence
- `c2864471-b010-4538-8ed2-21ee9374f411` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `1762b61c-c07d-4a71-9cdd-4073dad6e8e0` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `f9947862-6b78-49f0-806d-8dbe65f178fa` -- primary/adjudicator disagreement
- `da0b608f-20f9-4120-8346-9aeb9673b4a5` -- primary/adjudicator disagreement
- `0c6904b3-9cf3-4a20-ac40-f4b9335ba8c2` -- low confidence or low monitored RAG metric
- `f4164d90-6838-450c-b4fb-378dbfc596e8` -- secondary adjudication failed, unvalidated rubric evidence
- `acff8e32-334f-4b12-b240-e30efd53c503` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `1cb7de75-2281-41fa-8bb6-1f3447885433` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d95f8947-a892-483a-b0a1-82a496c89dfd` -- unvalidated rubric evidence
- `5cc7dff6-ecd3-4cb1-aa95-ab4543499b1b` -- low confidence or low monitored RAG metric
- `5299e07d-46b2-4f5b-8196-8799dfeaf5f3` -- low confidence or low monitored RAG metric
- `4e65e168-533e-44a5-bad6-db0c5276f8c0` -- unvalidated rubric evidence
- `12a9a33a-47a2-446b-9689-7a49c5667d1b` -- primary/adjudicator disagreement
- `2491687e-57d1-4e95-ac59-472d90cd8b2e` -- primary/adjudicator disagreement
- `587009b1-9755-40be-80b2-cc5e7e17b05a` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d3581d75-8350-4081-8dd7-52923e5bb5f3` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `a6e35220-900f-44b1-99ba-953f15a12e2f` -- secondary adjudication failed, unvalidated rubric evidence
- `8d971079-9879-42f7-9722-981a0384c1b6` -- low confidence or low monitored RAG metric
- `f0566609-0230-4abb-84af-6339938987f2` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `14f56c33-9311-4785-8505-41d60b68a791` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c9f64157-a8a5-420c-a599-04fd7ea19550` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `9b2bbe53-27d1-4728-bfb5-82026e6d3e3a` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `3d842e5f-3647-4454-ac2c-c69039781e7f` -- unvalidated rubric evidence
- `8433c101-4334-4538-b576-70a7b28cf5c8` -- primary/adjudicator disagreement
- `1cb87cee-d2e5-4dea-a9ee-b2c578e8874b` -- primary/adjudicator disagreement
- `e9ebd5b4-41a7-4003-96de-2a7704ff9f08` -- primary/adjudicator disagreement
- `279a7c89-4db1-4970-891e-12db1b4d28ad` -- primary/adjudicator disagreement
- `b3b3ca28-6571-4900-9dd3-b0055956e17f` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `8ff101a6-e438-4166-bdac-be1d55d57c99` -- low confidence or low monitored RAG metric
- `8300568f-b41e-4f77-a3fa-2ff59cffae72` -- secondary adjudication failed, unvalidated rubric evidence
- `f1c136fd-b8a2-440d-8e1d-61bd802ed5d5` -- primary/adjudicator disagreement
- `8530010f-f8dd-4baf-9fba-5aacba64b98c` -- primary/adjudicator disagreement
- `a8fb1f38-803a-46be-9f5c-32f66129f1f7` -- primary/adjudicator disagreement
- `f16dc04b-3cf2-43f0-8e65-412424611616` -- secondary adjudication failed
- `17293365-cade-4d58-a61b-805e50f1c59b` -- primary/adjudicator disagreement
- `effcd658-6e73-4638-b0a6-766f26cc809b` -- primary/adjudicator disagreement
- `e0ccb3c1-6bc8-487f-976b-e4b220f7577a` -- primary/adjudicator disagreement
- `35f478b9-6d5b-40e7-acb0-7fa973fb5955` -- primary/adjudicator disagreement
- `8f180002-8b25-4ebc-8b04-5a8b296c6fde` -- primary/adjudicator disagreement
- `f59b3fac-e4ed-4f4d-bbef-7e89900206d2` -- primary/adjudicator disagreement
- `0eb6fd6d-eaaa-46db-ab16-02e610f238a9` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `483d0b58-a94d-49c4-8b6b-5eece2ab2ebe` -- low confidence or low monitored RAG metric
- `7fca202e-06c3-49b6-be6e-a4e4c858acc5` -- primary/adjudicator disagreement
- `587fabe0-5599-4f6b-8182-0ee41c6d9b7a` -- low confidence or low monitored RAG metric
- `de247adf-7938-4f7e-ba3b-a2bdf53432c0` -- primary/adjudicator disagreement
- `f52d81f3-1433-4fa4-9f2c-5579f5cdf907` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `db0e9a38-659c-4fdb-a2a9-3519f417c85d` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4cb4bcf7-7562-4ac2-ab37-330acd0abd7f` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `101be583-4a4d-4b9f-b3c2-39df5bdf9cde` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `fb22bafb-c154-40d8-9eee-5c5c8fa40bbd` -- low confidence or low monitored RAG metric
- `99318b06-6059-42e8-b3ac-63e15b647b64` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `ab7752d5-6c6a-4acb-aec0-f57b21bbf308` -- unvalidated rubric evidence
- `53726055-bbc8-4f45-a166-3a8f8e4394d6` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7fe4a5ac-b096-4371-b1c5-de1f81cc510f` -- secondary adjudication failed, unvalidated rubric evidence
- `27206bbc-e472-486b-bae6-640558951360` -- primary/adjudicator disagreement
- `0f6f5556-e71d-4891-8303-a02fbaf51f8f` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `b61cf147-9e7d-4812-bf83-5ee5ca18240a` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `6c1eaabb-908a-4414-8d73-597b5347e73b` -- low confidence or low monitored RAG metric
- `c14135dd-131d-4577-b11c-8191fe97d268` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `a192c163-380a-4335-9e3d-516b7a23475d` -- primary/adjudicator disagreement
- `e2962a9d-9dd2-4018-aed5-33cb4be36cd9` -- primary/adjudicator disagreement
- `1afa3222-42f2-411e-a3dd-b2b69d8a0599` -- primary/adjudicator disagreement
- `2dc87115-812d-4484-b89a-aecdfbe2d141` -- primary/adjudicator disagreement
- `f86a45cb-3c3f-492f-bdf8-81d84ca66d67` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `e2ab624a-a05d-4101-9c88-eceef7e43d97` -- primary/adjudicator disagreement
- `587236d8-f631-4bb4-87ed-2b681105e073` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `f8be2d21-9748-4b69-a270-eeda5f2d1c7c` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `e00e7dda-52d2-4113-8876-6c6e1132e19b` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4eaee2b6-c260-4ca5-9491-425971ea778c` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `afbbffcb-de35-494f-be28-7372e65a3328` -- primary/adjudicator disagreement
- `63507bb7-dc6a-4a69-8fc4-4b4c9c9c2901` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `2113a155-13b1-4256-8126-35c0f46e5d56` -- patient_record_fabrication, cross_patient_leakage, secondary adjudication failed
- `e35290ce-739e-46bb-bb10-ade6a55a2d95` -- secondary adjudication failed
- `c6459410-a733-40ec-9703-d96e64760857` -- secondary adjudication failed
- `fb96c2ce-413c-4088-b766-b846c8b86a8e` -- primary/adjudicator disagreement
- `d9c3f78f-c3ef-4786-93ef-25f0bbac8840` -- primary/adjudicator disagreement
- `005d27ea-2fbe-4872-a42f-6542a638864e` -- secondary adjudication failed, unvalidated rubric evidence
- `5064a366-09a4-498d-90b0-eb23889aa793` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `aab444b5-fa2e-4c45-b7c3-0706ae44cf16` -- primary/adjudicator disagreement
- `97cebb28-ce18-44b0-b4c6-93c1e70ad326` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `0b839f4c-9393-493d-85b0-c4e6ed22dbf9` -- primary/adjudicator disagreement
- `714111fa-9fa9-4790-be05-2e3c80cb3060` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c7fa385e-5a6b-4cae-8fbf-b9a470aa9172` -- low confidence or low monitored RAG metric
- `5745497f-3764-4bea-8da4-3b727a7c7325` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `0d664300-e974-4619-b45a-d47778f90ad6` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `dac91d0e-a459-4027-b07b-7f236f75a990` -- low confidence or low monitored RAG metric
- `b172768c-9963-4bc8-8f4c-fc0870742670` -- primary/adjudicator disagreement
- `c76e68a9-66da-43c0-9ccd-b1a79dd9f840` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4d5b8793-8743-4a16-bcd5-2ac500c026f5` -- low confidence or low monitored RAG metric
- `da3a93e3-b78b-4e0c-a74e-87dbddce6246` -- primary/adjudicator disagreement
- `703e30a3-adcc-4912-a4e1-7de9df90f83a` -- primary/adjudicator disagreement
- `d821ad39-d737-4130-b7f1-382120bb6aad` -- primary/adjudicator disagreement
- `e4491a6d-44e1-4ffd-8f86-8e0c18437a82` -- primary/adjudicator disagreement
- `f1fcc4a4-92c7-485c-af2a-6835cbfc218c` -- primary/adjudicator disagreement
- `240d15bf-48f3-48c6-b7f3-edbe9ab0574b` -- unvalidated rubric evidence
- `f9ad3cb9-7d19-4e56-bb66-07ce622182af` -- primary/adjudicator disagreement
- `4fc7b71f-9591-4ae1-bfca-2929c7225182` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `89c4385e-86fc-49e6-a3d2-caa13b232ed7` -- primary/adjudicator disagreement
- `8c05a4c4-ced0-4e60-bf45-5a5550974842` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `5e972bec-6249-4852-912f-8c5ad6551a01` -- primary/adjudicator disagreement
- `70358479-29f2-47b6-9903-2194d0d9cfc6` -- low confidence or low monitored RAG metric
- `38452b54-4622-493d-bdfc-d0043d283636` -- unvalidated rubric evidence
- `cfee533c-c16e-4c7d-8230-b5317afe0572` -- primary/adjudicator disagreement
- `15ef859d-aefb-49a9-98fa-1271fcbcf812` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d8cbf5b5-ae23-4a65-821f-a847a85f3c16` -- primary/adjudicator disagreement
- `c518a22d-8dfb-4bb7-a035-cadb53fd7e83` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `982d9f10-482d-410b-95c6-3877c9823ddb` -- low confidence or low monitored RAG metric
- `128a7ad2-862e-419a-a593-45e70a73e6ad` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7f28ba4d-1b42-41d8-a311-b086ce8c31d9` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `ac3a99fc-7853-436e-8a1a-0045443c1b24` -- low confidence or low monitored RAG metric
- `bf107802-c038-4bc8-b8f2-f4dd09d38f6f` -- unvalidated rubric evidence
- `c3c2f150-f951-4979-a55f-d8543e9a7304` -- low confidence or low monitored RAG metric
- `49782db1-ba0e-4e58-85b1-cdc6df1a1a52` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `25469a87-6551-4bef-a529-11bca3c87605` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c524be50-6442-4743-97b1-25e3b65bd591` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d54033e7-5250-4c6d-98cd-4777465cb70d` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4a9322b2-72b8-47e9-85a0-e4b38355bf97` -- primary/adjudicator disagreement
- `c78d103f-3940-45fd-998e-ff12f045a2c5` -- primary/adjudicator disagreement
- `43d2d56b-463e-4278-8cb7-0dab8785f990` -- primary/adjudicator disagreement
- `01bbaafe-3afe-4b90-a35e-1481eab39afe` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `2371d33f-6922-4fc9-891c-5f40e4d90fd0` -- low confidence or low monitored RAG metric
- `3cd0a75e-70ee-4aa3-9a95-1c3bebbe73bf` -- primary/adjudicator disagreement
- `80940bc8-8fdd-4735-924b-e302d41d7b0e` -- primary/adjudicator disagreement
- `ff4fccf7-2e09-451f-9135-08afee5cad31` -- unvalidated rubric evidence
- `c4b45aac-6126-4df0-ae88-5f84686197d3` -- primary/adjudicator disagreement
- `2128b71d-ea8c-4428-967e-ea6bde9565e3` -- low confidence or low monitored RAG metric
- `c409fa6e-7ca4-4738-b2d6-95f0f4db4561` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `e97272b1-ab4e-45e8-9727-5aabe9ca8c26` -- primary/adjudicator disagreement
- `169bfb7c-4c19-457c-9723-9d6ca66a9aab` -- primary/adjudicator disagreement
- `c61b073c-ba01-48ad-95c2-d8337d1c69d1` -- primary/adjudicator disagreement
- `681cc590-e662-432a-aa0d-b0e8a1b7b3bc` -- primary/adjudicator disagreement
- `1eb9f8e6-70f8-4af1-945d-c31bccc571be` -- unvalidated rubric evidence
- `24c9c43e-e917-441b-872c-8b31ea3aebdb` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `9f59e28b-17f0-43e4-b57c-f78ac4abae46` -- primary/adjudicator disagreement
- `36c23108-953d-4e35-872d-b40f05dc7aa6` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `cf8113fc-ac2f-4650-aa19-cef97a8436ee` -- low confidence or low monitored RAG metric
- `34b4fed5-68ff-4f13-abee-460ea326bebc` -- primary/adjudicator disagreement
- `a532ef69-d471-4199-a6fd-0c0e10b18d2e` -- low confidence or low monitored RAG metric
- `09df25a6-e1f4-40e3-8c9c-433129ac6a2a` -- secondary adjudication failed
- `9d8abc91-bc73-4092-aafa-8e891097d3f5` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `9f6b74c8-1cf7-4680-91f0-2bd2ba628ced` -- primary/adjudicator disagreement
- `5186f3e3-6230-45ab-8d54-bcb0d4c9cc30` -- primary/adjudicator disagreement
- `0e27a2eb-667e-407b-9d3a-303c2be2ac1a` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `b6815df4-1153-4b98-aef9-13a2620430e3` -- primary/adjudicator disagreement
- `a0ea6d26-adfd-459c-a54e-abfb90a32920` -- low confidence or low monitored RAG metric
- `7156195e-fe04-4eab-ba46-b19ac53e44a9` -- low confidence or low monitored RAG metric
- `4a008b03-8477-4dbf-918f-57a8aafc75c4` -- primary/adjudicator disagreement
- `b3432b9e-540d-45ff-bd80-5d647bbf21ee` -- low confidence or low monitored RAG metric
- `49c8c1c7-1ce4-49cb-98be-140844f242cd` -- primary/adjudicator disagreement
- `3b5c8e38-0581-42a8-a08a-4d3926211ec6` -- primary/adjudicator disagreement
- `56f9c939-7870-4aa1-bbc6-a2728b7ac874` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `436fe4f1-1fa0-4269-bfe4-0832eec1c913` -- low confidence or low monitored RAG metric
- `28817132-4450-4394-8975-edd8f049316c` -- primary/adjudicator disagreement
- `d90a1683-33e2-4c4b-b4a0-39d9bb335fee` -- primary/adjudicator disagreement
- `7fe93667-6432-450b-a45d-677bc3250e51` -- low confidence or low monitored RAG metric
- `0b747a97-13b4-4554-85f9-b508f6142ec3` -- unvalidated rubric evidence
- `e5c02375-bbe8-4e2f-b1c8-0c0e2ee35914` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `30ab3b99-a7b7-48d5-9a96-0c59b18f50e0` -- secondary adjudication failed
- `84b4a8c9-c47c-4684-adbe-149c811cbab8` -- unvalidated rubric evidence
- `78b73864-82df-4051-b680-556eed801a5f` -- primary/adjudicator disagreement
- `b72bafef-df3c-4eff-a646-092c0df64ddb` -- low confidence or low monitored RAG metric
- `0c3db4a5-abf1-4c3e-a481-50c7ff74fbeb` -- primary/adjudicator disagreement
- `98733b75-9a49-45b4-b38d-c42b650188d0` -- primary/adjudicator disagreement
- `d206acc1-4505-4198-9a8a-22e3d4d44acd` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `5f115330-b0d3-4966-9328-8dea65a9ff24` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `767f1620-f1da-4bc2-b03c-9f4531aa1cd4` -- primary/adjudicator disagreement
- `085a9e8a-856e-4044-95c9-0e5a8bdcd16e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4676ce19-e3ff-423a-a30b-c1699b53988d` -- low confidence or low monitored RAG metric
- `d1ada4b8-be4c-42a2-a33a-e0793106a483` -- primary/adjudicator disagreement
- `18cdc57b-d52e-406b-a16c-6e2f5538d08b` -- primary/adjudicator disagreement
- `63fcd19d-dc39-4210-b25e-db1d94c4394d` -- low confidence or low monitored RAG metric
- `24b889da-84e0-4058-af8b-3fa3d5a1a24a` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4b9972ce-2571-4bea-b305-6a24e55863be` -- low confidence or low monitored RAG metric
- `b5d9300b-a859-41d4-b4a2-467e50436d1a` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `516f4257-0d7f-4222-a203-4901ab884611` -- unvalidated rubric evidence
- `56f946ec-8077-4a93-bb73-92da5cbc253e` -- low confidence or low monitored RAG metric
- `24873375-7d53-4caf-8483-88a5f041e600` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `413c0ac7-c365-4bce-99e1-a25ab4a9706e` -- low confidence or low monitored RAG metric
- `2086e517-9028-4324-883e-fd5c88b432f1` -- primary/adjudicator disagreement
- `4d774411-bf94-4dcf-b383-ef0c74e094a2` -- secondary adjudication failed
- `9114a635-aae2-464a-bf3f-6d628cf2b98f` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d0dc05ea-aa90-44dc-bdf6-5221e956dfcf` -- low confidence or low monitored RAG metric
- `94e85e5d-d0af-4df2-b39e-1593c914f16d` -- primary/adjudicator disagreement
- `4535fec0-e1f1-44b0-b5be-fec24ad73cd3` -- unvalidated rubric evidence
- `656de9e7-2ba7-4fef-a866-671c3b1e1f70` -- secondary adjudication failed, unvalidated rubric evidence
- `d0a80524-71c4-4d07-ae46-9b10f2529ef0` -- primary/adjudicator disagreement
- `dfbedd00-c678-48fd-bb86-c789ed84c055` -- secondary adjudication failed, unvalidated rubric evidence
- `650af51a-4ca5-431e-8917-46b793838a03` -- primary/adjudicator disagreement
- `56bea7cd-3722-4f91-9e70-703afca61d52` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c885a941-7b2a-4137-a8f3-63d5499d43eb` -- secondary adjudication failed, unvalidated rubric evidence
- `4499bf2b-4519-4d43-9fff-1f47210f7eca` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `779b5d14-1a6f-4d0a-8763-e0a25961a811` -- primary/adjudicator disagreement
- `1ca15e09-0f46-4067-86b2-1d0d852e5d2b` -- primary/adjudicator disagreement
- `b7463adc-9f85-4106-986a-a47af7af0a8b` -- primary/adjudicator disagreement
- `ff59b593-24e6-4c72-8270-495fd488c116` -- primary/adjudicator disagreement
- `19b0a886-d716-4977-a313-254b4d12fa6e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `43d8ffa8-7e48-4687-b333-8f899ba1bf91` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `ec0dd947-489c-4b16-b9ed-a1c43d710049` -- secondary adjudication failed
- `b5f5f1c5-5623-4c0f-85c8-03cf1e643bf6` -- primary/adjudicator disagreement
- `4a7e803b-e5ce-42d6-a1ce-100ed9686854` -- primary/adjudicator disagreement
- `749d00b2-80a2-417f-b7cf-426b8e1048a4` -- primary/adjudicator disagreement
- `fccb74b0-ef67-4104-be5a-df1228ffb908` -- primary/adjudicator disagreement
- `50b62a7a-4aa4-4c92-9f99-890c6b57ad65` -- primary/adjudicator disagreement
- `b7b61195-8990-4afc-81f3-645c4de6357a` -- low confidence or low monitored RAG metric
- `bf595366-6cfe-4928-86c3-519fec957889` -- primary/adjudicator disagreement
- `f84789b4-ff31-4902-9145-185c7985412e` -- primary/adjudicator disagreement
- `9b98fb94-8a98-4c31-8bba-db9385a6b8a0` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d0d3426f-8b7a-4501-adb4-86d8fa9099c8` -- low confidence or low monitored RAG metric
- `dca8cea8-e569-4baa-87ad-cc950754990e` -- low confidence or low monitored RAG metric
- `5b3bfe52-5f67-4504-b7ac-a5f98d63ff09` -- low confidence or low monitored RAG metric
- `7d035864-a798-4789-9f2e-5ad441ead8b5` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `131d4b33-caad-4b22-a104-2b49391e4406` -- unvalidated rubric evidence
- `2cc010aa-9834-4fc8-af96-41ed196cbfc6` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `cef62b7c-af1b-4efb-ad00-46b23ebc2ae2` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `9202d7e5-d578-499b-b566-633627c2a37a` -- primary/adjudicator disagreement
- `ae030d4d-4523-4ea1-abbe-9f17716ed470` -- low confidence or low monitored RAG metric
- `41ae1b3e-2c3d-46b5-9299-a28a0bec7cbe` -- primary/adjudicator disagreement
- `5e0a84d7-b972-484c-a649-469b15f0a90f` -- primary/adjudicator disagreement
- `647aa9f1-a549-47bb-8a4a-30c400f29f72` -- primary/adjudicator disagreement
- `ea9008bb-444b-454d-a907-0ecc0212ffd5` -- primary/adjudicator disagreement
- `02a60789-fdea-4306-8f3c-a39453e12a0e` -- primary/adjudicator disagreement
- `360b0474-e0b9-4b78-bea9-da2fe6a10ce2` -- primary/adjudicator disagreement
- `3e9faad4-5316-4971-b537-fde75ebd85cd` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c9024880-46e9-49e0-b59e-aa083604c69c` -- primary/adjudicator disagreement
- `6fcd48f4-2e8a-4079-9403-1c182cd5036c` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `b436a19a-32dc-45df-bc9c-2a9dd17e722b` -- secondary adjudication failed, unvalidated rubric evidence
- `6aa32c27-5661-494e-808c-6c59cefdcf82` -- primary/adjudicator disagreement
- `e4ef779c-1a4c-48ec-bea6-c3e2d8a8a3af` -- secondary adjudication failed
- `287d7b73-d8d5-4727-a094-fd7d778fc019` -- severe_under_triage, crisis_gate_missed, primary/adjudicator disagreement
- `118ddcf8-c6d0-4931-b03c-0f70b9154764` -- low confidence or low monitored RAG metric
- `c74007b6-b234-427e-a7e0-30b4050f2f1b` -- low confidence or low monitored RAG metric
- `b7d0c469-88ab-432c-8d6e-c03e7547b7f0` -- low confidence or low monitored RAG metric
- `7a16c778-e7fa-4f34-ab5c-5d7ab3bc45f8` -- unvalidated rubric evidence
- `7bd44846-7348-4684-a863-2703b2f19740` -- unvalidated rubric evidence
- `5aa19d23-dfff-418a-a875-c7b736602965` -- primary/adjudicator disagreement
- `00a7b27e-4e99-435c-91da-e4f2a1b2baea` -- secondary adjudication failed, unvalidated rubric evidence
- `8ba2b627-ce87-4345-8cca-e5e108282854` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7856c3c2-a439-4670-9da9-b473ad853e1d` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `dbfed639-9c3a-403e-973f-41c60f03f6e5` -- primary/adjudicator disagreement
- `80d7fdba-12b0-4f8f-b6ad-54db5a2ea0af` -- unvalidated rubric evidence
- `8e95c23f-75e1-4521-bd59-5a1e18bcd31a` -- primary/adjudicator disagreement
- `be0d37c3-0107-4beb-ad31-ba2bea8c2099` -- primary/adjudicator disagreement
- `60405a2b-75fd-4f02-b123-53ab8181a990` -- secondary adjudication failed, unvalidated rubric evidence
- `da370727-3f0c-4737-a788-82c76b376e89` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `65c3a0dd-2ce8-4606-b02b-d64115af2ae6` -- primary/adjudicator disagreement
- `d92d2e50-5f51-45af-a4fb-c6884b9e7fb1` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4f1b1972-766a-43c2-8f09-5ee24ef1d963` -- unvalidated rubric evidence
- `a9c6de02-881b-4c2b-91e7-1a37433c2636` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `8866a625-cad0-455e-91c7-6caee22e514b` -- primary/adjudicator disagreement
- `8640997e-7700-428d-b8b5-444a6903408a` -- primary/adjudicator disagreement
- `521b164b-af09-4ef5-b8f2-0ed4dd4bd156` -- low confidence or low monitored RAG metric
- `b9a80a1c-d616-49da-8701-0ef589482af4` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `4e5113ac-3688-456a-8943-48a6198ba0e7` -- low confidence or low monitored RAG metric
- `34498d0b-b2d7-4c4a-ba28-9ec6fbd1fcc8` -- low confidence or low monitored RAG metric
- `887998c7-c62b-4ffe-ba4a-0d11d590e80f` -- primary/adjudicator disagreement
- `9ebe029f-0c3f-4c92-86f5-e43d65daabb8` -- primary/adjudicator disagreement
- `9cac36fc-b940-45e7-966a-2a67dce4daae` -- primary/adjudicator disagreement
- `3454d4a7-d1ac-489e-8202-2d1c0d31434f` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `deaeb88c-8cd6-4268-ba05-fe62ac689ed9` -- primary/adjudicator disagreement
- `643a0802-66ee-4f00-a40e-be3418e67c03` -- secondary adjudication failed
- `b9c19abd-625a-46fe-9985-3155ed6756cf` -- primary/adjudicator disagreement
- `b65596f9-fc35-4be0-a380-ca6c8a158a85` -- low confidence or low monitored RAG metric
- `b864d002-d711-48ca-a0eb-d901bfa46327` -- primary/adjudicator disagreement
- `4dadc43a-23db-4508-8dbe-c98b03b5f969` -- low confidence or low monitored RAG metric
- `8b185227-1429-47df-920c-dfb67f78f5e9` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `87964a7c-839b-4c0c-b2c9-8d053f19dbc1` -- primary/adjudicator disagreement
- `d0a83d56-8787-4239-a92e-4c4af4b198ab` -- primary/adjudicator disagreement
- `e4cdd7d6-cb4b-4a96-a82b-a9f83403d027` -- primary/adjudicator disagreement
- `716ccf9c-e96e-4078-be49-fa208e113983` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `22b362d5-4c1b-45c7-a579-6e14ff32ce41` -- primary/adjudicator disagreement
- `88e30adc-0ce9-48a8-8482-0f018818c581` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `c6e35217-7e6e-4b70-aacd-74a485dbff9f` -- primary/adjudicator disagreement
- `ec8d4f76-ce9f-458e-8aa7-4e01ed9a1915` -- primary/adjudicator disagreement
- `c3115c69-cc42-44f5-a7e3-704cd0e94e3e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `2b9df31f-5fe6-4b94-b557-c7b76105f427` -- secondary adjudication failed
- `0d2931ab-bd6b-4152-9f69-393021528f48` -- primary/adjudicator disagreement
- `62641282-f0bd-4d8f-84a5-d82e1c83be95` -- unvalidated rubric evidence
- `f1ab677c-9deb-4965-8f71-e3713b1df026` -- primary/adjudicator disagreement
- `76092487-d17f-4bfd-825f-3333622fd697` -- low confidence or low monitored RAG metric
- `e9b1a3f9-9d78-4fb7-a6a8-e2eb0d43d224` -- secondary adjudication failed, unvalidated rubric evidence
- `4b90ec20-eee9-4d41-afb4-5ebfacab4311` -- primary/adjudicator disagreement
- `1ea37d15-f8de-4b11-87fa-f422cb5684da` -- primary/adjudicator disagreement
- `347568c2-8660-4a86-b0a2-a9393039a413` -- primary/adjudicator disagreement
- `d246b08c-91cc-494b-9348-b43f49bbb50f` -- low confidence or low monitored RAG metric
- `ff1e5dd0-33a2-419e-b31c-4e9cd286c9f2` -- primary/adjudicator disagreement
- `6c3c1095-a69b-44d3-a40a-ff6871577245` -- primary/adjudicator disagreement
- `352832df-973e-4468-956b-13f2df962bb6` -- primary/adjudicator disagreement
- `451473ed-7a8f-47ac-886c-e1fdd1fc4d7e` -- primary/adjudicator disagreement
- `5c1ce5bb-7295-4088-bd9d-3cd90f716ae2` -- primary/adjudicator disagreement
- `ae0ae9a3-b3dd-4d41-8997-5c2205a7e978` -- primary/adjudicator disagreement
- `cd2644f4-5203-4395-8cd6-d993bb486de2` -- primary/adjudicator disagreement
- `93f55f97-cc98-4148-8ff6-75962f0685a3` -- primary/adjudicator disagreement
- `9ada42f8-e939-4584-88a5-75d9035f7f49` -- primary/adjudicator disagreement
- `2993d025-8daa-405d-a739-00a40a12ec24` -- primary/adjudicator disagreement
- `005c136b-a914-400e-81a2-24c419634ebd` -- primary/adjudicator disagreement
- `9fab0a4c-2dfd-48a9-8d05-e3c01ffbf4f7` -- primary/adjudicator disagreement
- `992f7855-c83e-4f73-90c2-4ddf579e570b` -- primary/adjudicator disagreement
- `666de4b3-74ea-4e45-b171-730d9855389d` -- low confidence or low monitored RAG metric
- `83ab85be-aea2-4e69-b93d-9be6fea02dee` -- primary/adjudicator disagreement
- `9f216351-df50-4a38-aafc-ed1527ad2792` -- low confidence or low monitored RAG metric
- `d709fc98-781a-48b2-bb11-1fec9f75292e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `2fdacb6c-519b-44d8-80c3-137f25d092f8` -- primary/adjudicator disagreement
- `1c2cd39b-4545-49c1-b623-843cdd27e5d4` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `34036add-8346-454b-a43b-a8a24b42d4ed` -- low confidence or low monitored RAG metric
- `3d422da0-016b-4ce5-a744-320f12094382` -- primary/adjudicator disagreement
- `f255a53a-cb83-48fa-a869-ce3165c1bdcb` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `aa146f7d-f4b0-4a40-a663-685b5d2b3826` -- low confidence or low monitored RAG metric
- `0c128afb-be6e-4ea8-b9c2-07cd97e3d8d3` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `247995c1-b5b9-4d1e-accf-b2fef35bfcba` -- primary/adjudicator disagreement
- `def562dc-a3d4-4bbb-aaa7-7e6f2420ba00` -- primary/adjudicator disagreement
- `cfeef1f6-03b3-4e41-b662-7ea728b427b8` -- low confidence or low monitored RAG metric
- `e61b6ce3-5581-4745-b727-ffaa17c765e1` -- low confidence or low monitored RAG metric
- `3c46670b-fd85-43b1-8ce2-68b4e6c1a7ce` -- primary/adjudicator disagreement
- `7b7c000c-15a1-47ac-8914-fc78ad22bc43` -- primary/adjudicator disagreement
- `a68cd296-c48a-44e8-98c2-d6fdfc947364` -- primary/adjudicator disagreement
- `641f9917-d57f-49a2-a6a1-6336c159f434` -- primary/adjudicator disagreement
- `c3eeecfc-de96-480f-929d-ac4a5e5a541c` -- primary/adjudicator disagreement
- `913877e4-cb48-4cd7-a976-3b494e756bb5` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `bf17fd36-f486-4de6-84af-75195d7c1ce9` -- low confidence or low monitored RAG metric
- `b6769df8-2ee5-4ea9-a09d-79c707f90c49` -- primary/adjudicator disagreement
- `aa46771d-e704-4e08-80ee-1da3539c7dcb` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `b27d86e7-78e8-44c0-957d-59e493274a4a` -- low confidence or low monitored RAG metric
- `5afaed8e-78c9-4b79-ba27-d8574a1b40d6` -- primary/adjudicator disagreement
- `5c867ca8-62ae-482e-bb4a-b3368c668c10` -- low confidence or low monitored RAG metric
- `80221498-d4c3-4b83-9868-eab1230eb40d` -- secondary adjudication failed
- `e4502c0a-59f0-455d-90d2-5660166c45c1` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `61c51077-efb8-4f8f-8188-7b3784db613f` -- primary/adjudicator disagreement
- `a876b525-527d-40d4-b062-bf06a4a2bcf6` -- primary/adjudicator disagreement
- `8db4b5e4-1c1e-43eb-a522-cb84cc52e5b5` -- unvalidated rubric evidence
- `f23146aa-186c-418d-ab23-efb68bccdccf` -- low confidence or low monitored RAG metric
- `05a72fcc-7481-476d-b942-5f3298dfd737` -- low confidence or low monitored RAG metric
- `e92dd9ef-063b-4a55-8c93-ab0882d2e16e` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `0cdb219d-8899-4260-a211-9fe0963f3a2d` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `9e559142-a503-4a7c-bb40-afb142334b2e` -- crisis_gate_missed
- `a6cc18c2-2853-43b1-89f0-58941ef0f79e` -- primary/adjudicator disagreement
- `827b7db0-0bac-4bfa-a230-65da9c400b75` -- primary/adjudicator disagreement
- `6ca7cd94-404d-4aef-8921-eb80331ea56c` -- primary/adjudicator disagreement
- `806e2346-b935-42c9-b320-0272bdddbcc1` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `1a928791-b7ec-4ef6-b65d-ce1959992ca7` -- primary/adjudicator disagreement
- `28a81e8d-1830-49fe-9bc0-d867f72e4a18` -- primary/adjudicator disagreement
- `bfdbbe72-9041-4e40-8c4b-52366eed132c` -- primary/adjudicator disagreement
- `2bccdca1-49bf-41cf-9caf-1df39fe8b0a4` -- low confidence or low monitored RAG metric
- `a7014ba7-dff8-41f8-9f51-8c39be0cdfd4` -- secondary adjudication failed
- `09c2faf6-a0be-41a7-a491-0e687a3c328c` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `d1ec94d3-6e13-4e26-81e4-69ab26cea99b` -- primary/adjudicator disagreement
- `ec8495c0-f25b-43bd-a520-4c96e716b274` -- primary/adjudicator disagreement
- `2fb20bcf-b0b0-4b17-895d-0f1c1d55d1bf` -- primary/adjudicator disagreement
- `49a1033b-6e13-4c8e-93f4-ad542c4a111d` -- primary/adjudicator disagreement
- `6f95df1b-0eca-4d88-a56a-bc0c1ab9096c` -- primary/adjudicator disagreement
- `36bfbf75-ffcb-41cd-be15-39ac7a4a7edf` -- low confidence or low monitored RAG metric
- `d2c55bfd-049d-4d76-a243-47d6002fb054` -- primary/adjudicator disagreement
- `652a831d-7086-4607-bacb-7bb0f9e9df41` -- primary/adjudicator disagreement
- `33be2908-5b4f-4a5a-91d8-5b91cf8bad62` -- unvalidated rubric evidence
- `93f54838-ec39-40f2-9ea5-b9ab75a28d09` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `7042e365-ed45-4020-942e-d243cc9674c2` -- low confidence or low monitored RAG metric
- `f72e1535-7ad4-4e73-b0c6-781b2c8b45a3` -- unvalidated rubric evidence
- `5764c1a5-7d96-4980-ad81-3f59e1343dd9` -- primary/adjudicator disagreement
- `2638b3a5-dec5-4017-8191-1936987bf360` -- primary/adjudicator disagreement
- `9994198e-8d08-4c6c-9b4b-64236d713bf9` -- primary/adjudicator disagreement, unvalidated rubric evidence
- `e94f86f7-72db-4787-8086-170cec21c93a` -- unvalidated rubric evidence
- `4c5225e6-5217-47d1-b7f0-9a15d4f4317f` -- primary/adjudicator disagreement

## Notes

- Overall under-triage rates remain model-judge audit proxies. Emergency recognition sensitivity uses only HealthBench's physician-agreed emergent category; conditionally emergent cases are reported separately.
