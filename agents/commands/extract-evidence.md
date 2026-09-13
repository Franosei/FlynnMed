# Extract Evidence

Run the structured evidence extraction layer on retrieved medical articles.
Produces patient-specific ArticleEvidence JSON objects instead of raw text chunks.

## What it does
For each ranked source retrieved from PubMed or NHS/NICE guidance:
1. Sends the article snippet + patient profile + question to the configured `gpt-5.4-mini` model
2. Extracts: question_facts, patient_aligned_facts, contraindications, drug_interactions
3. Returns an ExtractedEvidenceDossier -- a structured JSON list of ArticleEvidence objects
4. This dossier replaces raw snippets in the LLM answer prompt

## Why this prevents hallucination
- The answer model only sees patient-specific extracted facts, not raw passages
- Each fact in question_facts must be explicitly present in the article
- Population mismatches (e.g., paediatric data for adult patient) are surfaced as low alignment_confidence
- The LLM prompt states: "do not cite facts not present here"

## Schema (ArticleEvidence)
```json
{
  "source_id": "S1",
  "title": "Article title",
  "evidence_tier": 1,
  "tier_label": "NHS/NICE",
  "answers_question": true,
  "question_facts": ["BP target <130/80 mmHg in adults with CKD"],
  "patient_aligned_facts": [
    {
      "category": "condition",
      "patient_value": "Type 2 diabetes",
      "article_statement": "...",
      "relevance_type": "direct_evidence"
    }
  ],
  "contraindications": [],
  "drug_interactions": [],
  "patient_relevant_summary": "...",
  "alignment_confidence": 0.91
}
```

## Integration
Runs automatically in ClinicalOrchestrator Step 10.5 after evidence ranking.
Falls back to raw snippet rendering if extraction fails (non-fatal).

## MCP tool
`extract_article_evidence(article_title, article_text, patient_question, patient_conditions, patient_medications, patient_age, evidence_tier)`
