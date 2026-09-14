# MCP Server

FlynnMed exposes authenticated clinical tools via its optional HTTP MCP endpoint.

## Tools exposed

### get_patient_context(username)
Returns the patient's full health context: profile, conditions, medications,
allergies, vitals, symptom logs, latest triage, longitudinal memory.

### extract_article_evidence(article_title, article_text, patient_question, patient_conditions, patient_medications, patient_age, evidence_tier)
Extracts structured ArticleEvidence JSON from a medical article matched to a patient.
Returns question_facts, patient_aligned_facts, contraindications, drug_interactions.

### generate_clinical_note(username, patient_question, conversation_summary, urgency_level, next_step)
Generates a SOAP note and saves it to the patient's account.
Returns the full note JSON including note_id.

### send_health_email(username, email_type, note_id, urgency_level, reason)
Sends email to the patient. email_type: "clinical_note" | "urgent_alert"
Requires SMTP configured in .env

### search_trials_for_patient(username, location, max_results)
Searches ClinicalTrials.gov for recruiting trials matched to the patient's
conditions and medications. Returns ranked trial results.

## Connect to Claude Desktop

### Deployed on Railway (recommended)
The MCP server is mounted automatically at `/mcp` on your Railway service.
No separate process needed -- it runs as part of the main app.

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "flynnmed": {
      "url": "https://<your-app>.railway.app/mcp",
      "headers": {
          "Authorization": "Bearer <FLYNNMED_ACCOUNT_JWT>"
      }
    }
  }
}
```

Set `MCP_ENABLED=true` and `MCP_AUTH_MODE=jwt` in Railway. The token must belong
to an active FlynnMed account. Patient accounts may select only themselves;
clinician accounts need an active patient consent grant.

Direct stdio mode is disabled because it cannot establish an authenticated
patient or clinician actor. Use the FastAPI HTTP endpoint locally as well.

## Railway environment variables
```
MCP_ENABLED=true
MCP_AUTH_MODE=jwt
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=FlynnMed <your@email.com>
```
