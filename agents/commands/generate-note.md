# Generate Clinical Note

Generate a SOAP (Subjective / Objective / Assessment / Plan) clinical note from the
current FlynnMed conversation and the patient's stored health record.

## What it does
1. Reads the last 4 messages from the active chat conversation
2. Fetches the patient's profile, conditions, medications, vitals, allergies, and latest triage
3. Calls the configured `gpt-5.4-mini` model to produce a structured SOAP note in standard UK GP/hospital format
4. Saves the note to the patient's account (visible in the Notes panel in chat)
5. Returns the note with a note_id that can be used to send it by email

## How to trigger
- Click "Generate SOAP note" in the Notes panel in the chat sidebar
- Or call: POST /api/notes with optional { question, conversation_summary, trace_id }

## Urgency handling
- If triage urgency is "high", "urgent", or "crisis" → requires_gp_visit = true
- The note will show a "Send GP alert" button (clinicians) or "Email me GP advice" button (patients)
- Urgent alert email: POST /api/email/urgent { reason, urgency_level }

## Editing
- Clinicians can edit any SOAP section inline before sharing
- Edit saved via: PUT /api/notes/{note_id} with updated fields

## Email delivery
- Send note by email: POST /api/notes/{note_id}/email
- Requires SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env
- Sends formatted HTML + plain text to the user's registered email address

## MCP tool equivalent
`generate_clinical_note(username, patient_question, conversation_summary, urgency_level, next_step)`
