<p align="center">
  <img src="image/identity.png" alt="FlynnMed" width="360" />
</p>

# FlynnMed

FlynnMed is a clinical safety and continuity workspace for patients and healthcare professionals. Its primary workflow reviews changes in patient records, identifies a small set of supported medicine and abnormal-result risks, links each finding to the exact patient facts and guidance behind it, prepares a bounded next step for confirmation and clinical review, and records whether follow-up happened. Evidence-based chat and the wider record tools support this workflow.

The Python backend (`backend/`) runs the full clinical workflow: crisis pre-screen, intent and risk classification, role routing, tiered evidence retrieval, hard policy gates, note generation, care-plan generation and email. The React/TypeScript frontend (`frontend/`) is a single-page app that talks to `/api/*` endpoints served by `backend/api.py`. Both are built and deployed as one ASGI service -- there is no separate frontend host.

## Signature safety workflow

The patient workspace now leads with **Safety review**:

```text
Saved records and daily changes
  -> locked safety checks
  -> exact patient facts and supporting guidance
  -> bounded next-step proposal
  -> patient confirmation
  -> clinician review required
  -> health-record write-back when SMART-on-FHIR is connected
  -> follow-up and patient-reported improvement
```

The first locked rule set is deliberately narrow. It covers severe and moderate potassium results, current emergency symptom entries, exact medicine-allergy conflicts, and warfarin recorded with selected NSAIDs. Unknown results are not assigned an invented reference range. The screen states uncertainty, never tells a patient to stop a prescribed medicine independently, and places emergency action before confirmation or approval.

`backend/safety_review.py` is deterministic and separately tested. Every clinical claim returned by it includes structured `patient_facts` and an `evidence` item with the source title, URL and relevant passage. Patient-controlled workflow states are limited to confirmation and follow-up. They cannot create clinician approval.

SMART-on-FHIR remains disconnected by default. Read and write capabilities are represented by the interface in `backend/fhir/`, but the safety screen clearly marks write-back as unavailable until a real sandbox connection, explicit clinician approval and production governance are in place.

This README is written for developers working on the codebase: it documents the system architecture, every backend module, the full REST API surface, environment configuration and deployment. If you're looking for end-user documentation, see the [Features](#features) section below instead.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-0f1f3d?style=for-the-badge&logo=python&logoColor=1f9c94" alt="Python 3.12+" />
  <img src="https://img.shields.io/badge/FastAPI-Uvicorn-0f1f3d?style=for-the-badge&logo=fastapi&logoColor=1f9c94" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18-0f1f3d?style=for-the-badge&logo=react&logoColor=1f9c94" alt="React 18" />
  <img src="https://img.shields.io/badge/OpenAI-gpt--4o--mini-0f1f3d?style=for-the-badge&logo=openai&logoColor=1f9c94" alt="OpenAI" />
  <br />
  <a href="https://github.com/Franosei/my_health_chatbot/actions/workflows/ci.yml"><img src="https://github.com/Franosei/my_health_chatbot/actions/workflows/ci.yml/badge.svg" alt="CI status" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-1f9c94?style=for-the-badge" alt="MIT License" /></a>
  <img src="https://img.shields.io/badge/Deploy-Railway%20%7C%20Docker-1f9c94?style=for-the-badge&logo=docker&logoColor=white" alt="Deploy on Railway or Docker" />
  <a href="https://github.com/Franosei/my_health_chatbot/stargazers"><img src="https://img.shields.io/github/stars/Franosei/my_health_chatbot?style=for-the-badge&color=e5c158&label=Stars" alt="GitHub stars" /></a>
</p>

---

## Table of Contents

- [Signature safety workflow](#signature-safety-workflow)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [User Roles](#user-roles)
- [Features](#features)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server)
- [Tech Stack](#tech-stack)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Deployment](#deployment)
- [PostgreSQL](#postgresql)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Important Note](#important-note)

---

## Architecture

FlynnMed is organised as seven layers, from the role-aware UI down to governance and audit. The reference diagram below is the source of truth for how a request moves through the system; the sections that follow map each box to the actual module that implements it.

![FlynnMed unified clinical AI platform architecture](image/architecture.png)

### 1. Experience layer
Role-aware React SPA (`frontend/src/App.tsx`) rendering the patient workspace, clinician dashboard, evidence chat, voice input, image upload, document upload, health timeline and trial search -- all from one component tree, gated on the signed-in user's `role`.

### 2. Access and API layer
`backend/api.py` is the single FastAPI application. It owns:
- **Auth & consent** -- JWT issuance/verification (HMAC-signed, `_token_secret()`), bcrypt password hashing, signup consent gate
- **Profile & sessions** -- `/api/me`, `/api/snapshot`, `/api/profile`
- **REST / streaming API** -- all `/api/*` endpoints, including Server-Sent Events streaming for chat (`/api/chat/stream`, `/api/chat/image/stream`)
- **MCP server** -- `backend/mcp_server.py` mounted at `/mcp` on the same process
- **Role-aware UI state** -- `backend/role_router.py` and `backend/product_config.py` resolve profile role strings into `RoleConfig` bundles the frontend renders against
- **Endpoint router** -- FastAPI also serves the built `frontend/dist` as static files, so one process handles both the API and the SPA

### 3. Governed clinical intelligence core
`backend/clinical_orchestrator.py` (`ClinicalOrchestrator`) is the central workflow engine every chat message passes through, in five stages:

| Stage | Responsibility | Key modules |
|---|---|---|
| A. Control plane | Moderation, crisis pre-screen, role resolution, intent & risk classification, policy gate evaluation, pathway selection | `moderation_ml.py`, `intent_risk_classifier.py`, `role_router.py`, `policy_engine.py`, `pathways/` |
| B. Reasoning plane | Bounded tool router -- an LLM tool-calling loop that decides which evidence sources to query before any answer is drafted | `clinical_orchestrator.py`, `official_guidance.py`, `pubmed_search.py`, `medication_checker.py`, `clinical_trials.py`, `memory_store.py` |
| C. Evidence handling | Provenance tracking, tiering, similarity scoring, rank & truncate into structured evidence | `evidence_ranker.py`, `evidence_schema.py` |
| D. Extraction boundary | Per-source, facts-only extraction with conflict detection and confidence scoring -- this is the anti-hallucination layer between raw sources and the LLM | `evidence_extractor.py` |
| E. Disclosure & synthesis | Role-scoped response schema, patient-facing vs clinician-facing construction, safety-netting text | `response_templates.py`, `summarizer.py` |

Authority tiers applied in stage C: **Tier 1** -- NHS guidance and NICE guidelines · **Tier 2** -- systematic reviews and Cochrane-style evidence · **Tier 3** -- primary research from Europe PMC / PubMed Central.

### 4. Clinical application services
Domain workflows built on top of the governed core: `clinical_notes.py` (SOAP notes), `gp_summary.py` (GP handover PDF export), `medication_checker.py` (openFDA interaction checks), `symptom_tracker.py` (trend summaries), `care_plan_agent.py` + `care_plan_store.py` (agentic care-plan generation), `triage_summary.py` (structured triage output), `email_service.py` (note and alert delivery), `image_analysis_agent.py` (medical image intake), `voice_transcriber.py` (Whisper transcription).

### 5. Health record and knowledge layer
Persistent patient context, all owned by `user_store.py` (`UserStore`): profile, conditions, medications, allergies, vitals/labs, uploaded records (`document_extractor.py` extracts structured facts on ingest), clinical notes history, cached trial results, and longitudinal memory (`memory_store.py`, refreshed incrementally as the account grows). `context_graph.py` builds a fast, no-LLM relevance graph over this record so retrieval can pull only what's relevant to the current question, and `patient_history.py` structures it into a compact context block for prompts.

### 6. External systems and evidence sources
OpenAI (chat, embeddings, vision, Whisper, image/video generation), NHS Conditions and NICE guidance, Europe PMC and PubMed Central, openFDA drug labels, ClinicalTrials.gov, SMTP (Gmail or any provider) for email, PyMuPDF for PDF parsing and export.

### 7. Governance, audit, and observability
Every gate decision, tool call and evidence tier is reconstructable: `policy_engine.py` logs each of its eight hard gates, `audit_models.py` defines `ClinicalAuditTrace` and `PolicyGateRecord`, and `feedback_store.py` persists anonymised thumbs-up/down quality signals (intent, risk level, role, pathway, evidence tiers, policy gates, alignment flags -- never question or answer text) to PostgreSQL for offline review.

### Component diagram

```mermaid
flowchart TD
    User[Signed-in user] --> React[React client -- frontend/src/App.tsx]
    React --> API[backend/api.py]

    API --> Store[backend/user_store.py]
    API --> RAG[backend/rag_system.py]
    API --> Notes[backend/clinical_notes.py]
    API --> CarePlan[backend/care_plan_agent.py]
    API --> Email[backend/email_service.py]
    API --> Trials[backend/clinical_trials.py]
    API --> GP[backend/gp_summary.py]
    API --> Voice[backend/voice_transcriber.py]
    API --> ImageAgent[backend/image_analysis_agent.py]
    API --> MCP[backend/mcp_server.py]

    RAG --> Orchestrator[backend/clinical_orchestrator.py]
    Orchestrator --> Control[Control plane: moderation_ml, intent_risk_classifier,\nrole_router, policy_engine, pathways/]
    Orchestrator --> Ranker[backend/evidence_ranker.py]
    Orchestrator --> Extractor[backend/evidence_extractor.py]
    Orchestrator --> Templates[backend/response_templates.py]
    Orchestrator --> LLM[backend/summarizer.py]
    Orchestrator --> Evidence[NHS/NICE, Europe PMC, PubMed Central, openFDA]
    Orchestrator --> Memory[backend/memory_store.py]
    Orchestrator --> ContextGraph[backend/context_graph.py]

    CarePlan --> CarePlanStore[backend/care_plan_store.py]
    API --> Feedback[backend/feedback_store.py]

    Store --> Local[users.json and data/uploads]
    Store --> Postgres[(PostgreSQL -- when DATABASE_URL is set)]
    Feedback --> Postgres
```

---

## Quick Start

### 1. Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_VISION_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Email (Gmail SMTP)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your-16-char-app-password
EMAIL_FROM=FlynnMed <your@gmail.com>

# Relational database (required for clinician/patient consent and MRNs)
DATABASE_URL=postgresql+psycopg://flynnmed:flynnmed_dev_only@localhost:5432/flynnmed
DATA_BACKEND=sql

# MCP API key (optional -- protects /mcp endpoint)
MCP_API_KEY=
```

Gmail requires an App Password, not your regular password. Generate one at:
`myaccount.google.com -> Security -> 2-Step Verification -> App passwords`

### 3. Database

Start PostgreSQL and apply the schema before starting FlynnMed:

```powershell
docker compose up -d db
py -m alembic upgrade head
```

For an existing account in `users.json`, migrate it before switching `DATA_BACKEND` to `sql`:

```powershell
# Keep DATA_BACKEND=legacy for these two commands.
py -m backend.scripts.migrate_json_to_sql
py -m backend.scripts.migrate_json_to_sql --verify
```

Then set `DATA_BACKEND=sql` in `.env`. This creates a relational patient row and MRN for each migrated patient account.

### 4. Frontend

```powershell
cd frontend
npm install
npm run build
cd ..
```

### 5. Start the server

```powershell
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

### 6. Frontend development

Keep the backend running on port 8000, then in `frontend/`:

```powershell
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`.

---

## User Roles

FlynnMed adapts its interface and responses to the signed-in user's role.

| Role | What they see |
|---|---|
| Patient | Personal health dashboard, clean response text, care plans, timeline, trial matching, consent request approval/revocation, simplified SOAP note view |
| Doctor | Separate clinical dashboard, consented patient list, MRN-based access requests, read-only patient charts, evidence review, full sources and triage metadata |
| Nurse | Role-adapted notes (Presenting concern / Observations / Nursing assessment / Care plan), editable |
| Midwife | Maternal-focused notes (Maternal concern / Maternal and fetal assessment / Risk assessment / Maternity plan), editable |
| Physiotherapist | MSK-focused notes (Presenting complaint / Physical assessment / Clinical impression / Treatment plan), editable |

Patients never see raw clinical metadata, source lists, evidence tiers, trace IDs or SOAP edit controls. Clinicians receive the full clinical picture. Role labels and per-role terms live in `backend/product_config.py`; the mapping from a stored profile role to UI/prompt behaviour lives in `backend/role_router.py`.

---

## Features

### Account and Access
- Role-aware sign-up with consent gate (GDPR-compliant)
- Role terms shown per clinical role at sign-up
- Password hashed with bcrypt; minimum 8 characters enforced
- Persistent session via JWT stored in localStorage
- Profile editing: display name, email, date of birth, biological sex, care context, organisation

### Evidence Chat
- Streaming evidence-based responses via GPT-4o-mini
- Role-aware clinical workflow: crisis pre-screen, intent classification, risk stratification, tiered evidence retrieval, policy gates and pathway logic applied before every answer
- Medical image analysis for JPG, PNG and WebP uploads: non-medical images are rejected, visual findings are screened first, then the agentic evidence pipeline searches guidance and research before answering
- Follow-up chips after each response: short first-person statements generated from the evidence that the user can tap to refine the answer
- Voice input via OpenAI Whisper (browser microphone permission required)
- Enter to send; Shift+Enter for a new line
- Patient view: clean response text with subtle urgency strip only for high, urgent or crisis levels
- Clinician view: collapsible triage card, source list and evidence basis chips after response text

### Evidence Tiers
Evidence is retrieved and ranked across three tiers:
- Tier 1: NHS guidance and NICE guidelines
- Tier 2: Systematic reviews and Cochrane-style evidence
- Tier 3: Primary research papers from Europe PMC and PubMed Central

### Care Plans
- Agentic, tool-calling care-plan generation (`backend/care_plan_agent.py`) that pulls NHS/NICE guidance and PubMed evidence for a named concern before synthesising a structured plan
- Shared clinical-context adjudication (`backend/clinical_context_guard.py`) resolves cross-specialty measurements from structured type/unit/document evidence, pauses on ambiguity, filters incompatible sources, validates generated plans, and safely migrates older plans that fail the gate
- Plans include goals, medication reminders, lab reminders, escalation thresholds and a missed-care checklist
- Task-level tracking: mark individual care-plan tasks complete via `PATCH /api/care-plans/{plan_id}/tasks/{task_id}`
- After-visit summary generation and GP-appointment prep notes generated from the saved plan
- Plans persisted per user in `backend/care_plan_store.py`

### SOAP Clinical Notes
- Generate a SOAP note from the current conversation at any time
- Role-specific section labels and LLM prompt guidance per clinical role
- Backend formats all fields as clean markdown -- no raw Python dicts or lists
- Clinicians can edit all four sections inline and save changes
- Patients see a simplified read-only view: "What was discussed" and "What happens next"
- Notes stored per user and restored on next login
- Email a note directly to the user's registered email address
- Send a GP alert email for notes flagged as requiring a GP visit

### Health Record
- Symptom log with dates, severity, triggers and notes, plus trend summarisation (`backend/symptom_tracker.py`)
- Medication list with dose, schedule and openFDA interaction checks
- Allergy and adverse drug reaction list
- Conditions list (active and past)
- Vitals and lab readings: blood pressure, heart rate, weight, blood glucose, oxygen saturation, temperature, HbA1c, eGFR and more
- All record sections editable from the chat side panel

### Document Uploads
- PDF upload with anonymisation and patient-name verification before extraction
- Structured extraction of measurements, allergies, medications, conditions, heights, weights and lab values from uploaded documents
- Extracted data added to the user's retrieval context automatically

### Health Timeline
- Scrollable timeline of conditions, medications, allergies, readings, triage summaries and uploaded records
- Trend cards for chartable vital types when at least two readings are saved

### GP Summary Export
- One-click PDF export of the user's saved health record, documents, longitudinal memory and recent triage summaries ready for a GP or hospital appointment

### Clinical Trial Search
- Searches ClinicalTrials.gov for recruiting trials matched to the user's saved conditions, medications and symptom logs
- Deterministic scoring plus model-based clinical alignment scoring
- Ranked results show trial title, phase, location, contact and link to the official record
- Results saved and restored on next login

### Email Delivery
- SOAP notes emailed as formatted HTML to the user's registered address
- Urgent care alert emails for high, urgent or crisis cases
- Sent via Gmail SMTP (or any SMTP provider) using App Password authentication
- Clear error messages if SMTP is not configured

### Feedback and Quality Signals
- Thumbs-up / thumbs-down rating on any assistant response (`POST /api/feedback`)
- Stores only anonymised quality metadata -- intent category, risk level, user role, pathway used, evidence tiers, source count, policy gates and alignment flags
- Never stores question text, answer text, username or email address
- Persisted to PostgreSQL via `backend/feedback_store.py` for offline model and prompt quality review

### Model Context Protocol (MCP) Server
- Mounted at `/mcp` on the same server process -- no separate service needed
- Works in Railway deployments (streamable HTTP) and locally (stdio)
- Optional API key guard via `MCP_API_KEY` environment variable
- Exposes five tools to AI agents and Claude Desktop -- see [Model Context Protocol (MCP) Server](#model-context-protocol-mcp-server) below

### Safety and Moderation
- Crisis pre-screen on every message before the main pipeline
- Eight hard policy gates: crisis, pregnancy, paediatric, medication, diagnosis, elderly, mental health, urgent
- Deterministic role-adaptive moderation rules in the core deployment
- Optional Detoxify scoring for local research environments via `requirements-ml.txt`
- Escalation triggers and safety-netting included in every clinical answer

---

## Project Structure

```text
backend/
  api.py                        FastAPI app: all /api/* endpoints, auth, static file serving, MCP mount
  anonymizer.py                 document redaction helpers
  audit_models.py               ClinicalAuditTrace and PolicyGateRecord dataclasses
  care_plan_agent.py            agentic, tool-calling care-plan generator (NHS/NICE + PubMed)
  care_plan_store.py            per-user care-plan persistence
  clinical_decision_support.py  ClinicalDecision objects with evidence-based triage guidance
  clinical_notes.py             SOAP note generation, role-specific prompts and coercion
  clinical_orchestrator.py      central clinical workflow engine (control/reasoning/evidence/disclosure)
  clinical_trials.py            ClinicalTrials.gov search and scoring
  context_graph.py              fast, no-LLM relevance graph over a user's prior health records
  document_extractor.py         structured extraction from uploaded PDFs
  email_service.py              SMTP sender for notes, verification codes and urgent alerts
  evidence_extractor.py         anti-hallucination, per-source article evidence extraction
  evidence_ranker.py            three-tier source ranking
  evidence_schema.py            Pydantic schema for extracted evidence dossiers
  feedback_store.py             anonymised response-quality signal store (Neon PostgreSQL)
  gp_summary.py                 GP handover PDF generation
  image_analysis_agent.py       medical image intake validation and analysis
  image_generator.py            GPT image generation integration
  intent_risk_classifier.py     intent and risk level classification (regex pre-screen + LLM)
  mcp_server.py                 Model Context Protocol server
  medication_checker.py         openFDA interaction checks
  memory_store.py                longitudinal memory refresh
  moderation_ml.py              role-adaptive Detoxify and regex moderation
  official_guidance.py          NHS and MedlinePlus retrieval
  patient_history.py            structures a patient's known history into a compact prompt context
  pathways/                     five specialty clinical pathways
    general_triage.py
    maternity.py
    msk.py
    medications.py
    chronic_conditions.py
  policy_engine.py              eight hard safety gates
  product_config.py             PRODUCT_NAME, role options, role terms, privacy notice text
  pubmed_search.py              Europe PMC and PubMed Central retrieval
  query_expander.py             query expansion for retrieval
  rag_system.py                 retrieval, generation and document ingestion engine
  response_templates.py         role-specific headings, personas and tier labels
  role_router.py                RoleConfig and RoleRouter per clinical role
  safety_review.py              locked longitudinal medicine and abnormal-result safety checks
  summarizer.py                 LLM wrapper, follow-up chips and SOAP generation helpers
  symptom_tracker.py            symptom trend summarisation
  triage_summary.py             structured triage output
  upload_verification.py        upload name checks and verification helpers
  user_store.py                 accounts, profiles, chat history, notes and persistence
  video_generator.py            medical video generation (Sora-2)
  voice_transcriber.py          voice transcription (Whisper)
  test_*.py                     unit tests (pytest) for the modules above

frontend/
  index.html                    Vite entry HTML
  package.json                  scripts and dependencies
  src/
    App.tsx                     full React app: all views, components and chat logic
    api.ts                      typed API client for all backend endpoints
    styles.css                  design system and component styles
    types.ts                    TypeScript types for all shared data shapes
    utils.ts                    formatting and helper functions
  public/                       static assets

agents/
  commands/                     Claude Code skills

image/
  identity.png                  FlynnMed logo
  architecture.png              reference architecture diagram (embedded above)

Dockerfile                      container build
Procfile                        Railway/Heroku start command
requirements.txt                Python dependencies
requirements-ml.txt             optional local Detoxify/PyTorch dependencies
scripts/start.sh                migrations, legacy account import and production server startup
```

---

## API Reference

All endpoints are served from `backend/api.py` and prefixed `/api` unless noted. Auth endpoints issue an HMAC-signed JWT that must be sent as `Authorization: Bearer <token>` on every other call.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness check |
| GET | `/api/config` | Public product config (name, role options) |
| POST | `/api/auth/signup` | Create an account (role, consent, password) |
| GET | `/api/access` | List the signed-in account's consent requests and active grants |
| POST | `/api/access/requests` | Clinician requests scoped patient access by MRN |
| POST | `/api/access/requests/{grant_id}/decision` | Patient approves or denies a pending request |
| DELETE | `/api/access/requests/{grant_id}` | Patient or clinician revokes/releases access |
| GET | `/api/clinician/patients/{patient_id}` | Read an approved patient chart under an active consent grant |
| POST | `/api/auth/login` | Exchange credentials for a JWT |
| GET | `/api/me` | Current user profile |
| GET | `/api/snapshot` | Full workspace snapshot, including active safety reviews, on load |
| PATCH | `/api/safety-reviews/{review_id}` | Confirm a safety proposal or record whether follow-up happened |
| PUT | `/api/profile` | Update profile fields |
| GET | `/api/terms/{role_label}` | Role-specific terms text shown at signup |
| DELETE | `/api/chat` | Clear chat history |
| POST | `/api/chat/stream` | Stream an evidence-based chat response (SSE) |
| POST | `/api/chat/image/stream` | Stream a response to a medical image upload (SSE) |
| POST | `/api/feedback` | Submit thumbs-up/down quality feedback on a response |
| POST | `/api/uploads` | Upload and process a clinical PDF |
| POST | `/api/voice/transcribe` | Transcribe voice input (Whisper) |
| POST | `/api/symptoms` | Add a symptom log entry |
| DELETE | `/api/symptoms/{log_id}` | Remove a symptom log entry |
| POST | `/api/conditions` | Add a condition |
| DELETE | `/api/conditions/{condition_id}` | Remove a condition |
| POST | `/api/medications` | Add a medication |
| DELETE | `/api/medications/{medication_id}` | Remove a medication |
| POST | `/api/allergies` | Add an allergy |
| DELETE | `/api/allergies/{allergy_id}` | Remove an allergy |
| POST | `/api/vitals` | Add a vitals/lab reading |
| DELETE | `/api/vitals/{vitals_id}` | Remove a vitals/lab reading |
| GET | `/api/care-plans` | List saved care plans |
| POST | `/api/care-plans/generate` | Generate a new agentic care plan for a condition |
| GET | `/api/care-plans/{plan_id}` | Get a single care plan |
| DELETE | `/api/care-plans/{plan_id}` | Delete a care plan |
| PATCH | `/api/care-plans/{plan_id}/tasks/{task_id}` | Toggle a care-plan task's completion state |
| POST | `/api/care-plans/{plan_id}/after-visit` | Generate an after-visit summary from a care plan |
| POST | `/api/care-plans/{plan_id}/gp-prep` | Generate GP-appointment prep notes from a care plan |
| GET | `/api/notes` | List saved SOAP notes |
| POST | `/api/notes` | Generate a SOAP note from the current conversation |
| GET | `/api/notes/{note_id}` | Get a single SOAP note |
| PUT | `/api/notes/{note_id}` | Edit a SOAP note (clinician roles) |
| DELETE | `/api/notes/{note_id}` | Delete a SOAP note |
| POST | `/api/notes/{note_id}/email` | Email a SOAP note to the user |
| POST | `/api/email/urgent` | Send an urgent care alert email |
| GET | `/api/export/account` | Export raw account data (JSON) |
| GET | `/api/export/summary.pdf` | Export a GP-ready health summary PDF |
| POST | `/api/trials/search` | Search ClinicalTrials.gov for matching trials |
| GET | `/api/trials/result` | Get the last saved trial search result |

---

## Model Context Protocol (MCP) Server

`backend/mcp_server.py` mounts at `/mcp` on the same server process -- no separate service needed. It works in Railway deployments (streamable HTTP) and locally (stdio), and is guarded by the optional `MCP_API_KEY` environment variable. It exposes context-scrutiny, validation, evidence, notes, email, and trial tools to AI agents and Claude Desktop:

| Tool | Description |
|---|---|
| `get_patient_context` | Full patient profile, structured measurements, adjudicated context and memory |
| `scrutinize_patient_context` | Resolve the clinical meaning/domain of a topic before retrieval or generation |
| `validate_clinical_output` | Post-generation specialty check for an answer, with a safe replacement if it fails |
| `validate_care_plan_output` | Post-generation specialty check for a care-plan JSON payload |
| `extract_article_evidence` | Structured evidence extraction from a medical article matched to a patient |
| `generate_clinical_note` | Generate and save a SOAP note from a consultation summary |
| `send_health_email` | Send a clinical note or urgent alert by email |
| `search_trials_for_patient` | Search ClinicalTrials.gov for a patient's conditions |

### Claude Desktop (deployed on Railway)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "flynnmed": {
      "url": "https://your-app.railway.app/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_MCP_API_KEY"
      }
    }
  }
}
```

### Claude Desktop (local stdio)

```json
{
  "mcpServers": {
    "flynnmed": {
      "command": "python",
      "args": ["-m", "backend.mcp_server"],
      "cwd": "/path/to/my_health_chatbot"
    }
  }
}
```

Install the `mcp` package first: `pip install mcp`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite |
| API | FastAPI, Uvicorn |
| LLM | OpenAI Chat Completions (gpt-4o-mini) |
| Embeddings | OpenAI text-embedding-3-small |
| Voice | OpenAI Whisper |
| Image analysis | OpenAI vision-capable chat model, then agentic evidence retrieval |
| Image generation | OpenAI gpt-image-1 |
| Video generation | OpenAI sora-2 |
| Biomedical literature | Europe PMC and PubMed Central |
| Official guidance | NHS Conditions and MedlinePlus |
| Drug interactions | openFDA drug label API |
| Clinical trials | ClinicalTrials.gov API v2 |
| Moderation | Deterministic rules, with optional local Detoxify scoring |
| PDF parsing and export | PyMuPDF |
| Email | Gmail SMTP via App Password (or any SMTP provider) |
| MCP | FastMCP streamable HTTP |
| Persistence | Local JSON or PostgreSQL |
| Deployment | Railway, Docker or any ASGI host |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `OPENAI_BASE_URL` | Yes | OpenAI base URL |
| `OPENAI_MODEL` | Yes | Chat model name |
| `OPENAI_VISION_MODEL` | No | Vision-capable model for medical image intake (default: OPENAI_MODEL or gpt-4o) |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model (default: text-embedding-3-small) |
| `DATABASE_URL` | For clinician access | PostgreSQL connection string required for MRNs, consent grants and clinician patient charts |
| `DATA_BACKEND` | For clinician access | Set to `sql` after migrations to enable relational accounts, consent grants, and clinician patient charts |
| `SMTP_HOST` | No | SMTP host (e.g. smtp.gmail.com) |
| `SMTP_PORT` | No | SMTP port (587 for STARTTLS, 465 for SSL) |
| `SMTP_USER` | No | Sender email address |
| `SMTP_PASSWORD` | No | App password for the sender account |
| `EMAIL_FROM` | No | Display name and address, e.g. `FlynnMed <you@gmail.com>` |
| `MCP_API_KEY` | No | Bearer token to restrict access to the /mcp endpoint |
| `APP_SECRET` / `SECRET_KEY` | No | HMAC secret used to sign session JWTs; falls back to a local dev secret if unset -- always set this in production |

---

## Testing

Both suites run in CI on every push/PR to `main` (`.github/workflows/ci.yml`).

### Backend

Unit tests live alongside the modules they cover (`backend/test_*.py`) and run with `pytest`:

```powershell
py -m pip install pytest ruff
py -m pytest backend/
ruff check backend/
```

Current coverage includes evidence ranking, response templates, moderation, image analysis, image generation, video generation, care-plan generation, the clinical orchestrator, intent/risk classification and the clinical-context guard. `backend/test_pubmed_search.py`, `backend/test_rag_engine.py`, and `backend/test_summarizer.py` are live-API manual smoke scripts rather than automated tests (no assertions, need a real `OPENAI_API_KEY` and network access) -- CI excludes them; run them manually with `python -m pytest backend/test_rag_engine.py` when you want to eyeball a real end-to-end answer.

### Frontend

Unit tests use [Vitest](https://vitest.dev/) and React Testing Library, covering the pure-logic modules (`frontend/src/utils.ts`, `frontend/src/api.ts`):

```powershell
cd frontend
npm install
npm test          # single run
npm run test:watch # watch mode
```

There is no component/E2E test suite yet -- `App.tsx` is a single large file with no separately exported components. Verify UI changes by running `npm run dev` against a local backend and exercising the affected flow in the browser.

---

## Deployment

### Docker

The production image uses separate Node build and Python runtime stages. Node, frontend source,
local patient data and optional CUDA/PyTorch moderation packages are not included in the runtime
image. Install `requirements-ml.txt` only in a local research environment that needs Detoxify.

```bash
docker build -t flynnmed .
docker run -p 8000:8000 --env-file .env flynnmed
```

### Railway

1. Connect the repository in Railway.
2. Add PostgreSQL and set `DATABASE_URL`, `JWT_SECRET_KEY` and the required application keys in Railway Variables.
3. Deploy from the repository Dockerfile. The startup script applies Alembic migrations, imports accounts from the former Railway JSON-blob table, switches to `DATA_BACKEND=sql`, and then starts Uvicorn.
4. Confirm the deployment log reaches `Starting FlynnMed` and the health check succeeds.
5. The MCP server is available at `https://your-app.railway.app/mcp`.

The legacy account import is idempotent. Existing patient accounts receive a relational patient row
and MRN during the first successful deployment, while subsequent deployments skip those accounts.

---

## PostgreSQL

For hosted or shared deployments, set `DATABASE_URL` so account data, consent grants, and clinician
patient access persist between deployments. The production startup script sets `DATA_BACKEND=sql`
after applying migrations.

1. Create a PostgreSQL database (Neon, Supabase or any provider).
2. Add `DATABASE_URL` to the environment.
3. Run `alembic upgrade head`.
4. If upgrading an existing JSON deployment, run `python -m backend.scripts.migrate_json_to_sql`
   and verify it before cutover.
5. Set `DATA_BACKEND=sql` and restart the server.

---

## Troubleshooting

**`OPENAI_API_KEY not found`**
Create `.env` with a valid API key and restart the server from the project root.

**Clinician Access shows `Database not connected` instead of an MRN**
Start PostgreSQL, set `DATABASE_URL`, run `py -m alembic upgrade head`, migrate any existing `users.json` accounts, set `DATA_BACKEND=sql`, and restart FlynnMed. An MRN is created with the relational patient row, so FlynnMed does not fabricate one while that database is unavailable.

**Accounts or trial results disappear after a deployment**
Set `DATABASE_URL` so the app uses PostgreSQL. Local `users.json` does not persist across Railway deployments.

**Email button returns an error**
Check that `SMTP_HOST`, `SMTP_USER` and `SMTP_PASSWORD` are set in `.env` and that the server was restarted after editing the file. For Gmail, use an App Password generated at `myaccount.google.com -> Security -> App passwords`, not your regular Gmail password.

**User gets "no email address saved" error**
The email is sent to the address stored in the user's FlynnMed profile. The user must have registered with a valid email or updated their profile email in app settings.

**PDF says the patient name cannot be found**
The upload checker reads the document text and filename. Make sure the full name on the account matches the name in the document. If it differs, the user can choose to continue anyway.

**Extracted data from a PDF looks wrong**
Review the entries in the trackers and remove anything incorrect. The extractor reads free-text documents and may occasionally misread a value, unit or date.

**Clinical trial search returns no results**
The trial finder needs saved health context such as conditions, symptoms or medications. Also confirm the server can make outbound HTTPS requests to `clinicaltrials.gov`.

**Voice input is unavailable**
Allow microphone access in the browser and confirm the backend has a valid OpenAI API key for Whisper transcription.

**MCP tools fail in Claude Desktop**
Confirm `MCP_API_KEY` in your environment matches the key in `claude_desktop_config.json`. For local stdio mode, install the `mcp` package with `pip install mcp` first.

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, code style, and pull request expectations. Everyone participating is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Found a security issue? Please follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

---

## License

MIT -- see [LICENSE](LICENSE).

---

## Important Note

FlynnMed is for health education, evidence review and clinical decision support. It is not a substitute for emergency care, a clinical diagnosis or a qualified clinician's judgement.

If someone may be seriously unwell, use the appropriate urgent care route -- NHS 111 or 999 in the UK.
