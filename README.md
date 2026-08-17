<p align="center">
  <img src="image/identity.png" alt="FlynnMed" width="360">
</p>

# FlynnMed

FlynnMed is a clinical safety, evidence review and continuity workspace for patients, carers and healthcare professionals. It combines a structured health record with role-aware chat, deterministic safety checks, clinical workflow tools and traceable evidence retrieval.

[![CI](https://github.com/Franosei/my_health_chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/Franosei/my_health_chatbot/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-1f9c94.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-0f1f3d.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/React-18-0f1f3d.svg)](https://react.dev/)

> [!IMPORTANT]
> FlynnMed provides health education and clinical decision support. It does not diagnose, prescribe, replace professional judgement or provide emergency care. If someone may be seriously unwell, use the appropriate urgent care route. In the UK, call NHS 111 for urgent advice or 999 for an emergency.

## Overview

The application supports two connected workspaces:

- Patients and carers can maintain a health record, review supported safety findings, ask evidence-based questions, create care plans, prepare clinical notes, search for clinical trials and control clinician access.
- Healthcare professionals can ask evidence questions, request consent-based access to patient records, prepare pre-visit summaries, discuss a consented record and draft medication proposals for patient release.

The React and TypeScript client is served by a FastAPI application. PostgreSQL stores relational accounts, patient records, consent grants, evidence lineage and audit data. The clinical pipeline retrieves live guidance and research, extracts relevant evidence, applies safety and policy checks, and returns a role-appropriate response.

## Safety and evidence model

FlynnMed separates deterministic record checks from generative clinical support.

### Safety review

The patient safety review uses locked rules in `backend/safety_review.py`. The current rule set is deliberately narrow and covers:

- severe and moderate potassium results;
- recently recorded emergency symptoms;
- exact medicine and allergy conflicts; and
- warfarin recorded with selected non-steroidal anti-inflammatory medicines.

Each finding links to the patient facts and guidance passage used to produce it. Emergency instructions appear before any confirmation step. Patient confirmation records agreement to send a proposal for review, but does not represent clinician approval.

The workflow is:

```text
Saved record changes
  -> deterministic safety checks
  -> linked patient facts and guidance
  -> bounded next-step proposal
  -> patient confirmation
  -> clinician review
  -> follow-up outcome
```

SMART on FHIR interfaces are present under `backend/fhir/`, but the only current provider is a non-connected stub. Health record write-back is therefore unavailable.

### Evidence chat

Chat requests pass through a governed pipeline that performs:

1. crisis screening, moderation, intent classification and risk classification;
2. role and clinical pathway selection;
3. retrieval from official guidance, biomedical literature and relevant clinical data services;
4. evidence extraction, ranking, contradiction handling and provenance capture;
5. policy checks and role-appropriate response generation; and
6. claim, evidence and interaction trace persistence where the relational workflow applies.

The main live sources are NHS and NICE guidance, MedlinePlus, Europe PMC, PubMed Central, openFDA and ClinicalTrials.gov. Results depend on the availability and quality of those external services.

### Clinical evidence pipeline

The following diagram shows how user input, patient context, evidence retrieval, evidence quality checks, claim verification and governance records connect across the application.

![FlynnMed clinical evidence and response pipeline](image/flynnmed_pipeline.png)

## Main capabilities

| Area | Capabilities |
| --- | --- |
| Accounts and roles | Patient, caregiver, doctor, nurse, midwife, physiotherapist and other clinician roles, with role-specific terms and views |
| Health record | Conditions, medicines, allergies, symptoms, observations, laboratory results, relationships, uploads and longitudinal summaries |
| Safety review | Deterministic supported-risk checks with fact and evidence lineage, confirmation and follow-up states |
| Evidence chat | Streaming responses, urgency handling, citations, follow-up prompts and role-specific presentation |
| Multimodal input | PDF health record ingestion, image analysis and voice transcription |
| Care planning | Evidence-informed care plans, task tracking, appointment preparation and after-visit notes |
| Clinical notes | SOAP-style and role-adapted notes, editing, PDF summary export and optional email delivery |
| Clinician workflows | MRN-based access requests, patient consent, pre-visit summaries, record-scoped chat and medication proposals |
| Trial matching | Recruiting study search using saved health context and ClinicalTrials.gov records |
| Governance | Consent records, audit events, evidence ledgers, answer-claim lineage and anonymised feedback metadata |
| MCP | Optional Model Context Protocol server for selected clinical tools over HTTP or local standard input/output |

## Architecture

![FlynnMed architecture](image/architecture.png)

At a high level, requests move through the following components:

```text
React client
  -> FastAPI routes and authentication
  -> patient record and consent services
  -> clinical orchestration and policy controls
  -> evidence retrieval, extraction and ranking
  -> role-aware response or workflow output
  -> PostgreSQL evidence, audit and application records
```

Key implementation areas are:

| Path | Purpose |
| --- | --- |
| `frontend/src/` | React user interface, API client, shared types and tests |
| `backend/api.py` | FastAPI application, REST endpoints, streaming endpoints, MCP mount and frontend hosting |
| `backend/clinical_orchestrator.py` | Central clinical control, retrieval and synthesis workflow |
| `backend/rag_system.py` | Retrieval-augmented generation facade and response assembly |
| `backend/safety_review.py` | Deterministic patient safety review rules |
| `backend/evidence_*.py` | Evidence schemas, extraction, ranking, quality checks, traces and persistence |
| `backend/policy_engine.py` | Hard policy gates for clinical responses |
| `backend/pathways/` | General triage, medicines, chronic condition, maternity and musculoskeletal pathways |
| `backend/models/` | SQLAlchemy relational models |
| `backend/repositories/` | PostgreSQL-backed application stores |
| `backend/fhir/` | EHR interface, FHIR resources and the current non-connected provider |
| `migrations/` | Alembic database migrations |
| `evaluations/` | HealthBench and tiered RAG evaluation harness |

## Technology

- Python 3.12 or later, FastAPI, Uvicorn, SQLAlchemy and Alembic
- PostgreSQL 16 for relational application data
- React 18, TypeScript and Vite
- OpenAI models for response generation, embeddings, vision and transcription
- PyMuPDF for PDF processing and export
- Optional Detoxify-based local moderation in addition to deterministic rules
- FastMCP for Model Context Protocol tools
- Pytest, Ruff, Vitest and React Testing Library for quality checks

## Getting started

### Prerequisites

- Python 3.12 or 3.13
- Node.js 20 or later
- Docker Desktop, or another accessible PostgreSQL 16 instance
- An OpenAI API key for AI-assisted features

### 1. Install the backend

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Optional local Detoxify moderation requires the larger machine-learning dependency set:

```powershell
py -m pip install -r requirements-ml.txt
```

The core application safely falls back to deterministic moderation rules when Detoxify is not installed.

### 2. Configure the environment

Copy the example file and add your credentials:

```powershell
Copy-Item .env.example .env
```

For a local SQL-backed setup, the essential values are:

```env
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql+psycopg://flynnmed:flynnmed_dev_only@localhost:5432/flynnmed
DATA_BACKEND=sql
ENVIRONMENT=development
APP_SECRET=replace_with_a_long_random_value
JWT_SECRET_KEY=replace_with_a_different_long_random_value
```

Do not commit `.env` or real patient data.

### 3. Start PostgreSQL and apply migrations

```powershell
docker compose up -d db
py -m alembic upgrade head
```

### 4. Start the backend

```powershell
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

The health endpoint is available at `http://127.0.0.1:8000/api/health`, and the interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 5. Start the frontend

In another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` to the backend on port 8000.

To serve a production frontend build from FastAPI instead:

```powershell
cd frontend
npm ci
npm run build
cd ..
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

## Configuration

The main environment variables are listed below. Evaluation-specific variables are documented in [`evaluations/README.md`](evaluations/README.md).

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required by the clinical pipeline and other AI-assisted features |
| `OPENAI_BASE_URL` | OpenAI-compatible API base URL, defaulting to `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Main chat model, defaulting to `gpt-4o-mini` |
| `OPENAI_VISION_MODEL` | Optional model override for document and image analysis |
| `OPENAI_EMBEDDING_MODEL` | Embedding model, defaulting to `text-embedding-3-small` |
| `DATABASE_URL` | PostgreSQL connection string for relational accounts, patient records, consent and audit workflows |
| `DATA_BACKEND` | Set to `sql` for the relational application store. `legacy` remains for migration and isolated evaluation use |
| `APP_SECRET` or `SECRET_KEY` | Signs the session tokens currently issued by `backend/api.py`. Set a strong value outside local development |
| `JWT_SECRET_KEY` | Secret used by the SQL-backed JWT utilities. Set a strong value for deployed environments |
| `ENVIRONMENT` | Runtime mode, normally `development` or `production` |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD` | Optional SMTP delivery for notes and urgent alerts |
| `EMAIL_FROM` | Optional sender name and address |
| `MCP_API_KEY` | Optional bearer token protecting the HTTP MCP endpoint. Set it before exposing `/mcp` |
| `EHR_PROVIDER` | EHR provider selection. Only `none` is implemented at present |
| `VITE_API_BASE_URL` | Optional frontend API base URL |
| `VITE_DEV_PROXY_TARGET` | Optional Vite development proxy target |

## Data migration

The SQL backend is the recommended application configuration. The legacy JSON store remains available for migration and isolated evaluation runs.

To inspect and migrate an existing `users.json` store:

```powershell
$env:DATA_BACKEND = "legacy"
py -m backend.scripts.migrate_json_to_sql --dry-run
py -m backend.scripts.migrate_json_to_sql
py -m backend.scripts.migrate_json_to_sql --verify
```

Restart the application with `DATA_BACKEND=sql` after verification. The migration is idempotent and does not delete the legacy files.

## Testing and evaluation

Install the development tools, then run the backend checks:

```powershell
py -m pip install pytest ruff==0.15.9
py -m ruff check backend/
py -m pytest backend/
```

Run the frontend checks from `frontend/`:

```powershell
npm test
npm run build
```

The GitHub Actions workflow contains the exact CI command and excludes manual live-API smoke scripts that require network access and a real API key.

The evaluation harness exercises the production RAG pipeline against HealthBench datasets and computes grounding, relevance, citation, calibration and safety metrics. It is an automated benchmark, not clinical validation. See [`evaluations/README.md`](evaluations/README.md) for configuration, datasets and reporting commands.

## Docker and deployment

The Dockerfile builds the React client in a Node stage, installs the Python runtime in a separate stage, applies database migrations at startup and serves the API and built client from one container.

For a local containerised deployment:

```powershell
Copy-Item .env.example .env
# Complete .env before continuing.
docker compose up --build
```

Open `http://127.0.0.1:8000` and confirm `http://127.0.0.1:8000/api/health` returns an OK response.

For Railway or another container platform:

1. provision PostgreSQL and set `DATABASE_URL`;
2. set `DATA_BACKEND=sql`, the OpenAI credentials and strong application secrets;
3. configure SMTP only if email delivery is required;
4. set `MCP_API_KEY` if the MCP endpoint will be exposed; and
5. deploy using the repository Dockerfile.

The startup script applies Alembic migrations before starting Uvicorn. It also performs an idempotent import from the former Railway legacy PostgreSQL store when that data is present.

## Model Context Protocol

The optional MCP server exposes selected clinical tools, including patient context retrieval, context checking, clinical output validation, note generation, trial search and email delivery.

- HTTP transport is mounted at `/mcp` by the FastAPI application.
- Local standard input/output transport runs with `python -m backend.mcp_server`.
- `MCP_API_KEY` protects the HTTP route with a bearer token.

Because these tools can access sensitive records and trigger email, do not expose the endpoint without authentication and appropriate operational controls. See [`agents/commands/mcp-server.md`](agents/commands/mcp-server.md) for client configuration examples.

## Known limitations

- The deterministic safety review covers a small, explicit rule set and is not a complete clinical review.
- Generative outputs and extracted document values can be incorrect and require human verification.
- The FHIR layer is an interface and stub only. No live EHR connection or write-back provider is implemented.
- Several features depend on live third-party APIs and may be unavailable when those services fail or rate-limit requests.
- The evaluation suite measures automated benchmark performance and does not establish clinical safety or regulatory approval.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow and pull request expectations. Contributors must follow the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Report security concerns through [`SECURITY.md`](SECURITY.md), not a public issue.

## Licence

FlynnMed is available under the MIT Licence. See [`LICENSE`](LICENSE).
