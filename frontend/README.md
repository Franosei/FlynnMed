# FlynnMed React Frontend

Mobile-first React client for the production branch.

The main patient journey is the **Safety review** view. It turns supported changes in saved results, symptoms, medicines and allergies into a traceable evidence-to-action review. Each review shows the facts used, guidance passage, uncertainty, bounded proposed action, required approver, write-back state and follow-up outcome.

Emergency findings are displayed first and do not wait for confirmation. Patient confirmation only records agreement to send a proposal for review; it does not represent clinician approval. SMART-on-FHIR write-back remains disabled while the backend reports no configured provider.

```powershell
npm install
npm run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`. Clinician access and MRNs also require PostgreSQL and the current migrations. From the repository root, start them before FastAPI:

```powershell
docker compose up -d db
py -m alembic upgrade head
py -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

For deployment, build the client:

```powershell
npm run build
```

FastAPI serves `frontend/dist` automatically when the build folder exists.
