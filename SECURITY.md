# Security Policy

FlynnMed handles health-related information and account credentials, so we
take security reports seriously and will respond promptly.

## Supported Versions

This project does not currently maintain multiple release branches. Security
fixes are applied to the `main` branch and the latest deployed version only.

| Version         | Supported          |
| --------------- | ------------------- |
| `main` (latest)  | :white_check_mark:  |
| Older commits    | :x:                  |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**
Public issues are visible to everyone immediately, including anyone who might
exploit the report before a fix ships.

Instead, report it privately using one of these channels, in order of
preference:

1. **GitHub Security Advisories** (preferred): open a
   [private security advisory](https://github.com/Franosei/FlynnMed/security/advisories/new)
   for this repository. This lets us discuss and fix the issue with you before
   it's public.
2. **Email**: send details to **oseifrancis633@gmail.com** with the subject
   line `SECURITY: <short description>`.

Please include as much of the following as you can:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof-of-concept (request/response, script, etc.)
- The affected file(s), endpoint(s), or component(s), if known
- Whether the issue requires authentication, and with what role/account type
- Any suggested remediation, if you have one

### What to expect

- **Acknowledgement** within 3 business days of your report.
- We will investigate, confirm severity, and aim to provide an initial
  assessment within 7 business days.
- We will keep you informed as a fix is developed and let you know before any
  public disclosure or advisory is published.
- We will credit reporters in the advisory/release notes unless you prefer to
  remain anonymous.

## Scope

Security reports are welcome for:

- Authentication/authorization bypass (`backend/auth/`, account JWTs,
  consent checks, or the optional MCP actor boundary)
- Injection issues (prompt injection into clinical output, path traversal in
  document uploads, SQL/command injection)
- Data exposure across accounts (one user's records, uploads, or chat history
  becoming visible to another)
- Anonymization/PII redaction bypass in uploaded documents
  (`backend/anonymizer.py`, `backend/anonymization_agent.py`)
- Denial-of-service vectors specific to this codebase (not generic
  infrastructure DoS)
- Supply-chain issues in this repository's own dependencies or CI

### Out of scope

- This is an educational/portfolio clinical-AI platform, **not** a certified
  medical device and not currently deployed with real patient data under a
  clinical governance framework. Reports asking us to attest to HIPAA/GDPR/
  MHRA certification are out of scope -- data-handling hardening suggestions
  are still welcome as regular issues or PRs.
- Vulnerabilities in third-party dependencies should be reported upstream
  (though letting us know so we can update is appreciated).
- Missing security headers/best-practices with no demonstrated exploit are
  best raised as a regular GitHub issue rather than a private report.

## Local Data Handling

The supported application path stores accounts and health data in PostgreSQL.
The former legacy stores exist only for offline migration and isolated test or
evaluation workflows. Treat database backups, local migration inputs, and
`.env` (API keys and `JWT_SECRET_KEY`) as sensitive and do not commit or share
them. Do not log raw patient prompts, patient records, or integration tool
arguments.

## Deployed Security Controls

- Access JWTs are short lived and bound to server-side sessions. Refresh
  tokens are held in `HttpOnly`, `SameSite=Strict` cookies, rotate on every
  use, and token reuse revokes the whole session family.
- New accounts cannot use patient-data routes until a short-lived email code
  is verified. Self-declared clinical roles remain patient accounts until an
  out-of-band administrator reviews their professional registration.
- Sensitive endpoints use Postgres-backed limits shared across all web
  processes. Production fails closed if the limiter database is unavailable.
- Clinical model calls pass through the PHI policy gateway. Production must
  explicitly approve the processor host and models and enable PHI processing;
  direct identifiers are redacted and logs contain request metadata only.
- PDF and image uploads are parsed and rewritten before analysis or storage;
  active PDF content, attachments, metadata, image EXIF, trailing bytes, and
  oversized page/pixel payloads are rejected or removed.
- Run `scripts/release.sh` once before deployment to apply migrations. Normal
  application startup never migrates, imports legacy users, or seeds accounts.
