#!/bin/sh
set -eu

if [ -n "${DATABASE_URL:-}" ]; then
    case "$DATABASE_URL" in
        postgresql://*)
            DATABASE_URL="postgresql+psycopg://${DATABASE_URL#postgresql://}"
            export DATABASE_URL
            ;;
    esac

    echo "Applying database migrations..."
    python -m alembic upgrade head
    echo "Migrating any legacy Railway accounts..."
    DATA_BACKEND=legacy python -m backend.scripts.migrate_json_to_sql --source legacy-postgres
    export DATA_BACKEND=sql
    if [ "${SEED_DEMO_ACCOUNTS:-true}" = "true" ]; then
        echo "Seeding fictional demo accounts..."
        python -m backend.scripts.seed_demo_accounts
    fi
else
    echo "DATABASE_URL is not set. Clinician access and MRNs will be unavailable."
fi

echo "Starting FlynnMed on port ${PORT:-8000}..."
exec uvicorn backend.api:app --host 0.0.0.0 --port "${PORT:-8000}"
