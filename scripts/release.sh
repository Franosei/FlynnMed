#!/bin/sh
# Run once per deployment, before application instances receive traffic.
set -eu

case "${DATABASE_URL:-}" in
    postgresql://*)
        DATABASE_URL="postgresql+psycopg://${DATABASE_URL#postgresql://}"
        export DATABASE_URL
        ;;
esac

if [ -z "${DATABASE_URL:-}" ]; then
    echo "DATABASE_URL is required for the release phase." >&2
    exit 1
fi

echo "Applying database migrations..."
python -m alembic upgrade head

if [ "${MIGRATE_LEGACY_ACCOUNTS:-false}" = "true" ]; then
    echo "Running the explicitly requested one-time legacy account migration..."
    DATA_BACKEND=legacy python -m backend.scripts.migrate_json_to_sql --source legacy-postgres
fi

if [ "${SEED_DEMO_ACCOUNTS:-false}" = "true" ]; then
    echo "Seeding fictional demo accounts..."
    DATA_BACKEND=sql python -m backend.scripts.seed_demo_accounts
fi
