#!/bin/sh
set -eu

case "${DATABASE_URL:-}" in
    postgresql://*)
        DATABASE_URL="postgresql+psycopg://${DATABASE_URL#postgresql://}"
        export DATABASE_URL
        ;;
esac

export DATA_BACKEND="${DATA_BACKEND:-sql}"

echo "Starting FlynnMed on port ${PORT:-8000}..."
exec uvicorn backend.api:app --host 0.0.0.0 --port "${PORT:-8000}"
