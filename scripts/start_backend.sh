#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ] || [ ! -f "$PROJECT_ROOT/.env" ]; then
  echo "Ejecuta scripts/setup_local.sh antes de iniciar la API." >&2
  exit 1
fi
"$PROJECT_ROOT/scripts/local_postgres.sh" start
set -a
. "$PROJECT_ROOT/.env"
set +a
exec "$PROJECT_ROOT/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
