#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIRECTORY="$PROJECT_ROOT/frontend"
if [ ! -d "$FRONTEND_DIRECTORY/node_modules" ] || [ ! -f "$FRONTEND_DIRECTORY/.env.local" ]; then
  echo "Ejecuta scripts/setup_local.sh antes de iniciar el frontend." >&2
  exit 1
fi
cd "$FRONTEND_DIRECTORY"
exec npm run dev -- --hostname 127.0.0.1 --port 3000
