#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIRECTORY="$PROJECT_ROOT/.venv"
FRONTEND_DIRECTORY="$PROJECT_ROOT/frontend"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Falta el comando requerido: $1." >&2
    exit 1
  fi
}

for command in python node npm initdb pg_ctl pg_isready psql; do
  require_command "$command"
done

if [ ! -d "$VENV_DIRECTORY" ]; then
  python -m venv "$VENV_DIRECTORY"
fi

"$VENV_DIRECTORY/bin/python" -m pip install --upgrade pip
"$VENV_DIRECTORY/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"

(cd "$FRONTEND_DIRECTORY" && npm ci)

"$PROJECT_ROOT/scripts/local_postgres.sh" start

if [ ! -f "$PROJECT_ROOT/.env" ]; then
  cat > "$PROJECT_ROOT/.env" <<EOF
DATABASE_URL=postgresql://$(id -un)@127.0.0.1:54329/fakenews-clima
EOF
  echo "Configuración local creada en .env."
else
  echo "Se preservó la configuración existente en .env."
fi

if [ ! -f "$FRONTEND_DIRECTORY/.env.local" ]; then
  cp "$FRONTEND_DIRECTORY/.env.example" "$FRONTEND_DIRECTORY/.env.local"
  echo "Configuración local creada en frontend/.env.local."
else
  echo "Se preservó la configuración existente en frontend/.env.local."
fi

set -a
# shellcheck disable=SC1091
. "$PROJECT_ROOT/.env"
set +a
"$VENV_DIRECTORY/bin/python" "$PROJECT_ROOT/scripts/bootstrap_database.py"

"$VENV_DIRECTORY/bin/python" - <<'PY'
from transformers import AutoModel, AutoTokenizer

AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
AutoModel.from_pretrained("prajjwal1/bert-tiny")
print("Encoder prajjwal1/bert-tiny disponible en la caché local.")
PY

echo "Preparación local completada."
