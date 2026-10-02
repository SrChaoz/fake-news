#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIRECTORY="$PROJECT_ROOT/.local/postgres/data"
SOCKET_DIRECTORY="$PROJECT_ROOT/.local/postgres/socket"
LOG_FILE="$PROJECT_ROOT/.local/postgres/postgres.log"
PORT="54329"
DATABASE_NAME="fakenews-clima"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Falta el comando requerido: $1. Instala PostgreSQL y vuelve a intentarlo." >&2
    exit 1
  fi
}

connection_url() {
  printf 'postgresql://%s@127.0.0.1:%s/%s\n' "$(id -un)" "$PORT" "$DATABASE_NAME"
}

is_running() {
  pg_isready --host 127.0.0.1 --port "$PORT" >/dev/null 2>&1
}

initialize() {
  require_command initdb
  mkdir -p "$(dirname "$DATA_DIRECTORY")" "$SOCKET_DIRECTORY"
  if [ -f "$DATA_DIRECTORY/PG_VERSION" ]; then
    echo "PostgreSQL local ya está inicializado en $DATA_DIRECTORY."
    return
  fi
  initdb --auth=trust --username="$(id -un)" --pgdata="$DATA_DIRECTORY"
  echo "PostgreSQL local inicializado en $DATA_DIRECTORY."
}

start() {
  require_command pg_ctl
  require_command pg_isready
  initialize
  if is_running; then
    echo "PostgreSQL local ya está en ejecución: $(connection_url)"
    return
  fi
  pg_ctl --pgdata="$DATA_DIRECTORY" --log="$LOG_FILE" start \
    --options="-h 127.0.0.1 -p $PORT -k $SOCKET_DIRECTORY -c listen_addresses=127.0.0.1"
  if ! is_running; then
    echo "PostgreSQL no pudo iniciar. Revisa $LOG_FILE." >&2
    exit 1
  fi
  echo "PostgreSQL local iniciado: $(connection_url)"
}

stop() {
  require_command pg_ctl
  if [ ! -f "$DATA_DIRECTORY/PG_VERSION" ]; then
    echo "PostgreSQL local no está inicializado."
    return
  fi
  if ! is_running; then
    echo "PostgreSQL local ya está detenido."
    return
  fi
  pg_ctl --pgdata="$DATA_DIRECTORY" stop --mode=fast
  echo "PostgreSQL local detenido."
}

status() {
  require_command pg_isready
  if is_running; then
    echo "PostgreSQL local disponible: $(connection_url)"
    return
  fi
  echo "PostgreSQL local no está en ejecución. Inícialo con: scripts/local_postgres.sh start" >&2
  exit 1
}

case "${1:-}" in
  init) initialize ;;
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "Uso: $0 {init|start|stop|status}" >&2
    exit 2
    ;;
esac
