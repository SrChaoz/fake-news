#!/usr/bin/env python3
"""Exporta datos reproducibles de Climate Veritas a SQL sin historial operativo.

Genera ``docs/database_data.sql`` usando pg_dump. Por defecto contiene solo
``dataset_records`` y ``dataset_evidence``; ``prediction_history`` se excluye
porque puede almacenar texto introducido por usuarios. Use ``--include-history``
solo si se revisó que esos registros son aptos para compartir.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "database_data.sql"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Incluye prediction_history; revisar antes de compartir por posible contenido de usuarios.",
    )
    return parser.parse_args()


def pg_dump_url(database_url: str) -> str:
    """Convierte la URL SQLAlchemy psycopg a una URL que entienda pg_dump."""
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def main() -> int:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL es obligatoria.", file=sys.stderr)
        return 2
    if not shutil.which("pg_dump"):
        print("ERROR: pg_dump no está instalado o no está en PATH.", file=sys.stderr)
        return 2

    parsed = urlparse(pg_dump_url(database_url))
    if parsed.scheme not in {"postgresql", "postgres"} or not parsed.path.strip("/"):
        print("ERROR: DATABASE_URL debe ser una URL PostgreSQL válida.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "pg_dump",
        "--data-only",
        "--inserts",
        "--no-owner",
        "--no-privileges",
        "--table=dataset_records",
        "--table=dataset_evidence",
        "--file",
        str(args.output),
        pg_dump_url(database_url),
    ]
    if args.include_history:
        command.insert(-3, "--table=prediction_history")

    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        args.output.unlink(missing_ok=True)
        print(result.stderr.strip() or "ERROR: pg_dump no pudo exportar la base de datos.", file=sys.stderr)
        return result.returncode

    # El archivo puede contener textos con contenido de datasets públicos, pero
    # nunca se imprime para evitar volcar sus datos en la consola.
    row_hint = "incluyendo historial" if args.include_history else "sin historial de predicciones"
    print(f"Exportación creada: {args.output} ({row_hint}).")
    print("Restaura con: psql -d <nombre_bd> -f docs/schema.sql && psql -d <nombre_bd> -f " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
