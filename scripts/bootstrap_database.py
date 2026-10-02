#!/usr/bin/env python3
"""Crea y carga la base local sin destruir datos existentes."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal
from urllib.parse import SplitResult, unquote, urlsplit, urlunsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "docs" / "schema.sql"
DATA_PATH = PROJECT_ROOT / "docs" / "database_data.sql"
DATABASE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z")


def seed_action(record_count: int, evidence_count: int) -> Literal["seed", "preserve"]:
    """Decide si es seguro cargar el volcado reproducible."""
    if record_count == 0 and evidence_count == 0:
        return "seed"
    if record_count > 0 and evidence_count > 0:
        return "preserve"
    raise RuntimeError(
        "La base contiene datos de entrenamiento parciales; no se modificó. "
        "Revísala antes de intentar restaurar el volcado."
    )


def parse_database_url(database_url: str) -> tuple[SplitResult, str]:
    """Valida la URL y devuelve una variante compatible con psycopg/psql."""
    parts = urlsplit(database_url)
    if parts.scheme not in {"postgresql", "postgresql+psycopg"}:
        raise RuntimeError("DATABASE_URL debe usar postgresql:// o postgresql+psycopg://.")
    database_name = unquote(parts.path.lstrip("/"))
    if not DATABASE_NAME_PATTERN.fullmatch(database_name):
        raise RuntimeError("El nombre de la base en DATABASE_URL contiene caracteres no permitidos.")
    plain_parts = parts._replace(scheme="postgresql")
    return plain_parts, database_name


def replace_database(parts: SplitResult, database_name: str) -> str:
    return urlunsplit(parts._replace(path=f"/{database_name}"))


def require_psycopg() -> object:
    try:
        import psycopg
    except ModuleNotFoundError as error:
        raise RuntimeError("Falta psycopg. Ejecuta scripts/setup_local.sh primero.") from error
    return psycopg


def run_psql(database_url: str, source: Path) -> None:
    subprocess.run(
        ["psql", "--set", "ON_ERROR_STOP=1", "--dbname", database_url, "--file", str(source)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def create_database_if_missing(database_url: str) -> None:
    parts, database_name = parse_database_url(database_url)
    psycopg = require_psycopg()
    maintenance_url = replace_database(parts, "postgres")
    try:
        with psycopg.connect(maintenance_url, autocommit=True) as connection:
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (database_name,)
            ).fetchone()
            if not exists:
                connection.execute(
                    psycopg.sql.SQL("CREATE DATABASE {} ").format(psycopg.sql.Identifier(database_name))
                )
                print(f"Base de datos '{database_name}' creada.")
    except Exception as error:
        raise RuntimeError(f"No se pudo conectar a PostgreSQL: {error}") from error


def dataset_counts(database_url: str) -> tuple[int, int]:
    psycopg = require_psycopg()
    try:
        with psycopg.connect(database_url) as connection:
            record_count = connection.execute("SELECT count(*) FROM dataset_records").fetchone()[0]
            evidence_count = connection.execute("SELECT count(*) FROM dataset_evidence").fetchone()[0]
    except Exception as error:
        raise RuntimeError(f"No se pudieron consultar los datos locales: {error}") from error
    return int(record_count), int(evidence_count)


def main() -> int:
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        print("DATABASE_URL es obligatoria. Consulta .env.example.", file=sys.stderr)
        return 1
    try:
        parts, _ = parse_database_url(database_url)
        plain_url = urlunsplit(parts)
        create_database_if_missing(plain_url)
        run_psql(plain_url, SCHEMA_PATH)
        action = seed_action(*dataset_counts(plain_url))
        if action == "seed":
            run_psql(plain_url, DATA_PATH)
            print("Esquema y datos de entrenamiento restaurados.")
        else:
            print("Datos existentes preservados; no se alteró el historial de predicciones.")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"No se pudo preparar la base de datos: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
