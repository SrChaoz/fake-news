#!/usr/bin/env python3
"""Crea las tablas de la aplicación cuando aún no existen."""

from __future__ import annotations

import sys


def main() -> int:
    """Inicializa el esquema y devuelve un código de salida apropiado."""
    try:
        from sqlalchemy.exc import SQLAlchemyError
    except ModuleNotFoundError as error:
        print(
            f"No se pudo inicializar la base de datos: falta una dependencia ({error}).",
            file=sys.stderr,
        )
        return 1

    try:
        from database import engine
        from models import Base  # Importa y registra todos los modelos en Base.metadata.

        Base.metadata.create_all(bind=engine)
    except (ModuleNotFoundError, RuntimeError, SQLAlchemyError) as error:
        print(f"No se pudo inicializar la base de datos: {error}", file=sys.stderr)
        return 1

    print("Base de datos inicializada correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
