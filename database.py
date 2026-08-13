"""Configuración de la conexión y sesiones de SQLAlchemy."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def get_database_url() -> str:
    """Obtiene la URL de conexión obligatoria desde el entorno.

    Raises:
        RuntimeError: Si ``DATABASE_URL`` no está definida o está vacía.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "La variable de entorno DATABASE_URL es obligatoria. "
            "Consulta .env.example para conocer el formato esperado."
        )
    return database_url


def get_sqlalchemy_url(database_url: str) -> str:
    """Adapta una URL PostgreSQL estándar al driver ``psycopg`` v3.

    De este modo, ``postgresql://...`` (el formato habitual en variables de
    entorno) funciona sin requerir el driver legado ``psycopg2``.
    """
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


DATABASE_URL = get_database_url()

# ``pool_pre_ping`` descarta conexiones inactivas antes de reutilizarlas.
engine: Engine = create_engine(get_sqlalchemy_url(DATABASE_URL), pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False)
