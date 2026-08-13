"""Modelos persistentes de la aplicación de detección de desinformación."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Clase base para todos los modelos ORM."""


class DatasetRecord(Base):
    """Registro original procedente de un conjunto de datos etiquetado."""

    __tablename__ = "dataset_records"
    __table_args__ = (
        CheckConstraint("label IN ('REAL', 'FAKE')", name="ck_dataset_records_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_claim: Mapped[str | None] = mapped_column(Text, nullable=True)
    label: Mapped[str] = mapped_column(String(10), nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(100), default="climate-fever", nullable=False)


class DatasetEvidence(Base):
    """Proveniencia y corrección científica asociada a un registro de entrenamiento."""

    __tablename__ = "dataset_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_record_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_dataset: Mapped[str] = mapped_column(String(100), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )


class PredictionHistory(Base):
    """Historial de predicciones realizadas por el modelo."""

    __tablename__ = "prediction_history"
    __table_args__ = (
        CheckConstraint("prediction IN ('REAL', 'FAKE')", name="ck_prediction_history_prediction"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_claim: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    detected_entities: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    ontological_conflicts: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    prediction: Mapped[str] = mapped_column(String(10), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
