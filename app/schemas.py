"""Esquemas Pydantic v2 de la API pública."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Solicitud de análisis de una publicación ambiental."""

    text: str = Field(min_length=1, max_length=20_000)
    claim: str | None = Field(default=None, max_length=20_000)


class BatchPredictionRequest(BaseModel):
    """Solicitud para analizar varios textos en una sola transacción."""

    texts: list[str] = Field(min_length=1, max_length=100)


class PredictionResponse(BaseModel):
    """Resultado explicable de una predicción individual."""

    extracted_claim: str
    category: str
    detected_entities: list[dict[str, Any]]
    ontological_conflicts: list[dict[str, Any]]
    prediction: Literal["REAL", "FAKE"]
    confidence_score: float = Field(ge=0, le=1)
    explanation: str
    probability_real: float = Field(ge=0, le=1)
    decision_threshold_real: float = Field(ge=0, le=1)
    model_prediction: Literal["REAL", "FAKE"]
    decision_source: str
    verification_status: Literal["SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE", "ONTOLOGY_CONFLICT", "ONTOLOGY_SUPPORT"]
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HistoryItem(BaseModel):
    """Fila serializable de ``prediction_history``."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    input_text: str
    extracted_claim: str | None
    category: str | None
    detected_entities: list[dict[str, Any]] | None
    ontological_conflicts: list[dict[str, Any]] | None
    prediction: Literal["REAL", "FAKE"]
    confidence_score: float
    explanation: str | None
    created_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
