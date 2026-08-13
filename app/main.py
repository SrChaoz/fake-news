"""API REST para inferencia y trazabilidad de desinformación ambiental."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.schemas import BatchPredictionRequest, BatchPredictionResponse, HistoryItem, HistoryResponse, PredictionRequest, PredictionResponse
from app.services.feature_extractor import build_feature_vector
from app.services.evidence_engine import verify_with_evidence
from app.services.ner_engine import extract_environmental_entities
from app.services.ontology_engine import detect_ontological_conflicts
from database import SessionLocal
from models import PredictionHistory

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_EVALUATION_PATH = MODEL_DIRECTORY / "holdout_evaluation.json"
PROMOTED_HOLDOUT_REPORT_PATH = MODEL_DIRECTORY / "experiments" / "holdout_bert" / "holdout_evaluation.json"
LOCAL_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8080", "http://127.0.0.1:8080"]
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

# La evaluación se inicia exclusivamente sobre el artefacto promocionado y el
# benchmark retenido local. Nunca recibe comandos, rutas ni argumentos del
# navegador. Así la UI puede pedir una evaluación sin convertirse en una vía
# para ejecutar procesos arbitrarios en el servidor.
_evaluation_lock = threading.Lock()
_holdout_evaluation_state: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
}


@asynccontextmanager
async def lifespan(_: FastAPI) -> Generator[None, None, None]:
    """Valida los artefactos esenciales al iniciar, sin cargar BERT aún."""
    try:
        load_artifacts()
        load_metrics()
    except RuntimeError as error:
        raise RuntimeError(f"No se pudo iniciar la API: {error}") from error
    yield


app = FastAPI(title="FakeNews Climático API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_ORIGINS,
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_db() -> Generator[Session, None, None]:
    """Proporciona una sesión por petición y siempre la cierra."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    """Carga el modelo, vectorizador y configuración de producción."""
    try:
        model = joblib.load(MODEL_DIRECTORY / "random_forest_model.pkl")
        vectorizer = joblib.load(MODEL_DIRECTORY / "tfidf.pkl")
        with (MODEL_DIRECTORY / "model_config.json").open(encoding="utf-8") as file:
            config = json.load(file)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("No hay artefactos de modelo válidos en app/models.") from error
    return model, vectorizer, config


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    """Carga las métricas persistidas por el entrenamiento de producción."""
    try:
        with (MODEL_DIRECTORY / "metrics.json").open(encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("No existen métricas de entrenamiento válidas.") from error


@lru_cache(maxsize=1)
def load_transformer_encoder(model_name: str) -> tuple[Any, Any, str]:
    """Carga el encoder local requerido por los artefactos BERT."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("El artefacto requiere torch y transformers instalados.") from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        encoder = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
        encoder.eval()
    except Exception as error:
        raise RuntimeError(f"No se pudo cargar el encoder Transformer {model_name}: {error}") from error
    return tokenizer, encoder, device


def build_transformer_embedding(text: str, model_name: str) -> np.ndarray:
    """Genera un embedding CLS coherente con el entrenamiento."""
    import torch

    tokenizer, encoder, device = load_transformer_encoder(model_name)
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        return encoder(**encoded).last_hidden_state[:, 0, :].cpu().numpy()


def normalize_claim(text: str, supplied_claim: str | None = None) -> str:
    """Usa el claim suministrado o una normalización ligera del texto."""
    return " ".join((supplied_claim if supplied_claim and supplied_claim.strip() else text).split())


def infer_category(claim: str) -> str:
    """Asigna una categoría ambiental de alto nivel."""
    lowered = claim.lower()
    if any(word in lowered for word in ("pollution", "pollutant", "plastic", "air quality")):
        return "Pollution"
    if any(word in lowered for word in ("forest", "deforestation", "species", "biodiversity", "reef")):
        return "Biodiversity"
    if any(word in lowered for word in ("energy", "solar", "wind", "coal", "fossil", "renewable")):
        return "Energy"
    return "Climate Change"


def build_inference_features(text: str, claim: str) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Replica el feature engineering usado por el modelo de producción."""
    model, vectorizer, config = load_artifacts()
    entities = extract_environmental_entities(claim)
    ontology = detect_ontological_conflicts(claim, entities)
    text_representation = vectorizer.transform([f"{text} [CLAIM] {claim}"])
    # Algunos experimentos competitivos usan TF-IDF de palabra + carácter con
    # regresión logística. Conservan las reglas ontológicas para la decisión,
    # pero no concatenan señales/embeddings estadísticos.
    if config.get("feature_mode") == "text_tfidf_only":
        features = text_representation.toarray().astype(np.float32)
        if features.shape[1] != getattr(model, "n_features_in_", features.shape[1]):
            raise RuntimeError("Los artefactos del modelo de texto no son compatibles entre sí; vuelve a entrenar.")
        return features, entities, ontology
    metadata = build_feature_vector(text, claim, ontology)["metadata"]
    symbolic = np.asarray([[metadata["text_word_count"], metadata["claim_word_count"], metadata["claim_readability"], metadata["linked_entity_count"], int(metadata["has_conflict"])]], dtype=np.float32)
    tfidf = text_representation.toarray().astype(np.float32)
    features = np.hstack([tfidf, symbolic])
    transformer = config.get("transformer", {})
    if transformer.get("enabled"):
        model_name = transformer.get("model")
        if not isinstance(model_name, str) or not model_name:
            raise RuntimeError("El artefacto Transformer no contiene el nombre del encoder.")
        features = np.hstack([features, build_transformer_embedding(f"{text} [CLAIM] {claim}", model_name)])
    if features.shape[1] != getattr(model, "n_features_in_", features.shape[1]):
        raise RuntimeError("Los artefactos del modelo no son compatibles entre sí; vuelve a entrenar.")
    return features, entities, ontology


def analyze_prediction(text: str, supplied_claim: str | None = None) -> PredictionResponse:
    """Ejecuta el pipeline neuro-simbólico sin acceder a PostgreSQL."""
    claim = normalize_claim(text, supplied_claim)
    model, _, config = load_artifacts()
    features, entities, ontology = build_inference_features(text, claim)
    probability_real = float(model.predict_proba(features)[0, 1])
    threshold = float(config.get("decision_threshold_real", 0.5))
    model_prediction = "REAL" if probability_real >= threshold else "FAKE"
    conflicts = ontology.get("conflicts", [])
    has_conflict = bool(ontology.get("has_conflict"))
    has_ontology_support = bool(ontology.get("has_support"))
    evidence_result = verify_with_evidence(claim)
    evidence_status = str(evidence_result.get("status", "INSUFFICIENT_EVIDENCE"))
    if evidence_status not in {"SUPPORTED", "REFUTED", "INSUFFICIENT_EVIDENCE"}:
        evidence_status = "INSUFFICIENT_EVIDENCE"
    evidence_matches = evidence_result.get("matches", [])
    prediction = "FAKE" if has_conflict else "REAL" if has_ontology_support or evidence_status == "SUPPORTED" else "FAKE" if evidence_status == "REFUTED" else model_prediction
    confidence = 1.0 if has_conflict else (
        1.0
        if has_ontology_support
        else
        max(float(evidence_result.get("top_similarity", 0.0)), 0.85)
        if evidence_status in {"SUPPORTED", "REFUTED"}
        else probability_real if prediction == "REAL" else 1 - probability_real
    )
    if has_conflict:
        explanation = "La afirmación contradice una relación ambiental conocida; se clasificó como FAKE por una regla ontológica."
        source = "ontology_rule_override"
        verification_status = "ONTOLOGY_CONFLICT"
    elif has_ontology_support:
        explanation = "La afirmación expresa una relación ambiental respaldada por una regla ontológica conocida."
        source = "ontology_rule_support"
        verification_status = "ONTOLOGY_SUPPORT"
    elif evidence_status == "SUPPORTED":
        explanation = "Se recuperó evidencia curada con alta similitud que respalda la afirmación."
        source = "evidence_retrieval_override"
        verification_status = "SUPPORTED"
    elif evidence_status == "REFUTED":
        explanation = "Se recuperó evidencia curada con alta similitud que refuta la afirmación."
        source = "evidence_retrieval_override"
        verification_status = "REFUTED"
    elif entities:
        names = ", ".join(entity["text"] for entity in entities[:3])
        explanation = f"No se detectaron conflictos ontológicos explícitos. Entidades analizadas: {names}."
        source = "hybrid_random_forest"
        verification_status = "INSUFFICIENT_EVIDENCE"
    else:
        explanation = "No se detectaron entidades ambientales específicas ni conflictos ontológicos explícitos."
        source = "hybrid_random_forest"
        verification_status = "INSUFFICIENT_EVIDENCE"
    # Las coincidencias débiles se descartan de la respuesta para no mostrar una
    # fuente tangencial como si fuese justificación de la predicción.
    evidence_for_response = evidence_matches if evidence_status in {"SUPPORTED", "REFUTED"} else []
    return PredictionResponse(extracted_claim=claim, category=infer_category(claim), detected_entities=entities, ontological_conflicts=conflicts, prediction=prediction, confidence_score=round(confidence, 4), explanation=explanation, probability_real=round(probability_real, 4), decision_threshold_real=threshold, model_prediction=model_prediction, decision_source=source, verification_status=verification_status, evidence=evidence_for_response)


def persist_predictions(db: Session, inputs: list[str], results: list[PredictionResponse]) -> None:
    """Almacena un lote de predicciones dentro de una sola transacción."""
    try:
        db.add_all(PredictionHistory(input_text=text, extracted_claim=result.extracted_claim, category=result.category, detected_entities=result.detected_entities, ontological_conflicts=result.ontological_conflicts, prediction=result.prediction, confidence_score=result.confidence_score, explanation=result.explanation) for text, result in zip(inputs, results, strict=True))
        db.commit()
    except SQLAlchemyError as error:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No se pudo guardar el historial de predicciones.") from error


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/evaluation/holdout")
def holdout_evaluation() -> dict[str, Any]:
    """Devuelve el estado y el último informe del benchmark retenido."""
    try:
        report = load_holdout_evaluation()
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    with _evaluation_lock:
        state = dict(_holdout_evaluation_state)
    return {"evaluation": state, "report": report}


@app.post("/evaluation/holdout", status_code=status.HTTP_202_ACCEPTED)
def start_holdout_evaluation() -> dict[str, Any]:
    """Inicia una evaluación asíncrona y segura del modelo actualmente promovido.

    El endpoint no admite rutas ni nombres de experimento: siempre ejecuta el
    benchmark retenido contra ``app/models`` para evitar ejecución arbitraria.
    """
    with _evaluation_lock:
        if _holdout_evaluation_state["status"] == "running":
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ya hay una evaluación holdout en ejecución.")
        _holdout_evaluation_state.update({
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
            "error": None,
        })
    threading.Thread(target=_run_holdout_evaluation, name="holdout-evaluation", daemon=True).start()
    return {"evaluation": dict(_holdout_evaluation_state), "message": "Evaluación iniciada; consulta GET /evaluation/holdout para ver el estado."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    """Analiza un texto y persiste el resultado en ``prediction_history``."""
    try:
        result = analyze_prediction(request.text, request.claim)
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    persist_predictions(db, [request.text], [result])
    return result


def load_holdout_evaluation() -> dict[str, Any] | None:
    """Lee el último informe de holdout del modelo promovido si existe."""
    try:
        return json.loads(HOLDOUT_EVALUATION_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # El artefacto se promovió desde holdout_bert. Antes de ejecutar una
        # reevaluación desde la UI, se conserva su informe validado original.
        try:
            return json.loads(PROMOTED_HOLDOUT_REPORT_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            raise RuntimeError("El informe de evaluación holdout no es válido.") from error
    except (OSError, ValueError) as error:
        raise RuntimeError("El informe de evaluación holdout no es válido.") from error


def _run_holdout_evaluation() -> None:
    """Ejecuta el evaluador aislado y actualiza el estado visible por API."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_holdout.py"),
        "--model-directory",
        str(MODEL_DIRECTORY),
        "--output",
        str(HOLDOUT_EVALUATION_PATH),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        with _evaluation_lock:
            _holdout_evaluation_state["finished_at"] = datetime.now(UTC).isoformat()
            if completed.returncode == 0:
                _holdout_evaluation_state.update({"status": "completed", "error": None})
            else:
                detail = (completed.stderr or completed.stdout or "El evaluador terminó con error.").strip()
                _holdout_evaluation_state.update({"status": "failed", "error": detail[-1_500:]})
    except (OSError, subprocess.SubprocessError) as error:
        with _evaluation_lock:
            _holdout_evaluation_state.update({
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": str(error),
            })


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest, db: Session = Depends(get_db)) -> BatchPredictionResponse:
    """Analiza hasta 100 textos y los persiste en una única transacción."""
    try:
        results = [analyze_prediction(text) for text in request.texts]
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    persist_predictions(db, request.texts, results)
    return BatchPredictionResponse(predictions=results)


@app.get("/history", response_model=HistoryResponse)
def history(limit: int = Query(default=20, ge=1, le=100), db: Session = Depends(get_db)) -> HistoryResponse:
    """Devuelve los últimos análisis almacenados."""
    try:
        records = db.scalars(select(PredictionHistory).order_by(PredictionHistory.created_at.desc()).limit(limit)).all()
    except SQLAlchemyError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No se pudo consultar el historial.") from error
    return HistoryResponse(items=[HistoryItem.model_validate(record) for record in records])


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Devuelve Accuracy, Precision, Recall, F1 y matriz de confusión."""
    data = load_metrics()
    return {"model": "full_hybrid", "metrics": data.get("ablation_study", {}).get("full_hybrid", {}), "cross_validation": data.get("cross_validation", {}), "dataset": data.get("dataset", {})}


@app.get("/ablation")
def ablation() -> dict[str, Any]:
    """Expone el estudio de ablación del modelo de producción."""
    return load_metrics().get("ablation_study", {})


@app.get("/experiments")
def experiments() -> dict[str, Any]:
    """Compara componentes disponibles y experimentos guardados con honestidad."""
    data = load_metrics()
    _, _, config = load_artifacts()
    experiment_dir = MODEL_DIRECTORY / "experiments"
    saved: list[dict[str, Any]] = []
    for path in sorted(experiment_dir.glob("*/metrics.json")) if experiment_dir.is_dir() else []:
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            saved.append({"name": path.parent.name, "dataset": content.get("dataset", {}), "cross_validation": content.get("cross_validation", {}).get("summary", {})})
        except (OSError, ValueError):
            continue
    return {"hybrid_model": data.get("ablation_study", {}).get("full_hybrid", {}), "transformer_baseline": data.get("baseline", {}), "bert": {"status": "enabled" if config.get("transformer", {}).get("enabled") else "not_enabled", "configuration": config.get("transformer", {})}, "roberta": {"status": "not_trained", "detail": "No existe un artefacto RoBERTa entrenado."}, "ontology_system": {"status": "enabled", "detail": "Reglas RDF aplicadas como restricción de alta prioridad."}, "saved_experiments": saved}
