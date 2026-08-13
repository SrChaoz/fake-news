#!/usr/bin/env python3
"""Entrena modelos híbridos para detección de desinformación ambiental.

Ejecutar desde la raíz del proyecto con DATABASE_URL configurada:
    python scripts/train_models.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.feature_extractor import build_feature_vector  # noqa: E402
from app.services.ner_engine import extract_environmental_entities  # noqa: E402
from app.services.ontology_engine import detect_ontological_conflicts  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402


LOGGER = logging.getLogger(__name__)
MODEL_DIRECTORY = PROJECT_ROOT / "app" / "models"
LABEL_TO_TARGET = {"FAKE": 0, "REAL": 1}
RANDOM_STATE = 42
# La negación cambia el significado factual de una afirmación. La lista por
# defecto de scikit-learn elimina ``no``, ``not`` y ``never``, por lo que no es
# apropiada para verificación de afirmaciones.
NEGATION_TERMS = frozenset({"no", "not", "nor", "never", "neither", "without"})
STOP_WORDS_PRESERVING_NEGATION = sorted(ENGLISH_STOP_WORDS - NEGATION_TERMS)


@dataclass(slots=True)
class TrainingRecord:
    """Registro tipado que alimenta el proceso de entrenamiento."""

    raw_text: str
    claim: str
    target: int
    original_label: str | None


def parse_arguments() -> argparse.Namespace:
    """Obtiene parámetros de entrenamiento reproducibles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de validación.")
    parser.add_argument("--estimators", type=int, default=200, help="Árboles por Random Forest.")
    parser.add_argument("--max-features", type=int, default=5_000, help="Máximo de n-gramas TF-IDF.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "climate-fever-dataset-r1-enriched.jsonl",
        help="JSONL usado para identificar etiquetas originales y casos ambiguos.",
    )
    parser.add_argument(
        "--include-ambiguous",
        action="store_true",
        help="Incluye NOT_ENOUGH_INFO/DISPUTED previamente mapeados a FAKE. Por defecto se excluyen.",
    )
    parser.add_argument(
        "--exclude-claims-file",
        type=Path,
        default=None,
        help="JSONL de benchmark cuyas afirmaciones se excluyen del entrenamiento.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds estratificados para validación cruzada.")
    parser.add_argument(
        "--experiment-name",
        default="production",
        help="Nombre de experimento; valores distintos de production se guardan sin reemplazar la API.",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="No descarga/carga BERT; usa una línea base TF-IDF para el modelo base.",
    )
    parser.add_argument(
        "--transformer-model",
        default=None,
        help="Modelo Hugging Face; por defecto bert-base-uncased con GPU o prajjwal1/bert-tiny en CPU.",
    )
    parser.add_argument(
        "--legacy-stop-words",
        action="store_true",
        help="Usa la lista inglesa original que elimina negaciones; solo para comparar experimentos históricos.",
    )
    arguments = parser.parse_args()
    if not 0 < arguments.test_size < 0.5:
        parser.error("--test-size debe estar entre 0 y 0.5.")
    if arguments.estimators < 1 or arguments.max_features < 1 or arguments.cv_folds < 2:
        parser.error("--estimators, --max-features y --cv-folds deben ser válidos.")
    if Path(arguments.experiment_name).name != arguments.experiment_name:
        parser.error("--experiment-name debe ser un nombre simple de directorio.")
    return arguments


def build_tfidf_vectorizer(max_features: int, legacy_stop_words: bool = False) -> TfidfVectorizer:
    """Crea TF-IDF preservando negaciones críticas para fact-checking."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english" if legacy_stop_words else STOP_WORDS_PRESERVING_NEGATION,
        max_features=max_features,
        lowercase=True,
        sublinear_tf=True,
    )


def set_random_seed() -> None:
    """Fija semillas para resultados comparables."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    try:
        import torch

        torch.manual_seed(RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_STATE)
    except ImportError:
        pass


def load_original_labels(dataset_path: Path) -> dict[str, str]:
    """Indexa la etiqueta original por claim_normalized sin tocar PostgreSQL."""
    if not dataset_path.is_file():
        raise RuntimeError(f"No se encontró el dataset de procedencia: {dataset_path}")
    labels: dict[str, str] = {}
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            item = json.loads(line)
            claim = item.get("claim_normalized") or item.get("claim")
            label = item.get("claim_label")
            if isinstance(claim, str) and isinstance(label, str):
                labels[claim.strip()] = label
    return labels


def load_excluded_claims(path: Path | None) -> set[str]:
    """Carga los claims de un benchmark retenido para impedir fuga de datos."""
    if path is None:
        return set()
    if not path.is_file():
        raise RuntimeError(f"No existe el archivo de exclusión: {path}")
    claims: set[str] = set()
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Benchmark inválido en línea {line_number}: {error}") from error
            claim = item.get("claim")
            if isinstance(claim, str) and claim.strip():
                claims.add(" ".join(claim.casefold().split()))
    if not claims:
        raise RuntimeError("El archivo de exclusión no contiene afirmaciones válidas.")
    return claims


def load_training_records(
    dataset_path: Path, include_ambiguous: bool, excluded_claims: set[str] | None = None
) -> list[TrainingRecord]:
    """Carga textos, afirmaciones y etiquetas binarias desde PostgreSQL."""
    try:
        from database import SessionLocal
        from models import DatasetRecord

        with SessionLocal() as session:
            rows = session.execute(
                select(
                    DatasetRecord.raw_text,
                    DatasetRecord.extracted_claim,
                    DatasetRecord.label,
                    DatasetRecord.source,
                ).where(
                    DatasetRecord.label.in_(LABEL_TO_TARGET)
                )
            ).all()
    except SQLAlchemyError as error:
        raise RuntimeError(f"No se pudieron leer los registros de PostgreSQL: {error}") from error

    original_labels = load_original_labels(dataset_path)
    records: list[TrainingRecord] = []
    excluded_ambiguous = 0
    excluded_holdout = 0
    excluded_claims = excluded_claims or set()
    for raw_text, extracted_claim, label, source in rows:
        if not raw_text or label not in LABEL_TO_TARGET:
            continue
        claim = extracted_claim or raw_text
        if " ".join(claim.casefold().split()) in excluded_claims:
            excluded_holdout += 1
            continue
        original_label = original_labels.get(claim)
        if (
            not include_ambiguous
            and source == "climate-fever-enriched"
            and original_label in {"NOT_ENOUGH_INFO", "DISPUTED"}
        ):
            excluded_ambiguous += 1
            continue
        records.append(
            TrainingRecord(
                raw_text=raw_text,
                claim=claim,
                target=LABEL_TO_TARGET[label],
                original_label=original_label,
            )
        )
    if excluded_ambiguous:
        LOGGER.info("Casos ambiguos excluidos del entrenamiento: %d", excluded_ambiguous)
    if excluded_holdout:
        LOGGER.info("Casos del benchmark retenido excluidos del entrenamiento: %d", excluded_holdout)
    if len({record.target for record in records}) < 2:
        raise RuntimeError("Se requieren registros REAL y FAKE para entrenar.")
    return records


def build_symbolic_features(records: list[TrainingRecord], use_claim: bool) -> np.ndarray:
    """Extrae longitud, legibilidad y señales ontológicas para cada registro."""
    features: list[list[float]] = []
    for index, record in enumerate(records, start=1):
        analysis_text = record.claim if use_claim else record.raw_text
        entities = extract_environmental_entities(analysis_text)
        ontology_results = detect_ontological_conflicts(analysis_text, entities)
        # Se reutiliza el servicio para garantizar que las métricas sean idénticas
        # a las que consumirá la API. El TF-IDF global se ajusta más abajo.
        extracted = build_feature_vector(record.raw_text, analysis_text, ontology_results)
        metadata = extracted["metadata"]
        row = [
            float(metadata["text_word_count"]),
            float(metadata["claim_word_count"]),
            float(metadata["claim_readability"]),
            float(metadata["linked_entity_count"]),
            float(metadata["has_conflict"]),
        ]
        features.append(row)
        if index % 100 == 0:
            LOGGER.info("Variables neuro-simbólicas procesadas: %d/%d", index, len(records))
    return np.asarray(features, dtype=np.float32)


def build_transformer_embeddings(
    texts: list[str], model_name: str | None, disabled: bool
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Obtiene embeddings [CLS] de BERT, sin ajustar el modelo base.

    En CPU selecciona ``prajjwal1/bert-tiny`` para una ejecución rápida. Si el
    modelo no está instalado ni puede descargarse, devuelve ``None`` y el resto
    del entrenamiento continúa con TF-IDF.
    """
    if disabled:
        return None, {"enabled": False, "reason": "deshabilitado por --no-transformer"}
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        return None, {"enabled": False, "reason": "transformers o torch no instalados"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_model = model_name or ("bert-base-uncased" if device == "cuda" else "prajjwal1/bert-tiny")
    try:
        tokenizer = AutoTokenizer.from_pretrained(selected_model)
        model = AutoModel.from_pretrained(selected_model).to(device)
        model.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 16):
                encoded = tokenizer(
                    texts[start : start + 16],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                batches.append(model(**encoded).last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(batches), {
            "enabled": True,
            "model": selected_model,
            "device": device,
            "embedding_dimension": int(batches[0].shape[1]),
        }
    except Exception as error:  # Descarga/caché de Hugging Face es opcional.
        LOGGER.warning("No se pudieron obtener embeddings Transformer: %s", error)
        return None, {"enabled": False, "reason": str(error), "requested_model": selected_model}


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Convierte métricas de clasificación a un objeto serializable JSON."""
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "macro_f1": round(float(np.mean(per_class_f1)), 4),
        "per_class": {
            "FAKE": {
                "precision": round(float(per_class_precision[0]), 4),
                "recall": round(float(per_class_recall[0]), 4),
                "f1_score": round(float(per_class_f1[0]), 4),
            },
            "REAL": {
                "precision": round(float(per_class_precision[1]), 4),
                "recall": round(float(per_class_recall[1]), 4),
                "f1_score": round(float(per_class_f1[1]), 4),
            },
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def fit_random_forest(
    name: str,
    matrix: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    estimators: int,
    threshold: float = 0.5,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Entrena y evalúa una variante del estudio de ablación."""
    classifier = RandomForestClassifier(
        n_estimators=estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(matrix[train_indices], targets[train_indices])
    probabilities = classifier.predict_proba(matrix[test_indices])[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    scores = metrics_for(targets[test_indices], predictions)
    scores["decision_threshold_real"] = round(threshold, 4)
    LOGGER.info("%s — accuracy=%.4f, f1=%.4f", name, scores["accuracy"], scores["f1_score"])
    return classifier, scores


def calibrate_threshold(
    combined_texts: list[str],
    symbolic_features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    max_features: int,
    estimators: int,
    embeddings: np.ndarray | None = None,
    target_fake_recall: float = 0.6,
    legacy_stop_words: bool = False,
) -> dict[str, Any]:
    """Selecciona un umbral de REAL con OOF del train, priorizando detectar FAKE."""
    train_targets = targets[train_indices]
    splits = min(5, int(min(np.sum(train_targets == 0), np.sum(train_targets == 1))))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probabilities = np.zeros(len(train_indices), dtype=np.float32)
    for fold, (inner_train, inner_validation) in enumerate(splitter.split(train_indices, train_targets), start=1):
        train_rows = train_indices[inner_train]
        validation_rows = train_indices[inner_validation]
        vectorizer = build_tfidf_vectorizer(max_features, legacy_stop_words)
        train_tfidf = vectorizer.fit_transform([combined_texts[index] for index in train_rows]).toarray()
        validation_tfidf = vectorizer.transform([combined_texts[index] for index in validation_rows]).toarray()
        classifier = RandomForestClassifier(
            n_estimators=estimators,
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
            class_weight="balanced",
        )
        train_features = np.hstack([train_tfidf, symbolic_features[train_rows]])
        validation_features = np.hstack([validation_tfidf, symbolic_features[validation_rows]])
        if embeddings is not None:
            train_features = np.hstack([train_features, embeddings[train_rows]])
            validation_features = np.hstack([validation_features, embeddings[validation_rows]])
        classifier.fit(train_features, targets[train_rows])
        oof_probabilities[inner_validation] = classifier.predict_proba(
            validation_features
        )[:, 1]

    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.3, 0.8, 26):
        candidate_metrics = metrics_for(train_targets, (oof_probabilities >= threshold).astype(np.int8))
        candidate_metrics["threshold"] = round(float(threshold), 2)
        candidates.append(candidate_metrics)

    eligible = [
        item for item in candidates if item["per_class"]["FAKE"]["recall"] >= target_fake_recall
    ]
    pool = eligible or candidates
    best = max(
        pool,
        key=lambda item: (
            item["macro_f1"],
            item["balanced_accuracy"],
            item["per_class"]["FAKE"]["recall"],
        ),
    )
    LOGGER.info(
        "Umbral calibrado=%.2f (macro-F1=%.4f, recall FAKE=%.4f)",
        best["threshold"],
        best["macro_f1"],
        best["per_class"]["FAKE"]["recall"],
    )
    return {
        "threshold_real": best["threshold"],
        "target_fake_recall": target_fake_recall,
        "selection_metrics": best,
        "candidates": candidates,
    }


def run_cross_validation(
    combined_texts: list[str],
    symbolic_features: np.ndarray,
    targets: np.ndarray,
    max_features: int,
    estimators: int,
    folds: int,
    threshold: float,
    embeddings: np.ndarray | None = None,
    legacy_stop_words: bool = False,
) -> dict[str, Any]:
    """Valida el híbrido con folds estratificados y TF-IDF ajustado por fold."""
    smallest_class = int(min(np.sum(targets == 0), np.sum(targets == 1)))
    actual_folds = min(folds, smallest_class)
    if actual_folds < 2:
        raise RuntimeError("No hay suficientes ejemplos por clase para validación cruzada.")

    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics: list[dict[str, Any]] = []
    for fold_number, (train_indices, test_indices) in enumerate(splitter.split(combined_texts, targets), start=1):
        vectorizer = build_tfidf_vectorizer(max_features, legacy_stop_words)
        train_tfidf = vectorizer.fit_transform([combined_texts[index] for index in train_indices]).toarray()
        test_tfidf = vectorizer.transform([combined_texts[index] for index in test_indices]).toarray()
        classifier = RandomForestClassifier(
            n_estimators=estimators,
            random_state=RANDOM_STATE + fold_number,
            n_jobs=-1,
            class_weight="balanced",
        )
        train_features = np.hstack([train_tfidf, symbolic_features[train_indices]])
        test_features = np.hstack([test_tfidf, symbolic_features[test_indices]])
        if embeddings is not None:
            train_features = np.hstack([train_features, embeddings[train_indices]])
            test_features = np.hstack([test_features, embeddings[test_indices]])
        classifier.fit(train_features, targets[train_indices])
        probabilities = classifier.predict_proba(test_features)[:, 1]
        scores = metrics_for(targets[test_indices], (probabilities >= threshold).astype(np.int8))
        scores["decision_threshold_real"] = round(threshold, 4)
        scores["fold"] = fold_number
        fold_metrics.append(scores)
        LOGGER.info("CV fold %d/%d — accuracy=%.4f, f1=%.4f", fold_number, actual_folds, scores["accuracy"], scores["f1_score"])

    return {
        "folds": fold_metrics,
        "summary": {
            metric: {
                "mean": round(float(np.mean([fold[metric] for fold in fold_metrics])), 4),
                "std": round(float(np.std([fold[metric] for fold in fold_metrics])), 4),
            }
            for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "macro_f1")
        },
    }


def save_error_analysis(
    records: list[TrainingRecord],
    targets: np.ndarray,
    classifier: RandomForestClassifier,
    matrix: np.ndarray,
    test_indices: np.ndarray,
    threshold: float,
) -> int:
    """Exporta falsos positivos y falsos negativos para revisión humana."""
    probabilities = classifier.predict_proba(matrix[test_indices])[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    errors: list[dict[str, Any]] = []
    for row_index, predicted, probability in zip(test_indices, predictions, probabilities, strict=True):
        if targets[row_index] == predicted:
            continue
        record = records[row_index]
        entities = extract_environmental_entities(record.claim)
        ontology_results = detect_ontological_conflicts(record.claim, entities)
        errors.append(
            {
                "raw_text": record.raw_text,
                "claim": record.claim,
                "original_label": record.original_label,
                "actual": "REAL" if targets[row_index] else "FAKE",
                "predicted": "REAL" if predicted else "FAKE",
                "confidence_real": round(float(probability), 4),
                "entities": entities,
                "ontological_conflicts": ontology_results["conflicts"],
            }
        )
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIRECTORY / "error_analysis.json").open("w", encoding="utf-8") as output_file:
        json.dump(errors, output_file, ensure_ascii=False, indent=2)
    summary = {
        "decision_threshold_real": threshold,
        "total_errors": len(errors),
        "false_negatives_fake": sum(item["actual"] == "FAKE" for item in errors),
        "false_positives_fake": sum(item["actual"] == "REAL" for item in errors),
        "errors_with_ontological_conflict": sum(bool(item["ontological_conflicts"]) for item in errors),
    }
    with (MODEL_DIRECTORY / "error_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
    return len(errors)


def main() -> int:
    """Ejecuta la carga, entrenamiento, ablación y persistencia de artefactos."""
    global MODEL_DIRECTORY
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_arguments()
    if arguments.experiment_name != "production":
        MODEL_DIRECTORY = MODEL_DIRECTORY / "experiments" / arguments.experiment_name
    set_random_seed()

    try:
        excluded_claims = load_excluded_claims(arguments.exclude_claims_file)
        records = load_training_records(arguments.dataset, arguments.include_ambiguous, excluded_claims)
        targets = np.asarray([record.target for record in records], dtype=np.int8)
        class_counts = {"FAKE": int(np.sum(targets == 0)), "REAL": int(np.sum(targets == 1))}
        LOGGER.info("Registros cargados: %d (FAKE=%d, REAL=%d)", len(records), class_counts["FAKE"], class_counts["REAL"])

        combined_texts = [f"{record.raw_text} [CLAIM] {record.claim}" for record in records]
        raw_texts = [record.raw_text for record in records]
        indices = np.arange(len(records))
        train_indices, test_indices = train_test_split(
            indices,
            test_size=arguments.test_size,
            random_state=RANDOM_STATE,
            stratify=targets,
        )
        tfidf = build_tfidf_vectorizer(arguments.max_features, arguments.legacy_stop_words)
        # El vectorizador se ajusta solo con entrenamiento para evitar que el
        # vocabulario e IDF del conjunto de validación influyan en la evaluación.
        tfidf.fit([combined_texts[index] for index in train_indices])
        combined_tfidf = tfidf.transform(combined_texts).toarray().astype(np.float32)
        raw_tfidf = tfidf.transform(raw_texts).toarray().astype(np.float32)

        full_symbolic = build_symbolic_features(records, use_claim=True)
        raw_symbolic = build_symbolic_features(records, use_claim=False)
        # Las últimas dos columnas son linked_entity_count y has_conflict.
        text_metrics = full_symbolic[:, :3]
        raw_text_metrics = raw_symbolic[:, :3]
        full_embeddings, transformer_info = build_transformer_embeddings(
            combined_texts, arguments.transformer_model, arguments.no_transformer
        )
        raw_embeddings = None
        if full_embeddings is not None:
            raw_embeddings, _ = build_transformer_embeddings(
                raw_texts, transformer_info.get("model"), disabled=False
            )
        threshold_calibration = calibrate_threshold(
            combined_texts,
            full_symbolic,
            targets,
            train_indices,
            arguments.max_features,
            arguments.estimators,
            embeddings=full_embeddings,
            legacy_stop_words=arguments.legacy_stop_words,
        )
        calibrated_threshold = float(threshold_calibration["threshold_real"])

        full_features = np.hstack([combined_tfidf, full_symbolic])
        without_ontology = np.hstack([combined_tfidf, text_metrics])
        without_claim = np.hstack([raw_tfidf, raw_symbolic])
        without_both = np.hstack([raw_tfidf, raw_text_metrics])
        if full_embeddings is not None:
            full_features = np.hstack([full_features, full_embeddings])
        if raw_embeddings is not None:
            without_claim = np.hstack([without_claim, raw_embeddings])
            without_both = np.hstack([without_both, raw_embeddings])
        if full_embeddings is not None:
            without_ontology = np.hstack([without_ontology, full_embeddings])

        # Línea base: embeddings BERT + regresión logística; si BERT no está
        # disponible, usa TF-IDF para no detener el experimento.
        baseline_features = full_embeddings if full_embeddings is not None else combined_tfidf
        baseline = LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE, class_weight="balanced")
        baseline.fit(baseline_features[train_indices], targets[train_indices])
        baseline_metrics = metrics_for(targets[test_indices], baseline.predict(baseline_features[test_indices]))
        LOGGER.info("Baseline Transformer/TF-IDF — accuracy=%.4f, f1=%.4f", baseline_metrics["accuracy"], baseline_metrics["f1_score"])

        variants = {
            "full_hybrid": full_features,
            "without_ontology": without_ontology,
            "without_claim_extraction": without_claim,
            "without_ontology_and_claim": without_both,
        }
        trained_models: dict[str, RandomForestClassifier] = {}
        ablation_metrics: dict[str, dict[str, Any]] = {}
        for name, matrix in variants.items():
            trained_models[name], ablation_metrics[name] = fit_random_forest(
                name,
                matrix,
                targets,
                train_indices,
                test_indices,
                arguments.estimators,
                threshold=calibrated_threshold if name == "full_hybrid" else 0.5,
            )

        cross_validation = run_cross_validation(
            combined_texts,
            full_symbolic,
            targets,
            arguments.max_features,
            arguments.estimators,
            arguments.cv_folds,
            calibrated_threshold,
            embeddings=full_embeddings,
            legacy_stop_words=arguments.legacy_stop_words,
        )
        error_count = save_error_analysis(
            records,
            targets,
            trained_models["full_hybrid"],
            full_features,
            test_indices,
            calibrated_threshold,
        )

        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        joblib.dump(trained_models["full_hybrid"], MODEL_DIRECTORY / "random_forest_model.pkl")
        for name, model in trained_models.items():
            joblib.dump(model, MODEL_DIRECTORY / f"random_forest_{name}.pkl")
        joblib.dump(tfidf, MODEL_DIRECTORY / "tfidf.pkl")
        joblib.dump(baseline, MODEL_DIRECTORY / "transformer_baseline_model.pkl")
        model_config = {
            "variant": "full_hybrid",
            "symbolic_feature_order": [
                "text_word_count",
                "claim_word_count",
                "claim_readability",
                "linked_entity_count",
                "has_conflict",
            ],
            "decision_threshold_real": calibrated_threshold,
            "transformer": transformer_info,
            "preprocessing": {
                "tfidf_stop_words": "english_legacy" if arguments.legacy_stop_words else "english_preserve_negation",
                "preserved_negations": [] if arguments.legacy_stop_words else sorted(NEGATION_TERMS),
            },
        }
        with (MODEL_DIRECTORY / "model_config.json").open("w", encoding="utf-8") as output_file:
            json.dump(model_config, output_file, ensure_ascii=False, indent=2)
        metrics = {
            "dataset": {
                "records": len(records),
                "class_counts": class_counts,
                "test_size": arguments.test_size,
                "ambiguous_records_included": arguments.include_ambiguous,
                "excluded_benchmark_claims": len(excluded_claims),
            },
            "transformer": transformer_info,
            "baseline": {
                "name": "Transformer embeddings + Logistic Regression"
                if full_embeddings is not None
                else "TF-IDF + Logistic Regression (Transformer no disponible)",
                "metrics": baseline_metrics,
            },
            "ablation_study": ablation_metrics,
            "threshold_calibration": threshold_calibration,
            "cross_validation": cross_validation,
            "error_analysis": {"misclassified_test_records": error_count, "file": "error_analysis.json"},
            "feature_dimensions": {name: int(matrix.shape[1]) for name, matrix in variants.items()},
        }
        with (MODEL_DIRECTORY / "metrics.json").open("w", encoding="utf-8") as output_file:
            json.dump(metrics, output_file, ensure_ascii=False, indent=2)

        print("\nEntrenamiento finalizado")
        print(f"Registros: {len(records)} | Train: {len(train_indices)} | Test: {len(test_indices)}")
        for name, scores in ablation_metrics.items():
            print(f"- {name}: accuracy={scores['accuracy']:.4f}, f1={scores['f1_score']:.4f}")
        cv_summary = cross_validation["summary"]
        print(
            "- CV híbrido: "
            f"accuracy={cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}, "
            f"f1={cv_summary['f1_score']['mean']:.4f} ± {cv_summary['f1_score']['std']:.4f}"
        )
        print(f"- Umbral de REAL calibrado: {calibrated_threshold:.2f}")
        print(f"Artefactos guardados en: {MODEL_DIRECTORY}")
        return 0
    except (RuntimeError, ValueError, SQLAlchemyError) as error:
        LOGGER.error("El entrenamiento falló: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
