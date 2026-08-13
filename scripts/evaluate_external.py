#!/usr/bin/env python3
"""Evalúa el modelo de producción en ClimateCheck test sin contaminar entrenamiento."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)
TEST_URL = (
    "https://huggingface.co/datasets/rabuahmad/climatecheck/resolve/main/"
    "data/test-00000-of-00001.parquet?download=true"
)
DEFAULT_DATASET = PROJECT_ROOT / "data" / "external" / "climatecheck-test.parquet"
OUTPUT_PATH = PROJECT_ROOT / "app" / "models" / "external_evaluation.json"
LABEL_MAP = {"supports": 1, "refutes": 0}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--download", action="store_true", help="Descarga el test oficial de ClimateCheck.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def download_dataset(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Descargando ClimateCheck test oficial...")
    urllib.request.urlretrieve(TEST_URL, destination)


def load_strict_consensus(dataset_path: Path) -> list[tuple[str, int]]:
    """Conserva solo claims con etiqueta binaria consensuada en el conjunto externo."""
    frame = pd.read_parquet(dataset_path, columns=["claim", "annotation"])
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        claim, annotation = getattr(row, "claim", None), getattr(row, "annotation", None)
        if isinstance(claim, str) and isinstance(annotation, str) and claim.strip():
            grouped[claim.strip()].add(annotation.strip().lower())
    records = [
        (claim, LABEL_MAP[next(iter(labels))])
        for claim, labels in grouped.items()
        if len(labels) == 1 and next(iter(labels)) in LABEL_MAP
    ]
    if len({label for _, label in records}) < 2:
        raise RuntimeError("El conjunto externo consensuado no contiene ambas clases.")
    return records


def evaluate(dataset_path: Path, output_path: Path) -> dict[str, object]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No existe el test externo: {dataset_path}")
    from app.main import build_inference_features, load_artifacts

    records = load_strict_consensus(dataset_path)
    model, _, config = load_artifacts()
    threshold = float(config.get("decision_threshold_real", 0.5))
    y_true = np.asarray([label for _, label in records])
    y_pred: list[int] = []
    overrides = 0
    for index, (claim, _) in enumerate(records, start=1):
        features, _, ontology = build_inference_features(claim, claim)
        probability_real = float(model.predict_proba(features)[0, 1])
        predicted = int(probability_real >= threshold)
        if ontology["has_conflict"]:
            predicted = 0
            overrides += 1
        y_pred.append(predicted)
        if index % 50 == 0:
            LOGGER.info("Evaluados %d/%d registros externos.", index, len(records))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    result: dict[str, object] = {
        "dataset": "ClimateCheck test (consenso estricto; nunca usado para entrenar)",
        "records": len(records),
        "class_counts": {"FAKE": int((y_true == 0).sum()), "REAL": int((y_true == 1).sum())},
        "decision_threshold_real": threshold,
        "ontology_overrides": overrides,
        "reliability_warning": (
            "La muestra externa de consenso estricto es menor de 50 registros; "
            "úsela solo como smoke test, no como estimación de generalización."
            if len(records) < 50
            else None
        ),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "fake": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1_score": round(float(f1[0]), 4)},
            "real": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1_score": round(float(f1[1]), 4)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()
    try:
        if args.download:
            download_dataset(args.dataset)
        result = evaluate(args.dataset, args.output)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        LOGGER.error("La evaluación externa falló: %s", error)
        return 1
    LOGGER.info("Evaluación externa completada: %s", args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
