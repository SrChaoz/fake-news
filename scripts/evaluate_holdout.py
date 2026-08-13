#!/usr/bin/env python3
"""Evalúa el modelo híbrido contra un benchmark JSONL retenido."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "evaluation" / "holdout_v1.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "app" / "models" / "holdout_evaluation.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=PROJECT_ROOT / "app" / "models",
        help="Directorio de artefactos a evaluar; permite validar un experimento sin promocionarlo.",
    )
    return parser.parse_args()


def project_relative_path(path: Path) -> str:
    """Devuelve una ruta relativa al proyecto cuando es posible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def main() -> int:
    args = parse_arguments()
    if not args.benchmark.is_file():
        print(f"No existe el benchmark: {args.benchmark}", file=sys.stderr)
        return 1
    import app.main as api

    if not (args.model_directory / "random_forest_model.pkl").is_file():
        print(f"No existe un modelo en: {args.model_directory}", file=sys.stderr)
        return 1
    # La API y el evaluador usan exactamente el mismo feature engineering. Se
    # redirige el cargador antes de la primera predicción, sin copiar artefactos
    # ni alterar producción.
    api.MODEL_DIRECTORY = args.model_directory.resolve()
    api.load_artifacts.cache_clear()
    api.load_transformer_encoder.cache_clear()

    records = [json.loads(line) for line in args.benchmark.read_text(encoding="utf-8").splitlines() if line]
    expected = np.asarray([1 if record["label"] == "REAL" else 0 for record in records], dtype=np.int8)
    predictions: list[int] = []
    errors: list[dict[str, Any]] = []
    for record, actual in zip(records, expected, strict=True):
        # La evaluación no persiste resultados de benchmark en el historial.
        result = api.analyze_prediction(record["text"], record["claim"]).model_dump()
        predicted = 1 if result["prediction"] == "REAL" else 0
        predictions.append(predicted)
        if predicted != actual:
            errors.append({
                "claim": record["claim"], "actual": record["label"], "predicted": result["prediction"],
                "probability_real": result["probability_real"], "decision_source": result["decision_source"],
                "category": record["category"], "source": record["source"],
            })
    precision, recall, f1, _ = precision_recall_fscore_support(expected, predictions, labels=[0, 1], zero_division=0)
    output: dict[str, Any] = {
        "benchmark": project_relative_path(args.benchmark),
        "model_directory": project_relative_path(args.model_directory),
        "records": len(records), "class_counts": dict(Counter(record["label"] for record in records)),
        "metrics": {"accuracy": round(float(accuracy_score(expected, predictions)), 4), "balanced_accuracy": round(float(balanced_accuracy_score(expected, predictions)), 4), "macro_f1": round(float(np.mean(f1)), 4), "fake": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1_score": round(float(f1[0]), 4)}, "real": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1_score": round(float(f1[1]), 4)}, "confusion_matrix": confusion_matrix(expected, predictions, labels=[0, 1]).tolist()},
        "error_count": len(errors),
        "errors_by_category": dict(Counter(error["category"] for error in errors)),
        "errors_by_source": dict(Counter(error["source"] for error in errors)),
        "errors": errors,
        "warning": "Este benchmark es válido solo para modelos entrenados con --exclude-claims-file apuntando a este mismo JSONL.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))
    print(f"Errores: {len(errors)} | Informe: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
