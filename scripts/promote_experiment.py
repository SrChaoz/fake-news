#!/usr/bin/env python3
"""Promueve artefactos de un experimento validado al directorio de producción."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "app" / "models"
REQUIRED_ARTIFACTS = ("random_forest_model.pkl", "tfidf.pkl", "model_config.json", "metrics.json")
OPTIONAL_ARTIFACTS = ("error_analysis.json", "error_summary.json", "transformer_baseline_model.pkl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="Nombre del experimento dentro de app/models/experiments/.")
    arguments = parser.parse_args()
    if Path(arguments.experiment).name != arguments.experiment:
        parser.error("El experimento debe ser un nombre simple de directorio.")

    source_directory = MODEL_DIRECTORY / "experiments" / arguments.experiment
    missing = [name for name in REQUIRED_ARTIFACTS if not (source_directory / name).is_file()]
    if missing:
        parser.error(f"El experimento no contiene los artefactos requeridos: {', '.join(missing)}")

    for artifact in REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS:
        source = source_directory / artifact
        if source.is_file():
            shutil.copy2(source, MODEL_DIRECTORY / artifact)
    print(f"Experimento '{arguments.experiment}' promovido a producción.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
