#!/usr/bin/env python3
"""Ingiere afirmaciones ClimateCheck de consenso estricto para uso académico."""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)
DATA_URL = (
    "https://huggingface.co/datasets/rabuahmad/climatecheck/resolve/main/"
    "data/train-00000-of-00001.parquet?download=true"
)
DEFAULT_DATASET = PROJECT_ROOT / "data" / "external" / "climatecheck-train.parquet"
LABEL_MAP = {"supports": "REAL", "refutes": "FAKE"}


def parse_arguments() -> argparse.Namespace:
    """Define las opciones de una ingesta reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--download", action="store_true", help="Descarga el Parquet oficial antes de ingerirlo.")
    parser.add_argument("--dry-run", action="store_true", help="Valida y muestra el resultado sin insertar.")
    return parser.parse_args()


def download_dataset(destination: Path) -> None:
    """Descarga el archivo oficial en una ubicación local explícita."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Descargando ClimateCheck desde Hugging Face...")
    urllib.request.urlretrieve(DATA_URL, destination)
    LOGGER.info("Archivo descargado: %s", destination)


def infer_category(claim: str) -> str:
    """Asigna una categoría de alto nivel sin depender de anotaciones externas."""
    normalized = claim.lower()
    if any(word in normalized for word in ("pollution", "pollutant", "air quality", "plastic")):
        return "Pollution"
    if any(word in normalized for word in ("forest", "deforestation", "species", "biodiversity", "reef")):
        return "Biodiversity"
    if any(word in normalized for word in ("energy", "solar", "wind", "fossil", "coal", "renewable")):
        return "Energy"
    return "Climate Change"


def consensus_claims(dataset_path: Path) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Conserva solo afirmaciones cuyas evidencias tienen una etiqueta binaria unánime."""
    try:
        frame = pd.read_parquet(dataset_path, columns=["claim", "annotation"])
    except (ImportError, ValueError) as error:
        raise RuntimeError("ClimateCheck requiere pyarrow: ejecuta python -m pip install pyarrow") from error

    grouped_labels: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        claim = getattr(row, "claim", None)
        annotation = getattr(row, "annotation", None)
        if isinstance(claim, str) and isinstance(annotation, str) and claim.strip():
            grouped_labels[claim.strip()].add(annotation.strip().lower())

    records: list[tuple[str, str, str]] = []
    summary = {"unique_claims": len(grouped_labels), "consensus_real": 0, "consensus_fake": 0, "excluded": 0}
    for claim, labels in grouped_labels.items():
        if len(labels) != 1:
            summary["excluded"] += 1
            continue
        mapped_label = LABEL_MAP.get(next(iter(labels)))
        if mapped_label is None:
            summary["excluded"] += 1
            continue
        records.append((claim, mapped_label, infer_category(claim)))
        summary["consensus_real" if mapped_label == "REAL" else "consensus_fake"] += 1
    return records, summary


def ingest(dataset_path: Path, dry_run: bool) -> int:
    """Deduplica e inserta el conjunto consensuado en una transacción."""
    if not dataset_path.is_file():
        LOGGER.error("No existe el archivo ClimateCheck: %s", dataset_path)
        return 1
    try:
        candidates, summary = consensus_claims(dataset_path)
        from database import SessionLocal
        from models import DatasetRecord

        with SessionLocal() as session:
            existing_claims = set(
                session.scalars(
                    select(DatasetRecord.extracted_claim).where(DatasetRecord.extracted_claim.is_not(None))
                ).all()
            )
        new_records = [record for record in candidates if record[0] not in existing_claims]
        inserted = {"REAL": 0, "FAKE": 0}
        for _, label, _ in new_records:
            inserted[label] += 1

        if not dry_run and new_records:
            with SessionLocal.begin() as session:
                session.add_all(
                    DatasetRecord(
                        raw_text=claim,
                        extracted_claim=claim,
                        label=label,
                        category=category,
                        source="climatecheck-2026-consensus",
                    )
                    for claim, label, category in new_records
                )

        LOGGER.info(
            "ClimateCheck: %d claims únicos; consenso REAL=%d, FAKE=%d; excluidos=%d.",
            summary["unique_claims"],
            summary["consensus_real"],
            summary["consensus_fake"],
            summary["excluded"],
        )
        LOGGER.info("Nuevos registros %s — REAL=%d, FAKE=%d; duplicados=%d.", "previstos" if dry_run else "insertados", inserted["REAL"], inserted["FAKE"], len(candidates) - len(new_records))
        return 0
    except (OSError, RuntimeError, SQLAlchemyError) as error:
        LOGGER.error("La ingesta ClimateCheck falló: %s", error)
        return 1


def main() -> int:
    """Punto de entrada CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_arguments()
    if arguments.download:
        try:
            download_dataset(arguments.dataset)
        except OSError as error:
            LOGGER.error("No se pudo descargar ClimateCheck: %s", error)
            return 1
    return ingest(arguments.dataset, arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
