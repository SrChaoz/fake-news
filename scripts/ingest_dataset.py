#!/usr/bin/env python3
"""Ingesta balanceada del dataset enriquecido Climate-FEVER a PostgreSQL.

Los registros SUPPORTS se almacenan como REAL y los REFUTES como FAKE. Como el
dataset contiene menos de 500 REFUTES, NOT_ENOUGH_INFO y DISPUTED se usan solo
para completar la cuota de FAKE solicitada; nunca se priorizan sobre REFUTES.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

# Permite ejecutar el script desde la raíz con ``python scripts/ingest_dataset.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import SessionLocal  # noqa: E402
from models import DatasetRecord  # noqa: E402


LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_PATH = PROJECT_ROOT / "climate-fever-dataset-r1-enriched.jsonl"
TOPIC_CATEGORIES = {
    "global_warming": "Climate Change",
    "greenhouse_gases": "Climate Change",
    "ice_and_glaciers": "Climate Change",
    "sea_level": "Climate Change",
    "natural_forcing": "Climate Change",
    "extreme_weather": "Climate Change",
    "ecosystems": "Biodiversity",
    "biodiversity": "Biodiversity",
    "pollution": "Pollution",
    "energy": "Energy",
    "renewable_energy": "Energy",
    "fossil_fuels": "Energy",
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """Registro validado y listo para persistirse."""

    raw_text: str
    extracted_claim: str
    label: str
    category: str
    is_ambiguous_fallback: bool = False


def parse_arguments() -> argparse.Namespace:
    """Define los parámetros de ejecución de la ingesta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Ruta al JSONL (por defecto: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=500,
        help="Mínimo balanceado para cada etiqueta (por defecto: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el resultado previsto sin insertar registros.",
    )
    arguments = parser.parse_args()
    if arguments.min_per_label < 1:
        parser.error("--min-per-label debe ser mayor que cero.")
    return arguments


def category_from_topics(topics: Any) -> str:
    """Convierte el primer tópico del dataset a una categoría de la aplicación."""
    if not isinstance(topics, list) or not topics:
        return "Climate Change"
    first_topic = topics[0]
    if not isinstance(first_topic, str):
        return "Climate Change"
    return TOPIC_CATEGORIES.get(first_topic.lower(), "Climate Change")


def iter_candidates(dataset_path: Path) -> Iterable[Candidate]:
    """Lee el JSONL y emite solamente registros aptos para clasificación binaria."""
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                LOGGER.warning("Línea %d omitida: JSON inválido (%s).", line_number, error)
                continue

            raw_text = item.get("claim")
            extracted_claim = item.get("claim_normalized")
            claim_label = item.get("claim_label")
            if not isinstance(raw_text, str) or not raw_text.strip():
                LOGGER.warning("Línea %d omitida: claim ausente.", line_number)
                continue
            if not isinstance(extracted_claim, str) or not extracted_claim.strip():
                extracted_claim = raw_text

            label_mapping = {"SUPPORTS": "REAL", "REFUTES": "FAKE"}
            label = label_mapping.get(claim_label)
            is_ambiguous_fallback = False
            if label is None:
                if claim_label not in {"NOT_ENOUGH_INFO", "DISPUTED"}:
                    LOGGER.warning(
                        "Línea %d omitida: etiqueta no reconocida (%r).",
                        line_number,
                        claim_label,
                    )
                    continue
                # Solo se seleccionan si faltan FAKE tras consumir los REFUTES.
                label = "FAKE"
                is_ambiguous_fallback = True

            yield Candidate(
                raw_text=raw_text.strip(),
                extracted_claim=extracted_claim.strip(),
                label=label,
                category=category_from_topics(item.get("topics")),
                is_ambiguous_fallback=is_ambiguous_fallback,
            )


def existing_counts_and_claims() -> tuple[dict[str, int], set[str]]:
    """Obtiene conteos actuales y afirmaciones ya persistidas para deduplicar."""
    with SessionLocal() as session:
        counts = {"REAL": 0, "FAKE": 0}
        for label, count in session.execute(
            select(DatasetRecord.label, func.count()).group_by(DatasetRecord.label)
        ):
            if label in counts:
                counts[label] = count

        existing_claims = set(
            session.scalars(
                select(DatasetRecord.extracted_claim).where(
                    DatasetRecord.extracted_claim.is_not(None)
                )
            ).all()
        )
    return counts, existing_claims


def select_balanced_candidates(
    candidates: Iterable[Candidate],
    existing_counts: dict[str, int],
    existing_claims: set[str],
    minimum_per_label: int,
) -> tuple[list[Candidate], dict[str, int], int]:
    """Selecciona candidatos sin duplicados hasta igualar ambos conteos."""
    target_per_label = max(minimum_per_label, *existing_counts.values())
    pending = {
        label: max(0, target_per_label - existing_counts[label])
        for label in ("REAL", "FAKE")
    }
    selected: list[Candidate] = []
    selected_claims = set(existing_claims)
    ambiguous_candidates: list[Candidate] = []

    # Primera pasada: usa únicamente las etiquetas binarias de confianza.
    for candidate in candidates:
        if candidate.extracted_claim in selected_claims:
            continue
        if candidate.is_ambiguous_fallback:
            ambiguous_candidates.append(candidate)
            continue
        if pending[candidate.label] == 0:
            continue
        selected.append(candidate)
        selected_claims.add(candidate.extracted_claim)
        pending[candidate.label] -= 1

    # Segunda pasada: completa exclusivamente la cuota FAKE que no cubrieron REFUTES.
    ambiguous_used = 0
    for candidate in ambiguous_candidates:
        if pending["FAKE"] == 0:
            break
        if candidate.extracted_claim in selected_claims:
            continue
        selected.append(candidate)
        selected_claims.add(candidate.extracted_claim)
        pending["FAKE"] -= 1
        ambiguous_used += 1

    return selected, pending, ambiguous_used


def run_ingestion(dataset_path: Path, minimum_per_label: int, dry_run: bool) -> int:
    """Selecciona, deduplica e inserta el lote balanceado en una transacción."""
    if not dataset_path.is_file():
        LOGGER.error("No se encontró el dataset: %s", dataset_path)
        return 1

    try:
        existing_counts, existing_claims = existing_counts_and_claims()
        selected, pending, ambiguous_used = select_balanced_candidates(
            iter_candidates(dataset_path),
            existing_counts,
            existing_claims,
            minimum_per_label,
        )

        inserted_counts = {"REAL": 0, "FAKE": 0}
        for candidate in selected:
            inserted_counts[candidate.label] += 1

        if not dry_run and selected:
            with SessionLocal.begin() as session:
                session.add_all(
                    DatasetRecord(
                        raw_text=candidate.raw_text,
                        extracted_claim=candidate.extracted_claim,
                        label=candidate.label,
                        category=candidate.category,
                        source="climate-fever-enriched",
                    )
                    for candidate in selected
                )

        final_counts = {
            label: existing_counts[label] + inserted_counts[label]
            for label in ("REAL", "FAKE")
        }
        action = "Simulación" if dry_run else "Ingesta"
        LOGGER.info("%s completada.", action)
        LOGGER.info("Insertados REAL: %d | FAKE: %d", inserted_counts["REAL"], inserted_counts["FAKE"])
        LOGGER.info("Totales REAL: %d | FAKE: %d", final_counts["REAL"], final_counts["FAKE"])
        if ambiguous_used:
            LOGGER.warning(
                "Se usaron %d registros NOT_ENOUGH_INFO/DISPUTED como FAKE para completar la cuota.",
                ambiguous_used,
            )
        if pending["REAL"] or pending["FAKE"]:
            LOGGER.warning(
                "No fue posible alcanzar la cuota: faltan REAL=%d, FAKE=%d.",
                pending["REAL"],
                pending["FAKE"],
            )
        return 0
    except (OSError, RuntimeError, SQLAlchemyError) as error:
        LOGGER.error("La ingesta falló: %s", error)
        return 1


def main() -> int:
    """Punto de entrada de línea de comandos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    arguments = parse_arguments()
    return run_ingestion(arguments.dataset, arguments.min_per_label, arguments.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
