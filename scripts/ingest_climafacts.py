#!/usr/bin/env python3
"""Ingiere mitos climáticos refutados por ClimaFactsKG con su evidencia."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, Namespace, RDF
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)
SCHEMA = Namespace("https://schema.org/")
DEFAULT_DATASET = PROJECT_ROOT / "data" / "external" / "climafacts-kg" / "data" / "climafacts_kg.ttl"
SOURCE = "climafactskg-ccby"


@dataclass(frozen=True)
class ClaimCandidate:
    """Afirmación falsa y corrección vinculada desde el grafo."""

    claim: str
    evidence: str
    source_url: str | None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="Muestra el resultado sin insertar.")
    return parser.parse_args()


def english_claim_text(graph: Graph, claim_node: object) -> str | None:
    """Obtiene la literal ``schema:text`` marcada explícitamente con idioma inglés."""
    for literal in graph.objects(claim_node, SCHEMA.text):
        if getattr(literal, "language", None) == "en":
            text = str(literal).strip()
            if text:
                return text
    return None


def infer_category(claim: str) -> str:
    text = claim.lower()
    if any(term in text for term in ("pollution", "plastic", "aerosol", "air quality")):
        return "Pollution"
    if any(term in text for term in ("forest", "deforestation", "species", "biodiversity", "coral", "reef")):
        return "Biodiversity"
    if any(term in text for term in ("energy", "solar", "wind", "coal", "fossil", "renewable")):
        return "Energy"
    return "Climate Change"


def extract_candidates(dataset_path: Path) -> list[ClaimCandidate]:
    """Extrae solo ClaimReview falsos con corrección textual del RDF oficial."""
    graph = Graph()
    graph.parse(dataset_path, format="turtle")
    candidates: dict[str, ClaimCandidate] = {}
    for review in graph.subjects(RDF.type, SCHEMA.ClaimReview):
        rating = graph.value(review, SCHEMA.reviewRating)
        rating_name = str(graph.value(rating, SCHEMA.name) or "").strip().lower()
        claim_node = graph.value(review, SCHEMA.claimReviewed)
        claim = english_claim_text(graph, claim_node)
        evidence = str(graph.value(review, SCHEMA.text) or "").strip()
        if rating_name != "false" or not evidence or not claim:
            continue
        key = " ".join(claim.casefold().split())
        candidates.setdefault(
            key,
            ClaimCandidate(claim=claim, evidence=evidence, source_url=_optional_value(graph.value(review, SCHEMA.url))),
        )
    return list(candidates.values())


def _optional_value(value: object) -> str | None:
    value_as_text = str(value or "").strip()
    return value_as_text or None


def ingest(dataset_path: Path, dry_run: bool) -> int:
    if not dataset_path.is_file():
        LOGGER.error("No existe el RDF de ClimaFactsKG: %s", dataset_path)
        return 1
    try:
        candidates = extract_candidates(dataset_path)
        from database import SessionLocal
        from models import DatasetEvidence, DatasetRecord

        with SessionLocal() as session:
            existing = {" ".join(value.casefold().split()) for value in session.scalars(
                select(DatasetRecord.extracted_claim).where(DatasetRecord.extracted_claim.is_not(None))
            ) if value}
        new_candidates = [candidate for candidate in candidates if " ".join(candidate.claim.casefold().split()) not in existing]
        if not dry_run and new_candidates:
            with SessionLocal.begin() as session:
                for candidate in new_candidates:
                    record = DatasetRecord(
                        raw_text=candidate.claim,
                        extracted_claim=candidate.claim,
                        label="FAKE",
                        category=infer_category(candidate.claim),
                        source=SOURCE,
                    )
                    session.add(record)
                    session.flush()
                    session.add(DatasetEvidence(
                        dataset_record_id=record.id,
                        source_dataset=SOURCE,
                        source_url=candidate.source_url,
                        evidence_text=candidate.evidence,
                    ))
        LOGGER.info("ClimaFactsKG: candidatos FALSE en inglés=%d.", len(candidates))
        LOGGER.info("Nuevos registros %s: FAKE=%d; duplicados=%d.", "previstos" if dry_run else "insertados", len(new_candidates), len(candidates) - len(new_candidates))
        return 0
    except (OSError, SQLAlchemyError, ValueError) as error:
        LOGGER.error("La ingesta ClimaFactsKG falló: %s", error)
        return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_arguments()
    return ingest(args.dataset, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
