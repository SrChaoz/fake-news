#!/usr/bin/env python3
"""Vincula abstracts científicos de ClimateCheck a registros ya ingeridos.

Solo asocia un abstract cuando su anotación binaria coincide con la etiqueta del
registro. Es idempotente y no crea registros de entrenamiento nuevos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "data" / "external" / "climatecheck-train.parquet"
LABELS = {"supports": "REAL", "refutes": "FAKE"}
SOURCE = "climatecheck-2026-consensus"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if not args.dataset.is_file():
        print(f"No existe el dataset: {args.dataset}", file=sys.stderr)
        return 1
    from database import SessionLocal
    from models import DatasetEvidence, DatasetRecord

    frame = pd.read_parquet(args.dataset, columns=["claim", "abstract", "abstract_id", "annotation"])
    candidates: dict[str, tuple[str, str]] = {}
    for row in frame.itertuples(index=False):
        claim, abstract, abstract_id, annotation = row
        label = LABELS.get(str(annotation).strip().lower())
        if isinstance(claim, str) and isinstance(abstract, str) and label and abstract.strip():
            candidates.setdefault(" ".join(claim.casefold().split()), (label, abstract.strip()))

    with SessionLocal() as session:
        records = session.execute(select(DatasetRecord).where(DatasetRecord.source == SOURCE)).scalars().all()
        existing_ids = set(session.scalars(select(DatasetEvidence.dataset_record_id).where(DatasetEvidence.source_dataset == SOURCE)).all())
        additions: list[DatasetEvidence] = []
        for record in records:
            key = " ".join((record.extracted_claim or record.raw_text).casefold().split())
            candidate = candidates.get(key)
            if candidate is None or record.id in existing_ids:
                continue
            label, abstract = candidate
            if label != record.label:
                continue
            additions.append(DatasetEvidence(dataset_record_id=record.id, source_dataset=SOURCE, source_url=None, evidence_text=abstract))
        if not args.dry_run and additions:
            session.add_all(additions)
            session.commit()
    print(f"Evidence ClimateCheck {'prevista' if args.dry_run else 'insertada'}: {len(additions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
