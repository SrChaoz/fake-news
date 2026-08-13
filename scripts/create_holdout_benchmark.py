#!/usr/bin/env python3
"""Crea un benchmark balanceado y retenido para evaluación externa reproducible."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "evaluation" / "holdout_v1.jsonl"
SOURCES = {"FAKE": "climafactskg-ccby", "REAL": "climatecheck-2026-consensus"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-label", type=int, default=100)
    args = parser.parse_args()
    if args.per_label < 1:
        parser.error("--per-label debe ser mayor que cero.")
    return args


def deterministic_key(claim: str) -> str:
    return hashlib.sha256(("holdout-v1:" + claim.casefold()).encode()).hexdigest()


def main() -> int:
    args = parse_arguments()
    try:
        from database import SessionLocal
        from models import DatasetRecord

        selected: list[dict[str, str]] = []
        with SessionLocal() as session:
            for label, source in SOURCES.items():
                rows = session.execute(
                    select(DatasetRecord.raw_text, DatasetRecord.extracted_claim, DatasetRecord.category)
                    .where(DatasetRecord.label == label, DatasetRecord.source == source)
                ).all()
                candidates = sorted(
                    (row for row in rows if row.extracted_claim or row.raw_text),
                    key=lambda row: deterministic_key(row.extracted_claim or row.raw_text),
                )
                if len(candidates) < args.per_label:
                    raise RuntimeError(f"{source} no tiene {args.per_label} registros {label} disponibles.")
                for raw_text, claim, category in candidates[: args.per_label]:
                    selected.append({
                        "text": raw_text,
                        "claim": claim or raw_text,
                        "label": label,
                        "category": category or "Climate Change",
                        "source": source,
                        "split": "holdout_v1",
                    })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output_file:
            for record in sorted(selected, key=lambda item: deterministic_key(item["claim"])):
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Benchmark creado: {args.output} (REAL={args.per_label}, FAKE={args.per_label}).")
        return 0
    except Exception as error:
        print(f"No se pudo crear el benchmark: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
