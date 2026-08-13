#!/usr/bin/env python3
"""Pruebas de regresión para contradicciones climáticas críticas."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.ner_engine import extract_environmental_entities
from app.services.ontology_engine import detect_ontological_conflicts


CASES = {
    "CO2 contributes to global warming.": False,
    "CO2 does not contribute to global warming.": True,
    "CO2 has no impact on global warming.": True,
    "El CO2 contribuye al calentamiento global.": False,
    "El CO2 no contribuye al calentamiento global.": True,
    "El dióxido de carbono no tiene impacto en el cambio climático.": True,
    "El metano no es un gas de efecto invernadero.": True,
}

SUPPORT_CASES = {
    "Carbon dioxide is a greenhouse gas that contributes to global warming.",
    "Methane is a greenhouse gas and reducing methane emissions can help limit warming.",
    "El dióxido de carbono contribuye al calentamiento global.",
}


def main() -> int:
    failures: list[str] = []
    for claim, expected in CASES.items():
        result = detect_ontological_conflicts(claim, extract_environmental_entities(claim))
        actual = bool(result["has_conflict"])
        status = "PASS" if actual == expected else "FAIL"
        print(f"[{status}] expected={expected} actual={actual} | {claim}")
        if actual != expected:
            failures.append(claim)
    for claim in SUPPORT_CASES:
        result = detect_ontological_conflicts(claim, extract_environmental_entities(claim))
        actual = bool(result["has_support"]) and not bool(result["has_conflict"])
        status = "PASS" if actual else "FAIL"
        print(f"[{status}] expected support=True actual={actual} | {claim}")
        if not actual:
            failures.append(claim)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
