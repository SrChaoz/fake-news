#!/usr/bin/env python3
"""Ejemplo ejecutable del pipeline neuro-simbólico ambiental."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.feature_extractor import build_feature_vector
from app.services.ner_engine import extract_environmental_entities
from app.services.ontology_engine import detect_ontological_conflicts


def main() -> None:
    """Ejecuta el ejemplo completo y muestra resultados legibles."""
    claim = "CO2 does not contribute to global warming."
    entities = extract_environmental_entities(claim)
    ontology_results = detect_ontological_conflicts(claim, entities)
    features = build_feature_vector(claim, claim, ontology_results)

    print("Entidades detectadas:")
    print(json.dumps(entities, indent=2, ensure_ascii=False))
    print("\nConceptos ontológicos vinculados:")
    print(json.dumps(ontology_results["linked_entities"], indent=2, ensure_ascii=False))
    print(f"\n¿Conflicto ontológico?: {ontology_results['has_conflict']}")
    if ontology_results["conflicts"]:
        print(json.dumps(ontology_results["conflicts"], indent=2, ensure_ascii=False))
    print(f"\nVector de características: {features['vector'].shape}")
    print(f"Metadatos: {features['metadata']}")


if __name__ == "__main__":
    main()
