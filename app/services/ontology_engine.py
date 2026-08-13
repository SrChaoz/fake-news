"""Enlace de entidades y validación de reglas ambientales con RDFLib."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef


ENV = Namespace("https://example.org/environmental-ontology/")


class OntologyMatcher:
    """Catálogo RDF local ampliable con conceptos de SWEET, ENVO y GEMET.

    El grafo inicial es deliberadamente pequeño para que el sistema funcione sin
    descargar ontologías externas. ``load_ontology`` permite incorporar archivos
    OWL/RDF reales cuando estén disponibles.
    """

    def __init__(self, ontology_path: str | None = None) -> None:
        self.graph = Graph()
        self._build_seed_graph()
        if ontology_path:
            self.load_ontology(ontology_path)

    def _add_concept(self, identifier: str, label: str, source: str, aliases: list[str]) -> None:
        concept = ENV[identifier]
        self.graph.add((concept, RDF.type, ENV.EnvironmentalConcept))
        self.graph.add((concept, RDFS.label, Literal(label)))
        self.graph.add((concept, ENV.ontology_source, Literal(source)))
        for alias in {label, *aliases}:
            self.graph.add((concept, ENV.alias, Literal(alias.lower())))

    def _build_seed_graph(self) -> None:
        self._add_concept("CarbonDioxide", "carbon dioxide", "SWEET", ["co2"])
        self._add_concept("Methane", "methane", "SWEET", ["ch4"])
        self._add_concept("GreenhouseGas", "greenhouse gas", "SWEET", ["greenhouse gases"])
        self._add_concept("GlobalWarming", "global warming", "GEMET", ["climate warming"])
        self._add_concept("ClimateChange", "climate change", "GEMET", [])
        self._add_concept("Deforestation", "deforestation", "AGROVOC", [])
        self._add_concept("Biodiversity", "biodiversity", "ENVO", [])
        self._add_concept("Ecosystem", "ecosystem", "ENVO", ["ecosystems"])
        self._add_concept("Pollution", "pollution", "GEMET", [])
        self._add_concept("Forest", "forest", "ENVO", ["forests"])

        self.graph.add((ENV.CarbonDioxide, ENV.is_a, ENV.GreenhouseGas))
        self.graph.add((ENV.Methane, ENV.is_a, ENV.GreenhouseGas))
        self.graph.add((ENV.CarbonDioxide, ENV.contributes_to, ENV.GlobalWarming))
        self.graph.add((ENV.Methane, ENV.contributes_to, ENV.GlobalWarming))
        self.graph.add((ENV.Deforestation, ENV.reduces, ENV.Forest))
        self.graph.add((ENV.Deforestation, ENV.harms, ENV.Biodiversity))
        self.graph.add((ENV.Pollution, ENV.harms, ENV.Ecosystem))

    def load_ontology(self, ontology_path: str) -> None:
        """Carga un recurso RDF/OWL adicional en el grafo actual."""
        self.graph.parse(ontology_path)

    def match_entities(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Vincula entidades NER a conceptos locales por sus alias RDF."""
        matches: list[dict[str, str]] = []
        seen: set[tuple[str, URIRef]] = set()
        for entity in entities:
            normalized = str(entity.get("normalized") or entity.get("text") or "").lower().strip()
            if not normalized:
                continue
            for concept in self.graph.subjects(ENV.alias, Literal(normalized)):
                key = (normalized, concept)
                if key in seen:
                    continue
                seen.add(key)
                label = self.graph.value(concept, RDFS.label)
                source = self.graph.value(concept, ENV.ontology_source) or Literal("external RDF/OWL")
                matches.append(
                    {
                        "entity": str(entity.get("text", normalized)),
                        "normalized": normalized,
                        "concept_uri": str(concept),
                        "concept_label": str(label),
                        "ontology": str(source),
                    }
                )
        return {"linked_entities": matches, "linked_count": len(matches)}


@lru_cache(maxsize=1)
def get_default_matcher() -> OntologyMatcher:
    """Comparte el grafo en memoria entre peticiones."""
    return OntologyMatcher()


CONFLICT_RULES = (
    {
        "subject": ("co2", "carbon dioxide", "dioxido de carbono"),
        "object": ("greenhouse gas", "greenhouse gases", "gas de efecto invernadero"),
        "relation": "is a greenhouse gas",
    },
    {
        "subject": ("co2", "carbon dioxide", "dioxido de carbono"),
        "object": ("global warming", "climate change", "calentamiento global", "cambio climatico"),
        "relation": "contributes to global warming",
    },
    {
        "subject": ("ch4", "methane", "metano"),
        "object": ("greenhouse gas", "greenhouse gases", "gas de efecto invernadero"),
        "relation": "is a greenhouse gas",
    },
    {
        "subject": ("ch4", "methane", "metano"),
        "object": ("global warming", "climate change", "calentamiento global", "cambio climatico"),
        "relation": "contributes to global warming",
    },
    {
        "subject": ("deforestation",),
        "object": ("forest", "forests", "forest cover"),
        "relation": "reduces forest cover",
    },
    {
        "subject": ("deforestation",),
        "object": ("biodiversity", "ecosystem", "ecosystems", "habitat", "habitats"),
        "relation": "harms biodiversity and habitats",
    },
    {
        "subject": ("pollution", "air pollution"),
        "object": ("ecosystem", "ecosystems", "biodiversity", "health"),
        "relation": "harms ecosystems and health",
    },
)
NEGATION_PATTERN = re.compile(
    r"\b(?:"
    r"does\s+not|do\s+not|is\s+not|are\s+not|isn't|aren't|doesn't|don't|never|"
    r"has\s+no\s+(?:impact|effect)|has\s+zero\s+impact|does\s+not\s+(?:affect|cause)|"
    r"no\s+(?:contribuye|afecta|tiene\s+impacto|tiene\s+efecto|causa)|"
    r"no\s+es\s+un\s+gas\s+de\s+efecto\s+invernadero|nunca"
    r")\b"
)
CONTRARY_RELATIONS = (
    {
        "subject": ("deforestation",),
        "phrases": ("increases forest cover", "creates forest", "improves biodiversity", "benefits biodiversity"),
        "relation": "reduces forest cover and harms biodiversity",
    },
    {
        "subject": ("pollution", "air pollution"),
        "phrases": ("improves ecosystems", "benefits ecosystems", "improves biodiversity", "benefits health"),
        "relation": "harms ecosystems and health",
    },
)
SUPPORT_RULES = (
    {
        "subject": ("co2", "carbon dioxide", "dioxido de carbono"),
        "object": ("greenhouse gas", "greenhouse gases", "gas de efecto invernadero"),
        "relation": "CO2 is a greenhouse gas",
    },
    {
        "subject": ("co2", "carbon dioxide", "dioxido de carbono"),
        "object": ("global warming", "climate change", "calentamiento global", "cambio climatico"),
        "relation": "CO2 contributes to global warming",
    },
    {
        "subject": ("ch4", "methane", "metano"),
        "object": ("greenhouse gas", "greenhouse gases", "gas de efecto invernadero"),
        "relation": "methane is a greenhouse gas",
    },
    {
        "subject": ("ch4", "methane", "metano"),
        "object": ("global warming", "climate change", "calentamiento global", "cambio climatico", "warming"),
        "relation": "methane contributes to global warming",
    },
)


def detect_ontological_conflicts(claim: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Detecta negaciones de relaciones ambientales respaldadas por la ontología."""
    # Eliminar acentos permite que "climático" y "climatico" activen la
    # misma regla, sin alterar el texto original mostrado al usuario.
    normalized_claim = "".join(
        character
        for character in unicodedata.normalize("NFD", claim.lower())
        if unicodedata.category(character) != "Mn"
    )
    conflicts: list[dict[str, str]] = []
    supporting_rules: list[dict[str, str]] = []
    has_negation = bool(NEGATION_PATTERN.search(normalized_claim))
    if has_negation:
        for rule in CONFLICT_RULES:
            if any(term in normalized_claim for term in rule["subject"]) and any(
                term in normalized_claim for term in rule["object"]
            ):
                conflicts.append(
                    {
                        "subject": rule["subject"][0],
                        "expected_relation": rule["relation"],
                        "reason": "La afirmación niega una relación ambiental conocida.",
                    }
                )

    for rule in CONTRARY_RELATIONS:
        if any(term in normalized_claim for term in rule["subject"]) and any(
            phrase in normalized_claim for phrase in rule["phrases"]
        ):
            conflicts.append(
                {
                    "subject": rule["subject"][0],
                    "expected_relation": rule["relation"],
                    "reason": "La afirmación expresa una relación contraria al conocimiento ambiental.",
                }
            )

    # Las relaciones positivas se aplican solo si el texto no contiene una
    # negación. Evita que "CO2 is a greenhouse gas but has no impact" reciba
    # soporte: la contradicción previa conserva prioridad absoluta.
    if not has_negation:
        for rule in SUPPORT_RULES:
            if any(term in normalized_claim for term in rule["subject"]) and any(
                term in normalized_claim for term in rule["object"]
            ):
                supporting_rules.append(
                    {
                        "subject": rule["subject"][0],
                        "supported_relation": rule["relation"],
                        "reason": "La afirmación expresa una relación ambiental respaldada por la ontología.",
                    }
                )

    ontology_matches = get_default_matcher().match_entities(entities)
    return {
        "has_conflict": bool(conflicts),
        "conflicts": conflicts,
        "has_support": bool(supporting_rules),
        "supporting_rules": supporting_rules,
        "linked_entities": ontology_matches["linked_entities"],
        "linked_count": ontology_matches["linked_count"],
    }
