"""Extracción híbrida de entidades ambientales con spaCy y reglas de dominio."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

try:
    import spacy
    from spacy.language import Language
except ImportError:  # Permite usar el reconocedor basado en reglas sin spaCy.
    spacy = None  # type: ignore[assignment]
    Language = Any  # type: ignore[misc,assignment]


LOGGER = logging.getLogger(__name__)

# término: (nombre canónico, tipo de entidad). Las expresiones se buscan sin
# distinguir mayúsculas, de modo que CO2 y co2 se vinculan al mismo concepto.
ENVIRONMENTAL_TERMS: dict[str, tuple[str, str]] = {
    "co2": ("carbon dioxide", "GREENHOUSE_GAS"),
    "carbon dioxide": ("carbon dioxide", "GREENHOUSE_GAS"),
    "dióxido de carbono": ("carbon dioxide", "GREENHOUSE_GAS"),
    "dioxido de carbono": ("carbon dioxide", "GREENHOUSE_GAS"),
    "ch4": ("methane", "GREENHOUSE_GAS"),
    "methane": ("methane", "GREENHOUSE_GAS"),
    "metano": ("methane", "GREENHOUSE_GAS"),
    "nitrous oxide": ("nitrous oxide", "GREENHOUSE_GAS"),
    "n2o": ("nitrous oxide", "GREENHOUSE_GAS"),
    "greenhouse gas": ("greenhouse gas", "ENVIRONMENTAL_CONCEPT"),
    "gas de efecto invernadero": ("greenhouse gas", "ENVIRONMENTAL_CONCEPT"),
    "global warming": ("global warming", "CLIMATE_PROCESS"),
    "climate change": ("climate change", "CLIMATE_PROCESS"),
    "calentamiento global": ("global warming", "CLIMATE_PROCESS"),
    "cambio climático": ("climate change", "CLIMATE_PROCESS"),
    "cambio climatico": ("climate change", "CLIMATE_PROCESS"),
    "deforestation": ("deforestation", "ENVIRONMENTAL_PROCESS"),
    "reforestation": ("reforestation", "ENVIRONMENTAL_PROCESS"),
    "biodiversity": ("biodiversity", "ENVIRONMENTAL_CONCEPT"),
    "ecosystem": ("ecosystem", "HABITAT"),
    "habitat": ("habitat", "HABITAT"),
    "wetland": ("wetland", "HABITAT"),
    "forest": ("forest", "HABITAT"),
    "ocean": ("ocean", "HABITAT"),
    "coral reef": ("coral reef", "HABITAT"),
    "sea level": ("sea level", "CLIMATE_INDICATOR"),
    "pollution": ("pollution", "ENVIRONMENTAL_PROCESS"),
    "air pollution": ("air pollution", "ENVIRONMENTAL_PROCESS"),
    "renewable energy": ("renewable energy", "ENERGY_SOURCE"),
    "fossil fuel": ("fossil fuel", "ENERGY_SOURCE"),
}


@lru_cache(maxsize=1)
def get_nlp() -> Language | None:
    """Carga el mejor modelo inglés disponible, con una alternativa ligera."""
    if spacy is None:
        LOGGER.warning("spaCy no está instalado; se usarán únicamente reglas ambientales.")
        return None

    for model_name in ("en_core_web_trf", "en_core_web_sm"):
        try:
            return spacy.load(model_name)
        except OSError:
            continue

    LOGGER.warning(
        "No se encontró un modelo spaCy en inglés; se usarán únicamente reglas ambientales. "
        "Instala uno con: python -m spacy download en_core_web_sm"
    )
    return spacy.blank("en")


def _entity_payload(
    text: str,
    normalized: str,
    label: str,
    source: str,
    start_char: int,
    end_char: int,
) -> dict[str, str | int]:
    return {
        "text": text,
        "normalized": normalized,
        "label": label,
        "source": source,
        "start_char": start_char,
        "end_char": end_char,
    }


def extract_environmental_entities(text: str) -> list[dict[str, str | int]]:
    """Extrae entidades generales de spaCy y términos del dominio ambiental.

    Las coincidencias del diccionario tienen prioridad sobre una entidad spaCy
    que ocupe el mismo intervalo, pues aportan un tipo ambiental más específico.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    rule_entities: list[dict[str, str | int]] = []
    # Orden descendente para priorizar "carbon dioxide" sobre términos parciales.
    for term, (normalized, label) in sorted(
        ENVIRONMENTAL_TERMS.items(), key=lambda item: len(item[0]), reverse=True
    ):
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if any(
                item["start_char"] <= span[0] and span[1] <= item["end_char"]
                for item in rule_entities
            ):
                continue
            rule_entities.append(
                _entity_payload(
                    match.group(), normalized, label, "domain_dictionary", match.start(), match.end()
                )
            )

    spacy_entities: list[dict[str, str | int]] = []
    nlp = get_nlp()
    if nlp is not None:
        document = nlp(text)
        for entity in document.ents:
            # Un término del diccionario reemplaza una coincidencia genérica de spaCy.
            if any(
                int(rule["start_char"]) < entity.end_char
                and entity.start_char < int(rule["end_char"])
                for rule in rule_entities
            ):
                continue
            spacy_entities.append(
                _entity_payload(
                    entity.text,
                    entity.text.strip().lower(),
                    f"SPACY_{entity.label_}",
                    "spacy",
                    entity.start_char,
                    entity.end_char,
                )
            )

    entities = [*spacy_entities, *rule_entities]
    return sorted(entities, key=lambda item: (int(item["start_char"]), int(item["end_char"])))
