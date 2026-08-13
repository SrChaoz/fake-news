"""Construcción de variables textuales y neuro-simbólicas para modelos ML."""

from __future__ import annotations

import re
from typing import Any

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer


WORD_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
SENTENCE_PATTERN = re.compile(r"[.!?]+")
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouyà-ÿ]+", re.IGNORECASE)


def _estimate_syllables(word: str) -> int:
    """Estima sílabas sin descargar recursos adicionales de NLTK."""
    return max(1, len(VOWEL_GROUP_PATTERN.findall(word)))


def _readability_score(text: str) -> float:
    """Calcula una aproximación Flesch Reading Ease para texto inglés."""
    words = WORD_PATTERN.findall(text)
    if not words:
        return 0.0
    sentences = max(1, len(SENTENCE_PATTERN.findall(text)))
    syllables = sum(_estimate_syllables(word) for word in words)
    return round(206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words)), 2)


def build_feature_vector(text: str, claim: str, ontology_results: dict[str, Any]) -> dict[str, Any]:
    """Genera TF-IDF, variables ontológicas y métricas de legibilidad.

    El vectorizador se ajusta al texto y la afirmación de la petición; durante
    entrenamiento se recomienda sustituirlo por un vectorizador ajustado sobre
    el corpus de entrenamiento y persistido junto al modelo.
    """
    text = text or ""
    claim = claim or ""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", lowercase=True)
    tfidf_matrix = vectorizer.fit_transform([text, claim])
    combined_tfidf = tfidf_matrix.mean(axis=0)

    linked_entities = ontology_results.get("linked_entities", [])
    linked_count = int(ontology_results.get("linked_count", len(linked_entities)))
    has_conflict = int(bool(ontology_results.get("has_conflict", False)))
    text_words = WORD_PATTERN.findall(text)
    claim_words = WORD_PATTERN.findall(claim)
    numeric_names = [
        "linked_entity_count",
        "has_ontological_conflict",
        "text_word_count",
        "claim_word_count",
        "claim_readability",
    ]
    numeric_features = csr_matrix(
        [[linked_count, has_conflict, len(text_words), len(claim_words), _readability_score(claim)]]
    )
    vector = hstack([csr_matrix(combined_tfidf), numeric_features], format="csr")

    return {
        "vector": vector,
        "feature_names": [*vectorizer.get_feature_names_out().tolist(), *numeric_names],
        "tfidf_vectorizer": vectorizer,
        "metadata": {
            "linked_entity_count": linked_count,
            "has_conflict": bool(has_conflict),
            "text_word_count": len(text_words),
            "claim_word_count": len(claim_words),
            "claim_readability": _readability_score(claim),
        },
    }
