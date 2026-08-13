"""Recuperación de evidencia curada y decisión selectiva basada en similitud.

No sustituye un modelo NLI entrenado: esta capa solo produce un veredicto de
evidencia cuando la recuperación supera umbrales conservadores. En los demás
casos devuelve ``INSUFFICIENT_EVIDENCE`` y deja el clasificador como apoyo.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select


HIGH_CONFIDENCE_SIMILARITY = 0.56
MINIMUM_MARGIN = 0.08


@dataclass(frozen=True)
class EvidenceDocument:
    claim: str
    label: str
    evidence_text: str
    source_dataset: str
    source_url: str | None


class EvidenceRetriever:
    """Índice TF-IDF local sobre evidencia científica y correcciones curadas."""

    def __init__(self, documents: list[EvidenceDocument]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=30_000
        )
        # Prioriza similitud entre claims; el abstract/corrección se devuelve
        # como justificación, no se usa para inflar artificialmente el match.
        self.matrix = self.vectorizer.fit_transform([document.claim for document in documents])

    def retrieve(self, claim: str, limit: int = 3) -> dict[str, Any]:
        if not self.documents or not claim.strip():
            return {"status": "INSUFFICIENT_EVIDENCE", "matches": []}
        scores = cosine_similarity(self.vectorizer.transform([claim]), self.matrix).ravel()
        indices = np.argsort(scores)[::-1][:limit]
        matches = [
            {
                "claim": self.documents[index].claim,
                "label": self.documents[index].label,
                "evidence_text": self.documents[index].evidence_text[:1_500],
                "source_dataset": self.documents[index].source_dataset,
                "source_url": self.documents[index].source_url,
                "similarity": round(float(scores[index]), 4),
            }
            for index in indices
            if scores[index] > 0
        ]
        if not matches:
            return {"status": "INSUFFICIENT_EVIDENCE", "matches": []}
        first = matches[0]
        second_score = matches[1]["similarity"] if len(matches) > 1 else 0.0
        strong = first["similarity"] >= HIGH_CONFIDENCE_SIMILARITY and first["similarity"] - second_score >= MINIMUM_MARGIN
        return {
            "status": "REFUTED" if strong and first["label"] == "FAKE" else "SUPPORTED" if strong else "INSUFFICIENT_EVIDENCE",
            "matches": matches,
            "top_similarity": first["similarity"],
            "strong_match": strong,
        }


def load_evidence_documents() -> list[EvidenceDocument]:
    """Lee la evidencia persistida desde PostgreSQL sin mezclar texto no curado."""
    from database import SessionLocal
    from models import DatasetEvidence, DatasetRecord

    with SessionLocal() as session:
        rows = session.execute(
            select(
                DatasetRecord.extracted_claim,
                DatasetRecord.raw_text,
                DatasetRecord.label,
                DatasetEvidence.evidence_text,
                DatasetEvidence.source_dataset,
                DatasetEvidence.source_url,
            ).join(DatasetEvidence, DatasetEvidence.dataset_record_id == DatasetRecord.id)
        ).all()
    return [
        EvidenceDocument(
            claim=(claim or raw_text).strip(), label=label, evidence_text=evidence.strip(),
            source_dataset=source, source_url=url,
        )
        for claim, raw_text, label, evidence, source, url in rows
        if (claim or raw_text) and evidence and label in {"REAL", "FAKE"}
    ]


@lru_cache(maxsize=1)
def get_evidence_retriever() -> EvidenceRetriever:
    """Construye el índice una vez por proceso; reiniciar tras una nueva ingesta."""
    documents = load_evidence_documents()
    if not documents:
        # TfidfVectorizer no admite un corpus vacío; el constructor se evita y
        # la API responderá evidencia insuficiente.
        raise RuntimeError("No hay evidencia curada indexable en PostgreSQL.")
    return EvidenceRetriever(documents)


def verify_with_evidence(claim: str) -> dict[str, Any]:
    """Devuelve evidencia recuperada y un estado selectivo verificable."""
    try:
        return get_evidence_retriever().retrieve(claim)
    except RuntimeError:
        return {"status": "INSUFFICIENT_EVIDENCE", "matches": [], "top_similarity": 0.0, "strong_match": False}
