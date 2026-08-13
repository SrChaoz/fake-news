#!/usr/bin/env python3
"""Pruebas de regresión del recuperador de evidencia curada."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.evidence_engine import get_evidence_retriever, verify_with_evidence


def main() -> int:
    retriever = get_evidence_retriever()
    document = retriever.documents[0]
    exact = verify_with_evidence(document.claim)
    unrelated = verify_with_evidence("A completely unrelated claim about quantum computers and football.")
    exact_ok = exact["status"] in {"SUPPORTED", "REFUTED"} and exact["matches"][0]["similarity"] == 1.0
    unrelated_ok = unrelated["status"] == "INSUFFICIENT_EVIDENCE"
    print(f"[{'PASS' if exact_ok else 'FAIL'}] coincidencia exacta: {exact['status']}")
    print(f"[{'PASS' if unrelated_ok else 'FAIL'}] abstención ante texto sin evidencia: {unrelated['status']}")
    return 0 if exact_ok and unrelated_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
