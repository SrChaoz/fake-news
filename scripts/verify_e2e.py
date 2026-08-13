#!/usr/bin/env python3
"""Verifica la tubería E2E: FastAPI, modelo, historial y métricas.

Requiere una instancia de FastAPI ya iniciada. Por defecto usa
``http://localhost:8000`` y devuelve código distinto de cero si alguna
verificación falla.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from typing import Any

import requests


REQUIRED_PREDICTION_FIELDS = {
    "extracted_claim",
    "category",
    "detected_entities",
    "ontological_conflicts",
    "prediction",
    "confidence_score",
    "explanation",
    "verification_status",
    "evidence",
}
REQUIRED_ABLATIONS = {
    "full_hybrid",
    "without_ontology",
    "without_claim_extraction",
    "without_ontology_and_claim",
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000", help="URL base de FastAPI.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Tiempo máximo por solicitud en segundos.")
    return parser.parse_args()


def request_json(method: str, url: str, timeout: float, **kwargs: Any) -> Any:
    response = requests.request(method, url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def add_check(results: list[CheckResult], name: str, condition: bool, detail: str) -> None:
    results.append(CheckResult(name=name, passed=condition, detail=detail))


def main() -> int:
    args = parse_arguments()
    base_url = args.base_url.rstrip("/")
    marker = f"[E2E-{uuid.uuid4().hex[:10]}]"
    text = f"{marker} Scientists hide evidence that CO2 has no impact on global warming."
    results: list[CheckResult] = []

    try:
        health = request_json("GET", f"{base_url}/health", args.timeout)
        add_check(results, "API disponible", health.get("status") == "ok", f"status={health.get('status')!r}")
    except requests.RequestException as error:
        add_check(results, "API disponible", False, str(error))
        report(results)
        return 1

    prediction: dict[str, Any] = {}
    try:
        payload = request_json("POST", f"{base_url}/predict", args.timeout, json={"text": text})
        if not isinstance(payload, dict):
            raise ValueError("La respuesta de /predict no es un objeto JSON.")
        prediction = payload
        missing = sorted(REQUIRED_PREDICTION_FIELDS - payload.keys())
        valid_label = payload.get("prediction") in {"REAL", "FAKE"}
        confidence = payload.get("confidence_score")
        valid_confidence = isinstance(confidence, (int, float)) and 0 <= confidence <= 1
        add_check(results, "POST /predict", not missing and valid_label and valid_confidence, f"faltantes={missing}; prediction={payload.get('prediction')}; confidence={confidence}")
    except (requests.RequestException, ValueError) as error:
        add_check(results, "POST /predict", False, str(error))

    try:
        history = request_json("GET", f"{base_url}/history?limit=50", args.timeout)
        items = history.get("items", []) if isinstance(history, dict) else []
        persisted = any(item.get("input_text") == text for item in items if isinstance(item, dict))
        add_check(results, "Persistencia en /history", persisted, f"historial consultado={len(items)}; marcador={marker}")
    except requests.RequestException as error:
        add_check(results, "Persistencia en /history", False, str(error))

    try:
        metrics = request_json("GET", f"{base_url}/metrics", args.timeout)
        model_metrics = metrics.get("metrics", {}) if isinstance(metrics, dict) else {}
        required = {"accuracy", "precision", "recall", "f1_score", "confusion_matrix"}
        add_check(results, "GET /metrics", required.issubset(model_metrics), f"campos={sorted(model_metrics.keys())}")
    except requests.RequestException as error:
        add_check(results, "GET /metrics", False, str(error))

    try:
        experiments = request_json("GET", f"{base_url}/experiments", args.timeout)
        required = {"bert", "roberta", "ontology_system", "hybrid_model"}
        add_check(results, "GET /experiments", isinstance(experiments, dict) and required.issubset(experiments), f"campos={sorted(experiments.keys()) if isinstance(experiments, dict) else []}")
    except requests.RequestException as error:
        add_check(results, "GET /experiments", False, str(error))

    try:
        ablation = request_json("GET", f"{base_url}/ablation", args.timeout)
        add_check(results, "GET /ablation", isinstance(ablation, dict) and REQUIRED_ABLATIONS.issubset(ablation), f"variantes={sorted(ablation.keys()) if isinstance(ablation, dict) else []}")
    except requests.RequestException as error:
        add_check(results, "GET /ablation", False, str(error))

    report(results)
    return 0 if all(result.passed for result in results) else 1


def report(results: list[CheckResult]) -> None:
    """Imprime un resumen claro apto para CI o ejecución manual."""
    print("\n=== Verificación E2E ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
    passed = sum(result.passed for result in results)
    print(f"\nResultado: {passed}/{len(results)} verificaciones aprobadas.")


if __name__ == "__main__":
    raise SystemExit(main())
