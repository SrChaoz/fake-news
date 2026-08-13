#!/usr/bin/env python3
"""Entrena un clasificador léxico consciente de negación para fact-checking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_models import (  # noqa: E402
    RANDOM_STATE,
    STOP_WORDS_PRESERVING_NEGATION,
    load_excluded_claims,
    load_training_records,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experiment-name", default="negation_word_char_logreg")
    result.add_argument("--exclude-claims-file", type=Path, default=None)
    result.add_argument("--cv-folds", type=int, default=5)
    return result


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), stop_words=STOP_WORDS_PRESERVING_NEGATION, max_features=5_000, sublinear_tf=True)),
        ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8_000, sublinear_tf=True)),
    ])


def make_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)


def metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, predictions, labels=[0, 1], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1_score": round(float(f1[1]), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "per_class": {"FAKE": {"precision": round(float(precision[0]), 4), "recall": round(float(recall[0]), 4), "f1_score": round(float(f1[0]), 4)}, "REAL": {"precision": round(float(precision[1]), 4), "recall": round(float(recall[1]), 4), "f1_score": round(float(f1[1]), 4)}},
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def find_threshold(texts: list[str], targets: np.ndarray, train_indices: np.ndarray) -> float:
    train_targets = targets[train_indices]
    probabilities = np.zeros(len(train_indices), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_fold, validation_fold in splitter.split(train_indices, train_targets):
        rows_train, rows_validation = train_indices[train_fold], train_indices[validation_fold]
        vectorizer = make_vectorizer()
        matrix_train = vectorizer.fit_transform([texts[index] for index in rows_train])
        matrix_validation = vectorizer.transform([texts[index] for index in rows_validation])
        classifier = make_classifier().fit(matrix_train, targets[rows_train])
        probabilities[validation_fold] = classifier.predict_proba(matrix_validation)[:, 1]
    candidates = np.linspace(0.3, 0.7, 21)
    eligible: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        candidate = metrics(train_targets, (probabilities >= threshold).astype(np.int8))
        if candidate["per_class"]["FAKE"]["recall"] >= 0.6:
            eligible.append((float(threshold), candidate))
    pool = eligible or [(float(threshold), metrics(train_targets, (probabilities >= threshold).astype(np.int8))) for threshold in candidates]
    return max(pool, key=lambda item: (item[1]["macro_f1"], item[1]["balanced_accuracy"], item[1]["per_class"]["FAKE"]["recall"]))[0]


def cross_validate(texts: list[str], targets: np.ndarray, threshold: float, folds: int) -> dict[str, Any]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, Any]] = []
    for number, (train_rows, test_rows) in enumerate(splitter.split(texts, targets), start=1):
        vectorizer = make_vectorizer()
        classifier = make_classifier().fit(vectorizer.fit_transform([texts[index] for index in train_rows]), targets[train_rows])
        result = metrics(targets[test_rows], (classifier.predict_proba(vectorizer.transform([texts[index] for index in test_rows]))[:, 1] >= threshold).astype(np.int8))
        result["fold"] = number
        rows.append(result)
    summary = {key: {"mean": round(float(np.mean([row[key] for row in rows])), 4), "std": round(float(np.std([row[key] for row in rows])), 4)} for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "macro_f1")}
    return {"folds": rows, "summary": summary}


def main() -> int:
    args = parser().parse_args()
    if Path(args.experiment_name).name != args.experiment_name:
        parser().error("--experiment-name debe ser un nombre simple.")
    excluded = load_excluded_claims(args.exclude_claims_file)
    records = load_training_records(PROJECT_ROOT / "climate-fever-dataset-r1-enriched.jsonl", False, excluded)
    texts = [f"{record.raw_text} [CLAIM] {record.claim}" for record in records]
    targets = np.asarray([record.target for record in records], dtype=np.int8)
    all_indices = np.arange(len(records))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, stratify=targets, random_state=RANDOM_STATE)
    threshold = find_threshold(texts, targets, train_indices)
    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform([texts[index] for index in train_indices])
    classifier = make_classifier().fit(matrix, targets[train_indices])
    test_matrix = vectorizer.transform([texts[index] for index in test_indices])
    test_metrics = metrics(targets[test_indices], (classifier.predict_proba(test_matrix)[:, 1] >= threshold).astype(np.int8))
    cv = cross_validate(texts, targets, threshold, args.cv_folds)
    output = PROJECT_ROOT / "app" / "models" / "experiments" / args.experiment_name
    output.mkdir(parents=True, exist_ok=True)
    # El nombre se conserva para compatibilidad con el promotor y la API; el
    # tipo real se documenta de forma explícita en model_config.json.
    joblib.dump(classifier, output / "random_forest_model.pkl")
    joblib.dump(vectorizer, output / "tfidf.pkl")
    config = {"variant": "negation_aware_word_char_logreg", "classifier": "LogisticRegression", "feature_mode": "text_tfidf_only", "decision_threshold_real": threshold, "transformer": {"enabled": False, "reason": "El experimento léxico no usa embeddings."}, "preprocessing": {"word_ngrams": [1, 2], "character_ngrams": [3, 5], "preserved_negations": sorted({"no", "not", "nor", "never", "neither", "without"})}}
    (output / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {"dataset": {"records": len(records), "class_counts": {"FAKE": int(np.sum(targets == 0)), "REAL": int(np.sum(targets == 1))}, "excluded_benchmark_claims": len(excluded), "test_size": 0.2}, "transformer": config["transformer"], "baseline": {"name": "Negation-aware word + character TF-IDF Logistic Regression", "metrics": test_metrics}, "ablation_study": {"full_hybrid": test_metrics}, "threshold_calibration": {"threshold_real": threshold, "target_fake_recall": 0.6}, "cross_validation": cv, "feature_dimensions": {"text_tfidf_only": int(matrix.shape[1])}}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Registros: {len(records)} | threshold={threshold:.2f}")
    print(f"Test: macro_f1={test_metrics['macro_f1']:.4f}, fake_f1={test_metrics['per_class']['FAKE']['f1_score']:.4f}")
    print(f"CV: macro_f1={cv['summary']['macro_f1']['mean']:.4f}, fake recall={np.mean([fold['per_class']['FAKE']['recall'] for fold in cv['folds']]):.4f}")
    print(f"Artefactos: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
