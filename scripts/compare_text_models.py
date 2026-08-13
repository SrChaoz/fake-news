#!/usr/bin/env python3
"""Compara preprocesamiento y modelos léxicos para fact-checking climático.

El objetivo es medir si preservar negaciones mejora la detección de FAKE antes
de gastar recursos en un entrenamiento Transformer completo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_models import (  # noqa: E402
    RANDOM_STATE,
    STOP_WORDS_PRESERVING_NEGATION,
    load_training_records,
)


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(np.mean(f1)), 4),
        "fake_precision": round(float(precision[0]), 4),
        "fake_recall": round(float(recall[0]), 4),
        "fake_f1": round(float(f1[0]), 4),
        "real_f1": round(float(f1[1]), 4),
    }


def word_legacy(train: list[str], test: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5_000, sublinear_tf=True)
    return vectorizer.fit_transform(train), vectorizer.transform(test)


def word_negation(train: list[str], test: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOP_WORDS_PRESERVING_NEGATION, max_features=5_000, sublinear_tf=True)
    return vectorizer.fit_transform(train), vectorizer.transform(test)


def word_character_negation(train: list[str], test: list[str]) -> tuple[np.ndarray, np.ndarray]:
    word = TfidfVectorizer(ngram_range=(1, 2), stop_words=STOP_WORDS_PRESERVING_NEGATION, max_features=5_000, sublinear_tf=True)
    character = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=8_000, sublinear_tf=True)
    return hstack([word.fit_transform(train), character.fit_transform(train)]), hstack([word.transform(test), character.transform(test)])


def evaluate(name: str, vectorize: Callable[[list[str], list[str]], tuple[np.ndarray, np.ndarray]], texts: list[str], targets: np.ndarray) -> dict[str, object]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict[str, float]] = []
    for train_rows, test_rows in splitter.split(texts, targets):
        train_texts = [texts[index] for index in train_rows]
        test_texts = [texts[index] for index in test_rows]
        train_matrix, test_matrix = vectorize(train_texts, test_texts)
        classifier = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=RANDOM_STATE)
        classifier.fit(train_matrix, targets[train_rows])
        folds.append(scores(targets[test_rows], classifier.predict(test_matrix)))
    summary = {key: round(float(np.mean([fold[key] for fold in folds])), 4) for key in folds[0]}
    print(f"{name}: macro_f1={summary['macro_f1']:.4f} fake_recall={summary['fake_recall']:.4f} fake_f1={summary['fake_f1']:.4f}")
    return {"folds": folds, "summary": summary}


def main() -> int:
    records = load_training_records(PROJECT_ROOT / "climate-fever-dataset-r1-enriched.jsonl", include_ambiguous=False)
    texts = [f"{record.raw_text} [CLAIM] {record.claim}" for record in records]
    targets = np.asarray([record.target for record in records], dtype=np.int8)
    variants = {
        "legacy_tfidf_logreg": word_legacy,
        "negation_aware_tfidf_logreg": word_negation,
        "negation_aware_word_char_logreg": word_character_negation,
    }
    results = {name: evaluate(name, vectorize, texts, targets) for name, vectorize in variants.items()}
    output = PROJECT_ROOT / "app" / "models" / "experiments" / "preprocessing_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"records": len(records), "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Resultados: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
