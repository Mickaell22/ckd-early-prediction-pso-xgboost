"""Tests de evaluacion: metricas en [0, 1] y especificidad correcta con un caso conocido."""

import numpy as np

from src.evaluation import compute_metrics


def test_metrics_within_unit_range():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=100)
    y_proba = rng.random(size=100)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    for name in ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]:
        assert 0.0 <= metrics[name] <= 1.0, name
    assert -1.0 <= metrics["mcc"] <= 1.0


def test_specificity_known_case():
    # Caso construido: TN=3, FP=1, FN=2, TP=4 -> especificidad = 3/(3+1) = 0.75
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    y_proba = y_pred.astype(float)
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["specificity"] == 0.75
    # Sensibilidad (recall) = TP/(TP+FN) = 4/6
    assert metrics["recall"] == 4 / 6


def test_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc_roc"] == 1.0
    assert metrics["specificity"] == 1.0
