"""Metricas de evaluacion, validacion cruzada, evaluacion multi-seed y graficas ROC / matriz de confusion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # backend sin display, seguro en servidores/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config

logger = logging.getLogger(__name__)

# Orden canonico de las metricas para reportes reproducibles.
METRIC_NAMES: list[str] = [
    "accuracy", "precision", "recall", "specificity", "f1", "auc_roc", "mcc",
]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> dict[str, float]:
    """Calcula el conjunto completo de metricas de clasificacion binaria.

    La especificidad no la da scikit-learn directamente, se calcula desde la
    matriz de confusion como TN / (TN + FP).

    Args:
        y_true: etiquetas reales (0/1).
        y_pred: etiquetas predichas (0/1).
        y_pred_proba: probabilidad de la clase positiva.

    Returns:
        Diccionario con accuracy, precision, recall, specificity, f1, auc_roc, mcc.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_pred_proba)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def cross_validated_metrics(
    model, X, y, cv: int = config.CV_FOLDS, random_state: int = config.RANDOM_SEED
) -> dict[str, float]:
    """Devuelve media y desviacion del AUC en validacion cruzada estratificada."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}


def multi_seed_evaluation(
    model_factory: Callable[[int], object],
    X_train,
    y_train,
    X_test,
    y_test,
    seeds: list[int] = config.SEEDS_MULTIRUN,
) -> dict[str, dict[str, float]]:
    """Corre el entrenamiento/evaluacion una vez por semilla y agrega media y desviacion.

    Args:
        model_factory: funcion que recibe una semilla y devuelve un modelo nuevo sin entrenar.
        X_train, y_train, X_test, y_test: particiones ya preprocesadas.
        seeds: lista de semillas.

    Returns:
        Diccionario {metrica: {"mean": ..., "std": ...}} mas la lista de corridas por metrica.
    """
    runs: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for seed in seeds:
        model = model_factory(seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(np.asarray(y_test), y_pred, y_proba)
        for m in METRIC_NAMES:
            runs[m].append(metrics[m])
    summary = {
        m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "runs": v}
        for m, v in runs.items()
    }
    logger.info("Evaluacion multi-seed sobre %d semillas completada", len(seeds))
    return summary


def plot_roc_curves(models_dict: dict, X_test, y_test, output_path: str | Path) -> None:
    """Superpone las curvas ROC de todos los modelos en una sola figura."""
    plt.figure(figsize=(7, 6))
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos (sensibilidad)")
    plt.title("Curvas ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Curvas ROC guardadas en %s", output_path)


def plot_confusion_matrix(
    y_true, y_pred, model_name: str, output_path: str | Path
) -> None:
    """Guarda la matriz de confusion de un modelo."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["notckd", "ckd"])
    ax.set_yticks([0, 1], labels=["notckd", "ckd"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Matriz de confusion guardada en %s", output_path)
