"""Modelos baseline (Regresion Logistica, SVM, Random Forest, XGBoost default) y entrenamiento."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import config
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


def get_baselines(random_state: int = config.RANDOM_SEED) -> dict[str, object]:
    """Devuelve los cuatro modelos baseline con su configuracion por defecto.

    XGBoost 2.x elimino el parametro `use_label_encoder`, por lo que no se pasa
    (el objetivo ya viene codificado a 0/1). Se fija `eval_metric='logloss'`.
    """
    return {
        "Regresion Logistica": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGBoost (default)": XGBClassifier(
            random_state=random_state, eval_metric="logloss"
        ),
    }


def train_and_evaluate(model, X_train, y_train, X_test, y_test) -> tuple[object, dict[str, float]]:
    """Entrena el modelo y devuelve (modelo_entrenado, metricas sobre el conjunto de prueba)."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(np.asarray(y_test), y_pred, y_proba)
    return model, metrics
