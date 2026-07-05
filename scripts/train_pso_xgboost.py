"""Ejecuta la optimizacion PSO, entrena el XGBoost final multi-seed y guarda modelo y metricas.

Flujo:
  1. Carga y preprocesa datos (fit del preprocesador solo con train, SMOTE solo en train).
  2. Optimiza con PSO usando la semilla 42 (una sola vez, no multi-seed).
  3. Con los mejores hiperparametros, entrena y evalua 5 veces (semillas 42-46).
  4. Guarda el modelo final y el preprocesador.
  5. Guarda las metricas agregadas.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from src import config
from src import data_loader as dl
from src import preprocessing as pp
from src.evaluation import multi_seed_evaluation
from src.pso_optimizer import PSOXGBoostOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_pso_xgboost")


def main() -> None:
    # 1. Datos
    df = dl.load_ckd_dataset()
    X_train, X_test, y_train, y_test = dl.split_data(df)
    preprocessor = pp.CKDPreprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    X_train_res, y_train_res = pp.apply_smote(X_train_p, y_train)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.PREPROCESSOR_PKL)
    logger.info("Preprocesador guardado en %s", config.PREPROCESSOR_PKL)

    # 2. Optimizacion PSO (semilla 42, una sola vez)
    optimizer = PSOXGBoostOptimizer(X_train_res, y_train_res, random_state=config.RANDOM_SEED)
    optimizer.optimize()
    best_params = optimizer.get_best_params()

    # 3. Evaluacion multi-seed con los mejores hiperparametros
    factory = lambda seed: optimizer.get_best_model(random_state=seed)
    summary = multi_seed_evaluation(
        factory, X_train_res, y_train_res, X_test_p, y_test, seeds=config.SEEDS_MULTIRUN
    )

    # 4. Modelo final (semilla 42) entrenado y guardado
    final_model = optimizer.get_best_model(random_state=config.RANDOM_SEED)
    final_model.fit(X_train_res, y_train_res)
    joblib.dump(final_model, config.PSO_MODEL_PKL)
    logger.info("Modelo final guardado en %s", config.PSO_MODEL_PKL)

    # 5. Metricas
    payload = {
        "model": "PSO-XGBoost",
        "best_params": best_params,
        "best_cv_auc": -optimizer.best_cost_,
        "metrics": summary,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PSO_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Metricas PSO-XGBoost guardadas en %s", config.METRICS_PSO_JSON)
    logger.info(
        "PSO-XGBoost -> AUC %.3f±%.3f | Sens %.3f | Espec %.3f",
        summary["auc_roc"]["mean"], summary["auc_roc"]["std"],
        summary["recall"]["mean"], summary["specificity"]["mean"],
    )


if __name__ == "__main__":
    main()
