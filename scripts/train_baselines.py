"""Entrena y evalua los cuatro modelos baseline con las 5 semillas; guarda metricas agregadas."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import data_loader as dl
from src import preprocessing as pp
from src.evaluation import multi_seed_evaluation
from src.models import get_baselines

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_baselines")


def main() -> None:
    df = dl.load_ckd_dataset()
    X_train, X_test, y_train, y_test = dl.split_data(df)

    preprocessor = pp.CKDPreprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    X_train_res, y_train_res = pp.apply_smote(X_train_p, y_train)

    # Persiste las particiones procesadas (utiles para los notebooks).
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_p.assign(classification=y_train.values).to_csv(config.TRAIN_CSV, index=False)
    X_test_p.assign(classification=y_test.values).to_csv(config.TEST_CSV, index=False)

    results: dict[str, dict] = {}
    for name in get_baselines():
        logger.info("Evaluando baseline: %s", name)
        factory = lambda seed, _name=name: get_baselines(seed)[_name]
        results[name] = multi_seed_evaluation(
            factory, X_train_res, y_train_res, X_test_p, y_test, seeds=config.SEEDS_MULTIRUN
        )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_BASELINES_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Metricas de baselines guardadas en %s", config.METRICS_BASELINES_JSON)

    for name, summary in results.items():
        logger.info(
            "%s -> AUC %.3f±%.3f | Sens %.3f | Espec %.3f",
            name, summary["auc_roc"]["mean"], summary["auc_roc"]["std"],
            summary["recall"]["mean"], summary["specificity"]["mean"],
        )


if __name__ == "__main__":
    main()
