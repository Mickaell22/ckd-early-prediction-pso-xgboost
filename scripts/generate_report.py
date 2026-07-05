"""Consolida los resultados en tablas comparativas y genera las figuras finales para el paper.

Requiere haber corrido antes `train_baselines.py` y `train_pso_xgboost.py`.
Genera la tabla comparativa (Markdown + CSV) y todas las figuras en `results/figures/`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src import data_loader as dl
from src import preprocessing as pp
from src.evaluation import plot_confusion_matrix, plot_roc_curves
from src.interpretability import run_shap_analysis
from src.models import get_baselines, train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_report")

# Metricas que van en la tabla comparativa tipo Tabla III del paper.
TABLE_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Sensibilidad",
    "specificity": "Especificidad", "f1": "F1", "auc_roc": "AUC",
}


def _prepare_data():
    df = dl.load_ckd_dataset()
    X_train, X_test, y_train, y_test = dl.split_data(df)
    preprocessor = pp.CKDPreprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)
    X_train_res, y_train_res = pp.apply_smote(X_train_p, y_train)
    return X_train_res, y_train_res, X_train_p, X_test_p, y_test


def _load_metrics() -> dict[str, dict[str, float]]:
    """Devuelve {modelo: {metrica: media}} a partir de los JSON guardados."""
    table: dict[str, dict[str, float]] = {}
    with open(config.METRICS_BASELINES_JSON, encoding="utf-8") as f:
        baselines = json.load(f)
    for name, summary in baselines.items():
        table[name] = {m: summary[m]["mean"] for m in TABLE_METRICS}
    with open(config.METRICS_PSO_JSON, encoding="utf-8") as f:
        pso = json.load(f)
    table["PSO-XGBoost"] = {m: pso["metrics"][m]["mean"] for m in TABLE_METRICS}
    return table


def _to_markdown(df: pd.DataFrame) -> str:
    """Serializa un DataFrame a tabla Markdown sin depender de `tabulate`."""
    headers = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:.3f}" for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_comparison_table(table: dict[str, dict[str, float]]) -> None:
    df = pd.DataFrame(table).T[TABLE_METRICS].rename(columns=METRIC_LABELS)
    df.index.name = "Modelo"
    df.round(3).to_csv(config.RESULTS_DIR / "comparison_table.csv")
    with open(config.RESULTS_DIR / "comparison_table.md", "w", encoding="utf-8") as f:
        f.write(_to_markdown(df))
    with open(config.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    logger.info("Tabla comparativa escrita (CSV, Markdown y metrics.json)")


def _plot_pso_convergence() -> None:
    if not config.PSO_CONVERGENCE_CSV.exists():
        logger.warning("No existe %s, se omite la figura de convergencia", config.PSO_CONVERGENCE_CSV)
        return
    hist = pd.read_csv(config.PSO_CONVERGENCE_CSV)
    plt.figure(figsize=(7, 5))
    plt.plot(hist["iteration"], hist["best_auc"], marker="o", ms=3, label="Mejor AUC")
    plt.xlabel("Iteracion")
    plt.ylabel("AUC-ROC (CV)")
    plt.title("Convergencia del PSO")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "pso_convergence.png", dpi=150)
    plt.close()
    logger.info("Figura de convergencia guardada")


def _plot_feature_importance(model, feature_names: list[str]) -> None:
    importances = model.feature_importances_
    order = importances.argsort()[::-1][:15]
    plt.figure(figsize=(7, 6))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Importancia (ganancia XGBoost)")
    plt.title("Importancia de variables — PSO-XGBoost")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "feature_importance_xgb.png", dpi=150)
    plt.close()
    logger.info("Figura de importancia XGBoost guardada")


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Tabla comparativa desde los JSON.
    table = _load_metrics()
    _write_comparison_table(table)

    # Datos y modelos para las figuras.
    X_train_res, y_train_res, X_train_p, X_test_p, y_test = _prepare_data()

    trained = {}
    for name, model in get_baselines(config.RANDOM_SEED).items():
        trained[name], _ = train_and_evaluate(model, X_train_res, y_train_res, X_test_p, y_test)

    pso_model = joblib.load(config.PSO_MODEL_PKL)
    trained["PSO-XGBoost"] = pso_model

    # Figuras.
    plot_roc_curves(trained, X_test_p, y_test, config.FIGURES_DIR / "roc_comparison.png")
    plot_confusion_matrix(
        y_test, pso_model.predict(X_test_p), "PSO-XGBoost",
        config.FIGURES_DIR / "confusion_matrix_pso_xgboost.png",
    )
    _plot_pso_convergence()
    _plot_feature_importance(pso_model, list(X_test_p.columns))

    ranking = run_shap_analysis(pso_model, X_train_res, X_test_p, config.FIGURES_DIR)
    ranking.to_csv(config.RESULTS_DIR / "shap_ranking.csv", index=False)
    logger.info("Reporte completo generado en %s", config.RESULTS_DIR)


if __name__ == "__main__":
    main()
