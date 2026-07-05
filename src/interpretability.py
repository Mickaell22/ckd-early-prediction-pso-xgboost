"""Analisis de interpretabilidad con SHAP (TreeExplainer, summary, bar y waterfall plots)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def run_shap_analysis(model, X_train, X_test, output_dir: str | Path) -> pd.DataFrame:
    """Ejecuta el analisis SHAP sobre un modelo de arboles y guarda las figuras.

    Genera y guarda en `output_dir`:
      - shap_summary.png     : summary plot (violin) de X_test.
      - shap_importance.png  : bar plot de importancia global (|SHAP| media).
      - shap_waterfall.png   : waterfall de un caso individual de ejemplo.

    Args:
        model: modelo de arboles ya entrenado (XGBoost).
        X_train: features de entrenamiento (contexto del explainer).
        X_test: features de prueba sobre las que se calculan los valores SHAP.
        output_dir: carpeta destino de las figuras.

    Returns:
        DataFrame con el ranking de features por importancia SHAP media (desc).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_test = pd.DataFrame(X_test).reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot (violin).
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot de importancia global.
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Waterfall de un caso individual (el primero del conjunto de prueba).
    plt.figure()
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0].values,
        feature_names=list(X_test.columns),
    )
    shap.plots.waterfall(explanation, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ranking por |SHAP| media.
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = (
        pd.DataFrame({"feature": list(X_test.columns), "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("SHAP top-5: %s", ", ".join(ranking["feature"].head(5)))
    return ranking
