"""Optimizacion de hiperparametros de XGBoost mediante PSO (pyswarms GlobalBestPSO)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyswarms as ps
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src import config

logger = logging.getLogger(__name__)

# Orden fijo de las dimensiones del vector de particulas.
PARAM_NAMES: list[str] = list(config.PSO_PARAM_BOUNDS.keys())


def decode_particle(vector: np.ndarray) -> dict[str, float | int]:
    """Convierte un vector de particula a un diccionario de hiperparametros de XGBoost.

    Los hiperparametros enteros se redondean; el resto quedan continuos.
    """
    params: dict[str, float | int] = {}
    for name, value in zip(PARAM_NAMES, vector):
        params[name] = int(round(value)) if name in config.INTEGER_PARAMS else float(value)
    return params


class PSOXGBoostOptimizer:
    """Optimiza los hiperparametros de XGBoost maximizando el AUC-ROC en CV con PSO.

    La funcion de fitness es el AUC-ROC promedio en `cv_folds` folds estratificados.
    pyswarms minimiza, por lo que se devuelve el negativo del AUC.
    """

    def __init__(
        self,
        X_train,
        y_train,
        cv_folds: int = config.CV_FOLDS,
        random_state: int = config.RANDOM_SEED,
        n_particles: int = config.N_PARTICLES,
        n_iterations: int = config.N_ITERATIONS,
    ) -> None:
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_particles = n_particles
        self.n_iterations = n_iterations

        self.dimensions = len(PARAM_NAMES)
        self.lb = np.array([config.PSO_PARAM_BOUNDS[p][0] for p in PARAM_NAMES], dtype=float)
        self.ub = np.array([config.PSO_PARAM_BOUNDS[p][1] for p in PARAM_NAMES], dtype=float)

        self.best_pos_: np.ndarray | None = None
        self.best_cost_: float | None = None
        self._mean_fitness_history: list[float] = []

    def _build_model(self, params: dict) -> XGBClassifier:
        return XGBClassifier(
            **params,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
        )

    def _fitness_function(self, particles: np.ndarray) -> np.ndarray:
        """Evalua una matriz de particulas (n_particles x 6) y devuelve -AUC por particula."""
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        costs = np.empty(particles.shape[0], dtype=float)
        for i, vector in enumerate(particles):
            model = self._build_model(decode_particle(vector))
            auc = cross_val_score(
                model, self.X_train, self.y_train, cv=skf, scoring="roc_auc"
            ).mean()
            costs[i] = -auc  # pyswarms minimiza
        self._mean_fitness_history.append(float(costs.mean()))
        return costs

    def optimize(self) -> tuple[np.ndarray, pd.DataFrame]:
        """Ejecuta el PSO y devuelve (mejor_vector, historial_de_convergencia).

        El historial se guarda ademas en `results/pso_convergence.csv`.
        """
        options = {"c1": config.C1, "c2": config.C2, "w": config.W_MAX}
        # Inercia decreciente linealmente de W_MAX a W_MIN (default de OptionsHandler).
        oh_strategy = {"w": "lin_variation"}
        # Velocidad maxima = 20% del rango de cada dimension.
        v_max = config.VELOCITY_FRACTION * (self.ub - self.lb)
        velocity_clamp = (-v_max, v_max)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=self.dimensions,
            options=options,
            bounds=(self.lb, self.ub),
            oh_strategy=oh_strategy,
            velocity_clamp=velocity_clamp,
        )
        self._mean_fitness_history = []
        best_cost, best_pos = optimizer.optimize(
            self._fitness_function, iters=self.n_iterations, verbose=True
        )
        self.best_cost_ = float(best_cost)
        self.best_pos_ = best_pos

        history = self._build_history(optimizer.cost_history)
        self._save_history(history)
        logger.info("PSO finalizado. Mejor AUC=%.4f", -best_cost)
        logger.info("Mejores hiperparametros: %s", decode_particle(best_pos))
        return best_pos, history

    def _build_history(self, cost_history: list[float]) -> pd.DataFrame:
        best_neg = np.asarray(cost_history, dtype=float)
        mean_neg = np.asarray(self._mean_fitness_history, dtype=float)
        n = len(best_neg)
        if len(mean_neg) != n:  # salvaguarda si pyswarms cambia el nº de evaluaciones
            mean_neg = np.resize(mean_neg, n)
        return pd.DataFrame({
            "iteration": np.arange(1, n + 1),
            "best_fitness_neg": best_neg,
            "best_auc": -best_neg,
            "mean_fitness_neg": mean_neg,
        })

    def _save_history(self, history: pd.DataFrame) -> None:
        config.PSO_CONVERGENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(config.PSO_CONVERGENCE_CSV, index=False)
        logger.info("Convergencia guardada en %s", config.PSO_CONVERGENCE_CSV)

    def get_best_params(self) -> dict[str, float | int]:
        """Devuelve los mejores hiperparametros ya decodificados (enteros redondeados)."""
        if self.best_pos_ is None:
            raise RuntimeError("Ejecuta optimize() antes de pedir los mejores hiperparametros.")
        return decode_particle(self.best_pos_)

    def get_best_model(self, random_state: int | None = None) -> XGBClassifier:
        """Instancia el XGBClassifier final con los mejores hiperparametros."""
        params = self.get_best_params()
        model = self._build_model(params)
        if random_state is not None:
            model.set_params(random_state=random_state)
        return model
