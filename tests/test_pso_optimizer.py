"""Tests del optimizador PSO: fitness escalar negativo, enteros redondeados, shape de optimize()."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src import config
from src.pso_optimizer import PARAM_NAMES, PSOXGBoostOptimizer, decode_particle


@pytest.fixture(scope="module")
def toy_data():
    X, y = make_classification(
        n_samples=120, n_features=8, n_informative=5, random_state=config.RANDOM_SEED
    )
    return X, y


def test_decode_rounds_integer_params():
    vector = [0.1, 123.7, 5.4, 3.9, 0.7, 0.8]
    params = decode_particle(np.array(vector))
    assert params["n_estimators"] == 124 and isinstance(params["n_estimators"], int)
    assert params["max_depth"] == 5 and isinstance(params["max_depth"], int)
    assert params["min_child_weight"] == 4 and isinstance(params["min_child_weight"], int)
    assert isinstance(params["learning_rate"], float)
    assert params["learning_rate"] == pytest.approx(0.1)


def test_fitness_returns_negative_scalars(toy_data):
    X, y = toy_data
    opt = PSOXGBoostOptimizer(X, y, cv_folds=3, n_particles=3, n_iterations=1)
    particles = np.array([[config.PSO_PARAM_BOUNDS[p][0] for p in PARAM_NAMES]] * 3)
    costs = opt._fitness_function(particles)
    assert costs.shape == (3,)
    # AUC en [0,1] -> costo (-AUC) en [-1,0].
    assert np.all(costs <= 0.0) and np.all(costs >= -1.0)


def test_optimize_returns_correct_shape(toy_data):
    X, y = toy_data
    opt = PSOXGBoostOptimizer(X, y, cv_folds=3, n_particles=4, n_iterations=2)
    best_pos, history = opt.optimize()
    assert best_pos.shape == (len(PARAM_NAMES),)
    assert list(history.columns) == [
        "iteration", "best_fitness_neg", "best_auc", "mean_fitness_neg"
    ]
    assert len(history) == 2
    # Los mejores hiperparametros enteros salen redondeados.
    params = opt.get_best_params()
    for name in config.INTEGER_PARAMS:
        assert isinstance(params[name], int)
