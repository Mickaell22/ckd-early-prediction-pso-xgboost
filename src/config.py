"""Constantes globales del proyecto: semillas, configuracion del PSO, rangos de hiperparametros y rutas.

Unico lugar donde tocar si algo cambia. Las rutas se derivan de la ubicacion de
este archivo para no hardcodear rutas absolutas.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibilidad ---
RANDOM_SEED: int = 42
SEEDS_MULTIRUN: list[int] = [42, 43, 44, 45, 46]
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# --- Configuracion del PSO ---
N_PARTICLES: int = 30
N_ITERATIONS: int = 50
W_MAX: float = 0.9          # inercia inicial
W_MIN: float = 0.4          # inercia final
C1: float = 2.0             # coeficiente cognitivo
C2: float = 2.0             # coeficiente social
VELOCITY_FRACTION: float = 0.2  # velocidad maxima = 20% del rango de cada dimension

# --- Espacio de busqueda de hiperparametros (orden fijo, lo usa el PSO) ---
# Cada entrada: (limite_inferior, limite_superior)
PSO_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (0.01, 0.30),
    "n_estimators": (50, 500),
    "max_depth": (3, 10),
    "min_child_weight": (1, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
}
# Hiperparametros que deben redondearse a entero antes de instanciar XGBoost.
INTEGER_PARAMS: frozenset[str] = frozenset({"n_estimators", "max_depth", "min_child_weight"})

# --- Columnas del dataset ---
# Numericas: imputacion KNN + MinMaxScaler.
NUMERIC_COLS: list[str] = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
]
# Nominales: one-hot encoding.
NOMINAL_COLS: list[str] = ["rbc", "pc", "pcc", "ba"]
# Binarias: mapeo a 0/1.
BINARY_COLS: list[str] = ["htn", "dm", "cad", "appet", "pe", "ane"]
TARGET_COL: str = "classification"

# Mapeo de valores binarios a 0/1 (cubre yes/no, present/notpresent, good/poor, normal/abnormal).
BINARY_MAP: dict[str, int] = {
    "yes": 1, "no": 0,
    "present": 1, "notpresent": 0,
    "good": 1, "poor": 0,
    "normal": 1, "abnormal": 0,
}

# --- Rutas (derivadas de la ubicacion del proyecto) ---
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw" / "ckd_uci.csv"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV: Path = DATA_PROCESSED_DIR / "ckd_train.csv"
TEST_CSV: Path = DATA_PROCESSED_DIR / "ckd_test.csv"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
MODELS_DIR: Path = RESULTS_DIR / "models"
PSO_CONVERGENCE_CSV: Path = RESULTS_DIR / "pso_convergence.csv"
METRICS_BASELINES_JSON: Path = RESULTS_DIR / "metrics_baselines.json"
METRICS_PSO_JSON: Path = RESULTS_DIR / "metrics_pso_xgboost.json"
METRICS_JSON: Path = RESULTS_DIR / "metrics.json"
PREPROCESSOR_PKL: Path = MODELS_DIR / "preprocessor.pkl"
PSO_MODEL_PKL: Path = MODELS_DIR / "pso_xgboost_final.pkl"
