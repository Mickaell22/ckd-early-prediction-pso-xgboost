"""Carga y limpieza del dataset UCI CKD, filtrado del subgrupo diabetico y particion estratificada."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)

# Columnas que el UCI guarda como numericas pero que llegan con basura de texto
# (tabuladores, '?'), por lo que hay que forzar la conversion a numerico.
_NUMERIC = config.NUMERIC_COLS


def load_ckd_dataset(path: str | Path = config.DATA_RAW) -> pd.DataFrame:
    """Lee el CSV crudo del UCI CKD y devuelve un DataFrame limpio, sin imputar ni escalar.

    Limpieza aplicada:
    - '?' y cadenas vacias se convierten en NaN.
    - Espacios y tabuladores sobrantes en celdas de texto se recortan.
    - Las columnas numericas se fuerzan a numerico (valores no parseables -> NaN).
    - El objetivo se normaliza a {'ckd', 'notckd'}.

    Args:
        path: ruta al CSV crudo.

    Returns:
        DataFrame con las 25 columnas originales, tipado pero con valores faltantes.
    """
    df = pd.read_csv(path, na_values=["?", "", "\t?"], skipinitialspace=True)
    logger.info("Dataset crudo cargado: %d filas x %d columnas", *df.shape)

    # Recorta espacios/tabuladores en las columnas de texto (las no numericas).
    text_cols = [c for c in df.columns if c not in _NUMERIC]
    for col in text_cols:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()

    # Fuerza numerico donde corresponde; lo no parseable queda como NaN.
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normaliza el objetivo: cualquier valor que no sea exactamente 'ckd' se trata
    # como 'notckd' (el UCI trae una fila corrupta con 'no').
    if config.TARGET_COL in df.columns:
        df[config.TARGET_COL] = df[config.TARGET_COL].apply(
            lambda v: "ckd" if str(v).strip() == "ckd" else "notckd"
        )

    return df


def filter_diabetic_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo los pacientes diabeticos (dm == 'yes').

    Nota: en el dataset UCI todos los diabeticos tienen ERC, por lo que este
    subconjunto tiene una sola clase y no sirve para clasificacion. Se conserva
    para analisis descriptivo (EDA), no para modelado.
    """
    subset = df[df["dm"] == "yes"].copy()
    logger.info("Subgrupo diabetico: %d filas", len(subset))
    return subset


def encode_target(y: pd.Series) -> pd.Series:
    """Codifica el objetivo a 1 (ckd) / 0 (notckd)."""
    return (y.astype(str).str.strip() == "ckd").astype(int)


def split_data(
    df: pd.DataFrame,
    target_col: str = config.TARGET_COL,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Particion estratificada train/test sobre el objetivo.

    Args:
        df: DataFrame limpio.
        target_col: nombre de la columna objetivo.
        test_size: proporcion del conjunto de prueba.
        random_state: semilla.

    Returns:
        (X_train, X_test, y_train, y_test). El objetivo sale codificado a 0/1.
    """
    X = df.drop(columns=[target_col])
    y = encode_target(df[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Particion estratificada: train=%d, test=%d (positivos train=%.2f, test=%.2f)",
        len(X_train), len(X_test), y_train.mean(), y_test.mean(),
    )
    return X_train, X_test, y_train, y_test
