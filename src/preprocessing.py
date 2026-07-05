"""Preprocesamiento: CKDPreprocessor (imputacion, codificacion, escalado) y aplicacion de SMOTE."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from src import config

logger = logging.getLogger(__name__)


class CKDPreprocessor:
    """Preprocesador estilo scikit-learn para el dataset CKD.

    Encapsula, en este orden:
      1. Imputacion KNN (k=5) para variables numericas.
      2. Imputacion por moda para variables categoricas.
      3. Codificacion binaria (yes/no -> 1/0) para las columnas binarias.
      4. One-hot encoding para las variables nominales (rbc, pc, pcc, ba).
      5. MinMaxScaler para las numericas.

    Los imputadores, el encoder y el scaler se ajustan SOLO con datos de
    entrenamiento; `transform` reutiliza esos parametros para el conjunto de prueba.
    """

    def __init__(self, knn_neighbors: int = 5) -> None:
        self.knn_neighbors = knn_neighbors
        self.numeric_cols = list(config.NUMERIC_COLS)
        self.nominal_cols = list(config.NOMINAL_COLS)
        self.binary_cols = list(config.BINARY_COLS)

        self._knn_imputer: KNNImputer | None = None
        self._cat_imputer: SimpleImputer | None = None
        self._ohe: OneHotEncoder | None = None
        self._scaler: MinMaxScaler | None = None
        self.feature_names_: list[str] = []
        self._fitted = False

    # -- API estilo sklearn ----------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CKDPreprocessor":
        """Ajusta imputadores, encoder y scaler con el conjunto de entrenamiento."""
        cat_cols = self.nominal_cols + self.binary_cols

        self._knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors)
        self._knn_imputer.fit(X[self.numeric_cols])

        self._cat_imputer = SimpleImputer(strategy="most_frequent")
        self._cat_imputer.fit(X[cat_cols])

        # Nominales imputadas -> one-hot.
        cat_imputed = pd.DataFrame(
            self._cat_imputer.transform(X[cat_cols]), columns=cat_cols, index=X.index
        )
        self._ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self._ohe.fit(cat_imputed[self.nominal_cols])

        # Scaler sobre numericas imputadas.
        num_imputed = self._knn_imputer.transform(X[self.numeric_cols])
        self._scaler = MinMaxScaler()
        self._scaler.fit(num_imputed)

        ohe_names = list(self._ohe.get_feature_names_out(self.nominal_cols))
        self.feature_names_ = self.numeric_cols + self.binary_cols + ohe_names
        self._fitted = True
        logger.info("CKDPreprocessor ajustado: %d features de salida", len(self.feature_names_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma un DataFrame con los parametros ya ajustados."""
        if not self._fitted:
            raise RuntimeError("CKDPreprocessor no ajustado: llama a fit() primero.")

        cat_cols = self.nominal_cols + self.binary_cols
        cat_imputed = pd.DataFrame(
            self._cat_imputer.transform(X[cat_cols]), columns=cat_cols, index=X.index
        )

        # Numericas: imputar + escalar.
        num_imputed = self._knn_imputer.transform(X[self.numeric_cols])
        num_scaled = self._scaler.transform(num_imputed)
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_cols, index=X.index)

        # Binarias: mapear a 0/1 (fallback a 0 para valores no previstos).
        bin_df = cat_imputed[self.binary_cols].apply(
            lambda col: col.str.strip().map(config.BINARY_MAP)
        ).fillna(0).astype(int)

        # Nominales: one-hot.
        ohe_arr = self._ohe.transform(cat_imputed[self.nominal_cols])
        ohe_names = self._ohe.get_feature_names_out(self.nominal_cols)
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_names, index=X.index)

        out = pd.concat([num_df, bin_df, ohe_df], axis=1)
        out = out[self.feature_names_]  # orden estable
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Equivalente a fit(X).transform(X)."""
        return self.fit(X, y).transform(X)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica SMOTE SOLO al conjunto de entrenamiento para balancear las clases.

    Args:
        X_train: features de entrenamiento ya preprocesadas.
        y_train: objetivo de entrenamiento.
        random_state: semilla.

    Returns:
        (X_resampled, y_resampled) con las clases balanceadas.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE aplicado: %d -> %d filas (positivos %.2f -> %.2f)",
        len(X_train), len(X_res), np.mean(y_train), np.mean(y_res),
    )
    return X_res, y_res
