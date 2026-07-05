"""Tests del preprocesamiento: sin NaN al final, dimensiones correctas, fit_transform == fit + transform."""

import numpy as np
import pytest

from src import data_loader as dl
from src import preprocessing as pp


@pytest.fixture(scope="module")
def splits():
    df = dl.load_ckd_dataset()
    return dl.split_data(df)


def test_no_nan_after_transform(splits):
    X_train, X_test, _, _ = splits
    prep = pp.CKDPreprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_test_p = prep.transform(X_test)
    assert X_train_p.isna().sum().sum() == 0
    assert X_test_p.isna().sum().sum() == 0


def test_dimensions_consistent(splits):
    X_train, X_test, _, _ = splits
    prep = pp.CKDPreprocessor()
    X_train_p = prep.fit_transform(X_train)
    X_test_p = prep.transform(X_test)
    # Mismo numero de columnas en train y test, e igual a feature_names_.
    assert X_train_p.shape[1] == X_test_p.shape[1] == len(prep.feature_names_)
    assert X_train_p.shape[0] == len(X_train)
    assert list(X_train_p.columns) == list(X_test_p.columns)


def test_fit_transform_equals_fit_then_transform(splits):
    X_train, _, _, _ = splits
    a = pp.CKDPreprocessor().fit_transform(X_train)
    b = pp.CKDPreprocessor().fit(X_train).transform(X_train)
    np.testing.assert_allclose(a.values, b.values)


def test_values_scaled_to_unit_range(splits):
    X_train, _, _, _ = splits
    X_train_p = pp.CKDPreprocessor().fit_transform(X_train)
    assert X_train_p.values.min() >= 0.0
    assert X_train_p.values.max() <= 1.0 + 1e-9


def test_smote_balances_classes(splits):
    X_train, _, y_train, _ = splits
    X_train_p = pp.CKDPreprocessor().fit_transform(X_train)
    X_res, y_res = pp.apply_smote(X_train_p, y_train)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)
