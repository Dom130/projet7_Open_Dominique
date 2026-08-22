"""
Tests unitaires - Projet7_Open_Dominique
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─── Tests metrics ────────────────────────────────────────────────────

def test_business_cost_all_correct():
    """Coût 0 si toutes les prédictions sont correctes."""
    from src.metrics import compute_business_cost
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_business_cost(y_true, y_pred) == 0


def test_business_cost_fn_weight():
    """FN coûte 10, FP coûte 1."""
    from src.metrics import compute_business_cost, COST_FN, COST_FP
    y_true = np.array([1, 0])
    y_pred = np.array([0, 1])  # 1 FN + 1 FP
    cost = compute_business_cost(y_true, y_pred)
    assert cost == COST_FN + COST_FP


def test_normalized_cost_range():
    """Le coût normalisé est entre 0 et 1."""
    from src.metrics import compute_normalized_cost
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    cost = compute_normalized_cost(y_true, y_pred)
    assert 0.0 <= cost <= 1.0


def test_find_optimal_threshold():
    """Le seuil optimal doit être dans [0, 1]."""
    from src.metrics import find_optimal_threshold
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.4, 0.6])
    threshold, cost, df = find_optimal_threshold(y_true, y_proba)
    assert 0.0 < threshold < 1.0
    assert cost >= 0
    assert len(df) > 0


def test_compute_all_metrics():
    """Les métriques sont calculées correctement."""
    from src.metrics import compute_all_metrics
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
    metrics = compute_all_metrics(y_true, y_proba, threshold=0.5)
    assert "auc_roc" in metrics
    assert "accuracy" in metrics
    assert "business_cost" in metrics
    assert 0 <= metrics["auc_roc"] <= 1
    assert 0 <= metrics["accuracy"] <= 1


# ─── Tests preprocessing ──────────────────────────────────────────────

def test_create_application_features():
    """Feature engineering crée les bonnes colonnes."""
    from src.preprocessing import create_application_features
    df = pd.DataFrame({
        "AMT_INCOME_TOTAL": [100000.0],
        "AMT_CREDIT": [300000.0],
        "AMT_ANNUITY": [15000.0],
        "AMT_GOODS_PRICE": [280000.0],
        "DAYS_BIRTH": [-12000],
        "DAYS_EMPLOYED": [-3000],
        "EXT_SOURCE_1": [0.5],
        "EXT_SOURCE_2": [0.6],
        "EXT_SOURCE_3": [0.55],
        "CODE_GENDER": ["F"],
        "FLAG_OWN_CAR": ["Y"],
        "FLAG_OWN_REALTY": ["Y"],
    })
    result = create_application_features(df)
    assert "CREDIT_INCOME_RATIO" in result.columns
    assert "ANNUITY_INCOME_RATIO" in result.columns
    assert "EXT_SOURCE_MEAN" in result.columns
    assert "AGE_YEARS" in result.columns
    assert result["AGE_YEARS"].iloc[0] > 0
    assert result["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(3.0, abs=0.01)


def test_preprocessor_fit_transform():
    """Le préprocesseur s'entraîne et transforme sans erreur."""
    from src.preprocessing import CreditScoringPreprocessor
    df = pd.DataFrame({
        "A": [1.0, 2.0, np.nan, 4.0],
        "B": ["cat1", "cat2", "cat1", np.nan],
        "C": [10.0, 20.0, 30.0, 40.0],
    })
    prep = CreditScoringPreprocessor()
    prep.fit(df)
    result = prep.transform(df)
    assert result.shape[0] == 4
    assert result.isnull().sum().sum() == 0


def test_preprocessor_handles_unseen_categories():
    """Le préprocesseur gère les catégories inconnues au transform."""
    from src.preprocessing import CreditScoringPreprocessor
    df_train = pd.DataFrame({"cat": ["A", "B", "A", "B"], "num": [1.0, 2.0, 3.0, 4.0]})
    df_test = pd.DataFrame({"cat": ["A", "C", "UNKNOWN"], "num": [1.0, 2.0, 3.0]})
    prep = CreditScoringPreprocessor()
    prep.fit(df_train)
    result = prep.transform(df_test)
    assert result.shape[0] == 3
    assert result.isnull().sum().sum() == 0
