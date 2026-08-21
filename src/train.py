"""
Module d'entraînement LightGBM + MLflow
Projet7_Open_Dominique
"""
import json
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.metrics import (
    COST_FN, COST_FP,
    compute_all_metrics,
    find_optimal_threshold,
    generate_metrics_report,
)
from src.preprocessing import CreditScoringPreprocessor

warnings.filterwarnings('ignore')

EXPERIMENT_NAME = "home-credit-scoring-dominique"
MODEL_NAME = "credit-scoring-lgbm-dominique"


def setup_mlflow(tracking_uri: str = None, experiment_name: str = None):
    """Configure MLflow."""
    if tracking_uri is None:
        # Chemin par défaut : mlruns/ à la racine du projet
        root = Path(__file__).resolve().parent.parent
        tracking_uri = str(root / "mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    exp_name = experiment_name or EXPERIMENT_NAME
    mlflow.set_experiment(exp_name)
    exp = mlflow.get_experiment_by_name(exp_name)
    print(f"✅ MLflow configuré:")
    print(f"   - Tracking URI: {tracking_uri}")
    print(f"   - Experiment: {exp_name}")
    if exp:
        print(f"   - Experiment ID: {exp.experiment_id}")
    return exp_name


def get_default_lgb_params() -> dict:
    """Paramètres LightGBM par défaut optimisés pour le scoring crédit."""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_estimators": 500,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, params=None):
    """Entraîne un modèle LightGBM."""
    if params is None:
        params = get_default_lgb_params()

    model = lgb.LGBMClassifier(**params)

    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_with_mlflow(X_train, y_train, params=None, test_size=0.2, run_name=None):
    """
    Entraîne avec tracking MLflow complet.
    Retourne (model, preprocessor, threshold, metrics_dict).
    """
    if params is None:
        params = get_default_lgb_params()

    # Split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=test_size, stratify=y_train, random_state=42
    )

    with mlflow.start_run(run_name=run_name or "lgbm_run"):
        # Prétraitement
        preprocessor = CreditScoringPreprocessor()
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)

        # Entraînement
        model = train_lightgbm(X_tr_proc, y_tr, X_val_proc, y_val, params)

        # Prédictions
        y_proba = model.predict_proba(X_val_proc)[:, 1]

        # Seuil optimal
        best_threshold, best_cost, _ = find_optimal_threshold(y_val, y_proba)

        # Métriques
        metrics = compute_all_metrics(y_val, y_proba, threshold=best_threshold)

        # Log MLflow
        mlflow.log_params(params)
        mlflow.log_param("threshold", best_threshold)
        mlflow.log_metric("auc_roc", metrics["auc_roc"])
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("business_cost", metrics["business_cost"])
        mlflow.log_metric("normalized_cost", metrics["normalized_cost"])

        # Log modèle
        mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name=MODEL_NAME)

        run_id = mlflow.active_run().info.run_id
        print(f"✅ Run MLflow: {run_id}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   Seuil optimal: {best_threshold:.4f}")
        print(f"   Coût métier: {metrics['business_cost']:.0f}")

    return model, preprocessor, best_threshold, metrics


def save_artifacts(model, preprocessor, threshold, feature_names,
                   models_dir: Path = None):
    """Sauvegarde les artefacts dans le dossier models/."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    # Modèle
    model_path = models_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Préprocesseur
    prep_path = models_dir / "preprocessor.pkl"
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)

    # Config JSON
    config = {
        "threshold": float(threshold),
        "model_type": "LightGBM",
        "feature_names": list(feature_names),
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "version": "1.0.0",
    }
    config_path = models_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Artefacts sauvegardés dans {models_dir}")
    print(f"   - model.pkl")
    print(f"   - preprocessor.pkl")
    print(f"   - model_config.json")
    return model_path, prep_path, config_path
