"""
API FastAPI - Home Credit Scoring
Projet7_Open_Dominique

Endpoints:
  GET  /            → Page d'accueil
  GET  /health      → Health check
  POST /predict     → Prédiction unique
  POST /predict/batch  → Prédictions batch
  POST /predict/explain → Prédiction + SHAP
  GET  /model/info  → Infos modèle
  GET  /model/features → Liste des features
"""
import json
import os
import pickle
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ajouter le dossier racine au PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import create_application_features, CreditScoringPreprocessor
from api.models import (
    PredictionRequest, PredictionResponse,
    ExplainResponse, BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)

# ─── Logging ───────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Application FastAPI ───────────────────
app = FastAPI(
    title="🏦 Home Credit Scoring API",
    description="API de scoring de crédit - Projet7_Open_Dominique",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Chargement des artefacts ──────────────
MODEL = None
PREPROCESSOR = None
CONFIG = None
SHAP_EXPLAINER = None


def get_models_dir() -> Path:
    """Cherche le dossier models/ (local ou Docker)."""
    candidates = [
        Path("/app/models"),
        ROOT / "models",
        Path(os.getenv("MODELS_DIR", "")),
    ]
    for p in candidates:
        if p.is_dir() and (p / "model.pkl").exists():
            return p
    return ROOT / "models"


def load_artifacts():
    """Charge le modèle, le préprocesseur et la config au démarrage."""
    global MODEL, PREPROCESSOR, CONFIG

    models_dir = get_models_dir()
    logger.info(f"Chargement des artefacts depuis {models_dir}")

    try:
        with open(models_dir / "model.pkl", "rb") as f:
            MODEL = pickle.load(f)
        logger.info("✅ Modèle chargé")
    except FileNotFoundError:
        logger.warning("⚠️ model.pkl non trouvé — mode dégradé (entraîner d'abord le modèle)")

    try:
        with open(models_dir / "preprocessor.pkl", "rb") as f:
            PREPROCESSOR = pickle.load(f)
        logger.info("✅ Préprocesseur chargé")
    except FileNotFoundError:
        logger.warning("⚠️ preprocessor.pkl non trouvé")

    try:
        with open(models_dir / "model_config.json", "r") as f:
            CONFIG = json.load(f)
        logger.info(f"✅ Config chargée — seuil: {CONFIG.get('threshold', 0.44)}")
    except FileNotFoundError:
        CONFIG = {
            "threshold": 0.44,
            "model_type": "LightGBM",
            "feature_names": [],
            "cost_fn": 10,
            "cost_fp": 1,
            "version": "1.0.0",
        }
        logger.warning("⚠️ model_config.json non trouvé — config par défaut")


@app.on_event("startup")
async def startup_event():
    load_artifacts()


# ─── Helpers ───────────────────────────────

def get_threshold() -> float:
    return CONFIG.get("threshold", 0.44) if CONFIG else 0.44


def classify_risk(prob: float, threshold: float) -> str:
    if prob < threshold * 0.5:
        return "low"
    elif prob < threshold:
        return "medium"
    else:
        return "high"


def prepare_features(features_dict: Dict[str, Any]) -> pd.DataFrame:
    """Transforme le dict features en DataFrame prêt pour le modèle."""
    df = pd.DataFrame([features_dict])
    df = create_application_features(df)

    if PREPROCESSOR is not None:
        # Aligner avec les colonnes attendues
        expected_cols = CONFIG.get("feature_names", []) if CONFIG else []
        if expected_cols:
            for col in expected_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[expected_cols]
        df = PREPROCESSOR.transform(df)
    return df


# ─── Endpoints ─────────────────────────────

@app.get("/", summary="Page d'accueil")
def root():
    return {
        "message": "🏦 Home Credit Scoring API - Projet7_Open_Dominique",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    return HealthResponse(
        status="ok" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        preprocessor_loaded=PREPROCESSOR is not None,
        threshold=get_threshold(),
        version=CONFIG.get("version", "1.0.0") if CONFIG else "1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, summary="Prédiction unique")
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Entraîner le modèle d'abord (notebook 03)."
        )

    try:
        X = prepare_features(request.features)
        prob = float(MODEL.predict_proba(X)[0, 1])
        threshold = get_threshold()
        pred = int(prob >= threshold)
        decision = "REJECTED" if pred == 1 else "ACCEPTED"
        risk = classify_risk(prob, threshold)

        client_id = request.features.get("SK_ID_CURR")

        return PredictionResponse(
            client_id=int(client_id) if client_id is not None else None,
            probability=round(prob, 4),
            prediction=pred,
            decision=decision,
            risk_category=risk,
            threshold=threshold,
        )
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, summary="Prédictions batch")
def predict_batch(request: BatchPredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    predictions = []
    for client_dict in request.clients:
        try:
            X = prepare_features(client_dict)
            prob = float(MODEL.predict_proba(X)[0, 1])
            threshold = get_threshold()
            pred = int(prob >= threshold)
            decision = "REJECTED" if pred == 1 else "ACCEPTED"
            risk = classify_risk(prob, threshold)
            client_id = client_dict.get("SK_ID_CURR")

            predictions.append(PredictionResponse(
                client_id=int(client_id) if client_id is not None else None,
                probability=round(prob, 4),
                prediction=pred,
                decision=decision,
                risk_category=risk,
                threshold=threshold,
            ))
        except Exception as e:
            logger.warning(f"Client ignoré (erreur): {e}")

    accepted = sum(1 for p in predictions if p.decision == "ACCEPTED")
    rejected = len(predictions) - accepted

    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        accepted=accepted,
        rejected=rejected,
    )


@app.post("/predict/explain", response_model=ExplainResponse, summary="Prédiction + SHAP")
def predict_explain(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    try:
        X = prepare_features(request.features)
        prob = float(MODEL.predict_proba(X)[0, 1])
        threshold = get_threshold()
        pred = int(prob >= threshold)
        decision = "REJECTED" if pred == 1 else "ACCEPTED"
        risk = classify_risk(prob, threshold)
        client_id = request.features.get("SK_ID_CURR")

        # SHAP
        shap_values = {}
        top_features = []
        try:
            import shap
            global SHAP_EXPLAINER
            if SHAP_EXPLAINER is None:
                SHAP_EXPLAINER = shap.TreeExplainer(MODEL)

            sv = SHAP_EXPLAINER.shap_values(X)
            if isinstance(sv, list):
                sv = sv[1]

            feature_names = X.columns.tolist() if hasattr(X, 'columns') else []
            shap_values = {
                name: round(float(val), 6)
                for name, val in zip(feature_names, sv[0])
            }
            # Top 10
            top_features = sorted(
                [{"feature": k, "shap_value": v, "impact": "positive" if v > 0 else "negative"}
                 for k, v in shap_values.items()],
                key=lambda x: abs(x["shap_value"]),
                reverse=True
            )[:10]
        except Exception as shap_err:
            logger.warning(f"SHAP non disponible: {shap_err}")

        return ExplainResponse(
            client_id=int(client_id) if client_id is not None else None,
            probability=round(prob, 4),
            prediction=pred,
            decision=decision,
            risk_category=risk,
            threshold=threshold,
            shap_values=shap_values,
            top_features=top_features,
        )
    except Exception as e:
        logger.error(f"Erreur explain: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/model/info", response_model=ModelInfoResponse, summary="Informations modèle")
def model_info():
    if CONFIG is None:
        raise HTTPException(status_code=503, detail="Config non disponible.")
    features = CONFIG.get("feature_names", [])
    return ModelInfoResponse(
        model_type=CONFIG.get("model_type", "LightGBM"),
        version=CONFIG.get("version", "1.0.0"),
        threshold=CONFIG.get("threshold", 0.44),
        cost_fn=CONFIG.get("cost_fn", 10),
        cost_fp=CONFIG.get("cost_fp", 1),
        n_features=len(features),
        feature_names=features,
    )


@app.get("/model/features", summary="Liste des features")
def model_features():
    if CONFIG is None:
        return {"features": [], "count": 0}
    features = CONFIG.get("feature_names", [])
    return {"features": features, "count": len(features)}
