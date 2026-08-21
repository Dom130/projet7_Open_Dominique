"""
Modèles Pydantic pour l'API - Projet7_Open_Dominique
Compatible Pydantic v2
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Requête de prédiction. Envoyer les features dans le dict 'features'."""
    features: Dict[str, Any] = Field(..., description="Features du client (17 minimum)")

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    client_id: Optional[int] = None
    probability: float = Field(..., description="Probabilité de défaut (0-1)")
    prediction: int = Field(..., description="0 = pas de défaut, 1 = défaut")
    decision: str = Field(..., description="ACCEPTED ou REJECTED")
    risk_category: str = Field(..., description="low / medium / high")
    threshold: float = Field(..., description="Seuil de décision utilisé")


class ExplainResponse(BaseModel):
    """Réponse avec explication SHAP."""
    client_id: Optional[int] = None
    probability: float
    prediction: int
    decision: str
    risk_category: str
    threshold: float
    shap_values: Dict[str, float] = Field(default_factory=dict)
    top_features: List[Dict[str, Any]] = Field(default_factory=list)


class BatchPredictionRequest(BaseModel):
    """Requête batch : liste de clients."""
    clients: List[Dict[str, Any]] = Field(..., description="Liste de dicts features")


class BatchPredictionResponse(BaseModel):
    """Réponse batch."""
    predictions: List[PredictionResponse]
    count: int
    accepted: int
    rejected: int


class HealthResponse(BaseModel):
    """Health check."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    threshold: float
    version: str


class ModelInfoResponse(BaseModel):
    """Informations sur le modèle."""
    model_type: str
    version: str
    threshold: float
    cost_fn: int
    cost_fp: int
    n_features: int
    feature_names: List[str]
