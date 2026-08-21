# 🏦 Projet 7 OpenClassrooms — Home Credit Scoring
**Projet7_Open_Dominique**

[![CI/CD](https://github.com/Dom130/projet7_Open_Dominique/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Dom130/projet7_Open_Dominique/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Description

Projet complet de **scoring de crédit** basé sur le dataset [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) de Kaggle.

### 🎯 Objectif métier
Prédire la **probabilité de défaut de paiement** d'un client, avec une décision optimisée par un coût métier asymétrique :
- **Faux Négatif (FN)** : accepter un client qui fera défaut → **Coût 10**
- **Faux Positif (FP)** : refuser un bon client → **Coût 1**

## 🏗️ Architecture

```
projet7_Open_Dominique/
├── 📁 api/                  # API FastAPI de prédiction
│   ├── main.py              # Endpoints /predict, /explain, /health
│   ├── models.py            # Schémas Pydantic v2
│   ├── requirements.txt
│   └── Dockerfile
├── 📁 streamlit_app/        # Dashboard interactif
│   ├── app.py               # 3 pages: Scoring, Drift, Documentation
│   ├── requirements.txt
│   └── Dockerfile
├── 📁 mlflow/               # Suivi des expériences ML
│   └── Dockerfile
├── 📁 src/                  # Code source Python
│   ├── preprocessing.py     # Chargement, feature engineering, pipeline
│   ├── train.py             # Entraînement LightGBM + MLflow
│   ├── metrics.py           # Métriques métier (coût FN/FP)
│   └── feature_importance.py # SHAP global et local
├── 📁 notebooks/            # Notebooks Jupyter
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Training_MLflow.ipynb
│   └── 04_Drift_Evidently.ipynb
├── 📁 models/               # Artefacts modèle (model.pkl, preprocessor.pkl)
├── 📁 reports/              # Rapports Evidently, figures, métriques
├── 📁 tests/                # Tests unitaires pytest
├── .github/workflows/       # CI/CD GitHub Actions
├── render.yaml              # Déploiement Render (3 services)
├── run.py                   # Lancement local
├── environment.yml          # Environnement conda
└── pyproject.toml           # Config projet
```

## 🚀 Installation locale

### 1. Cloner le repo
```bash
git clone https://github.com/Dom130/projet7_Open_Dominique.git
cd projet7_Open_Dominique
```

### 2. Créer l'environnement
```bash
# Avec conda (recommandé)
conda env create -f environment.yml
conda activate projet7-dominique

# Ou avec pip
pip install -r api/requirements.txt
```

### 3. Données
Placer les fichiers CSV Kaggle dans `data/` :
```
data/
├── application_train.csv
├── application_test.csv
├── bureau.csv
├── bureau_balance.csv
├── previous_application.csv
├── POS_CASH_balance.csv
├── credit_card_balance.csv
└── installments_payments.csv
```
Ou définir la variable `DATA_DIR` pointant vers ton dossier de données :
```bash
set DATA_DIR=C:\Users\domir\Downloads\Projet 7 OPEN\Projet+Mise+en+prod+-+home-credit-default-risk
```

### 4. Entraîner le modèle (requis avant de lancer l'API)
```bash
python run.py train
```

### 5. Lancer les services
```bash
# Tout en même temps
python run.py all

# Ou séparément
python run.py api        # http://localhost:8000
python run.py dashboard  # http://localhost:8501
python run.py mlflow     # http://localhost:5002
```

## 📊 Résultats du modèle

| Métrique | Valeur |
|---|---|
| AUC-ROC | ~0.76 |
| Seuil optimal | ~0.44 |
| Accuracy | ~0.70 |
| Coût métier | Minimisé |

## 🌐 API — Endpoints

| Endpoint | Méthode | Description |
|---|---|---|
| `/` | GET | Page d'accueil |
| `/health` | GET | Health check |
| `/predict` | POST | Prédiction unique |
| `/predict/batch` | POST | Prédictions en lot |
| `/predict/explain` | POST | Prédiction + SHAP |
| `/model/info` | GET | Infos modèle |

### Exemple de requête
```json
POST /predict
{
  "features": {
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "AMT_ANNUITY": 25000,
    "AMT_GOODS_PRICE": 450000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -3000,
    "CNT_CHILDREN": 1,
    "CODE_GENDER_M": 0,
    "FLAG_OWN_CAR": 1,
    "FLAG_OWN_REALTY": 1,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.55,
    "REGION_RATING_CLIENT": 2,
    "CREDIT_INCOME_RATIO": 3.33,
    "ANNUITY_INCOME_RATIO": 0.167,
    "EXT_SOURCE_MEAN": 0.55
  }
}
```

### Exemple de réponse
```json
{
  "probability": 0.23,
  "prediction": 0,
  "decision": "ACCEPTED",
  "risk_category": "low",
  "threshold": 0.44
}
```

## 🧪 Tests
```bash
pytest tests/ -v
```

## ☁️ Déploiement Render

Le fichier `render.yaml` configure **3 services** déployés automatiquement :

| Service | Nom | Port |
|---|---|---|
| API FastAPI | projet7-dominique-api | 8000 |
| Dashboard Streamlit | projet7-dominique-dashboard | 8501 |
| MLflow UI | projet7-dominique-mlflow | 5000 |

**Étapes :**
1. Pusher sur GitHub
2. Sur [render.com](https://render.com) → New → Blueprint → connecter le repo
3. Render déploie automatiquement les 3 services
4. Mettre à jour `API_URL` dans les variables d'env du dashboard

## 📓 Notebooks

| Notebook | Contenu |
|---|---|
| `01_EDA.ipynb` | Exploration, variable cible, distributions |
| `02_Preprocessing.ipynb` | Feature engineering, encodage, normalisation |
| `03_Training_MLflow.ipynb` | Entraînement LightGBM, optimisation seuil |
| `04_Drift_Evidently.ipynb` | Détection du data drift |

## 📄 Licence
MIT — voir [LICENSE](LICENSE)

---
*Réalisé dans le cadre du Projet 7 OpenClassrooms "Implémentez un modèle de scoring"*
