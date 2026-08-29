# API Home Credit Scoring

## Objectif

Cette API expose le modèle de scoring crédit afin de calculer :
- la probabilité de défaut d’un client
- la décision binaire (ACCEPTÉ / REFUSÉ) selon un seuil métier optimisé

## Structure du dossier

- `main.py` : endpoints FastAPI (prédiction, batch, explication, infos modèle)
- `models.py` : schémas Pydantic (requêtes / réponses)
- `requirements.txt` : liste des packages utilisés par l’API
- `Dockerfile` : image Docker compatible Render (plan gratuit)

## Artefacts requis

L’API charge automatiquement les artefacts présents dans `models/` à la racine du projet :
- `models/lgbm_model.joblib`
- `models/preprocessor.joblib`
- `models/model_config.json`

## Endpoints principaux

- `/health` : état de l’API et chargement du modèle
- `/predict` : prédiction d’un client (format unique: `features`)
- `/predict/batch` : prédictions multiples
- `/predict/explain` : explication locale des features
- `/model/info` : informations modèle
- `/model/features` : importance des variables
- `/model/feature-names` : liste des features internes (après prétraitement)

## Format de requête

```json
{
  "features": {
    "AMT_INCOME_TOTAL": 150000,
    "AMT_CREDIT": 500000,
    "DAYS_BIRTH": -12000
  }
}
```

**Important** : l’API attend **uniquement** la clé `features`.
En cas de payload invalide, l’API renvoie une **400** (pas de 422).

## Variables attendues pour l’inférence

Le modèle est entraîné sur les **colonnes d’origine Home Credit** (dataset `application_*` + agrégations).  
Le pipeline applique `create_application_features()` (ratios, âge, stats EXT_SOURCE, etc.) et complète les colonnes manquantes par `NaN` puis imputation.
Dans les notebooks (`02_Preprocessing_Features` et `03_Model_Training_MLflow`), l’entraînement utilise `include_supplementary=True`, ce qui ajoute des agrégations issues des tables auxiliaires. À l’inférence, ces colonnes sont **optionnelles** et seront imputées si absentes.

Fonctions ajoutées automatiquement par `create_application_features()` (exemples) :
- `AGE_YEARS`, `EMPLOYED_YEARS`
- `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_GOODS_RATIO`
- `EXT_SOURCE_MEAN`, `EXT_SOURCE_STD`, `EXT_SOURCE_MIN`, `EXT_SOURCE_MAX`
- `DOCUMENTS_COUNT`, `CONTACTS_COUNT`

### ✅ Minimum recommandé (17 features)

Finances : `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`  
Temporel : `DAYS_BIRTH`, `DAYS_EMPLOYED`  
Personnel : `CNT_CHILDREN`, `CODE_GENDER_M`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`  
Scores : `EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `REGION_RATING_CLIENT`  
Ratios : `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `EXT_SOURCE_MEAN`

### ✅ Colonnes optionnelles (si disponibles)

- Catégorielles : `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`, `NAME_FAMILY_STATUS`,  
  `NAME_HOUSING_TYPE`, `NAME_CONTRACT_TYPE`, `OCCUPATION_TYPE`, `ORGANIZATION_TYPE`, etc.
- Flags : `FLAG_*`, `REG_*`, `LIVE_*`, `DEF_*`, `OBS_*`, etc.
- Agrégées : préfixes `BUREAU_`, `PREV_`, `INST_`, `POS_`, `CC_` (si vous les avez déjà agrégées).

➡️ **Le pipeline n’agrège pas les tables auxiliaires au runtime** : si vous voulez utiliser ces colonnes, fournissez-les déjà agrégées.

## Libellés côté dashboard

Le dashboard Streamlit utilise des **libellés explicites** (mapping UI).  
L’API attend **toujours les noms de colonnes d’origine** dans `features`.

## Données et rapports

- Les données sont téléchargées dans l’image Docker au build (pas de Git LFS)
- Le rapport Evidently est servi par `/data/drift` si `reports/` est présent

## Liens utiles

- [README principal](../README.md)
- [Guide Render](../RENDER_SETUP.md)
- [README Dashboard](../streamlit_app/README.md)
- [README MLflow](../mlflow/README.md)
