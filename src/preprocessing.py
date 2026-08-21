"""
Preprocessing module - Home Credit Scoring
Projet7_Open_Dominique
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# Localisation automatique des données
# ─────────────────────────────────────────────

def locate_data_folder() -> Path:
    """
    Cherche automatiquement le dossier contenant les CSV Kaggle.
    Priorité : variable d'env DATA_DIR > dossier data/ dans le projet > Download typique.
    """
    # 1. Variable d'environnement
    env_path = os.getenv("DATA_DIR")
    if env_path and Path(env_path).is_dir():
        return Path(env_path).resolve()

    # 2. Chemins relatifs au fichier courant
    current = Path(__file__).resolve().parent.parent
    candidates = [
        current / "data",
        current.parent / "data",
        Path(r"C:\Users\domir\Downloads\Projet 7 OPEN\Projet+Mise+en+prod+-+home-credit-default-risk"),
        Path("/app/data"),  # Docker
    ]
    for p in candidates:
        if p.is_dir() and (p / "application_train.csv").exists():
            return p.resolve()

    # 3. Recherche récursive depuis le dossier parent
    for directory in [current, *current.parents]:
        results = list(directory.rglob("application_train.csv"))
        if results:
            return results[0].parent.resolve()

    raise FileNotFoundError(
        "Impossible de trouver le dossier des données. "
        "Définissez la variable d'environnement DATA_DIR ou placez les CSV dans data/."
    )


# ─────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────

def load_application_data(data_dir: Path = None):
    """Charge application_train.csv et application_test.csv."""
    if data_dir is None:
        data_dir = locate_data_folder()
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "application_train.csv")
    test = pd.read_csv(data_dir / "application_test.csv")
    print(f"✅ Train: {train.shape}, Test: {test.shape}")
    return train, test


def load_bureau_data(data_dir: Path = None):
    """Charge et agrège bureau.csv + bureau_balance.csv."""
    if data_dir is None:
        data_dir = locate_data_folder()
    data_dir = Path(data_dir)

    bureau = pd.read_csv(data_dir / "bureau.csv")
    bureau_balance = pd.read_csv(data_dir / "bureau_balance.csv")

    # Agrégation bureau_balance → bureau
    bb_agg = bureau_balance.groupby("SK_ID_BUREAU")["STATUS"].agg(
        BUREAU_BALANCE_COUNT="count",
        BUREAU_BALANCE_ACTIVE=lambda x: (x == "C").sum(),
    ).reset_index()

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")

    # Agrégation bureau → application
    num_cols = bureau.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "SK_ID_CURR"]

    agg_dict = {c: ["mean", "sum", "max"] for c in num_cols}
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_agg.columns = ["BUREAU_" + "_".join(c).upper() for c in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()

    print(f"✅ Bureau agrégé: {bureau_agg.shape}")
    return bureau_agg


def load_previous_applications(data_dir: Path = None):
    """Charge et agrège previous_application.csv."""
    if data_dir is None:
        data_dir = locate_data_folder()
    data_dir = Path(data_dir)

    prev = pd.read_csv(data_dir / "previous_application.csv")
    num_cols = prev.select_dtypes(include=np.number).columns.tolist()
    num_cols = [c for c in num_cols if c != "SK_ID_CURR"]

    agg_dict = {c: ["mean", "max"] for c in num_cols[:20]}  # top 20 pour limiter
    prev_agg = prev.groupby("SK_ID_CURR").agg(agg_dict)
    prev_agg.columns = ["PREV_" + "_".join(c).upper() for c in prev_agg.columns]
    prev_agg = prev_agg.reset_index()

    print(f"✅ Previous agrégé: {prev_agg.shape}")
    return prev_agg


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def create_application_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les features dérivées des colonnes application."""
    df = df.copy()

    # Ratios financiers
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / (df["AMT_CREDIT"] + 1)
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 1)

    # Âge et emploi
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_YEARS"] = df["DAYS_EMPLOYED"].apply(
        lambda x: -x / 365 if x < 0 else 0
    )

    # Score externe moyen
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    available = [c for c in ext_cols if c in df.columns]
    if available:
        df["EXT_SOURCE_MEAN"] = df[available].mean(axis=1)
        df["EXT_SOURCE_STD"] = df[available].std(axis=1).fillna(0)

    # Encodage genre
    if "CODE_GENDER" in df.columns:
        df["CODE_GENDER_M"] = (df["CODE_GENDER"] == "M").astype(int)

    # Flags numériques
    for col in ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col] == "Y").astype(int)

    return df


def build_full_dataset(data_dir: Path = None, use_auxiliary: bool = True) -> tuple:
    """Construit les datasets complets train/test avec toutes les features."""
    if data_dir is None:
        data_dir = locate_data_folder()

    train, test = load_application_data(data_dir)

    if use_auxiliary:
        try:
            bureau_agg = load_bureau_data(data_dir)
            train = train.merge(bureau_agg, on="SK_ID_CURR", how="left")
            test = test.merge(bureau_agg, on="SK_ID_CURR", how="left")
            print("✅ Bureau joint")
        except Exception as e:
            print(f"⚠️ Bureau ignoré: {e}")

        try:
            prev_agg = load_previous_applications(data_dir)
            train = train.merge(prev_agg, on="SK_ID_CURR", how="left")
            test = test.merge(prev_agg, on="SK_ID_CURR", how="left")
            print("✅ Previous joint")
        except Exception as e:
            print(f"⚠️ Previous ignoré: {e}")

    train = create_application_features(train)
    test = create_application_features(test)

    print(f"✅ Dataset final - Train: {train.shape}, Test: {test.shape}")
    return train, test


def prepare_train_test_data(data_dir: Path = None, use_auxiliary: bool = True):
    """Retourne X_train, X_test, y_train avec séparation TARGET."""
    train, test = build_full_dataset(data_dir, use_auxiliary)

    y_train = train["TARGET"]
    X_train = train.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
    X_test = test.drop(columns=["SK_ID_CURR"], errors="ignore")

    return X_train, X_test, y_train


# ─────────────────────────────────────────────
# Preprocessor sklearn-compatible
# ─────────────────────────────────────────────

class CreditScoringPreprocessor(BaseEstimator, TransformerMixin):
    """
    Pipeline de prétraitement complet :
    - Encodage des catégorielles (LabelEncoder)
    - Imputation des valeurs manquantes (médiane)
    """

    def __init__(self):
        self.label_encoders_ = {}
        self.imputer_ = None
        self.feature_names_ = None
        self.cat_cols_ = []
        self.num_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        # Colonnes catégorielles
        self.cat_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        # Encodage
        for col in self.cat_cols_:
            le = LabelEncoder()
            X[col] = X[col].astype(str).fillna("__NAN__")
            le.fit(X[col])
            self.label_encoders_[col] = le

        # Imputation numérique
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(X[self.num_cols_])

        self.feature_names_ = self.num_cols_ + self.cat_cols_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Encodage
        for col in self.cat_cols_:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("__NAN__")
                le = self.label_encoders_[col]
                known = set(le.classes_)
                X[col] = X[col].apply(lambda v: v if v in known else "__NAN__")
                if "__NAN__" not in known:
                    le.classes_ = np.append(le.classes_, "__NAN__")
                X[col] = le.transform(X[col])

        # Imputation numérique
        num_present = [c for c in self.num_cols_ if c in X.columns]
        X[num_present] = self.imputer_.transform(X[num_present])

        return X

    def get_feature_names_out(self):
        return self.feature_names_
