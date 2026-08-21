"""
Feature importance globale (LightGBM) et locale (SHAP)
Projet7_Open_Dominique
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')


def compute_global_importance(model, feature_names, top_n=20, save_path=None) -> pd.DataFrame:
    """
    Calcule et visualise l'importance globale des features (gain).
    """
    importance = model.feature_importances_
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Plot top N
    top = df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="steelblue")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {top_n} features les plus importantes")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return df


def compute_shap_values(model, X_transformed: pd.DataFrame, max_display=20, save_path=None):
    """
    Calcule les valeurs SHAP pour explicabilité locale et globale.
    """
    try:
        import shap
    except ImportError:
        print("⚠️ shap non installé. Faites : pip install shap")
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    # Pour classification binaire LightGBM, shap_values est une liste [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_transformed, max_display=max_display, show=False)
    plt.title("SHAP - Importance globale")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return sv, explainer


def explain_single_prediction(explainer, X_single: pd.DataFrame, feature_names=None):
    """
    Explique une prédiction individuelle avec SHAP.
    Retourne un dict {feature: shap_value}.
    """
    try:
        import shap
    except ImportError:
        return {}

    shap_vals = explainer.shap_values(X_single)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    names = feature_names if feature_names is not None else X_single.columns.tolist()
    result = dict(zip(names, sv.tolist()))
    return dict(sorted(result.items(), key=lambda x: abs(x[1]), reverse=True))


def save_feature_importance_csv(df_importance: pd.DataFrame, save_path=None):
    """Sauvegarde le CSV de feature importance."""
    if save_path is None:
        save_path = Path(__file__).resolve().parent.parent / "reports" / "feature_importance.csv"
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True)
    df_importance.to_csv(save_path, index=False)
    print(f"✅ Feature importance sauvegardée : {save_path}")
