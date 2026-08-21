"""
Dashboard Streamlit - Home Credit Scoring
Projet7_Open_Dominique

Interface pour les chargés de clientèle :
- Scoring d'un client
- Explication SHAP locale
- Comparaison avec d'autres clients
- Monitoring du drift (Evidently)
"""
import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 Scoring Crédit - Projet7 Dominique",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5002")

ROOT = Path(__file__).resolve().parent.parent

# ─── Libellés explicites des features ─────────────────────────────────
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu annuel total (€)",
    "AMT_CREDIT": "Montant du crédit demandé (€)",
    "AMT_ANNUITY": "Annuité mensuelle (€)",
    "AMT_GOODS_PRICE": "Prix du bien financé (€)",
    "DAYS_BIRTH": "Âge (jours négatifs)",
    "DAYS_EMPLOYED": "Ancienneté emploi (jours négatifs)",
    "CNT_CHILDREN": "Nombre d'enfants",
    "CODE_GENDER_M": "Genre masculin (1=Oui, 0=Non)",
    "FLAG_OWN_CAR": "Possède une voiture (1=Oui)",
    "FLAG_OWN_REALTY": "Possède un bien immobilier (1=Oui)",
    "EXT_SOURCE_1": "Score externe 1 (0-1)",
    "EXT_SOURCE_2": "Score externe 2 (0-1)",
    "EXT_SOURCE_3": "Score externe 3 (0-1)",
    "REGION_RATING_CLIENT": "Notation de la région (1-3)",
    "CREDIT_INCOME_RATIO": "Ratio crédit/revenu",
    "ANNUITY_INCOME_RATIO": "Ratio annuité/revenu",
    "EXT_SOURCE_MEAN": "Moyenne scores externes",
}


def label(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


# ─── Helpers API ──────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def predict(features: dict) -> dict:
    r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=15)
    r.raise_for_status()
    return r.json()


def predict_explain(features: dict) -> dict:
    r = requests.post(f"{API_URL}/predict/explain", json={"features": features}, timeout=30)
    r.raise_for_status()
    return r.json()


# ─── Gauge de probabilité ─────────────────────────────────────────────

def draw_gauge(probability: float, threshold: float):
    """Dessine une jauge de risque."""
    fig, ax = plt.subplots(figsize=(5, 2.5), subplot_kw=dict(polar=False))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Fond coloré
    ax.barh(0.5, threshold, left=0, height=0.4, color="#2ecc71", alpha=0.7)
    ax.barh(0.5, 1 - threshold, left=threshold, height=0.4, color="#e74c3c", alpha=0.7)

    # Flèche probabilité
    ax.annotate("", xy=(probability, 0.5), xytext=(probability, 0.95),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # Texte
    color = "#e74c3c" if probability >= threshold else "#27ae60"
    ax.text(probability, 0.1, f"{probability:.1%}", ha="center", fontsize=13,
            fontweight="bold", color=color)
    ax.text(threshold, 0.97, f"Seuil {threshold:.0%}", ha="center", fontsize=8, color="gray")

    ax.text(0.02, 0.5, "✅ Accepté", va="center", fontsize=9, color="#27ae60")
    ax.text(0.98, 0.5, "❌ Refusé", va="center", ha="right", fontsize=9, color="#e74c3c")

    return fig


# ─── Sidebar : formulaire client ──────────────────────────────────────

def sidebar_form() -> dict:
    st.sidebar.header("📋 Informations client")

    with st.sidebar.expander("💰 Finances", expanded=True):
        amt_income = st.number_input("Revenu annuel (€)", 10000, 10000000, 150000, step=5000)
        amt_credit = st.number_input("Montant crédit (€)", 10000, 5000000, 500000, step=10000)
        amt_annuity = st.number_input("Annuité mensuelle (€)", 1000, 200000, 25000, step=500)
        amt_goods = st.number_input("Prix du bien (€)", 10000, 5000000, 450000, step=10000)

    with st.sidebar.expander("👤 Personnel", expanded=True):
        age_years = st.slider("Âge (années)", 18, 75, 35)
        employed_years = st.slider("Ancienneté emploi (années)", 0, 40, 5)
        cnt_children = st.number_input("Nombre d'enfants", 0, 10, 0)
        gender_m = st.radio("Genre", ["Femme", "Homme"]) == "Homme"
        own_car = st.checkbox("Possède une voiture")
        own_realty = st.checkbox("Possède un bien immobilier")

    with st.sidebar.expander("📊 Scores externes", expanded=True):
        ext1 = st.slider("Score externe 1", 0.0, 1.0, 0.50, 0.01)
        ext2 = st.slider("Score externe 2", 0.0, 1.0, 0.55, 0.01)
        ext3 = st.slider("Score externe 3", 0.0, 1.0, 0.50, 0.01)
        region = st.selectbox("Notation région", [1, 2, 3], index=1)

    # Calcul des ratios
    credit_income = amt_credit / (amt_income + 1)
    annuity_income = amt_annuity / (amt_income + 1)
    ext_mean = np.mean([ext1, ext2, ext3])

    return {
        "AMT_INCOME_TOTAL": amt_income,
        "AMT_CREDIT": amt_credit,
        "AMT_ANNUITY": amt_annuity,
        "AMT_GOODS_PRICE": amt_goods,
        "DAYS_BIRTH": int(-age_years * 365),
        "DAYS_EMPLOYED": int(-employed_years * 365),
        "CNT_CHILDREN": cnt_children,
        "CODE_GENDER_M": int(gender_m),
        "FLAG_OWN_CAR": int(own_car),
        "FLAG_OWN_REALTY": int(own_realty),
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "REGION_RATING_CLIENT": region,
        "CREDIT_INCOME_RATIO": round(credit_income, 4),
        "ANNUITY_INCOME_RATIO": round(annuity_income, 4),
        "EXT_SOURCE_MEAN": round(ext_mean, 4),
    }


# ─── Page : Scoring ───────────────────────────────────────────────────

def page_scoring(features: dict):
    st.title("🏦 Scoring de Crédit")
    st.caption("Projet7_Open_Dominique — Home Credit Default Risk")

    # Health check
    health = check_api_health()
    if health is None:
        st.error(f"❌ API non accessible sur {API_URL}")
        st.info("Vérifiez que l'API est démarrée : `python run.py api`")
        return

    if health.get("model_loaded"):
        st.success(f"✅ API connectée | Seuil: {health['threshold']:.2f} | v{health['version']}")
    else:
        st.warning("⚠️ API en mode dégradé — modèle non chargé (entraîner d'abord le modèle)")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Résumé client")
        df_client = pd.DataFrame([
            {"Paramètre": label(k), "Valeur": v}
            for k, v in features.items()
        ])
        st.dataframe(df_client, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🎯 Décision de crédit")
        if st.button("🔍 Analyser ce client", type="primary", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                try:
                    result = predict_explain(features)

                    prob = result["probability"]
                    decision = result["decision"]
                    risk = result["risk_category"]
                    threshold = result["threshold"]

                    # Décision
                    if decision == "ACCEPTED":
                        st.success(f"✅ **CRÉDIT ACCORDÉ**")
                    else:
                        st.error(f"❌ **CRÉDIT REFUSÉ**")

                    # Métriques
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Probabilité défaut", f"{prob:.1%}")
                    m2.metric("Seuil", f"{threshold:.0%}")
                    risk_icons = {"low": "🟢 Faible", "medium": "🟡 Moyen", "high": "🔴 Élevé"}
                    m3.metric("Niveau de risque", risk_icons.get(risk, risk))

                    # Jauge
                    st.pyplot(draw_gauge(prob, threshold))

                    # SHAP
                    if result.get("top_features"):
                        st.subheader("🔍 Facteurs les plus influents")
                        top = result["top_features"][:8]
                        df_shap = pd.DataFrame(top)
                        df_shap["feature"] = df_shap["feature"].apply(label)
                        df_shap["couleur"] = df_shap["shap_value"].apply(
                            lambda x: "#e74c3c" if x > 0 else "#2ecc71"
                        )
                        fig_shap, ax = plt.subplots(figsize=(7, 4))
                        colors = ["#e74c3c" if x > 0 else "#2ecc71"
                                  for x in df_shap["shap_value"]]
                        ax.barh(df_shap["feature"][::-1], df_shap["shap_value"][::-1],
                                color=colors[::-1])
                        ax.axvline(0, color="black", linewidth=0.8)
                        ax.set_xlabel("Impact SHAP (+ = risque, - = sécurité)")
                        ax.set_title("Explication locale de la décision")
                        ax.grid(axis="x", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig_shap)

                        st.info(
                            "🔴 Barres rouges = facteurs qui **augmentent** le risque\n\n"
                            "🟢 Barres vertes = facteurs qui **diminuent** le risque"
                        )

                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de joindre l'API.")
                except Exception as e:
                    st.error(f"Erreur: {e}")


# ─── Page : Drift ─────────────────────────────────────────────────────

def page_drift():
    st.title("📉 Monitoring du Data Drift")
    st.caption("Rapport Evidently — Détection de dérive des données")

    report_path = ROOT / "reports" / "evidently_full_report.html"

    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    else:
        st.warning("📊 Rapport Evidently non disponible.")
        st.info(
            "Pour générer le rapport :\n"
            "1. Ouvrir le notebook `notebooks/04_Drift_Evidently.ipynb`\n"
            "2. Exécuter toutes les cellules\n"
            "3. Le rapport sera créé dans `reports/evidently_full_report.html`"
        )


# ─── Page : Documentation ─────────────────────────────────────────────

def page_documentation():
    st.title("📚 Documentation")

    st.header("🎯 Objectif métier")
    st.write("""
    Prédire la probabilité de **défaut de paiement** d'un client demandant un crédit.

    **Coût métier :**
    - 🔴 **Faux Négatif** (FN) = Accepter un mauvais client → **Coût 10**
    - 🟡 **Faux Positif** (FP) = Refuser un bon client → **Coût 1**

    Le seuil de décision est optimisé pour minimiser ce coût asymétrique.
    """)

    st.header("🔧 Architecture")
    st.code("""
    projet7_Open_Dominique/
    ├── api/            # API FastAPI (prédictions)
    ├── streamlit_app/  # Ce dashboard
    ├── mlflow/         # Suivi des expériences
    ├── src/            # Code source ML
    ├── notebooks/      # EDA, Prétraitement, Entraînement, Drift
    ├── models/         # Modèle entraîné + artefacts
    └── reports/        # Rapports Evidently + figures
    """, language="")

    st.header("📡 Endpoints API")
    endpoints = {
        "GET /health": "Vérification de l'état de l'API",
        "POST /predict": "Prédiction unique",
        "POST /predict/batch": "Prédictions en lot",
        "POST /predict/explain": "Prédiction + explication SHAP",
        "GET /model/info": "Informations sur le modèle",
    }
    for ep, desc in endpoints.items():
        st.markdown(f"- **`{ep}`** : {desc}")

    st.header("📊 Features minimales")
    features_min = list(FEATURE_LABELS.items())
    df = pd.DataFrame(features_min, columns=["Feature technique", "Description"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.header("🔗 Liens utiles")
    col1, col2, col3 = st.columns(3)
    col1.link_button("📖 Documentation API", f"{API_URL}/docs")
    col2.link_button("📈 MLflow UI", MLFLOW_URL)
    col3.link_button("💻 GitHub", "https://github.com/Dom130/projet7_Open_Dominique")


# ─── Navigation principale ────────────────────────────────────────────

def main():
    features = sidebar_form()

    page = st.sidebar.radio(
        "Navigation",
        ["🎯 Scoring Client", "📉 Data Drift", "📚 Documentation"],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.caption(f"API: {API_URL}")
    st.sidebar.caption(f"MLflow: {MLFLOW_URL}")

    if page == "🎯 Scoring Client":
        page_scoring(features)
    elif page == "📉 Data Drift":
        page_drift()
    elif page == "📚 Documentation":
        page_documentation()


if __name__ == "__main__":
    main()
