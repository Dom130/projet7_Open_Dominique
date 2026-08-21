"""
Métriques métier - Home Credit Scoring
Coût FN=10 (accepter un mauvais payeur), FP=1 (refuser un bon client)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix,
    classification_report, roc_curve
)

# ─── Paramètres métier ───────────────────────
COST_FN = 10   # Faux Négatif : accepter un client qui fera défaut
COST_FP = 1    # Faux Positif : refuser un bon client


def compute_business_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    """Calcule le coût métier total."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return fn * cost_fn + fp * cost_fp


def compute_normalized_cost(y_true, y_pred, cost_fn=COST_FN, cost_fp=COST_FP) -> float:
    """Coût métier normalisé (entre 0 et 1)."""
    cost = compute_business_cost(y_true, y_pred, cost_fn, cost_fp)
    max_cost = len(y_true) * max(cost_fn, cost_fp)
    return cost / max_cost if max_cost > 0 else 0.0


def find_optimal_threshold(y_true, y_proba, cost_fn=COST_FN, cost_fp=COST_FP) -> tuple:
    """
    Trouve le seuil optimal minimisant le coût métier.
    Retourne (seuil_optimal, coût_min, df_résultats).
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = compute_business_cost(y_true, y_pred, cost_fn, cost_fp)
        norm_cost = compute_normalized_cost(y_true, y_pred, cost_fn, cost_fp)
        auc = roc_auc_score(y_true, y_proba)
        acc = accuracy_score(y_true, y_pred)
        results.append({
            "threshold": t,
            "business_cost": cost,
            "normalized_cost": norm_cost,
            "auc_roc": auc,
            "accuracy": acc,
        })

    df = pd.DataFrame(results)
    best_idx = df["business_cost"].idxmin()
    best_threshold = df.loc[best_idx, "threshold"]
    best_cost = df.loc[best_idx, "business_cost"]

    return float(best_threshold), float(best_cost), df


def compute_all_metrics(y_true, y_proba, threshold=0.44) -> dict:
    """Calcule toutes les métriques pour un seuil donné."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc_roc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": threshold,
        "business_cost": compute_business_cost(y_true, y_pred),
        "normalized_cost": compute_normalized_cost(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
    }


def plot_threshold_optimization(df_results, best_threshold=None, save_path=None):
    """Courbe coût métier vs seuil."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df_results["threshold"], df_results["business_cost"], "b-", lw=2)
    if best_threshold:
        ax1.axvline(x=best_threshold, color="red", linestyle="--",
                    label=f"Seuil optimal: {best_threshold:.3f}")
    ax1.set_xlabel("Seuil")
    ax1.set_ylabel("Coût métier")
    ax1.set_title("Optimisation du coût métier")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df_results["threshold"], df_results["auc_roc"], "g-", lw=2, label="AUC-ROC")
    ax2.plot(df_results["threshold"], df_results["accuracy"], "b--", lw=2, label="Accuracy")
    ax2.set_xlabel("Seuil")
    ax2.set_ylabel("Score")
    ax2.set_title("AUC-ROC & Accuracy vs Seuil")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Affiche la matrice de confusion."""
    import seaborn as sns
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Prédit 0", "Prédit 1"],
                yticklabels=["Réel 0", "Réel 1"], ax=ax)
    ax.set_title("Matrice de Confusion")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_roc_curve(y_true, y_proba, save_path=None):
    """Courbe ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def generate_metrics_report(metrics: dict, save_path=None) -> str:
    """Génère un rapport texte des métriques."""
    lines = [
        "=" * 50,
        "  RAPPORT DE MÉTRIQUES - HOME CREDIT SCORING",
        "  Projet7_Open_Dominique",
        "=" * 50,
        f"  AUC-ROC              : {metrics.get('auc_roc', 0):.4f}",
        f"  Accuracy             : {metrics.get('accuracy', 0):.4f}",
        f"  Seuil optimal        : {metrics.get('threshold', 0):.4f}",
        f"  Coût métier total    : {metrics.get('business_cost', 0):.0f}",
        f"  Coût normalisé       : {metrics.get('normalized_cost', 0):.4f}",
        "-" * 50,
        f"  Vrais Négatifs (TN)  : {metrics.get('tn', 0)}",
        f"  Faux Positifs (FP)   : {metrics.get('fp', 0)}  (coût: {metrics.get('fp',0) * COST_FP})",
        f"  Faux Négatifs (FN)   : {metrics.get('fn', 0)}  (coût: {metrics.get('fn',0) * COST_FN})",
        f"  Vrais Positifs (TP)  : {metrics.get('tp', 0)}",
        "-" * 50,
        f"  Précision            : {metrics.get('precision', 0):.4f}",
        f"  Rappel               : {metrics.get('recall', 0):.4f}",
        "=" * 50,
    ]
    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)

    return report
