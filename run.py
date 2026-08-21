"""
Orchestrateur local - Projet7_Open_Dominique
Usage:
    python run.py api
    python run.py dashboard
    python run.py mlflow
    python run.py all
    python run.py train
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_api():
    print("🚀 Démarrage API FastAPI sur http://localhost:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ], cwd=ROOT)


def run_dashboard():
    print("🎨 Démarrage Dashboard Streamlit sur http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "streamlit_app" / "app.py"),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ], cwd=ROOT)


def run_mlflow():
    print("📊 Démarrage MLflow UI sur http://localhost:5002")
    mlruns_path = ROOT / "mlruns"
    mlruns_path.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--host", "0.0.0.0",
        "--port", "5002",
        "--backend-store-uri", str(mlruns_path),
    ], cwd=ROOT)


def run_train():
    print("🧠 Lancement de l'entraînement...")
    subprocess.run([
        sys.executable, "-c",
        """
import sys
sys.path.insert(0, '.')
from src.preprocessing import prepare_train_test_data
from src.train import setup_mlflow, train_with_mlflow, save_artifacts

print("Chargement des données...")
X_train, X_test, y_train = prepare_train_test_data()

setup_mlflow()
model, preprocessor, threshold, metrics = train_with_mlflow(X_train, y_train)
save_artifacts(model, preprocessor, threshold, X_train.columns.tolist())
print("Entraînement terminé!")
"""
    ], cwd=ROOT)


def run_all():
    """Lance API + Dashboard + MLflow en parallèle."""
    import threading
    threads = [
        threading.Thread(target=run_api, daemon=True),
        threading.Thread(target=run_dashboard, daemon=True),
        threading.Thread(target=run_mlflow, daemon=True),
    ]
    print("🚀 Démarrage de tous les services...")
    print("   API:       http://localhost:8000")
    print("   Dashboard: http://localhost:8501")
    print("   MLflow:    http://localhost:5002")

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n⛔ Arrêt des services.")


COMMANDS = {
    "api": run_api,
    "dashboard": run_dashboard,
    "mlflow": run_mlflow,
    "train": run_train,
    "all": run_all,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python run.py [api|dashboard|mlflow|train|all]")
        print("\nCommandes disponibles:")
        for cmd in COMMANDS:
            print(f"  {cmd}")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
