from pathlib import Path
import numpy as np
import joblib

MODELS_PATH = Path(__file__).parents[1] / "data" / "models"


def save_model(model, name: str) -> Path:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    path = MODELS_PATH / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"  Saved → {path}")
    return path


def load_model(name: str):
    path = MODELS_PATH / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model '{name}' not found at {path}.\n"
            "Run 02_baseline_models.py first."
        )
    return joblib.load(path)


def mae(pred, actual) -> float:
    return float(np.abs(np.asarray(pred, float) - np.asarray(actual, float)).mean())


def mape(pred, actual) -> float:
    p, a = np.asarray(pred, float), np.asarray(actual, float)
    return float((np.abs(p - a) / a).mean() * 100)


def scaled_mae(pred, actual, naive) -> float:
    return mae(pred, actual) / mae(naive, actual)


def eval_table(rows, title="Evaluation"):
    """rows: list of (name, pred_array, actual_array, naive_array)"""
    print(f"\n── {title} ────────────────────────────────────────────────────────")
    print(f"{'Model':<38} {'MAE':>8} {'MAPE%':>7} {'Scaled MAE':>12}")
    print("─" * 70)
    for name, pred, actual, naive in rows:
        m  = mae(pred, actual)
        mp = mape(pred, actual)
        s  = m / mae(naive, actual)
        print(f"{name:<38} {m:>8.1f} {mp:>7.2f} {s:>12.3f}")
    print("─" * 70)
    print("Scaled MAE < 1.0 → beats lag-52 naive")
