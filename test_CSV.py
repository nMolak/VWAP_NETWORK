"""
Ewaluacja wytrenowanego modelu na zbiorze testowym (CSV).

Wczytuje model .pt + scaler .pkl, generuje predykcje i wypisuje
classification report + confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from filters import filters
from load_data import add_VWAP
from parameters import sigma_val
from training_weights import (
    apply_cooldown,
    data_diagnostics,
    load_vwap_model,
    predict,
)


# ---------------------------------------------------------------------------
# Przygotowanie danych
# ---------------------------------------------------------------------------

def prepare_single_file(
    df: pd.DataFrame,
    label_col: str,
    sigma: float,
    feature_prefix: str = "feature_",
    log: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Przygotowuje X, y z pojedynczego DataFrame."""

    def logprint(*args, **kwargs):
        if log:
            print(*args, **kwargs)

    df = df.copy()
    logprint(f"Rekordow przed filtrami: {len(df)}")
    df = add_VWAP(df, sigma)

    df["is_extreme"] = (
        (df["close"] > df[f"vwap_plus_{sigma}_sigma"])
        | (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
    )

    for f in filters:
        before = len(df)
        df = f(df)
        after = len(df)
        logprint(f"Filtr {f.__name__}: {before} -> {after} ({(before - after) / before:.2%} usunieto)")

    df = df[df["is_extreme"]].copy()
    logprint(f"Po ograniczeniu do ekstremow: {len(df)} rekordow")

    if df.empty:
        print("Po filtrach brak danych!")
        return None, None, []

    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[feature_cols].values
    y = df[label_col].values.astype(int)

    data_diagnostics(df, feature_cols)
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Skrypt
# ---------------------------------------------------------------------------

data_path = r"data/1m/test_data"
scaler_path = r"scalers/scaler.pkl"
model_path = r"data/models/vwap_model.pt"

label = "label1_N10"
sigma = sigma_val

df = pd.read_csv(data_path)
X, y_true, feature_cols = prepare_single_file(df, label, sigma, "feature_")

scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)
model = load_vwap_model(model_path)

y_pred_probs = predict(model, X_scaled)
y_pred = np.argmax(y_pred_probs, axis=1) if y_pred_probs.ndim > 1 else (y_pred_probs > 0.5).astype(int)

mapping = {-1: 0, 0: 1, 1: 2}
y_true_mapped = np.array([mapping[val] for val in y_true])

print(">>> Classification Report")
print(classification_report(y_true_mapped, y_pred, target_names=["-1", "0", "1"]))

print(">>> Confusion Matrix")
print(confusion_matrix(y_true_mapped, y_pred))
