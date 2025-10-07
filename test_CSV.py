import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from parameters import sigma_val
from filters import filters
from load_data import add_VWAP

# --- pomocnicze funkcje ---

def apply_cooldown(mask_extreme, seq_len=10, cooldown=30):
    mask = mask_extreme.to_numpy().copy()
    true_idx = np.where(mask)[0]
    i = 0
    while i < len(true_idx):
        start = true_idx[i]
        j = i
        while j + 1 < len(true_idx) and true_idx[j+1] == true_idx[j] + 1:
            j += 1
        end = true_idx[j]
        length = end - start + 1
        if length >= seq_len:
            mask[start:end+cooldown+1] = False
            i = j + 1
        else:
            i += 1
    return mask

def data_diagnostics(df, feature_cols):
    import numpy as np
    import pandas as pd
    print(">>> Diagnostyka danych")
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not numeric.empty:
            print(f"{col}: min={numeric.min():.6f}, max={numeric.max():.6f}, "
                  f"mean={numeric.mean():.6f}, std={numeric.std():.6f}")



def prepare_single_file(df, label_col, sigma, feature_prefix="feature_", log=False):
    def logprint(*args, **kwargs):
        if log:
            print(*args, **kwargs)

    df = df.copy()
    logprint(f"Rekordów przed filtrami: {len(df)}")
    df = add_VWAP(df, sigma)



    df["is_extreme"] = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
                       (df["close"] < df[f"vwap_minus_{sigma}_sigma"])

    # zastosuj filtry (lista z importu)
    for f in filters:
        before = len(df)
        df = f(df)
        after = len(df)
        logprint(f"Filtr {f.__name__}: {before} → {after} rekordów ({(before - after)/before:.2%} usunięto)")

    # ogranicz do samych ekstremów
    df = df[df["is_extreme"]].copy()
    logprint(f"Po ograniczeniu do ekstremów: {len(df)} rekordów")

    if df.empty:
        print("❌ Po filtrach brak danych!")
        return None, None, []

    # wybierz cechy i etykiety
    feature_cols = [c for c in df.columns if c.startswith(feature_prefix)]
    X = df[feature_cols].values
    y = df[label_col].values.astype(int)

    data_diagnostics(df, feature_cols)
    return X, y, feature_cols




# --- ścieżki ---
data_path = r"data/1m/test_data"
scaler_path = r"scalers"
model_path = r"data/models/vwap_model.h5"

label = "label1_N10"
sigma = sigma_val

# --- przygotowanie danych ---
X, y_true, feature_cols = prepare_single_file(data_path, label, sigma, "feature_")

# --- scaler + model ---
scaler = joblib.load(scaler_path)
X_scaled = scaler.transform(X)
model = load_model(model_path)

# --- predykcje ---
y_pred_probs = model.predict(X_scaled)
y_pred = y_pred_probs.argmax(axis=1)

# mapowanie ręczne
mapping = {-1: 0, 0: 1, 1: 2}
y_true_mapped = np.array([mapping[val] for val in y_true])

print(">>> Classification Report")
print(classification_report(
    y_true_mapped,
    y_pred,
    target_names=["-1","0","1"]
))

print(">>> Confusion Matrix")
print(confusion_matrix(y_true_mapped, y_pred))

