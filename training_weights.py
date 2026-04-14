"""
Modul treningowy — budowa, trening i zapis modeli MLP (PyTorch).

Eksportowane API:
    VWAPModel          — klasa nn.Module (MLP)
    load_vwap_model    — odtworzenie modelu z checkpointu .pt
    predict            — numpy-in / numpy-out wrapper inferencji
    build_model_random — losowa architektura MLP + metadane
    train              — pelny pipeline: dane → trening → zapis
    make_model_name    — kodowanie hiperparametrow w nazwie pliku
    apply_cooldown     — tlumienie powtorzonych sygnalow ekstremalnych
    get_effective_accuracy — precision per klasa
    data_diagnostics   — walidacja kolumn feature w DataFrame
"""

from __future__ import annotations

import copy
import random
from datetime import datetime
from functools import partial
from pathlib import Path
from time import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from calculations import calc_indicators
from filters import filter_clean, filters
from labels import calc_label9, calc_label10
from load_data import add_VWAP, repair_volume
from parameters import sigma_val


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VWAPModel(nn.Module):
    """MLP do klasyfikacji binarnej lub wieloklasowej.

    Wyjscie to surowe logity (bez sigmoid/softmax) —
    sigmoid/softmax aplikowane sa w kryterium strat lub w ``predict()``.
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
        dropouts: list[float],
        activation: str,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        act_cls = nn.ReLU if activation == "relu" else nn.Tanh
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for units, drop in zip(layer_sizes, dropouts):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(act_cls())
            layers.append(nn.Dropout(drop))
            prev_dim = units

        if n_classes == 2:
            layers.append(nn.Linear(prev_dim, 1))
        else:
            layers.append(nn.Linear(prev_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_vwap_model(
    path: str | Path,
    device: torch.device | None = None,
) -> VWAPModel:
    """Laduje checkpoint ``.pt`` i zwraca model w trybie eval."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    meta = checkpoint["meta"]
    model = VWAPModel(
        input_dim=checkpoint["input_dim"],
        layer_sizes=meta["layer_sizes"],
        dropouts=meta["dropouts"],
        activation=meta["activation"],
        n_classes=meta["n_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict(
    model: VWAPModel,
    X: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Numpy-in / numpy-out inferencja.

    Zwraca:
        Binary  (n_classes==2): ``(N,)`` float z prawdopodobienstwami klasy 1.
        Multi   (n_classes>2):  ``(N, C)`` float z prawdopodobienstwami klas.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_t)
        if model.n_classes == 2:
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        else:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# ---------------------------------------------------------------------------
# Nazewnictwo modelu
# ---------------------------------------------------------------------------

def make_model_name(
    timestamp: datetime,
    label: Any,
    sigma: float,
    layers: list[int],
    accuracies: list[float] | float,
    optimizer: str,
    dropouts: list[float] | None = None,
    activation: str | None = None,
    loss: str | None = None,
) -> str:
    """Koduje hiperparametry w nazwie pliku modelu.

    Format: ``ts_label{params}_sigma_layers-dropouts_activation_loss_acc1-acc2_optimizer``
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    sigma_str = f"{sigma:.2f}"
    layers_str = "-".join(map(str, layers))

    if hasattr(label, "func") and hasattr(label, "keywords"):
        base_name = label.func.__name__
        params_str = ";".join(f"{k}={v}" for k, v in label.keywords.items()) if label.keywords else ""
        label_str = f"{base_name}{{{params_str}}}"
    else:
        label_str = str(label)

    dropouts_str = "-".join(f"{d:.2f}" for d in dropouts) if dropouts else ""

    if isinstance(accuracies, (list, tuple)):
        accs_str = "-".join(f"{acc:.4f}" for acc in accuracies)
    else:
        accs_str = f"{accuracies:.4f}"

    return f"{ts_str}_{label_str}_{sigma_str}_{layers_str}_{dropouts_str}_{activation}_{loss}_{accs_str}_{optimizer}"


# ---------------------------------------------------------------------------
# Diagnostyka danych
# ---------------------------------------------------------------------------

def data_diagnostics(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Wypisuje statystyki kolumn feature i liczbe blednych wierszy."""
    print(">>> Diagnostyka danych\n")
    for col in feature_cols:
        if col not in df.columns:
            print(f"Kolumna {col} nie istnieje w ramce!")
            continue
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if numeric.empty:
            print(f"{col}: brak poprawnych wartosci liczbowych")
        else:
            print(
                f"{col}: min={numeric.min():.6f}, max={numeric.max():.6f}, "
                f"median={numeric.median():.6f}, mean={numeric.mean():.6f}, std={numeric.std():.6f}"
            )

    bad_mask = df[feature_cols].replace("", np.nan).map(
        lambda x: not np.isfinite(x) if isinstance(x, (int, float, np.number)) else pd.isna(x)
    )
    bad_rows = bad_mask.any(axis=1).sum()
    print(f"\nLacznie blednych wierszy (NaN/Inf/puste): {bad_rows}")


def data_diagnostics_labels(y: np.ndarray, name: str = "y") -> None:
    print(f">>> Diagnostyka etykiet {name}")
    print(f"Shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Label {u}: {c} ({c / len(y):.2%})")
    print()


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------

def apply_cooldown(
    mask_extreme: pd.Series,
    seq_len: int = 10,
    cooldown: int = 30,
) -> np.ndarray:
    """Tlumi powtorzone ciagi ekstremalnych swiec."""
    mask = mask_extreme.to_numpy().copy()
    true_idx = np.where(mask)[0]
    i = 0
    while i < len(true_idx):
        start = true_idx[i]
        j = i
        while j + 1 < len(true_idx) and true_idx[j + 1] == true_idx[j] + 1:
            j += 1
        end = true_idx[j]
        length = end - start + 1
        if length >= seq_len:
            mask[start : end + cooldown + 1] = False
            i = j + 1
        else:
            i += 1
    return mask


# ---------------------------------------------------------------------------
# Ewaluacja modelu
# ---------------------------------------------------------------------------

def get_effective_accuracy(
    model: VWAPModel,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler | None = None,
) -> list[float]:
    """Zwraca precision per klasa (precyzja kolumnowa macierzy pomylek)."""
    if scaler is not None:
        X_val = scaler.transform(X_val)

    probs = predict(model, X_val)

    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        y_pred = (probs.ravel() > 0.5).astype(int)
        n_classes = 2
    else:
        y_pred = np.argmax(probs, axis=1)
        n_classes = probs.shape[1]

    cm = confusion_matrix(y_val, y_pred, labels=np.arange(n_classes))
    accs = []
    for i in range(n_classes):
        total_pred = cm[:, i].sum()
        accs.append(cm[i, i] / total_pred if total_pred > 0 else 0.0)

    print(f"get_effective_accuracy: wykryto {n_classes} klasy, accs={accs}")
    return accs


def get_model_layers(model: VWAPModel) -> list[int]:
    """Zwraca rozmiary warstw Linear z modelu."""
    return [m.out_features for m in model.net if isinstance(m, nn.Linear)]


def get_max_class_percent(data: pd.Series | np.ndarray) -> float:
    """Zwraca udzial najwiekszej klasy (0–1)."""
    if isinstance(data, pd.Series):
        if data.empty:
            return 0.0
        return data.value_counts(normalize=True).max()
    if len(data) == 0:
        return 0.0
    _, counts = np.unique(data, return_counts=True)
    return counts.max() / len(data)


# ---------------------------------------------------------------------------
# Przygotowanie danych
# ---------------------------------------------------------------------------

def prepare_all_data(
    data_path: str,
    sigma: float,
    filter_list: list,
    features_filename: str,
    split_ratio: float = 0.70,
    label_func=None,
    recalc_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Wczytuje CSV, liczy VWAP, filtruje, buduje X/y, skaluje."""

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    path = Path(data_path)
    feature_cols: list[str] = []

    print(f"=== START prepare_all_data ===")
    print(f"Sciezka danych: {data_path}")
    print(f"Sigma: {sigma}")
    print(f"Filtry: {[f.__name__ for f in filter_list]}")

    for ticker_dir in sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")):
        print(f"\n=== TICKER {ticker_dir.name} ===")

        csv_file = next(ticker_dir.glob("*.csv"), None)
        if csv_file is None:
            raise ValueError(f"Brak pliku CSV w {ticker_dir}")

        df = pd.read_csv(csv_file)
        print(f"Wczytano {len(df):,} rekordow z {csv_file.name}")

        df = repair_volume(df, 30, log=True)
        zeros_after = (df["volume"] == 0).sum()
        if zeros_after > 0:
            print(f"  {zeros_after} zerowych wolumenow pozostalo po naprawie!")

        # 1. VWAP
        df = add_VWAP(df, sigma)
        df["is_extreme"] = (
            (df["close"] > df[f"vwap_plus_{sigma}_sigma"])
            | (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
        )
        print(f"  Dodano VWAP + maske is_extreme (True={df['is_extreme'].sum()})")

        # 2. Label
        if label_func is None:
            raise ValueError("Brak funkcji etykietujacej (label_func)")
        t0 = time()
        y_series = label_func(df)
        df["y_series"] = y_series
        n_invalid = (df["y_series"] == -1).sum()
        print(f"  Label obliczony w {time() - t0:.2f}s ({len(df) - n_invalid:,}/{len(df):,} poprawnych)")

        # 3. Czyszczenie
        before = len(df)
        df = filter_clean()(df)
        df = df[df["y_series"] != -1]
        print(f"  filter_clean + usuwanie y=-1: {before:,} -> {len(df):,}")

        # 4. Filtry globalne
        for f in filter_list:
            before = len(df)
            df = f(df)
            print(f"  Filtr {f.__name__}: {before:,} -> {len(df):,}")

        # 5. Ekstrema
        before = len(df)
        df = df[df["is_extreme"]]
        print(f"  Ekstrema: {before:,} -> {len(df):,}")
        if df.empty:
            print(f"  Brak ekstremalnych rekordow — pomijam {ticker_dir.name}")
            continue

        # 6. X, y
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        X = df[feature_cols].to_numpy()
        y = df["y_series"].astype(int).to_numpy()
        X_list.append(X)
        y_list.append(y)
        print(f"  Dodano {X.shape[0]:,} rekordow")

    # Scalenie
    print("\n=== SCALENIE ===")
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"Calosc: X_all={X_all.shape}, y_all={y_all.shape}")

    # Diagnostyka
    mask_nan = np.isnan(X_all).sum(axis=0)
    mask_inf = np.isinf(X_all).sum(axis=0)
    if mask_nan.any() or mask_inf.any():
        print("Znaleziono problemy w X_all:")
        for i, (n_nan, n_inf) in enumerate(zip(mask_nan, mask_inf)):
            if n_nan or n_inf:
                print(f"  Kolumna {i}: NaN={n_nan:,} | inf={n_inf:,}")

    # Podzial bez shufflowania
    n = len(X_all)
    train_size = int(n * 0.7)
    start = int(n * 0.15)
    end = start + train_size

    X_train = X_all[start:end]
    y_train = y_all[start:end]
    X_test = np.concatenate([X_all[:start], X_all[end:]])
    y_test = np.concatenate([y_all[:start], y_all[end:]])

    print(f"Split: train={len(X_train):,} ({len(X_train)/n:.2%}), test={len(X_test):,} ({len(X_test)/n:.2%})")

    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "scalers/scaler.pkl")
    print("Scaler zapisany (scalers/scaler.pkl)")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Budowa modelu
# ---------------------------------------------------------------------------

def build_model_random(input_dim: int, n_classes: int = 2) -> tuple[VWAPModel, dict]:
    """Tworzy losowy model MLP i zwraca ``(model, meta)``.

    Architektura 3–5 warstw, losowy schemat (pyramid/inverted/wide_start/flat/bottleneck),
    malejace dropouty, aktywacja relu/tanh, optimizer Adam/RMSprop.
    """
    n_layers = random.randint(3, 5)

    big = [128, 256, 512]
    mid = [64, 96, 128]
    small = [16, 32, 48]

    pattern_type = random.choice(["pyramid", "inverted", "wide_start", "flat", "bottleneck"])

    if pattern_type == "pyramid":
        layer_sizes = [random.choice(small), random.choice(big), random.choice(big)]
        while len(layer_sizes) < n_layers:
            layer_sizes.append(random.choice(small))

    elif pattern_type == "inverted":
        start_size = random.choice(big)
        step = random.choice([0.5, 0.6, 0.75])
        layer_sizes = [max(8, int(start_size * (step ** i))) for i in range(n_layers)]

    elif pattern_type == "wide_start":
        layer_sizes = [random.choice([256, 512, 768])]
        for _ in range(1, n_layers):
            next_size = max(16, int(layer_sizes[-1] * random.uniform(0.5, 0.8)))
            layer_sizes.append(next_size)

    elif pattern_type == "flat":
        val = random.choice(mid)
        layer_sizes = [val] * n_layers

    else:  # bottleneck
        half = n_layers // 2
        layer_sizes = [random.choice(big)]
        for i in range(1, n_layers):
            layer_sizes.append(random.choice(small) if i == half else random.choice(mid))

    print(f">>> Wzorzec warstw: {pattern_type}, warstwy: {layer_sizes}")

    # Dropouty malejace
    start_dropout = round(random.uniform(0.2, 0.3), 2)
    drop_step = round(random.uniform(0.03, 0.07), 2)
    dropouts = [max(0.05, round(start_dropout - i * drop_step, 2)) for i in range(n_layers)]

    activation = random.choice(["relu", "tanh"])

    # Konfiguracja optimizera (tworzony pozniej, po model.to(device))
    optimizer_name = random.choice(["adam", "rmsprop"])
    lr = random.choice([1e-3, 5e-4, 1e-4])

    # Loss
    loss_name = "bce_with_logits" if n_classes == 2 else "cross_entropy"

    model = VWAPModel(
        input_dim=input_dim,
        layer_sizes=layer_sizes,
        dropouts=dropouts,
        activation=activation,
        n_classes=n_classes,
    )

    meta = {
        "input_dim": input_dim,
        "layer_sizes": layer_sizes,
        "dropouts": dropouts,
        "activation": activation,
        "optimizer": optimizer_name,
        "lr": lr,
        "loss": loss_name,
        "n_classes": n_classes,
    }

    print(f">>> build_model_random: {meta}")
    return model, meta


# ---------------------------------------------------------------------------
# Trening
# ---------------------------------------------------------------------------

def _build_optimizer(model: VWAPModel, meta: dict) -> torch.optim.Optimizer:
    if meta["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=meta["lr"])
    return torch.optim.RMSprop(model.parameters(), lr=meta["lr"])


def train(data_path: str, label, features_filename: str, batch_size: int) -> None:
    """Pelny pipeline: przygotowanie danych → trening → zapis modelu .pt."""

    start_time = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train, X_test, y_train, y_test, feature_cols = prepare_all_data(
        data_path,
        sigma=sigma_val,
        filter_list=filters,
        features_filename=features_filename,
        label_func=label,
        recalc_features=False,
    )

    # Wazenie klas
    classes = np.unique(y_train)
    n_classes = len(classes)
    class_weights_array = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, class_weights_array))
    print(f"Klasy: {n_classes}, class_weights: {class_weights}")

    # Model
    model, meta = build_model_random(X_train.shape[1], n_classes=n_classes)
    model.to(device)
    optimizer = _build_optimizer(model, meta)

    # Kryterium
    if n_classes == 2:
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        weights_tensor = torch.tensor(
            [class_weights[c] for c in sorted(class_weights)],
            dtype=torch.float32,
            device=device,
        )
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Tensory
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    if n_classes == 2:
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)
    else:
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # Early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15
    best_state: dict | None = None

    for epoch in range(100):
        # --- Trening ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            if n_classes == 2:
                loss = criterion(output.squeeze(1), y_batch)
            else:
                loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)
        epoch_loss /= len(X_train_t)

        # --- Walidacja ---
        model.eval()
        with torch.no_grad():
            val_out = model(X_test_t)
            if n_classes == 2:
                val_loss = criterion(val_out.squeeze(1), y_test_t).item()
            else:
                val_loss = criterion(val_out, y_test_t).item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d} | train_loss={epoch_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping po epoce {epoch + 1}")
                break

    # Przywrocenie najlepszych wag
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Ewaluacja
    accuracies = get_effective_accuracy(model, X_test, y_test, scaler=None)

    model_name = make_model_name(
        datetime.now(),
        label,
        sigma_val,
        meta["layer_sizes"],
        accuracies,
        meta["optimizer"],
        dropouts=meta["dropouts"],
        activation=meta["activation"],
        loss=meta["loss"],
    )

    # Zapis
    models_path = Path("models")
    class_dir = Path("2_class")
    model_dir = models_path / class_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    final_path = model_dir / f"{model_name}.pt"
    model.cpu()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "meta": meta,
            "input_dim": meta["input_dim"],
        },
        final_path,
    )
    print(f"Model zapisano: {final_path}")

    elapsed = (time() - start_time) / 60
    print(f"Trening zakonczony w {elapsed:.1f} minut")

    # Raport
    probs = predict(model, X_test)
    if n_classes == 2:
        y_pred_classes = (probs > 0.5).astype(int)
        target_names = ["0", "1"]
    else:
        y_pred_classes = np.argmax(probs, axis=1)
        target_names = [str(c) for c in classes]

    print(">>> Classification Report")
    print(classification_report(y_test, y_pred_classes, target_names=target_names))
    print(">>> Confusion Matrix")
    print(confusion_matrix(y_test, y_pred_classes, labels=classes))


# ---------------------------------------------------------------------------
# Skrypt uruchamiajacy
# ---------------------------------------------------------------------------

label_partials = [
    partial(calc_label9, T=40, alpha=0.72, use_atr=False),
    partial(calc_label9, T=35, alpha=0.72, use_atr=False),
    partial(calc_label9, T=40, alpha=0.80, use_atr=False),
    partial(calc_label9, T=45, alpha=0.775, use_atr=False),
]

data_path = "data/training_data"
features_filename = "features00"
batch_size = 64

for label in label_partials:
    for _ in range(10):
        train(data_path, label, features_filename, batch_size)
