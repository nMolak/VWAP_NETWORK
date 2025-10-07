"""
Główny moduł odpowiadający za trening sieci i przygotowanie danych.

make_model_name -> Zwraca stringa zawierającego parametry danej sieci
get_model_layers -> Zwraca tablicę rozmiarów warstw modelu

data_diagnostics -> Walidacja istnienia kolumn featurów, podstawowe miary
                    statystyczne, informacja o ilości błędnych/pustych rekordów.
data_diagnostics_numpy -> do wypierdolenia

prepare_all_data -> # Przygotowanie danych do wejścia sieci.

build_model_random -> Zwraca model z losowymi parametrami + metadane

train -> Proces treningu i zapisu modelu

"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from pathlib import Path
import joblib
import random

from datetime import datetime
from time import time


from calculations import calc_indicators
from functools import partial
from load_data import add_VWAP
from parameters import sigma_val
from filters import filters

from labels import calc_label9





def make_model_name(timestamp: datetime,
                    label,
                    sigma: float,
                    layers: list,
                    accuracies: list,
                    optimizer: str,
                    dropouts: list = None,
                    activation: str = None,
                    loss: str = None) -> str:
    """
    Składa nazwę modelu z dodatkowymi hiperparametrami losowanymi.
    Format:
    ts_label{params}_sigma_layers-dropouts_activation_loss_acc1-acc2-acc3_optimizer
    """
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    sigma_str = f"{sigma:.2f}"
    layers_str = "-".join(map(str, layers))

    # --- label ---
    if hasattr(label, "func") and hasattr(label, "keywords"):
        base_name = label.func.__name__
        if label.keywords:
            params_str = ";".join(f"{k}={v}" for k, v in label.keywords.items())
        else:
            params_str = ""
        label_str = f"{base_name}{{{params_str}}}"
    else:
        label_str = str(label)

    # --- dropouts ---
    dropouts_str = "-".join(f"{d:.2f}" for d in dropouts) if dropouts else ""

    # --- accuracies ---
    if isinstance(accuracies, (list, tuple)):
        accs_str = "-".join(f"{acc:.4f}" for acc in accuracies)
    else:
        accs_str = f"{accuracies:.4f}"

    # --- składanie ---
    return f"{ts_str}_{label_str}_{sigma_str}_{layers_str}_{dropouts_str}_{activation}_{loss}_{accs_str}_{optimizer}"


def get_model_layers(model):
    layer_sizes = []
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            layer_sizes.append(layer.units)
    return layer_sizes

def data_diagnostics(df, feature_cols):

    print(">>> Diagnostyka danych\n")

    for col in feature_cols:
        if col not in df.columns:
            print(f"⚠️ Kolumna {col} nie istnieje w ramce!")
            continue

        series = df[col]

        # wartości liczbowe (bez NaN/Inf)
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        if numeric.empty:
            print(f"{col}: brak poprawnych wartości liczbowych")
        else:
            print(
                f"{col}: min={numeric.min():.6f}, max={numeric.max():.6f}, "
                f"median={numeric.median():.6f}, mean={numeric.mean():.6f}, std={numeric.std():.6f}"
            )

    # sprawdzenie błędnych wierszy
    bad_mask = df[feature_cols].replace("", np.nan).applymap(
        lambda x: not np.isfinite(x) if isinstance(x, (int, float, np.number)) else pd.isna(x)
    )
    bad_rows = bad_mask.any(axis=1).sum()

    print(f"\nŁącznie błędnych wierszy (NaN/Inf/puste): {bad_rows}")

def data_diagnostics_numpy(X, name="X"):
    print(f">>> Diagnostyka macierzy {name}")
    print(f"Shape: {X.shape}")

    # min/max/mean/std dla każdej kolumny
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    for i in range(X.shape[1]):
        print(f"Feature {i}: min={mins[i]:.6f}, max={maxs[i]:.6f}, "
              f"mean={means[i]:.6f}, std={stds[i]:.6f}")
    print()


def data_diagnostics_labels(y, name="y"):
    print(f">>> Diagnostyka etykiet {name}")
    print(f"Shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Label {u}: {c} ({c/len(y):.2%})")
    print()


def apply_cooldown(mask_extreme, seq_len=10, cooldown=30):
    # zamień na numpy array (bool)
    mask = mask_extreme.to_numpy().copy()
    true_idx = np.where(mask)[0]

    i = 0
    while i < len(true_idx):
        start = true_idx[i]
        j = i
        # znajdź koniec sekwencji kolejnych świec
        while j + 1 < len(true_idx) and true_idx[j+1] == true_idx[j] + 1:
            j += 1
        end = true_idx[j]

        length = end - start + 1
        if length >= seq_len:
            # wyzeruj sekwencję + cooldown
            mask[start:end+cooldown+1] = False
            i = j + 1
        else:
            i += 1
    return mask

def get_effective_accuracy(model, X_val, y_val, scaler=None):
    """
    Zwraca accuracy w sensie:
    'jeśli model przewidział klasę k, to w ilu % przypadków miał rację'.

    - dla 2 klas: [acc_klasy_0, acc_klasy_1]
    - dla 3 klas: [acc_klasy_0, acc_klasy_1, acc_klasy_2]
    """

    # --- Skalowanie (jeśli podano scaler)
    if scaler is not None:
        X_val = scaler.transform(X_val)

    # --- Predykcje
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    n_classes = model.output_shape[-1]

    # --- Macierz pomyłek
    cm = confusion_matrix(y_val, y_pred, labels=np.arange(n_classes))

    # --- Accuracy per klasa (precision)
    accs = []
    for i in range(n_classes):
        total_pred = cm[:, i].sum()  # ile razy model przewidział klasę i
        if total_pred == 0:
            accs.append(0.0)
        else:
            accs.append(cm[i, i] / total_pred)  # trafienia / wszystkie predykcje tej klasy

    return accs
# def prepare_all_data(data_path, sigma, filters,
#                      features_filename, split_ratio=0.70,
#                      label_func=None, labels_filename=None, recalc_features=False):
#     """
#     Przygotowanie danych do wejścia sieci.
#     - Iteracja po danych treningowych, wykonianie add_VWAP, dodanie maski is_extreme
#     - Możliwość zczytania gotowych featurów / policzenie ich w locie
#     - Możliwość zczytania gotowego labela / policzenie go w locie
#     - Przeskalowanie danych, zapisanie scalera, końcowa diagnostyka i zwrócenie danych
#     """
#
#     X_list, y_list = [], []
#
#     path = Path(data_path)
#     for ticker_file in path.iterdir():
#         if ticker_file.is_dir():
#             if ticker_file.name.startswith("."):
#                 continue
#
#             print(f"Znajdujemy się w folderze {ticker_file}")
#             csv_file = next(ticker_file.glob("*.csv"), None)
#
#             if csv_file is None:
#                 raise ValueError(f"Nie istnieje plik df w {ticker_file}")
#
#             # --- 1. Wczytaj dane
#             df = pd.read_csv(csv_file).copy()
#             print(f"Dane z pliku {ticker_file} pobrane - mają {len(df)} rekordów")
#
#             # --- 2. VWAP + sigma
#             print(f"Dodajemy do ramki danych VWAP + odległości sigma {sigma_val}")
#             df = add_VWAP(df, sigma)
#             df["is_extreme"] = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
#                                (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
#             from filters import filter_clean
#
#             df = filter_clean()(df)
#             # --- 3. Label
#             if label_func is not None:
#                 print("Obliczam label!")
#                 start_time_label = time()
#                 y_series = label_func(df).dropna()
#                 print(f"Label obliczony! ({len(y_series)} rekordów)")
#                 print(f"Policzenie labela zajęło {time() - start_time_label:.3f} s")
#             else:
#                 raise ValueError("Brak funkcji etykietującej (label_func)")
#
#             # przycięcie df do indeksu etykiet
#             df = df.loc[y_series.index].copy()
#
#             # --- 4. Filtry globalne (np. czyszczenie, godziny, FOMC)
#             for f in filters:
#                 before = len(df)
#                 df = f(df)
#                 print(f"Filtr {f.__name__} → {before:,} → {len(df):,}")
#
#             # --- 5. Przycięcie do ekstremów
#             df = df[df["is_extreme"]].copy()
#             if df.empty:
#                 raise ValueError("Brak ekstremalnych rekordów po filtrach!")
#
#             # --- 6. Dodaj cechy (features)
#             if recalc_features:
#                 indicators = calc_indicators(df)
#                 for k, v in indicators.items():
#                     df[f"feature_{k}"] = v
#                 feature_cols = [f"feature_{k}" for k in indicators.keys()]
#                 print(f"Cechy policzone w locie: {feature_cols}")
#             else:
#                 features_path = ticker_file / "features" / f"{features_filename}.parquet"
#                 print(f"Szukam ścieżki {features_path}!")
#                 if features_path.exists():
#                     features_df = pd.read_parquet(features_path)
#                     features_df = features_df.add_prefix("feature_")
#                     df = pd.concat([df, features_df], axis=1)
#                     feature_cols = list(features_df.columns)
#                     print(f"Wczytano featury z {features_path}: {feature_cols}")
#                     if "feature_timestamp" in feature_cols:
#                         print("⚠️ Usuwam kolumnę feature_timestamp (nienumeryczna)")
#                         feature_cols.remove("feature_timestamp")
#                 else:
#                     raise FileNotFoundError(f"Ścieżka {features_path} nie istnieje!")
#
#             # dopasowanie etykiet do przefiltrowanego df
#             y_series = y_series.loc[df.index]
#
#             # --- X, y ---
#             X = df[feature_cols].values
#             y = y_series.astype(int).values
#
#             # --- zapis ---
#             X_list.append(X)
#             y_list.append(y)
#
#             print(f"Dane wczytano i zapisano do tablic - było to {X.shape[0]} rekordów")
#
#     X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
#
#     for X, y in zip(X_list, y_list):
#         X_tr, X_te, y_tr, y_te = train_test_split(
#             X, y,
#             test_size=(1 - split_ratio),
#             shuffle=True,
#             random_state=42  # dla powtarzalności
#         )
#         X_train_list.append(X_tr)
#         X_test_list.append(X_te)
#         y_train_list.append(y_tr)
#         y_test_list.append(y_te)
#
#     # sklej całość
#     X_train = np.vstack(X_train_list)
#     X_test = np.vstack(X_test_list)
#     y_train = np.concatenate(y_train_list)
#     y_test = np.concatenate(y_test_list)
#
#     print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
#     print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
#
#     # --- skalowanie ---
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#
#     scaler_path = r"scalers/scaler.pkl"
#     print("Zapisuję scaler...")
#     joblib.dump(scaler, scaler_path)
#     print(f"Scaler zapisany pod ścieżką: {scaler_path}")
#
#     # --- logi ---
#     print(">>> StandardScaler parametry (tylko z danych treningowych):")
#     for i, col in enumerate(feature_cols):
#         print(f"Feature {col}: mean={scaler.mean_[i]:.6f}, std={scaler.scale_[i]:.6f}")
#
#     data_diagnostics_numpy(X_train_scaled, "X_train_scaled")
#     data_diagnostics_numpy(X_test_scaled, "X_test_scaled")
#     data_diagnostics_labels(y_train, "y_train")
#     data_diagnostics_labels(y_test, "y_test")
#
#     return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
def prepare_all_data(data_path, sigma, filters,
                              features_filename, split_ratio=0.70,
                              label_func=None, labels_filename=None, recalc_features=False):

    X_list, y_list = [], []
    path = Path(data_path)

    print(f"=== 🚀 START prepare_all_data_detailed ===")
    print(f"Ścieżka danych: {data_path}")
    print(f"Sigma: {sigma}")
    print(f"Filtry: {[f.__name__ for f in filters]}")

    for ticker_file in path.iterdir():
        if ticker_file.is_dir() and not ticker_file.name.startswith("."):
            print(f"\n📁 === TICKER {ticker_file.name} ===")

            csv_file = next(ticker_file.glob("*.csv"), None)
            if csv_file is None:
                raise ValueError(f"❌ Brak pliku CSV w {ticker_file}")

            df = pd.read_csv(csv_file)
            print(f"✅ Wczytano {len(df):,} rekordów z {csv_file.name}")

            # 1️⃣ VWAP
            from load_data import add_VWAP
            df = add_VWAP(df, sigma)
            df["is_extreme"] = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
                               (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
            print(f"➕ Dodano VWAP + maskę is_extreme (True={df['is_extreme'].sum()})")

            # 2️⃣ Czyszczenie
            from filters import filter_clean
            from filters import filter_no_zero_inf_nan
            before = len(df)
            df = filter_clean()(df)
            df = filter_no_zero_inf_nan()(df)


            df.reset_index(drop=True, inplace=True)
            print(f"🧹 filter_clean: {before:,} → {len(df):,}")

            # 3️⃣ Label
            if label_func is not None:
                print("🏷️ Obliczam label...")
                start_time = time()
                y_series = label_func(df).dropna()
                print(f"✅ Label obliczony ({len(y_series):,}) w {time()-start_time:.2f}s")
            else:
                raise ValueError("❌ Brak funkcji etykietującej (label_func)")

            df = df.loc[y_series.index].copy()

            # 4️⃣ Filtry globalne
            for f in filters:
                before = len(df)
                df = f(df)
                print(f"🔧 Filtr {f.__name__}: {before:,} → {len(df):,}")

            # 5️⃣ Ekstrema
            before = len(df)
            df = df[df["is_extreme"]]
            print(f"📊 Ekstrema: {before:,} → {len(df):,}")
            if df.empty:
                print(f"⚠️ Brak ekstremalnych rekordów – pomijam {ticker_file.name}")
                continue

            # 6️⃣ Featury
            features_path = ticker_file / "features" / f"{features_filename}.parquet"
            if features_path.exists():
                features_df = pd.read_parquet(features_path)
                print(f"✅ Wczytano {len(features_df):,} wierszy z {features_path}")
                if "timestamp" in features_df.columns:
                    print("⚠️ Usuwam kolumnę timestamp z features")
                    features_df = features_df.drop(columns=["timestamp"])
                features_df = features_df.add_prefix("feature_")
                df = pd.concat([df, features_df], axis=1)
            else:
                print(f"❌ Brak pliku {features_path}")
                continue

            feature_cols = [c for c in df.columns if c.startswith("feature_")]
            print(f"🧠 {len(feature_cols)} kolumn feature_*: {feature_cols}")

            # 🔎 Diagnostyka NaN/inf po złączeniu
            bad_cols = []
            for c in feature_cols:
                n_nan = df[c].isna().sum()
                n_inf = np.isinf(df[c]).sum()
                if n_nan or n_inf:
                    bad_cols.append((c, n_nan, n_inf))
            if bad_cols:
                print("⚠️ Znaleziono kolumny z NaN/inf:")
                for c, n_nan, n_inf in bad_cols:
                    print(f"   {c:25s} NaN={n_nan:,} | inf={n_inf:,}")

            df = filter_clean()(df)
            y_series = y_series.loc[df.index]

            df = filter_no_zero_inf_nan()(df)
            y_series = y_series.loc[df.index]

            # dopasuj label
            y_series = y_series.loc[df.index]

            X = df[feature_cols].to_numpy()
            y = y_series.astype(int).to_numpy()
            X_list.append(X)
            y_list.append(y)
            print(f"✅ Dodano {X.shape[0]:,} rekordów (X,y)")

    # === SCALENIE ===
    print("\n=== 🔗 SCALENIE ===")
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    print(f"Całość: X_all={X_all.shape}, y_all={y_all.shape}")

    # 🔎 Diagnostyka przed split
    mask_nan = np.isnan(X_all).sum(axis=0)
    mask_inf = np.isinf(X_all).sum(axis=0)
    if mask_nan.any() or mask_inf.any():
        print("⚠️ Znaleziono problemy w X_all:")
        for i, (n, infn) in enumerate(zip(mask_nan, mask_inf)):
            if n or infn:
                print(f"Kolumna {i}: NaN={n:,} | inf={infn:,}")

    # === PODZIAŁ ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=(1 - split_ratio),
        shuffle=True,
        random_state=42
    )
    print(f"📦 Split: X_train={X_train.shape}, X_test={X_test.shape}")

    # === SCALER ===
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        print("❌ Błąd podczas skalowania!")
        print(str(e))
        np.save("X_train_debug.npy", X_train)
        print("💾 Zapisano X_train_debug.npy do analizy")
        raise

    joblib.dump(scaler, "scalers/scaler.pkl")
    print("✅ Scaler zapisany (scalers/scaler.pkl)")

    print("=== ✅ KONIEC prepare_all_data_detailed ===")
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def explain_with_shap(model, X_sample, feature_names=None, log=True):
    """
    Liczy wartości SHAP dla modelu Keras i zwraca wyniki.

    model         : tf.keras.Model
    X_sample      : numpy array z danymi wejściowymi (np. fragment X_test)
    feature_names : lista nazw cech (opcjonalnie)
    log           : jeśli True, to używa logprint, w innym wypadku print()
    """

    logger = print if log else (lambda *a, **k: None)

    logger(">>> Tworzę explainer SHAP (DeepExplainer)...")
    explainer = shap.DeepExplainer(model, X_sample[:200])  # bierzemy kawałek jako "tło"

    logger(">>> Liczę wartości SHAP...")
    shap_values = explainer.shap_values(X_sample)

    # SHAP dla binary classification -> lista z jednym elementem
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    logger(">>> Wyliczono wartości SHAP")
    logger(f"Shape shap_values: {shap_values.shape}")

    # Średnie absolutne wartości shap dla każdej cechy
    mean_abs = np.mean(np.abs(shap_values), axis=0)

    logger(">>> Średni wpływ cech (|SHAP|):")
    for i, val in enumerate(mean_abs):
        fname = feature_names[i] if feature_names is not None else f"feature_{i}"
        logger(f"{fname}: {val:.6f}")

    return shap_values


def get_max_class_percent(data) -> float:
    """
    Zwraca udział największej klasy (w [0,1]).
    Obsługuje zarówno pd.Series, jak i np.ndarray.
    """
    if isinstance(data, pd.Series):
        if data.empty:
            return 0.0
        counts = data.value_counts(normalize=True)
        return counts.max()
    else:  # np.ndarray albo lista
        if len(data) == 0:
            return 0.0
        unique, counts = np.unique(data, return_counts=True)
        return counts.max() / len(data)


def build_model_random(input_dim, n_classes=2):
    """
    Tworzy losowy model MLP do klasyfikacji binarnej lub wieloklasowej.
    - 3–5 warstw ukrytych
    - układ "duża-duża-mała" lub "mała-duża-duża-mała"
    - dropout 0.05–0.30 (malejący)
    - aktywacja: relu / tanh
    - optimizer: Adam / RMSprop
    - loss: binary_crossentropy (dla 2 klas) lub sparse_categorical_crossentropy (dla >2)
    """
    # --- losowanie liczby warstw ---
    n_layers = random.randint(3, 5)

    big = [128, 256, 512]
    mid = [64, 96, 128]
    small = [16, 32, 48]

    # --- wybór schematu architektury ---
    pattern_type = random.choice([
        "pyramid",  # mała → duża → mała
        "inverted",  # duża → mała
        "wide_start",  # bardzo szeroko na początku
        "flat",  # wszystkie podobne
        "bottleneck"  # duża → mała → duża (autoencoder-like)
    ])

    if pattern_type == "pyramid":
        # klasyczna piramida
        layer_sizes = [random.choice(small), random.choice(big), random.choice(big)]
        while len(layer_sizes) < n_layers:
            layer_sizes.append(random.choice(small))

    elif pattern_type == "inverted":
        # malejące rozmiary
        start = random.choice(big)
        step = random.choice([0.5, 0.6, 0.75])
        layer_sizes = [max(8, int(start * (step ** i))) for i in range(n_layers)]

    elif pattern_type == "wide_start":
        # bardzo szeroki początek
        layer_sizes = [random.choice([256, 512, 768])]
        for i in range(1, n_layers):
            next_size = max(16, int(layer_sizes[-1] * random.uniform(0.5, 0.8)))
            layer_sizes.append(next_size)

    elif pattern_type == "flat":
        # wszystkie podobne rozmiary
        val = random.choice(mid)
        layer_sizes = [val for _ in range(n_layers)]

    elif pattern_type == "bottleneck":
        # duża → mała → duża
        half = n_layers // 2
        layer_sizes = [random.choice(big)]
        for i in range(1, n_layers):
            if i == half:
                layer_sizes.append(random.choice(small))
            else:
                layer_sizes.append(random.choice(mid))

    print(f">>> Wybrany wzorzec warstw: {pattern_type}, warstwy: {layer_sizes}")

    # --- dropouty malejące ---
    start_dropout = round(random.uniform(0.2, 0.3), 2)
    step = round(random.uniform(0.03, 0.07), 2)
    dropouts = [max(0.05, round(start_dropout - i * step, 2)) for i in range(n_layers)]

    # --- aktywacja i optymalizator ---
    activation = random.choice(['relu', 'tanh'])
    optimizer_choice = random.choice([
        optimizers.Adam(learning_rate=random.choice([1e-3, 5e-4, 1e-4])),
        optimizers.RMSprop(learning_rate=random.choice([1e-3, 5e-4]))
    ])
    optimizer_name = optimizer_choice.get_config()['name']

    # --- budowa modelu ---
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units, d in zip(layer_sizes, dropouts):
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(d))

    # --- output i loss w zależności od liczby klas ---
    if n_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss_choice = 'binary_crossentropy'
    else:
        model.add(layers.Dense(n_classes, activation='softmax'))
        loss_choice = 'sparse_categorical_crossentropy'

    model.compile(optimizer=optimizer_choice, loss=loss_choice, metrics=['accuracy'])

    # --- meta ---
    meta = {
        "layer_sizes": layer_sizes,
        "dropouts": dropouts,
        "activation": activation,
        "optimizer": optimizer_name,
        "loss": loss_choice,
        "n_classes": n_classes
    }

    print(">>> build_model_random")
    print(meta)

    return model, meta





def train(data_path, label, features_filename, batch_size):

    tolerance_error = 0.3

    start_time = time()

    X_train, X_test, y_train, y_test, feature_cols = prepare_all_data(
        data_path,
        sigma=sigma_val,
        filters=filters,
        features_filename=features_filename,
        label_func=label,
        recalc_features=False
    )

    # --- Automatyczne skorzystanie z ważenia klas przy progu decyzyjnym ---
    classes = np.unique(y_train)
    n_classes = len(classes)
    max_class = get_max_class_percent(y_train)
    expected = 1 / n_classes

    train_on_weights = True

    if max_class > expected * (1 + tolerance_error) or max_class < expected * (1 - tolerance_error) or train_on_weights:
        print(f"Wykryto nierówności w klasach!")
        print(f"Liczba klas: {n_classes}")
        print(f"Największy przydział klasy: {max_class}")
        print(f"Natomiast rozkład oczekiwany to {expected}")

        classes = np.unique(y_train)
        class_weights_array = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, class_weights_array))
        class_weights[2] *= 3
        print(">>> Wyliczone class weights:", class_weights)
    else:
        class_weights = None

    # --- Budowa modelu ---
    model, meta = build_model_random(X_train.shape[1], n_classes=n_classes)
    # model.compile(
    #     optimizer=optimizer,
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # --- Trening ---
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=batch_size,
        callbacks=[callback],
        class_weight=class_weights,
        verbose=1
    )

    # ============== ZAPIS MODELU ================
    models_path = Path("models")
    model_layers = get_model_layers(model)

    # >>> tu zamieniam X_val, y_val na X_test, y_test
    accuracies = get_effective_accuracy(model, X_test, y_test, scaler=None)

    model_name = make_model_name(
        datetime.now(), label, sigma_val,
        meta["layer_sizes"], accuracies, meta["optimizer"],
        dropouts=meta["dropouts"], activation=meta["activation"], loss=meta["loss"]
    )

    class_dir = Path("3_class")

    model_dir = models_path / class_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    final_path = model_dir / f"{model_name}.keras"

    model.save(final_path)
    print(f"Model zapisano pod ścieżką {final_path}")

    end_time = time()

    print(f"Wczytanie danych, trening i zapis modelu zajął łącznie {(end_time - start_time)/60} minut")

    y_pred = model.predict(X_test, verbose=0)

    if n_classes == 2:
        y_pred_classes = (y_pred > 0.5).astype(int).ravel()
        target_names = ["0", "1"]
    else:
        # y_pred shape: (N, n_classes)
        y_pred_classes = np.argmax(y_pred, axis=1)
        target_names = [str(c) for c in classes]

    print(">>> Classification Report")
    print(classification_report(y_test, y_pred_classes, target_names=target_names))

    print(">>> Confusion Matrix")
    print(confusion_matrix(y_test, y_pred_classes, labels=classes))

    # do_diagnostics = False
    # if do_diagnostics:
    #     #========== SHAP ============
    #
    #     # Wybieramy próbkę danych (np. 1000 rekordów z testu)
    #     X_sample = X_test[:1000]
    #
    #     # Liczymy shap_values
    #     shap_values = explain_with_shap(model, X_sample, feature_names=feature_cols, log=True)
    #
    #     # Wykresy podsumowania
    #     print(">>> Rysuję summary_plot")
    #     shap.summary_plot(shap_values, X_sample, feature_names=feature_cols)
    #
    #     print(">>> Rysuję summary_plot (bar)")
    #     shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type="bar")
    #
    #     # ============================== DIAGNOSTYKA Cech ==============================
    #     import os, gc
    #     import matplotlib.pyplot as plt
    #     from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    #
    #     print(">>> [DIAG] Tworzę folder diagnostics/")
    #     os.makedirs("diagnostics", exist_ok=True)
    #
    #     # 0) DataFrames
    #     print(">>> [DIAG] Tworzę DataFrame'y Xtr_df, Xte_df")
    #     Xtr_df = pd.DataFrame(X_train, columns=feature_cols)
    #     Xte_df = pd.DataFrame(X_test,  columns=feature_cols)
    #
    #     # 1) Korelacje
    #     print(">>> [DIAG] Liczę macierz korelacji na X_train")
    #     corr = Xtr_df.corr(method="pearson")
    #     fig, ax = plt.subplots(figsize=(12, 10))
    #     im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    #     ax.set_xticks(range(len(feature_cols))); ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
    #     ax.set_yticks(range(len(feature_cols))); ax.set_yticklabels(feature_cols, fontsize=8)
    #     ax.set_title("Macierz korelacji cech (X_train)")
    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    #     fig.tight_layout()
    #     fig.savefig("diagnostics/correlation_matrix.png", dpi=160)
    #     plt.close(fig)
    #     print(">>> [DIAG] Zapisano correlation_matrix.png")
    #
    #     pairs = []
    #     thr = 0.90
    #     for i in range(len(feature_cols)):
    #         for j in range(i+1, len(feature_cols)):
    #             r = float(corr.iat[i, j])
    #             if abs(r) >= thr:
    #                 pairs.append((feature_cols[i], feature_cols[j], r))
    #     pd.DataFrame(pairs, columns=["feat_a", "feat_b", "corr"])\
    #       .sort_values("corr", key=lambda s: s.abs(), ascending=False)\
    #       .to_csv("diagnostics/high_corr_pairs.csv", index=False)
    #     print(f">>> [DIAG] Zapisano high_corr_pairs.csv (liczba par: {len(pairs)})")
    #
    #     # 2) SHAP dependence
    #     print(">>> [DIAG] Generuję wykresy SHAP dependence")
    #     import shap as _shap
    #     shap_arr = shap_values
    #     N = shap_arr.shape[0]
    #     X_for_shap = pd.DataFrame(X_sample[:N], columns=feature_cols)
    #
    #     shap_importance = pd.Series(np.abs(shap_arr).mean(axis=0), index=feature_cols)\
    #                         .sort_values(ascending=False)
    #     shap_importance.to_csv("diagnostics/shap_mean_abs.csv")
    #     print(">>> [DIAG] Zapisano shap_mean_abs.csv")
    #
    #     topk = min(12, len(feature_cols))
    #     for feat in shap_importance.index[:topk]:
    #         print(f">>> [DIAG] Rysuję shap.dependence_plot dla {feat}")
    #         plt.figure(figsize=(6, 4))
    #         _shap.dependence_plot(feat, shap_arr, X_for_shap, show=False)
    #         plt.title(f"SHAP dependence: {feat}")
    #         plt.tight_layout()
    #         plt.savefig(f"diagnostics/shap_dependence_{feat}.png", dpi=160)
    #         plt.close()
    #
    #     # 3) PFI
    #     print(">>> [DIAG] Liczę Permutation Feature Importance (PFI)")
    #     def predict_proba_binary(m, Xdf):
    #         p = m.predict(Xdf.values, verbose=0)
    #         return np.asarray(p).ravel()
    #
    #     def eval_metrics(y_true, p):
    #         y_hat = (p >= 0.5).astype(int)
    #         return {
    #             "accuracy": accuracy_score(y_true, y_hat),
    #             "f1": f1_score(y_true, y_hat),
    #             "auc": roc_auc_score(y_true, p),
    #         }
    #
    #     base_p = predict_proba_binary(model, Xte_df)
    #     base = eval_metrics(y_test, base_p)
    #     print(f">>> [DIAG] Wyniki bazowe na X_test: {base}")
    #
    #     rng = np.random.default_rng(123)
    #     pfi_rows = []
    #     for idx, col in enumerate(feature_cols, 1):
    #         print(f"    [PFI] permutuję {idx}/{len(feature_cols)}: {col}")
    #         Xp = Xte_df.copy()
    #         Xp[col] = rng.permutation(Xp[col].values)
    #         p = predict_proba_binary(model, Xp)
    #         m = eval_metrics(y_test, p)
    #         pfi_rows.append({
    #             "feature": col,
    #             "delta_accuracy": base["accuracy"] - m["accuracy"],
    #             "delta_f1":       base["f1"]       - m["f1"],
    #             "delta_auc":      base["auc"]      - m["auc"],
    #         })
    #     pfi_df = pd.DataFrame(pfi_rows).sort_values("delta_auc", ascending=False)
    #     pfi_df.to_csv("diagnostics/pfi.csv", index=False)
    #     print(">>> [DIAG] Zapisano pfi.csv")
    #
    #     fig, ax = plt.subplots(figsize=(8, 6))
    #     ax.barh(pfi_df["feature"][::-1], pfi_df["delta_auc"][::-1])
    #     ax.set_xlabel("Spadek AUC po permutacji (ΔAUC)")
    #     ax.set_title("Permutation Feature Importance (X_test)")
    #     fig.tight_layout()
    #     fig.savefig("diagnostics/pfi_auc.png", dpi=160)
    #     plt.close(fig)
    #     print(">>> [DIAG] Zapisano pfi_auc.png")
    #
    #     # 4) RFE
    #     print(">>> [DIAG] Startuję Recursive Feature Elimination (RFE)")
    #     from tensorflow.keras.callbacks import EarlyStopping
    #
    #     def build_model_diag(input_dim: int) -> tf.keras.Model:
    #         m = models.Sequential([
    #             layers.Input(shape=(input_dim,)),
    #             layers.Dense(256, activation='relu'),
    #             layers.Dropout(0.4),
    #             layers.Dense(128, activation='relu'),
    #             layers.Dropout(0.3),
    #             layers.Dense(64, activation='relu'),
    #             layers.Dropout(0.2),
    #             layers.Dense(32, activation='relu'),
    #             layers.Dropout(0.1),
    #             layers.Dense(1, activation='sigmoid')
    #         ])
    #         m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #         return m
    #
    #     def run_rfe_pfi(build_fn, Xtr, ytr, Xval, yval,
    #                     start_feats, min_features=5, epochs=8, batch_size=1024, patience=2):
    #         feats = list(start_feats)
    #         history = []
    #         best_auc = -np.inf
    #         best_subset = feats.copy()
    #
    #         while len(feats) >= min_features:
    #             print(f"    [RFE] Trenuję model na {len(feats)} cechach...")
    #             mdl = build_fn(len(feats))
    #             es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
    #             mdl.fit(Xtr[feats].values, ytr,
    #                     validation_data=(Xval[feats].values, yval),
    #                     epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    #
    #             p0 = predict_proba_binary(mdl, Xval[feats])
    #             base_m = eval_metrics(yval, p0)
    #             print(f"       Wyniki: AUC={base_m['auc']:.4f}, ACC={base_m['accuracy']:.4f}, F1={base_m['f1']:.4f}")
    #
    #             # PFI w aktualnym podzbiorze
    #             rng = np.random.default_rng(123)
    #             deltas = {}
    #             for f in feats:
    #                 Xp = Xval[feats].copy()
    #                 Xp[f] = rng.permutation(Xp[f].values)
    #                 p = predict_proba_binary(mdl, Xp)
    #                 m = eval_metrics(yval, p)
    #                 deltas[f] = base_m["auc"] - m["auc"]
    #
    #             history.append({
    #                 "n_features": len(feats),
    #                 "features": ",".join(feats),
    #                 "auc": base_m["auc"],
    #                 "accuracy": base_m["accuracy"],
    #                 "f1": base_m["f1"],
    #             })
    #
    #             if base_m["auc"] > best_auc:
    #                 best_auc = base_m["auc"]
    #                 best_subset = feats.copy()
    #
    #             # usuń cechę o najmniejszym wpływie
    #             worst = sorted(deltas.items(), key=lambda kv: kv[1])[0][0]
    #             print(f"       Usuwam {worst} (najmniejszy wpływ ΔAUC={deltas[worst]:.6f})")
    #             feats.remove(worst)
    #
    #             del mdl; gc.collect()
    #
    #         hist_df = pd.DataFrame(history)
    #         hist_df.to_csv("diagnostics/rfe_history.csv", index=False)
    #         print(">>> [DIAG] Zapisano rfe_history.csv")
    #
    #         fig, ax = plt.subplots(figsize=(7,5))
    #         ax.plot(hist_df["n_features"], hist_df["auc"], marker="o")
    #         ax.set_xlabel("Liczba cech")
    #         ax.set_ylabel("AUC (walidacja)")
    #         ax.set_title("RFE sterowane PFI – krzywa AUC")
    #         ax.invert_xaxis()
    #         fig.tight_layout()
    #         fig.savefig("diagnostics/rfe_auc_curve.png", dpi=160)
    #         plt.close(fig)
    #         print(">>> [DIAG] Zapisano rfe_auc_curve.png")
    #
    #         return hist_df, best_subset
    #
    #     rfe_hist, rfe_best = run_rfe_pfi(
    #         build_fn=build_model_diag,
    #         Xtr=Xtr_df, ytr=y_train,
    #         Xval=Xte_df, yval=y_test,
    #         start_feats=feature_cols,
    #         min_features=max(5, int(len(feature_cols)*0.3)),
    #         epochs=8, batch_size=1024, patience=2
    #     )
    #
    #     with open("diagnostics/rfe_best_features.txt", "w", encoding="utf-8") as f:
    #         for feat in rfe_best:
    #             f.write(feat + "\n")
    #     print(">>> [DIAG] Zapisano rfe_best_features.txt")
    #
    #     print("\n=== DIAGNOSTYKA ZAKOŃCZONA ===")
    #     print("Top SHAP (mean|SHAP|):")
    #     print(shap_importance.head(10))
    #     print("\nTop PFI (ΔAUC):")
    #     print(pfi_df.head(10))
    #     print("\nRFE – najlepszy podzbiór cech:")
    #     print(r   fe_best)

from labels import calc_label10
from functools import partial

label_partials = [
    partial(calc_label10, T=70, alpha=0.53, use_atr=True),
    partial(calc_label10, T=61, alpha=0.50, use_atr=False)
]

data_path = "data/training_data"
features_filename = "features00"

batch_size = 128

for label in label_partials:
    for _ in range(10):
        train(data_path, label, features_filename, batch_size)



