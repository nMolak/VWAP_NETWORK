import ccxt
import time
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from pathlib import Path
import pandas as pd
import numpy as np
import os

from calculations import *
from utils import logprint


#Funkcja musi przyjmować i zwracać df
def modify_all_csv(relative_data_path, func):
    path = Path(relative_data_path)

    for file in path.iterdir():
        if file.is_file() and file.suffix == ".csv":
            func_name = getattr(func, "__name__", repr(func))

            print(f"Obecnie przetwarzam plik {file.name} w folderze {path} - wykonuję funkcję {func_name}")
            df = pd.read_csv(file)
            df = func(df)
            df.to_csv(file, index=False)
            print(f"Przetwarzanie zakończone!")


from parameters import *
from filters import apply_filters


def fetch_ohlcv_df(ticker, interval, start_year, start_month, start_day):
    exchange = ccxt.binance()
    exchange.load_markets()

    # zamieniamy np. "2023-01-01 00:00:00" na timestamp w ms
    since = int(datetime(start_year, start_month, start_day).timestamp() * 1000)
    limit = 1000
    all_data = []

    while True:
        ohlcv = exchange.fetch_ohlcv(ticker, timeframe=interval, since=since, limit=limit)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # przesuwamy się dalej
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def get_prefix_upper(s: str) -> str:
    return s.split("_", 1)[0].upper()


def iterate_over_folder_and_save(tickers, interval, start_year, start_month, start_day, path, log=False):
    folder_path = Path(path)
    folder_path.mkdir(parents=True, exist_ok=True)

    logprint(f"Będę działał w folderze {path} - przechodzę do pętli")
    for ticker in tickers:
        logprint(f"Ściągam teraz dane {ticker}")
        df = fetch_ohlcv_df(ticker, interval, start_year, start_month, start_day)

        safe_ticker = ticker.replace("/", "_")
        logprint(f"Safe ticker: {safe_ticker}")

        ticker_dir = folder_path / safe_ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        new_filename = f"{safe_ticker.upper()}_{interval}_data.csv"
        file_path = ticker_dir / new_filename

        logprint(f"Przechodzę do zapisywania jako {file_path}")
        df.to_csv(file_path, index=False)

        logprint(f"Do folderu {ticker_dir} zapisano plik {new_filename}")
        logprint(f"Długość ramki danych: {len(df)} | "
                 f"najwcześniejsza data: {df.iloc[0]['timestamp']} | "
                 f"najpóźniejsza: {df.iloc[-1]['timestamp']}")




from pathlib import Path
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    """Zwraca kopię df z kolumną 'timestamp' (ms int64), posortowaną i bez duplikatów."""
    df = df.copy()

    # 1) Źródła czasu: kolumna 'timestamp' / 'time' / index datetime
    if 'timestamp' in df.columns:
        ts = df['timestamp']
    elif 'time' in df.columns:
        ts = df['time']
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    else:
        raise ValueError("Brak kolumny czasu ('timestamp' lub 'time') i brak DatetimeIndex.")

    # 2) Normalizacja → ms int64
    if np.issubdtype(ts.dtype, np.datetime64):
        ts_ms = ts.view('int64') // 1_000_000  # ns → ms
    else:
        # Spróbuj sparsować stringi/obiekty na daty (jeśli nie są liczbami)
        if ts.dtype == object:
            try:
                ts_parsed = pd.to_datetime(ts, utc=True, errors='raise')
                ts_ms = ts_parsed.view('int64') // 1_000_000
            except Exception:
                # Zakładamy, że to już ms/s – spróbujmy rzutować na int
                ts_num = pd.to_numeric(ts, errors='coerce')
                if ts_num.isna().any():
                    raise ValueError("Nie mogę zinterpretować kolumny czasu jako daty ani liczby.")
                # Heurystyka: jeśli wygląda na sekundy, przemnożymy do ms
                if ts_num.max() < 10**12:
                    ts_ms = (ts_num.astype('int64') * 1_000).astype('int64')
                else:
                    ts_ms = ts_num.astype('int64')
        else:
            # liczbowy typ: sprawdź czy sekundy czy ms
            ts_num = ts.astype('int64')
            if ts_num.max() < 10**12:
                ts_ms = (ts_num * 1_000).astype('int64')
            else:
                ts_ms = ts_num

    df['timestamp'] = ts_ms.astype('int64')

    # 3) Porządkowanie i deduplikacja
    before = len(df)
    df = df.sort_values('timestamp').drop_duplicates('timestamp', keep='first')
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"[timestamp] usunieto duplikaty: {dropped}")

    return df

def iterate_over_folder_and_save_features(
    data_path,
    calc_func,
    log: bool = True,
    filename_prefix: str = "features",
    compression: str = "snappy"
):
    """
    Przechodzi po folderach tickerów (każdy zawiera 1 plik CSV),
    liczy featury funkcją `calc_func(df)` i zapisuje do Parquet z kolumną 'timestamp'.

    Wymagania:
      - CSV musi dać się znormalizować do posiadania kolumny 'timestamp' (ms int64)
      - calc_func zwraca dict lub DataFrame; długość = długość wejściowego df po normalizacji.
    """
    def logprint(*args, **kwargs):
        if log:
            logger.info(" ".join(str(a) for a in args))

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Ścieżka nie istnieje: {data_path}")

    for ticker_folder in sorted([p for p in data_path.iterdir() 
                             if p.is_dir() and not p.name.startswith(".")]):

        logprint(f"===> Przechodzę do folderu: {ticker_folder}")

        # pokaż jakie pliki są w folderze
        files_in_folder = list(ticker_folder.iterdir())
        logprint(f"[{ticker_folder.name}] zawartość folderu: {[f.name for f in files_in_folder]}")

        # szukamy pliku CSV
        csv_file = next(ticker_folder.glob("*.csv"), None)

        if not csv_file:
            logprint(f"[{ticker_folder.name}] !!! Nie znaleziono pliku CSV w {ticker_folder}")
            raise ValueError(
                f"[{ticker_folder.name}] Brak pliku CSV w folderze {ticker_folder}. "
                f"Szukano wzorca '*.csv', znaleziono: {[f.name for f in files_in_folder]}")

        logprint(f"[{ticker_folder.name}] wczytuję plik CSV: {csv_file.name}")
        df = pd.read_csv(csv_file)


        # Upewnij się, że mamy timestamp w ms i brak duplikatów
        df = _ensure_timestamp_col(df)
        n_df = len(df)
        t0, t1 = df['timestamp'].iloc[0], df['timestamp'].iloc[-1]
        logprint(f"[{ticker_folder.name}] zakres: {t0} → {t1} (ms), n={n_df}")

        # Oblicz featury
        out = calc_func(df.copy())
        if isinstance(out, dict):
            features_df = pd.DataFrame(out, index=df.index)
        elif isinstance(out, pd.DataFrame):
            features_df = out.copy()
        else:
            raise TypeError("calc_func musi zwracać dict lub pandas.DataFrame")

        # Walidacja długości i indeksu
        if len(features_df) != n_df:
            # Jeżeli calc_func samo dorzuciło timestamp i można zmergować – spróbujmy:
            can_merge = 'timestamp' in features_df.columns
            if can_merge:
                tmp = df[['timestamp']].merge(
                    features_df, on='timestamp', how='inner', validate='one_to_one'
                )
                logprint(f"[{ticker_folder.name}] align po 'timestamp': n={len(tmp)} (df={n_df})")
                if len(tmp) != n_df:
                    raise ValueError(
                        f"[{ticker_folder.name}] Po align n_features={len(tmp)} != n_df={n_df}. "
                        f"Ujednolić pipeline (dropna/obcięcia) przed calc_func.")
                # Po align – zachowujemy kolejność jak w df
                features_df = tmp.set_index(df.index)
            else:
                raise ValueError(
                    f"[{ticker_folder.name}] Długość outputu calc_func ({len(features_df)}) "
                    f"≠ długości df ({n_df}) i brak kolumny 'timestamp' w featurach do merge.")

        # Wstrzyknij timestamp jako pierwszą kolumnę – zawsze na bazie df
        if 'timestamp' in features_df.columns:
            features_df = features_df.drop(columns=['timestamp'])

        # używamy datetime64 z df, żeby było spójne w całym pipeline
        features_df.insert(0, 'timestamp', pd.to_datetime(df['timestamp'], errors='coerce'))

        # Zapis
        feature_folder = ticker_folder / "features"
        feature_folder.mkdir(parents=True, exist_ok=True)

        existing = sorted([f for f in feature_folder.iterdir() if f.suffix == ".parquet"])
        features_number = len(existing)
        filepath = feature_folder / f"{filename_prefix}{features_number:02d}.parquet"

        features_df.to_parquet(filepath, index=False, compression=compression)

        # Log końcowy: len(df) vs len(parquet)
        n_parq = len(features_df)
        logprint(f"[{ticker_folder.name}] Zapisano: {filepath.name} | len(df)={n_df}, len(parquet)={n_parq}")
        if n_parq != n_df:
            raise AssertionError(
                f"[{ticker_folder.name}] len(parquet) ({n_parq}) != len(df) ({n_df}) – sprawdź pipeline!"
                )


# from calculations import calc_indicators
# calc_func = calc_indicators
# iterate_over_folder_and_save_features(
#     data_path="data/training_data",
#     calc_func=calc_func,
#     log=True
# )


def remove_all_features(data_path, log=True):
    data_path = Path(data_path)
    for ticker_folder in data_path.iterdir():
        if ticker_folder.is_dir():
            feature_folder = ticker_folder / "features"
            if feature_folder.exists():
                for parquet_file in feature_folder.glob("*.parquet"):
                    parquet_file.unlink()
                    if log:
                        print(f"Usunięto {parquet_file}")


def add_VWAP(df, sigma_mult=2.15) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    hl2 = (df["high"] + df["low"]) / 2

    # grupowanie po dacie (bez zmiany indeksu)
    grouped_date = df["timestamp"].dt.date
    volumesum = df["volume"].groupby(grouped_date).cumsum()
    v2sum     = (df["volume"] * hl2**2).groupby(grouped_date).cumsum()

    vwap = calc_vwap(df)
    variance = v2sum / volumesum - vwap**2
    sigma = variance.clip(lower=0) ** 0.5

    df["vwap"] = vwap
    df[f"sigma_{sigma_mult}"] = sigma  # <--- DODAJ TO
    df[f"vwap_plus_{sigma_mult}_sigma"] = vwap + sigma_mult * sigma
    df[f"vwap_minus_{sigma_mult}_sigma"] = vwap - sigma_mult * sigma

    return df

def add_indicators(df) -> pd.DataFrame:
    ind = calc_indicators(df)
    for k, v in ind.items():
        df[f"feature_{k}"] = v
    return df

def analyze_indicators(df, filters, log=False):

    df = df.copy()
    df = apply_filters(df, filters)

    ind = calc_indicators(df)

    # --- analiza rozkładu ---
    for name, series in ind.items():
        s = series.dropna()  # wyrzuć NaN-y na początku
        if len(s) == 0:
            print(f"{name}: brak danych\n")
            continue

        print(f"--- {name} ---")
        print(f"  min     = {s.min():.6f}")
        print(f"  max     = {s.max():.6f}")
        print(f"  mean    = {s.mean():.6f}")
        print(f"  median  = {s.median():.6f}")
        print(f"  std     = {s.std():.6f}")
        print(f"  skew    = {s.skew():.6f}")   # skośność
        print(f"  kurt    = {s.kurt():.6f}")   # kurtoza
        q25, q75 = s.quantile([0.25, 0.75])
        print(f"  IQR     = {q75 - q25:.6f}")
        print(f"  count   = {len(s)}\n")



def get_info(df, name: str, sigma: float, filters: list):
    df_len = len(df)
    dfc = df.copy()
    print(f"Wczytałem ramkę danych {name} - początkowa długość ramki: {df_len}")

    dfc = apply_filters(dfc, filters)

    # --- maski ekstremów ---
    mask_above = dfc['close'] > dfc[f"vwap_plus_{sigma}_sigma"]
    mask_below = dfc['close'] < dfc[f"vwap_minus_{sigma}_sigma"]
    mask_extreme = mask_above | mask_below

    df_extreme = dfc[mask_extreme].copy()

    above_sigma = mask_above.sum()
    below_sigma = mask_below.sum()
    extreme_vals = above_sigma + below_sigma

    print(f"Rekordów nad sigmą jest {above_sigma}")
    print(f"Rekordów pod sigmą jest {below_sigma}")

    if extreme_vals > 0:
        above_sigma_percentage = (above_sigma / extreme_vals) * 100
        below_sigma_percentage = (below_sigma / extreme_vals) * 100
    else:
        above_sigma_percentage = below_sigma_percentage = 0

    print(f"Rozkład above/below to: {above_sigma_percentage:.2f}% / {below_sigma_percentage:.2f}%")
    print(f"Łącznie {extreme_vals} świec jest ekstremalnych - jest to {(extreme_vals/df_len) * 100:.2f}% wszystkich wartości.")
    print(f"Najstarsza data: {df.iloc[0]['timestamp']}, najpóźniejsza: {df.iloc[-1]['timestamp']}\n")

    return df_extreme, extreme_vals



def check_features_correlation(df: pd.DataFrame, threshold: float = 0.8):
    # wybieramy kolumny z prefixem "feature_"
    features = [col for col in df.columns if col.startswith("feature_")]

    if not features:
        print("Brak kolumn z prefiksem 'feature_'")
        return None

    # macierz korelacji
    corr_matrix = df[features].corr()

    # maska do uniknięcia diagonali i duplikatów
    mask = np.triu(np.ones(corr_matrix.shape), k=0).astype(bool)
    corr_vals = corr_matrix.where(~mask).abs().stack()

    # metryki
    mean_corr = corr_vals.mean()
    max_corr = corr_vals.max()
    high_corr_ratio = (corr_vals > 0.7).mean()

    print(f"Średnia bezwzględna korelacja: {mean_corr:.3f}")
    print(f"Maksymalna bezwzględna korelacja: {max_corr:.3f}")
    print(f"Odsetek par z korelacją > 0.7: {high_corr_ratio:.2%}")

    # wypisz pary powyżej progu
    high_pairs = corr_vals[corr_vals > threshold]
    if not high_pairs.empty:
        print(f"\nPary featurów z korelacją > {threshold:.0%}:")
        for (f1, f2), val in high_pairs.items():
            print(f"  {f1} ↔ {f2}: {val:.3f}")
    else:
        print(f"\nBrak par z korelacją > {threshold:.0%}")
    print("\n")

    return corr_matrix

def get_correlation_info(path):
    data_path = Path(path)
    for file in data_path.iterdir():
        if file.is_file() and file.name.endswith(".csv"):
            print(f"Obecnie analizuję korelację w pliku {file}")
            df = pd.read_csv(file)
            check_features_correlation(df)


def analyze_labels(df, filters, N_list=[30, 40, 50], labels=None):
    """
    Analizuje wybrane etykiety (lista funkcji w argumencie labels).
    Funkcje labelujące przekazujemy jako partial (z wklejonymi parametrami).
    """
    if not labels:
        raise ValueError("Nie podano żadnych funkcji etykiet do analizy")

    from filters import filters as global_filters

    # --- licz labele na pełnym df (bez filtrów) ---
    print(">>> Start analizy etykiet (przed filtrami)")
    print(f"Pełny df: {len(df)} rekordów")

    df = df.copy()
    df = add_VWAP(df, sigma_val)

    all_labels = {N: {} for N in N_list}
    for idx_N, N in enumerate(N_list, 1):
        print(f"[{idx_N}/{len(N_list)}] Liczenie etykiet dla okna N={N}...")
        for lab_func in labels:
            series = lab_func(df, N=N)  # parametry już w partial
            name = lab_func.func.__name__ if hasattr(lab_func, "func") else lab_func.__name__
            all_labels[N][name] = series

    # --- ewentualnie na końcu zastosuj filtry do df ---
    if filters:
        print(">>> Zastosowanie filtrów...")
        df = apply_filters(df.copy(), filters, True)
        print(f"Po filtrach: {len(df)} rekordów")

    # --- podsumowania ---
    def summarize(series, name):
        arr = series.dropna().to_numpy()
        counts = dict(zip(*np.unique(arr, return_counts=True)))
        total = sum(counts.values())
        if total == 0:
            print(f"{name}: brak danych\n")
            return
        percents = {k: 100*v/total for k, v in counts.items()}
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs)) if probs.size > 1 else 0
        dominant_class = max(counts, key=counts.get)
        dominant_share = percents[dominant_class]
        print(f"{name}:")
        for k in sorted(counts.keys()):
            print(f"  klasa {k}: {counts[k]} ({percents[k]:.1f}%)")
        print(f"  -> entropia: {entropy:.3f}, "
              f"dominująca klasa: {dominant_class} "
              f"({dominant_share:.1f}%)\n")

    print(">>> Start podsumowań statystyk (na pełnym df)")
    for N in N_list:
        print(f"\n===== Analiza etykiet (okno N={N}) =====")
        for lab_func in labels:
            name = lab_func.func.__name__ if hasattr(lab_func, "func") else lab_func.__name__
            summarize(all_labels[N][name], name)
        print("-----------------------------------------")

    print(">>> Analiza zakończona")
    return all_labels

def analyze_labels_with_filters(df, filters, label_functions):

    print(f">>> Start analizy, oryginalna liczba rekordów: {len(df)}")

    df = apply_filters(df, filters, True)

    # --- analiza labeli ---
    for func in label_functions:
        print(f"\n>>> Dodaję i analizuję etykiety: {func.__name__}")
        df_tmp = df.copy()
        df_tmp = func(df_tmp)  # dodaj kolumny labeli

        # znajdź świeżo dodane kolumny
        new_cols = [c for c in df_tmp.columns if c.startswith(func.__name__.replace("add_", ""))]
        if not new_cols:
            print(f"⚠️ Funkcja {func.__name__} nie dodała żadnych kolumn!")
            continue

        # analiza każdej kolumny
        for col in new_cols:
            arr = df_tmp[col].dropna().to_numpy()
            if arr.size == 0:
                print(f"{col}: brak danych\n")
                continue

            counts = dict(zip(*np.unique(arr, return_counts=True)))
            total = arr.size
            percents = {k: v/total*100 for k,v in counts.items()}
            probs = np.array(list(counts.values())) / total
            entropy = -np.sum(probs * np.log2(probs)) if probs.size > 1 else 0
            dominant_class = max(counts, key=counts.get)
            dominant_share = percents[dominant_class]

            print(f"{col}:")
            for k in sorted(counts.keys()):
                print(f"  klasa {k}: {counts[k]} ({percents[k]:.1f}%)")
            print(f"  entropia: {entropy:.3f}, dominująca klasa: {dominant_class} "
                  f"({dominant_share:.1f}%)\n")



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
#
# def analyze_various_sigma_values(data_path, sigma, start_hour=0, end_hour=10,
#                                  future_window=30, phi=0.002):
#     min_vals_list, max_vals_list = [], []
#     above_vals_list, below_vals_list = [], []
#
#     path = Path(data_path)
#     for file in path.iterdir():
#         if file.is_file() and file.suffix == ".csv":
#             df = pd.read_csv(file, parse_dates=["timestamp"]).copy()
#             print(f"Wczytałem plik {file}")
#             df = df.replace("", np.nan)
#             df = df.replace([np.inf, -np.inf], np.nan)
#
#             df = add_VWAP(df, sigma)
#
#             # upewnij się, że indeks jest datetime
#             if not isinstance(df.index, pd.DatetimeIndex):
#                 df.index = pd.to_datetime(df.index, errors="coerce")
#
#             # filtr godzin
#             df = df[~df["timestamp"].dt.hour.between(start_hour, end_hour)]
#
#
#             # --- maski ekstremów ---
#             mask_extreme = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
#                            (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
#             mask_cooldown = apply_cooldown(mask_extreme, seq_len=10, cooldown=30)
#             mask_final = mask_extreme & mask_cooldown
#             df = df[mask_final].copy()
#
#             entry_price = df["close"]
#
#             # przyszłe min/max jako wartości względne (% zmiana względem ceny wejściowej)
#             future_min = (df["close"].shift(-1).rolling(future_window, min_periods=1).min() / entry_price - 1) * 100
#             future_max = (df["close"].shift(-1).rolling(future_window, min_periods=1).max() / entry_price - 1) * 100
#
#             col_min = f"future_min_sigma_{sigma}"
#             col_max = f"future_max_sigma_{sigma}"
#             df[col_min] = np.nan
#             df[col_max] = np.nan
#
#             mask_above = df["close"] > df[f"vwap_plus_{sigma}_sigma"]
#             mask_below = df["close"] < df[f"vwap_minus_{sigma}_sigma"]
#
#             df.loc[mask_above, col_min] = future_min[mask_above]
#             df.loc[mask_below, col_max] = future_max[mask_below]
#
#             # % zmiany potrzebnej do powrotu do VWAP ± phi
#             vwap_upper = df["vwap"] * (1 + phi)
#             vwap_lower = df["vwap"] * (1 - phi)
#             df[f"to_vwap_phi_pct_{sigma}"] = np.nan
#             df.loc[mask_above, f"to_vwap_phi_pct_{sigma}"] = \
#                 (vwap_upper[mask_above] - df["close"][mask_above]) / df["close"][mask_above] * 100
#             df.loc[mask_below, f"to_vwap_phi_pct_{sigma}"] = \
#                 (vwap_lower[mask_below] - df["close"][mask_below]) / df["close"][mask_below] * 100
#
#             # zbierz dane
#             min_vals_list.append(df[col_min].values)
#             max_vals_list.append(df[col_max].values)
#             above_vals_list.append(df.loc[mask_above, f"to_vwap_phi_pct_{sigma}"].to_numpy())
#             below_vals_list.append(df.loc[mask_below, f"to_vwap_phi_pct_{sigma}"].to_numpy())
#
#     # --- scal listy tablic (poza pętlą) ---
#     min_all = np.concatenate(min_vals_list) if min_vals_list else np.array([])
#     max_all = np.concatenate(max_vals_list) if max_vals_list else np.array([])
#     above_all = np.concatenate(above_vals_list) if above_vals_list else np.array([])
#     below_all = np.concatenate(below_vals_list) if below_vals_list else np.array([])
#
#     def log_stats(name, arr):
#         if arr.size == 0:
#             print(f"[{name}] brak danych")
#             return
#         q = np.nanquantile(arr, [0, 0.25, 0.5, 0.75, 1])
#         print(f"\n[{name}] n={arr.size}")
#         print(f"  min={q[0]:.6f}, Q1={q[1]:.6f}, median={q[2]:.6f}, "
#               f"Q3={q[3]:.6f}, max={q[4]:.6f}")
#
#     print("\n=== Diagnostyka sigma =", sigma, "===")
#     log_stats("future_min", min_all)
#     log_stats("future_max", max_all)
#     log_stats("to_vwap_phi_above", above_all)
#     log_stats("to_vwap_phi_below", below_all)
#
#     return min_all, max_all, above_all, below_all

def analyze_sigma(df: pd.DataFrame, sigma: float, future_window: int = 30, phi: float = 0.002):
    """
    Analiza zachowania świec ekstremalnych względem VWAP ± sigma.
    Zakłada, że df ma kolumny 'vwap', 'close', 'vwap_plus_x_sigma', 'vwap_minus_x_sigma'.
    """
    entry_price = df["close"]

    # przyszłe min/max (% zmiana względem ceny wejścia)
    future_min = (df["close"].shift(-1).rolling(future_window, min_periods=1).min() / entry_price - 1) * 100
    future_max = (df["close"].shift(-1).rolling(future_window, min_periods=1).max() / entry_price - 1) * 100

    mask_above = df["close"] > df[f"vwap_plus_{sigma}_sigma"]
    mask_below = df["close"] < df[f"vwap_minus_{sigma}_sigma"]

    # ile % potrzeba do powrotu do VWAP ± phi
    vwap_upper = df["vwap"] * (1 + phi)
    vwap_lower = df["vwap"] * (1 - phi)

    to_vwap_above = ((vwap_upper[mask_above] - df.loc[mask_above, "close"]) / df.loc[mask_above, "close"]) * 100
    to_vwap_below = ((vwap_lower[mask_below] - df.loc[mask_below, "close"]) / df.loc[mask_below, "close"]) * 100

    # --- logi ---
    print(f"[Analiza sigma={sigma}]")
    print(f"Świece above: {mask_above.sum()}, below: {mask_below.sum()}")
    if not to_vwap_above.empty:
        print(f"  Powrót ABOVE → median={to_vwap_above.median():.3f}%, 90%={to_vwap_above.quantile(0.9):.3f}%")
    if not to_vwap_below.empty:
        print(f"  Powrót BELOW → median={to_vwap_below.median():.3f}%, 90%={to_vwap_below.quantile(0.9):.3f}%")
    if not future_min.empty:
        print(f"  future_min (okno {future_window}) → {future_min.median():.3f}% (mediana)")
    if not future_max.empty:
        print(f"  future_max (okno {future_window}) → {future_max.median():.3f}% (mediana)")
    print("-" * 60)

def read_model_names(folder_path, output_file="model_names.txt"):
    """
    Zapisuje nazwy modeli z folderu, posortowane wg accuracy (malejąco),
    do pliku tekstowego w katalogu roboczym.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} nie jest katalogiem!")

    model_names = os.listdir(folder_path)

    def extract_acc(name: str) -> float:
        try:
            parts = name.split("_")
            # accuracy to przedostatni element przed nazwą optymalizatora
            return float(parts[-2])
        except Exception:
            return -1.0  # jeśli nie uda się sparsować

    # sortowanie po accuracy malejąco
    model_names_sorted = sorted(model_names, key=extract_acc, reverse=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for name in model_names_sorted:
            f.write(name + "\n")

    print(f"Zapisano {len(model_names_sorted)} nazw do {output_file} (posortowanych po accuracy)")

#read_model_names("models")

import os

def remove_models_below_threshold(folder_path, min_acc: float = 0.59):
    """
    Usuwa modele z folderu, których accuracy < min_acc.
    Accuracy pobierane jest z nazwy pliku (przedostatni element przed optymalizatorem).
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} nie jest katalogiem!")

    removed = []
    kept = []

    def extract_acc(name: str) -> float:
        try:
            parts = name.split("_")
            return float(parts[-2])
        except Exception:
            return -1.0

    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        acc = extract_acc(name)
        if acc < min_acc:
            try:
                os.remove(path)
                removed.append(name)
            except Exception as e:
                print(f"⚠️ Nie udało się usunąć {name}: {e}")
        else:
            kept.append(name)

    print(f"Usunięto {len(removed)} modeli (accuracy < {min_acc}), pozostawiono {len(kept)}.")

    return removed, kept

#remove_models_below_threshold("models")
#read_model_names("models")


# tickers = ["BTC/USDT"]
# interval = "1m"
# start_year, start_month, start_day = 2021, 1, 1
# data_path = "data/1m/training_data"
#
#
# sigma_vals = [1.85, 2, 2.15, 2.30, 2.45, 2.60, 2.75]
#
# for sigma in sigma_vals:
#     analyze_various_sigma_values(data_path, sigma, 0,  12)

# tickers = ["ETH/USDT", "XRP/USDT", "LINK/USDT", "LTC/USDT", "SOL/USDT", "ADA/USDT", "AVAX/USDT"]
# interval = "1m"
# start_year = 2021
# start_month = 1
# start_day = 1
# data_path = "data/training_data"
# #iterate_over_folder_and_save(tickers, interval, start_year, start_month, start_day, data_path, log=True)
# from calculations import calc_indicators
# remove_all_features(data_path)
# iterate_over_folder_and_save_features(data_path, calc_indicators)

#
# data_path = r"data/1m/training_data"
# sigmas = [2.0, 2.15, 2.3, 2.45]
# path = Path(data_path)
#
# #filter_unique_extremes_in_window(5, "is_extreme")

# df = pd.read_csv(r"data/1m/training_data/XRP_USDT_1m_data.csv")
#
#
# sigma = 2.15
# df = add_VWAP(df, sigma)
# df["is_extreme"] = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
#                    (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
# functions = [add_label1]
# label_names = ["label1", "label2", "label3"]
# analyze_labels_with_filters(df, filters, functions)
#
# for sigma in sigmas:
#     df_lens = []
#     for file in path.iterdir():
#         if file.is_file() and file.suffix == ".csv":
#             df = pd.read_csv(file)
#             df = add_VWAP(df, sigma)
#
#             # dodajemy kolumnę is_extreme
#             df["is_extreme"] = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
#                                (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
#
#             df_extreme, df_len = get_info(df, file.name, sigma, filters)
#             analyze_sigma(df_extreme, sigma, future_window=30, phi=0.005)
#
#
#     print(f"Łącznie rekordów treningowych jest: {sum(df_lens)}")


#get_info(data_path)

#df = pd.read_csv("data/1m/test_data/ADA_USDT_1m_data.csv")
#modify_all_csv(data_path, add_VWAP)
#df = 3add_VWAP(df, 2.15)
#analyze_indicators(df)

#modify_all_csv(data_path, add_indicators)

#path = r"data/1m/test_data"
#modify_all_csv(path, lambda df: add_VWAP(df, 2.15))
#modify_all_csv(path, lambda df: add_label1(df, phi=0.0075, N_list=[10, 20, 30]))


#check_features_correlation(df)

#get_correlation_info(data_path)

# path = Path(data_path)
#
# for file in path.iterdir():
#     if file.suffix == ".csv" and file.is_file():
#         print(f'Analizuję plik {file}')
#         df = pd.read_csv(file)
#         analyze_existing_labels(df, which=("label2",))




#get_info(data_path)

#data_path = r"data/1m/training_data"

#modify_all_csv(data_path, add_label1)
#modify_all_csv(data_path, add_label2)
#modify_all_csv(data_path, add_label3)

#df = pd.read_csv(f"data/1m/AVAX_USDT_1m_data.csv")
# print(f"Przed modyfikacją plik ma kolumny: {df.columns}")
# df = add_label1(df)
# df = add_label2(df)
# df = add_label3(df)
# df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
#print(f"Po modyfikacji plik ma kolumny: {df.columns}")
# df.to_csv(f"data/1m/AVAX_USDT_1m_data.csv", index=False)





def diagnose_csvs(data_path="data/1m/training_data"):
    path = Path(data_path)
    report = []

    for file in path.glob("*.csv"):
        df = pd.read_csv(file)

        # usuń śmieciowe kolumny
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        info = {}
        info["file"] = file.name
        info["rows"] = len(df)
        info["n_cols"] = len(df.columns)
        info["cols"] = set(df.columns)

        # daty (zakładam, że timestamp jest stringiem ISO lub datą)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            info["date_min"] = df["timestamp"].min()
            info["date_max"] = df["timestamp"].max()

        # ile NaN
        info["nan_rows"] = df.isna().any(axis=1).sum()

        # rozkład etykiet (jeśli istnieją)
        label_stats = {}
        for col in df.columns:
            if col.startswith("label"):
                counts = df[col].value_counts(dropna=True).to_dict()
                label_stats[col] = counts
        info["labels"] = label_stats

        report.append(info)

    # --- podsumowanie ---
    print(">>> Diagnostyka CSV-ów")
    print(f"Znaleziono {len(report)} plików\n")

    # sprawdź kolumny
    all_cols = [r["cols"] for r in report]
    common_cols = set.intersection(*all_cols)
    union_cols = set.union(*all_cols)

    print("Wszystkie pliki mają te wspólne kolumny:")
    print(sorted(list(common_cols)))
    print("\nPełny zestaw kolumn (mogą być różnice):")
    print(sorted(list(union_cols)))
    print()

    # raport per plik
    for r in report:
        print(f"Plik: {r['file']}")
        print(f"  Wiersze: {r['rows']}, Kolumny: {r['n_cols']}")
        if "date_min" in r:
            print(f"  Zakres dat: {r['date_min']} → {r['date_max']}")
        print(f"  Wiersze z ≥1 NaN: {r['nan_rows']}")
        if r["labels"]:
            print("  Rozkład etykiet:")
            for col, counts in r["labels"].items():
                print(f"    {col}: {counts}")
        print("-" * 40)


#diagnose_csvs()


def analyze_vwap_dist(df: pd.DataFrame,
                      model_path=r"C:\Users\norbe\PycharmProjects\GRU_2\data\models\vwap_model.h5"):
    # kopia i czyszczenie
    df = df.copy()
    df.replace(["-", np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[DEBUG] Początkowa liczba rekordów: {len(df)}")

    # filtracja seriami
    original_length = len(df)
    mask_extreme = (df["close"] > df["vwap_plus_3_sigma"]) | (df["close"] < df["vwap_minus_3_sigma"])
    mask_cooldown = apply_cooldown(mask_extreme, seq_len=10, cooldown=30)
    mask_final = mask_extreme & mask_cooldown
    df = df.loc[mask_final]
    print(f"[DEBUG] Po filtracji mask_final: {len(df)} / {original_length} obserwacji")

    if df.empty:
        print("[DEBUG] Brak danych po filtracji")
        return

    # kolumny cech
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].to_numpy()

    # model
    model = load_model(model_path)

    # predykcje
    y_pred = model.predict(X, verbose=0)
    y_class = np.argmax(y_pred, axis=1)

    # wybór tylko klasa 1
    mask = y_class == 1
    df_sel = df.loc[mask]
    print(f"[DEBUG] Po predykcji, klasa 1: {len(df_sel)} rekordów")

    if df_sel.empty:
        print("[DEBUG] Brak obserwacji z klasą 1")
        return

    # ile % brakuje do VWAP
    delta_pct = (df_sel["vwap"].to_numpy() - df_sel["close"].to_numpy()) / df_sel["close"].to_numpy() * 100

    # podział
    below = delta_pct[df_sel["close"].to_numpy() < df_sel["vwap"].to_numpy()]
    above = delta_pct[df_sel["close"].to_numpy() > df_sel["vwap"].to_numpy()]

    # debug: statystyki
    print("[DEBUG] Statystyki globalne delta_pct:")
    print(pd.Series(delta_pct).describe(percentiles=[0.01,0.25,0.5,0.75,0.99]))

    if len(below) > 0:
        print("[DEBUG] Statystyki świec poniżej VWAP:")
        print(pd.Series(below).describe(percentiles=[0.01,0.25,0.5,0.75,0.99]))
    if len(above) > 0:
        print("[DEBUG] Statystyki świec powyżej VWAP:")
        print(pd.Series(above).describe(percentiles=[0.01,0.25,0.5,0.75,0.99]))

    qs = [0.25, 0.5, 0.75]

    # osobny histogram: poniżej VWAP
    if len(below) > 0:
        plt.figure(figsize=(8,5))
        plt.hist(below, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title("Świece poniżej VWAP (klasa 1)")
        plt.xlabel("Procent brakujący do VWAP")
        plt.ylabel("Liczność")
        plt.xlim(-4, 4)
        q_vals = np.quantile(below, qs)
        for q, v in zip(qs, q_vals):
            plt.axvline(v, color="red", linestyle="--")
            plt.text(v, plt.ylim()[1]*0.9, f"{int(q*100)}%={v:.2f}%", rotation=90, color="red")
        plt.show()

    # osobny histogram: powyżej VWAP
    if len(above) > 0:
        plt.figure(figsize=(8,5))
        plt.hist(above, bins=50, color="lightgreen", edgecolor="black", alpha=0.7)
        plt.title("Świece powyżej VWAP (klasa 1)")
        plt.xlabel("Procent brakujący do VWAP")
        plt.ylabel("Liczność")
        plt.xlim(-4, 4)
        q_vals = np.quantile(above, qs)
        for q, v in zip(qs, q_vals):
            plt.axvline(v, color="red", linestyle="--")
            plt.text(v, plt.ylim()[1]*0.9, f"{int(q*100)}%={v:.2f}%", rotation=90, color="red")
        plt.show()

    return below, above


#df = pd.read_csv(r"data/1m/training_data/BTC_USDT_1m_data.csv")
#print(df.columns.tolist())
#add_VWAP(df, 2.51)
#df = df.rename(columns={"vwap_plus_2_51_sigma": "vwap_plus_3_sigma"})
#df = df.rename(columns={"vwap_minus_2_51_sigma": "vwap_minus_3_sigma"})

#analyze_vwap_dist(df, r"C:\Users\norbe\PycharmProjects\GRU_2\data\models\vwap_model.h5")

#-----------------------------------------------------

#modify_all_csv(data_path, add_indicators)

#-------------------------------------------------------

from labels import all_labels

# df = pd.read_csv("data/training_data/BTC_USDT_1m_data.csv")
# df = add_VWAP(df, sigma_val)
# # analyze_labels(df, labels=all_labels, do_filter=True)
#
# #-------------------------------------------------------
#
# from filters import filter_extreme_values
# from filters import filter_hours
# from filters import filter_clean
# from filters import filter_remove_long_series
#
# start_hour = 0
# end_hour = 8
#
# N = 30
# max_len = 10
# sigma = sigma_val
#
# filters = [filter_clean(),
#            filter_hours(start_hour, end_hour),
#            filter_remove_long_series(max_len, sigma_val),
#            filter_extreme_values(sigma)]
#
# from functools import partial
#
# def add_label_wrapper(df, func):
#     ser = func(df)
#     colname = func.func.__name__
#     df = df.copy()
#     df[colname] = ser
#     return df
#
# # budujemy listę z nazwami
# wrapped_labels = []
# for f in all_labels:
#     w = partial(add_label_wrapper, func=f)
#     w.__name__ = f.func.__name__  # przypisanie nazwy, np. "calc_label4"
#     wrapped_labels.append(w)
#
# analyze_labels_with_filters(df, filters, wrapped_labels)
#

