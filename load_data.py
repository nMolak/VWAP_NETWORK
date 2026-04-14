"""
Funkcje ladowania danych, diagnostyki i analizy VWAP/labeli.

Obsluguje pobieranie OHLCV z gield, generowanie featurow,
diagnostyke CSV-ow i analize rozkladow etykiet.
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calculations import calc_indicators, calc_vwap
from filters import apply_filters
from parameters import LOG_ENABLED, features_data_mode, sigma_val
from utils import (
    check_feature_consistency,
    display_dataframe,
    get_func_name,
    make_logprint,
)


def modules_diagnostics() -> None:
    print("=== SYSTEM / PYTHON ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")

    packages = [
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "joblib",
        "matplotlib",
        "ccxt",
        "pandas_ta",
    ]

    print("=== PACKAGE VERSIONS ===")
    for pkg in packages:
        try:
            module = importlib.import_module(pkg)
            ver = getattr(module, "__version__", "no __version__ attr")
            print(f"{pkg:15s}: {ver}")
        except ImportError:
            print(f"{pkg:15s}: NOT INSTALLED")
    print()

    try:
        import torch

        print("=== PYTORCH DETAILS ===")
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("PyTorch check failed:", e)


def modify_all_csv(relative_data_path: str, func) -> None:
    path = Path(relative_data_path)

    for file in path.iterdir():
        if file.is_file() and file.suffix == ".csv":
            func_name = getattr(func, "__name__", repr(func))
            print(
                f"Obecnie przetwarzam plik {file.name} w folderze {path} "
                f"- wykonuje funkcje {func_name}"
            )
            df = pd.read_csv(file)
            df = func(df)
            df.to_csv(file, index=False)
            print("Przetwarzanie zakonczone!")


def repair_volume(
    df: pd.DataFrame, N: int = 10, log: bool = True
) -> pd.DataFrame:
    """
    Uzupelnia zerowe wolumeny na podstawie sredniej z N najblizszych
    niezerowych wartosci po lewej i po prawej stronie.
    """
    logprint = make_logprint(log)
    t0 = time.time()

    assert "volume" in df.columns, "Brak kolumny 'volume' w ramce danych."
    df = df.copy()
    vol = df["volume"].to_numpy(dtype=float)
    n = len(vol)
    affected = 0

    zero_idx = np.where(vol == 0)[0]
    if len(zero_idx) == 0:
        logprint("[repair_volume] Brak zerowych wolumenow — nic nie naprawiono.")
        return df

    for i in zero_idx:
        left_vals, right_vals = [], []

        j = i - 1
        while j >= 0 and len(left_vals) < N:
            if vol[j] != 0:
                left_vals.append(vol[j])
            j -= 1

        j = i + 1
        while j < n and len(right_vals) < N:
            if vol[j] != 0:
                right_vals.append(vol[j])
            j += 1

        neigh = left_vals + right_vals
        if neigh:
            vol[i] = np.mean(neigh)
            affected += 1

    df["volume"] = vol

    dt = time.time() - t0
    logprint(
        f"[repair_volume] Naprawiono {affected:,} zerowych wolumenow "
        f"z {n:,} ({affected / n:.4%})."
    )
    logprint(f"[repair_volume] Czas wykonania: {dt:.3f} s")

    return df


def fetch_ohlcv_df(
    ticker: str,
    interval: str,
    start_year: int,
    start_month: int,
    start_day: int,
) -> pd.DataFrame:
    exchange = ccxt.binance()
    exchange.load_markets()

    since = int(datetime(start_year, start_month, start_day).timestamp() * 1000)
    limit = 1000
    all_data = []

    while True:
        ohlcv = exchange.fetch_ohlcv(
            ticker, timeframe=interval, since=since, limit=limit
        )
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def iterate_over_folder_and_save(
    tickers: list[str],
    interval: str,
    start_year: int,
    start_month: int,
    start_day: int,
    path: str,
    log: bool = False,
) -> None:
    folder_path = Path(path)
    folder_path.mkdir(parents=True, exist_ok=True)

    logprint = make_logprint(log)

    logprint(f"Bede dzialal w folderze {path} - przechodze do petli")
    for ticker in tickers:
        logprint(f"Sciagam teraz dane {ticker}")
        df = fetch_ohlcv_df(ticker, interval, start_year, start_month, start_day)

        safe_ticker = ticker.replace("/", "_")
        logprint(f"Safe ticker: {safe_ticker}")

        ticker_dir = folder_path / safe_ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        new_filename = f"{safe_ticker.upper()}_{interval}_data.csv"
        file_path = ticker_dir / new_filename

        logprint(f"Przechodze do zapisywania jako {file_path}")
        df.to_csv(file_path, index=False)

        logprint(
            f"Do folderu {ticker_dir} zapisano plik {new_filename}"
        )
        logprint(
            f"len(df = {len(df)} | "
            f"najwczesniejsza data: {df.iloc[0]['timestamp']} | "
            f"najpozniejsza: {df.iloc[-1]['timestamp']}"
        )


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    """Upewnia sie, ze kolumna 'timestamp' istnieje i jest typu datetime64[ns]."""
    df = df.copy()

    if "timestamp" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "timestamp"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df["timestamp"] = df.index
        else:
            raise ValueError("Brak kolumny czasu 'timestamp' ani DatetimeIndex")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    return df


def iterate_over_folder_and_save_features(
    data_path: str,
    calc_func,
    do_backup: bool = False,
    filename_prefix: str = "features",
    compression: str = "snappy",
    log: bool = True,
) -> None:
    """
    Przechodzi po folderach tickerow (kazdy zawiera 1 plik CSV),
    liczy featury funkcja `calc_func(df)` i zapisuje do Parquet
    z kolumna 'timestamp'.
    """
    logprint = make_logprint(log)

    def get_csv(ticker_folder: Path) -> pd.DataFrame:
        logprint(f"===> Przechodze do folderu: {ticker_folder}")

        files_in_folder = list(ticker_folder.iterdir())
        logprint(
            f"[{ticker_folder.name}] zawartosc folderu: "
            f"{[f.name for f in files_in_folder]}"
        )

        csv_file = next(ticker_folder.glob("*.csv"), None)

        if not csv_file:
            logprint(
                f"[{ticker_folder.name}] !!! Nie znaleziono pliku CSV "
                f"w {ticker_folder}"
            )
            raise ValueError(
                f"[{ticker_folder.name}] Brak pliku CSV w folderze "
                f"{ticker_folder}. Szukano wzorca '*.csv', znaleziono: "
                f"{[f.name for f in files_in_folder]}"
            )

        logprint(f"[{ticker_folder.name}] wczytuje plik CSV: {csv_file.name}")
        df = pd.read_csv(csv_file)

        df = _ensure_timestamp_col(df)
        n_df = len(df)
        t0, t1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]
        logprint(f"[{ticker_folder.name}] zakres: {t0} -> {t1} (ms), n={n_df}")

        return df

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Sciezka nie istnieje: {data_path}")

    if features_data_mode == "parquet":
        for ticker_folder in sorted(
            p for p in data_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ):
            df = get_csv(ticker_folder)
            n_df = len(df)

            out = calc_func(df)
            if isinstance(out, dict):
                features_df = pd.DataFrame(out, index=df.index)
            elif isinstance(out, pd.DataFrame):
                features_df = out.copy()
            else:
                raise TypeError("calc_func musi zwracac dict lub pandas.DataFrame")

            features_df.columns = [
                f"feature_{col}" if col != "timestamp" else col
                for col in features_df.columns
            ]

            if len(features_df) != n_df:
                logprint(
                    f"Dlugosc ramki zbudowanej z funkcji {calc_func.__name__} "
                    f"nie zgadza sie: {len(features_df)} != {n_df}"
                )
                can_merge = "timestamp" in features_df.columns
                if can_merge:
                    tmp = df[["timestamp"]].merge(
                        features_df, on="timestamp", how="inner",
                        validate="one_to_one",
                    )
                    logprint(
                        f"[{ticker_folder.name}] align po 'timestamp': "
                        f"n={len(tmp)} (df={n_df})"
                    )
                    if len(tmp) != n_df:
                        raise ValueError(
                            f"[{ticker_folder.name}] Po align n_features="
                            f"{len(tmp)} != n_df={n_df}. Ujednolicic pipeline "
                            f"(dropna/obciecia) przed calc_func."
                        )
                    features_df = tmp.set_index(df.index)
                else:
                    raise ValueError(
                        f"[{ticker_folder.name}] Dlugosc outputu calc_func "
                        f"({len(features_df)}) != dlugosci df ({n_df}) "
                        f"i brak kolumny 'timestamp' w featurach do merge."
                    )

            if "timestamp" in features_df.columns:
                features_df = features_df.drop(columns=["timestamp"])

            features_df.insert(0, "timestamp", df["timestamp"])

            feature_folder = ticker_folder / "features"
            feature_folder.mkdir(parents=True, exist_ok=True)

            existing = [
                f for f in feature_folder.iterdir() if f.suffix == ".parquet"
            ]
            features_number = len(existing)
            filepath = (
                feature_folder / f"{filename_prefix}{features_number:02d}.parquet"
            )

            features_df.to_parquet(filepath, index=False, compression=compression)

            n_parq = len(features_df)
            logprint(
                f"[{ticker_folder.name}] Zapisano: {filepath.name} | "
                f"len(df)={n_df}, len(parquet)={n_parq}"
            )
            if n_parq != n_df:
                raise AssertionError(
                    f"[{ticker_folder.name}] len(parquet) ({n_parq}) != "
                    f"len(df) ({n_df}) - sprawdz pipeline!"
                )

    elif features_data_mode == "dataframe":
        for ticker_folder in sorted(
            p for p in data_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ):
            df = get_csv(ticker_folder)
            n_df = len(df)

            out = calc_func(df)
            if isinstance(out, dict):
                features_df = pd.DataFrame(out, index=df.index)
            elif isinstance(out, pd.DataFrame):
                features_df = out.copy()
            else:
                raise TypeError("calc_func musi zwracac dict lub pandas.DataFrame")

            features_df.columns = [
                f"feature_{col}" if col != "timestamp" else col
                for col in features_df.columns
            ]

            if len(features_df) != n_df:
                logprint(
                    f"Dlugosc ramki zbudowanej z funkcji {calc_func.__name__} "
                    f"nie zgadza sie: {len(features_df)} != {n_df}"
                )
                if "timestamp" in features_df.columns:
                    tmp = df[["timestamp"]].merge(
                        features_df, on="timestamp", how="inner",
                        validate="one_to_one",
                    )
                    logprint(
                        f"[{ticker_folder.name}] Align po timestamp: "
                        f"n={len(tmp)} (df={n_df})"
                    )
                    features_df = tmp.set_index(df.index)
                else:
                    raise ValueError(
                        f"[{ticker_folder.name}] Dlugosc outputu calc_func "
                        f"({len(features_df)}) != dlugosci df ({n_df}) "
                        f"i brak kolumny 'timestamp' do wyrownania."
                    )
            else:
                logprint(
                    f"[{ticker_folder.name}] Dlugosc df i features_df zgodna: "
                    f"{n_df}"
                )

            merged_df = pd.concat([df, features_df], axis=1)

            csv_file = next(ticker_folder.glob("*.csv"), None)
            if csv_file is None:
                raise FileNotFoundError(
                    f"[{ticker_folder.name}] Brak pliku CSV do nadpisania."
                )

            if do_backup:
                backup_path = csv_file.with_suffix(".bak.csv")
                csv_file.rename(backup_path)
                merged_df.to_csv(csv_file, index=False)
                logprint(
                    f"[{ticker_folder.name}] Dopisano kolumny feature_ "
                    f"i zapisano z powrotem do CSV ({csv_file.name})"
                )
            else:
                merged_df.to_csv(csv_file, index=False)


def diagnose_mean_and_variance(
    data_path: str, log: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    print(
        f"Wykonuje sie funkcja {get_func_name()} "
        f"- znajdujemy sie w sciezce {data_path}"
    )

    check_feature_consistency(data_path)

    data_path = Path(data_path)
    feature_cols = None
    logprint = make_logprint(log)

    if features_data_mode == "parquet":
        raise NotImplementedError("Tryb feature parquet obecnie niedostepny!")
    elif features_data_mode == "dataframe":
        local_stats_mean, local_stats_dev = {}, {}
        for ticker_folder in sorted(
            p for p in data_path.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ):
            local_stats_mean[ticker_folder.name] = []
            local_stats_dev[ticker_folder.name] = []
            csv = next(ticker_folder.glob("*.csv"), None)
            df = pd.read_csv(csv)
            logprint(f"Analizuje plik {csv}")

            feature_cols = [col for col in df.columns if col.startswith("feature_")]
            logprint(f"Oto featury: {feature_cols}")

            for feature in feature_cols:
                local_stats_mean[ticker_folder.name].append(np.mean(df[feature]))
                local_stats_dev[ticker_folder.name].append(np.std(df[feature]))

        if feature_cols is None:
            raise ValueError("feature_cols is None -> blad w funkcji!")

        mean_values = pd.DataFrame(local_stats_mean, index=feature_cols)
        dev_values = pd.DataFrame(local_stats_dev, index=feature_cols)

        display_dataframe(mean_values, title="SREDNIE LOKALNE", precision=4)
        display_dataframe(dev_values, title="ODCHYLENIA LOKALNE", precision=4)

        mean_values_std = mean_values.std(axis=1)
        dev_values_std = dev_values.std(axis=1)

        mean_values_std = pd.DataFrame(
            mean_values_std, columns=["mean_std_between_files"]
        )
        dev_values_std = pd.DataFrame(
            dev_values_std, columns=["dev_std_between_files"]
        )

        display_dataframe(
            mean_values_std,
            title="Odchylenia srednich wzgledem plikow",
        )
        display_dataframe(
            dev_values_std,
            title="Odchylenia odchylen wzgledem plikow",
        )

        mean_values_sum_row = pd.DataFrame(mean_values.sum(axis=0)).T
        mean_values_sum_row.index = ["Suma srednich po pliku"]

        dev_values_sum_row = pd.DataFrame(dev_values.sum(axis=0)).T
        dev_values_sum_row.index = ["Suma odchylen po pliku"]

        display_dataframe(
            mean_values_sum_row,
            title="Suma srednich po pliku",
        )
        min_val = mean_values_sum_row.iloc[0].min()
        max_val = mean_values_sum_row.iloc[0].max()
        print(f"Min: {min_val}, Max: {max_val}, Range: {max_val - min_val}")

        display_dataframe(
            dev_values_sum_row,
            title="Suma odchylen po pliku",
        )
        min_val = dev_values_sum_row.iloc[0].min()
        max_val = dev_values_sum_row.iloc[0].max()
        print(f"Min: {min_val}, Max: {max_val}, Range: {max_val - min_val}")

        print()

        return mean_values, dev_values


def remove_all_features(data_path: str, log: bool = True) -> None:
    """Usuwa pliki .parquet zawierajace featury."""
    data_path = Path(data_path)
    for ticker_folder in data_path.iterdir():
        if ticker_folder.is_dir():
            feature_folder = ticker_folder / "features"
            if feature_folder.exists():
                for parquet_file in feature_folder.glob("*.parquet"):
                    parquet_file.unlink()
                    if log:
                        print(f"Usunieto {parquet_file}")


def add_VWAP(df: pd.DataFrame, sigma_mult: float = 2.15) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    hl2 = (df["high"] + df["low"]) / 2

    grouped_date = df["timestamp"].dt.date
    volumesum = df["volume"].groupby(grouped_date).cumsum()
    v2sum = (df["volume"] * hl2 ** 2).groupby(grouped_date).cumsum()

    vwap = calc_vwap(df)
    variance = v2sum / volumesum - vwap ** 2
    sigma = variance.clip(lower=0) ** 0.5

    df["vwap"] = vwap
    df[f"sigma_{sigma_mult}"] = sigma
    df[f"vwap_plus_{sigma_mult}_sigma"] = vwap + sigma_mult * sigma
    df[f"vwap_minus_{sigma_mult}_sigma"] = vwap - sigma_mult * sigma

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    ind = calc_indicators(df)

    existing_features = [col for col in df.columns if col.startswith("feature_")]
    valid_features = [f"feature_{k}" for k in ind.keys()]
    to_drop = [col for col in existing_features if col not in valid_features]
    df = df.drop(columns=to_drop, errors="ignore")

    for k, v in ind.items():
        df[f"feature_{k}"] = v

    return df


def analyze_indicators(
    df: pd.DataFrame, filters: list, log: bool = False
) -> None:
    df = df.copy()
    df = apply_filters(df, filters)

    ind = calc_indicators(df)

    for name, series in ind.items():
        s = series.dropna()
        if len(s) == 0:
            print(f"{name}: brak danych\n")
            continue

        print(f"--- {name} ---")
        print(f"  min     = {s.min():.6f}")
        print(f"  max     = {s.max():.6f}")
        print(f"  mean    = {s.mean():.6f}")
        print(f"  median  = {s.median():.6f}")
        print(f"  std     = {s.std():.6f}")
        print(f"  skew    = {s.skew():.6f}")
        print(f"  kurt    = {s.kurt():.6f}")
        q25, q75 = s.quantile([0.25, 0.75])
        print(f"  IQR     = {q75 - q25:.6f}")
        print(f"  count   = {len(s)}\n")


def get_info(
    df: pd.DataFrame, name: str, sigma: float, filters: list
) -> tuple[pd.DataFrame, int]:
    df_len = len(df)
    dfc = df.copy()
    print(f"Wczytalem ramke danych {name} - poczatkowa dlugosc ramki: {df_len}")

    dfc = apply_filters(dfc, filters)

    mask_above = dfc["close"] > dfc[f"vwap_plus_{sigma}_sigma"]
    mask_below = dfc["close"] < dfc[f"vwap_minus_{sigma}_sigma"]
    mask_extreme = mask_above | mask_below

    df_extreme = dfc[mask_extreme].copy()

    above_sigma = mask_above.sum()
    below_sigma = mask_below.sum()
    extreme_vals = above_sigma + below_sigma

    print(f"Rekordow nad sigma jest {above_sigma}")
    print(f"Rekordow pod sigma jest {below_sigma}")

    if extreme_vals > 0:
        above_pct = (above_sigma / extreme_vals) * 100
        below_pct = (below_sigma / extreme_vals) * 100
    else:
        above_pct = below_pct = 0

    print(f"Rozklad above/below to: {above_pct:.2f}% / {below_pct:.2f}%")
    print(
        f"Lacznie {extreme_vals} swiec jest ekstremalnych "
        f"- jest to {(extreme_vals / df_len) * 100:.2f}% wszystkich wartosci."
    )
    print(
        f"Najstarsza data: {df.iloc[0]['timestamp']}, "
        f"najpozniejsza: {df.iloc[-1]['timestamp']}\n"
    )

    return df_extreme, extreme_vals


def check_features_correlation(
    df: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame | None:
    features = [col for col in df.columns if col.startswith("feature_")]

    if not features:
        print("Brak kolumn z prefiksem 'feature_'")
        return None

    corr_matrix = df[features].corr()

    mask = np.triu(np.ones(corr_matrix.shape), k=0).astype(bool)
    corr_vals = corr_matrix.where(~mask).abs().stack()

    mean_corr = corr_vals.mean()
    max_corr = corr_vals.max()
    high_corr_ratio = (corr_vals > 0.7).mean()

    print(f"Srednia bezwzgledna korelacja: {mean_corr:.3f}")
    print(f"Maksymalna bezwzgledna korelacja: {max_corr:.3f}")
    print(f"Odsetek par z korelacja > 0.7: {high_corr_ratio:.2%}")

    high_pairs = corr_vals[corr_vals > threshold]
    if not high_pairs.empty:
        print(f"\nPary featurow z korelacja > {threshold:.0%}:")
        for (f1, f2), val in high_pairs.items():
            print(f"  {f1} <-> {f2}: {val:.3f}")
    else:
        print(f"\nBrak par z korelacja > {threshold:.0%}")
    print("\n")

    return corr_matrix


def get_correlation_info(path: str) -> None:
    data_path = Path(path)
    for file in data_path.iterdir():
        if file.is_file() and file.name.endswith(".csv"):
            print(f"Obecnie analizuje korelacje w pliku {file}")
            df = pd.read_csv(file)
            check_features_correlation(df)


def analyze_labels(
    df: pd.DataFrame,
    filters: list,
    N_list: list[int] | None = None,
    labels=None,
) -> dict:
    """
    Analizuje wybrane etykiety (lista funkcji w argumencie labels).
    Funkcje labelujace przekazujemy jako partial (z wklejonymi parametrami).
    """
    if N_list is None:
        N_list = [30, 40, 50]
    if not labels:
        raise ValueError("Nie podano zadnych funkcji etykiet do analizy")

    print(">>> Start analizy etykiet (przed filtrami)")
    print(f"Pelny df: {len(df)} rekordow")

    df = df.copy()
    df = add_VWAP(df, sigma_val)

    all_labels = {N: {} for N in N_list}
    for idx_N, N in enumerate(N_list, 1):
        print(f"[{idx_N}/{len(N_list)}] Liczenie etykiet dla okna N={N}...")
        for lab_func in labels:
            series = lab_func(df, N=N)
            name = (
                lab_func.func.__name__
                if hasattr(lab_func, "func")
                else lab_func.__name__
            )
            all_labels[N][name] = series

    if filters:
        print(">>> Zastosowanie filtrow...")
        df = apply_filters(df.copy(), filters, True)
        print(f"Po filtrach: {len(df)} rekordow")

    def summarize(series, name):
        arr = series.dropna().to_numpy()
        counts = dict(zip(*np.unique(arr, return_counts=True)))
        total = sum(counts.values())
        if total == 0:
            print(f"{name}: brak danych\n")
            return
        percents = {k: 100 * v / total for k, v in counts.items()}
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs)) if probs.size > 1 else 0
        dominant_class = max(counts, key=counts.get)
        dominant_share = percents[dominant_class]
        print(f"{name}:")
        for k in sorted(counts.keys()):
            print(f"  klasa {k}: {counts[k]} ({percents[k]:.1f}%)")
        print(
            f"  -> entropia: {entropy:.3f}, "
            f"dominujaca klasa: {dominant_class} "
            f"({dominant_share:.1f}%)\n"
        )

    print(">>> Start podsumowania statystyk (na pelnym df)")
    for N in N_list:
        print(f"\n===== Analiza etykiet (okno N={N}) =====")
        for lab_func in labels:
            name = (
                lab_func.func.__name__
                if hasattr(lab_func, "func")
                else lab_func.__name__
            )
            summarize(all_labels[N][name], name)
        print("-----------------------------------------")

    print(">>> Analiza zakonczona")
    return all_labels


def analyze_labels_with_filters(
    df: pd.DataFrame, filters: list, label_functions: list
) -> None:
    print(f">>> Start analizy, oryginalna liczba rekordow: {len(df)}")

    df = apply_filters(df, filters, True)

    for func in label_functions:
        print(f"\n>>> Dodaje i analizuje etykiety: {func.__name__}")
        df_tmp = df.copy()
        df_tmp = func(df_tmp)

        new_cols = [
            c
            for c in df_tmp.columns
            if c.startswith(func.__name__.replace("add_", ""))
        ]
        if not new_cols:
            print(f"Funkcja {func.__name__} nie dodala zadnych kolumn!")
            continue

        for col in new_cols:
            arr = df_tmp[col].dropna().to_numpy()
            if arr.size == 0:
                print(f"{col}: brak danych\n")
                continue

            counts = dict(zip(*np.unique(arr, return_counts=True)))
            total = arr.size
            percents = {k: v / total * 100 for k, v in counts.items()}
            probs = np.array(list(counts.values())) / total
            entropy = -np.sum(probs * np.log2(probs)) if probs.size > 1 else 0
            dominant_class = max(counts, key=counts.get)
            dominant_share = percents[dominant_class]

            print(f"{col}:")
            for k in sorted(counts.keys()):
                print(f"  klasa {k}: {counts[k]} ({percents[k]:.1f}%)")
            print(
                f"  entropia: {entropy:.3f}, dominujaca klasa: "
                f"{dominant_class} ({dominant_share:.1f}%)\n"
            )


def analyze_labels_in_folder(
    base_path: str,
    label_functions: list,
    filters: list | None = None,
    sigma_val: float = 2.45,
) -> tuple[dict, pd.DataFrame]:
    """
    Analizuje rozklad etykiet dla wszystkich plikow CSV w folderze base_path.
    """
    all_counts = {}
    summary_rows = []
    all_moves = {0: [], 1: [], 2: []}

    base_path = Path(base_path)
    csv_files = sorted(base_path.rglob("*.csv"))
    print(f"\nPrzetwarzam {len(csv_files)} plikow z folderu: {base_path}")

    for csv_path in csv_files:
        if csv_path.name.startswith("."):
            continue

        print(f"\n=== Analiza pliku: {csv_path.name} ===")
        df = pd.read_csv(csv_path)
        print(f"   -> wczytano {len(df):,} rekordow")

        df = add_VWAP(df, sigma_val)
        df["is_extreme"] = (
            (df["close"] > df[f"vwap_plus_{sigma_val}_sigma"])
            | (df["close"] < df[f"vwap_minus_{sigma_val}_sigma"])
        )

        for func in label_functions:
            func_name = func.__name__
            T = func.keywords.get("T", 40)
            print(f"\n   Uruchamiam {func_name} (T={T})...")

            try:
                df_tmp = df.copy()
                y_series = func(df_tmp).dropna()

                if y_series.empty:
                    print(f"      Brak wynikow dla {func_name}")
                    continue

                df_tmp = df_tmp.loc[y_series.index].copy()
                df_tmp["label"] = y_series

                if filters:
                    before = len(df_tmp)
                    for f in filters:
                        df_tmp = f(df_tmp)
                    print(
                        f"      -> po filtrach: {len(df_tmp):,}/{before:,} "
                        f"rekordow"
                    )

                before_ext = len(df_tmp)
                df_tmp = df_tmp[df_tmp["is_extreme"]].copy()
                print(
                    f"      -> po ekstremach: {len(df_tmp):,}/{before_ext:,}"
                )

                if df_tmp.empty:
                    print(
                        f"      Brak ekstremow po filtrach — pomijam "
                        f"{func_name}"
                    )
                    continue

                rets = np.full(len(df_tmp), np.nan)
                closes_arr = df_tmp["close"].to_numpy(float)
                for i in range(len(df_tmp) - T):
                    rets[i] = (
                        (closes_arr[i + T] - closes_arr[i]) / closes_arr[i]
                    )
                df_tmp["price_delta_T"] = np.abs(rets)

                arr = df_tmp["label"].dropna().to_numpy()
                counts = dict(zip(*np.unique(arr, return_counts=True)))
                for k, v in counts.items():
                    all_counts.setdefault(func_name, {}).setdefault(k, 0)
                    all_counts[func_name][k] += v

                for k in [0, 1, 2]:
                    mask = df_tmp["label"] == k
                    moves = df_tmp.loc[mask, "price_delta_T"].dropna().to_numpy()
                    if moves.size > 0:
                        all_moves[k].extend(moves.tolist())

                print(f"      {func_name}: {counts}")

            except Exception as e:
                print(f"      Blad podczas {func_name}: {e}")

    q2_deltas = {
        k: np.nanmedian(all_moves[k]) if all_moves[k] else np.nan
        for k in [0, 1, 2]
    }
    mean_deltas = {
        k: np.nanmean(all_moves[k]) if all_moves[k] else np.nan
        for k in [0, 1, 2]
    }

    print("\n\n=== PODSUMOWANIE GLOBALNE ===")
    for col, counts in all_counts.items():
        total = sum(counts.values())
        perc = {k: v / total * 100 for k, v in counts.items()}
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs)) if probs.size > 1 else 0
        dominant_class = max(counts, key=counts.get)
        dominant_share = perc[dominant_class]

        print(f"\n{col}:")
        for k in sorted(counts):
            print(f"   klasa {k}: {counts[k]:,} ({perc[k]:.1f}%)")

            summary_rows.append({
                "label": col,
                "class": k,
                "count": counts[k],
                "percent": perc[k],
                "entropy": entropy,
                "dominant_class": dominant_class,
                "dominant_share": dominant_share,
                "Q_2_delta_0": q2_deltas[0],
                "Q_2_delta_1": q2_deltas[1],
                "Q_2_delta_2": q2_deltas[2],
                "mean_delta_0": mean_deltas[0],
                "mean_delta_1": mean_deltas[1],
                "mean_delta_2": mean_deltas[2],
            })

        print(
            f"   entropia: {entropy:.3f}, dominujaca klasa: "
            f"{dominant_class} ({dominant_share:.1f}%)"
        )

    summary_df = pd.DataFrame(summary_rows)
    output_csv = Path("label_summary.csv")
    summary_df.to_csv(output_csv, index=False, float_format="%.6f")

    print(f"\nZapisano pelne podsumowanie do pliku: {output_csv.resolve()}")
    return all_counts, summary_df


def apply_cooldown(
    mask_extreme: pd.Series, seq_len: int = 10, cooldown: int = 30
) -> np.ndarray:
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


def analyze_sigma(
    df: pd.DataFrame,
    sigma: float,
    future_window: int = 30,
    phi: float = 0.002,
) -> None:
    """
    Analiza zachowania swiec ekstremalnych wzgledem VWAP +/- sigma.
    """
    entry_price = df["close"]

    future_min = (
        (
            df["close"].shift(-1).rolling(future_window, min_periods=1).min()
            / entry_price
            - 1
        )
        * 100
    )
    future_max = (
        (
            df["close"].shift(-1).rolling(future_window, min_periods=1).max()
            / entry_price
            - 1
        )
        * 100
    )

    mask_above = df["close"] > df[f"vwap_plus_{sigma}_sigma"]
    mask_below = df["close"] < df[f"vwap_minus_{sigma}_sigma"]

    vwap_upper = df["vwap"] * (1 + phi)
    vwap_lower = df["vwap"] * (1 - phi)

    to_vwap_above = (
        (vwap_upper[mask_above] - df.loc[mask_above, "close"])
        / df.loc[mask_above, "close"]
        * 100
    )
    to_vwap_below = (
        (vwap_lower[mask_below] - df.loc[mask_below, "close"])
        / df.loc[mask_below, "close"]
        * 100
    )

    print(f"[Analiza sigma={sigma}]")
    print(f"Swiece above: {mask_above.sum()}, below: {mask_below.sum()}")
    if not to_vwap_above.empty:
        print(
            f"  Powrot ABOVE -> median={to_vwap_above.median():.3f}%, "
            f"90%={to_vwap_above.quantile(0.9):.3f}%"
        )
    if not to_vwap_below.empty:
        print(
            f"  Powrot BELOW -> median={to_vwap_below.median():.3f}%, "
            f"90%={to_vwap_below.quantile(0.9):.3f}%"
        )
    if not future_min.empty:
        print(
            f"  future_min (okno {future_window}) -> "
            f"{future_min.median():.3f}% (mediana)"
        )
    if not future_max.empty:
        print(
            f"  future_max (okno {future_window}) -> "
            f"{future_max.median():.3f}% (mediana)"
        )
    print("-" * 60)


def save_model_names(
    folder_path: str,
    output_file: str = "model_names.txt",
    n_classes: int = 3,
) -> None:
    """
    Zapisuje nazwy modeli z folderu, posortowane wg wartosci
    agregowanej z accuracy.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} nie jest katalogiem!")

    model_names = os.listdir(folder_path)

    def parse_accuracies(name: str) -> list[float]:
        match = re.search(r"_([\d\.\-]+)_[A-Za-z]+$", name)
        if not match:
            return []
        try:
            parts = match.group(1).split("-")
            return [float(x) for x in parts[-n_classes:]]
        except Exception:
            return []

    def aggregate_acc(acc_list: list[float]) -> float:
        if not acc_list:
            return -1.0
        return float(np.mean(acc_list))

    def extract_acc(name: str) -> float:
        return aggregate_acc(parse_accuracies(name))

    model_names_sorted = sorted(model_names, key=extract_acc, reverse=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for name in model_names_sorted:
            accs = parse_accuracies(name)
            agg = aggregate_acc(accs)
            f.write(f"{name}\t{agg:.4f}\n")

    print(
        f"Zapisano {len(model_names_sorted)} nazw do {output_file} "
        f"(sortowanie po agregowanym accuracy)"
    )


def remove_models_below_threshold(
    folder_path: str, min_acc: float = 0.59
) -> tuple[list[str], list[str]]:
    """Usuwa modele z folderu, ktorych accuracy < min_acc."""
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
                print(f"Nie udalo sie usunac {name}: {e}")
        else:
            kept.append(name)

    print(
        f"Usunieto {len(removed)} modeli (accuracy < {min_acc}), "
        f"pozostawiono {len(kept)}."
    )

    return removed, kept


def diagnose_csvs(data_path: str = "data/1m/training_data") -> None:
    path = Path(data_path)
    report = []

    for file in path.glob("*.csv"):
        df = pd.read_csv(file)

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        info = {}
        info["file"] = file.name
        info["rows"] = len(df)
        info["n_cols"] = len(df.columns)
        info["cols"] = set(df.columns)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            info["date_min"] = df["timestamp"].min()
            info["date_max"] = df["timestamp"].max()

        info["nan_rows"] = df.isna().any(axis=1).sum()

        label_stats = {}
        for col in df.columns:
            if col.startswith("label"):
                counts = df[col].value_counts(dropna=True).to_dict()
                label_stats[col] = counts
        info["labels"] = label_stats

        report.append(info)

    print(">>> Diagnostyka CSV-ow")
    print(f"Znaleziono {len(report)} plikow\n")

    all_cols = [r["cols"] for r in report]
    common_cols = set.intersection(*all_cols)
    union_cols = set.union(*all_cols)

    print("Wszystkie pliki maja te wspolne kolumny:")
    print(sorted(list(common_cols)))
    print("\nPelny zestaw kolumn (moga byc roznice):")
    print(sorted(list(union_cols)))
    print()

    for r in report:
        print(f"Plik: {r['file']}")
        print(f"  Wiersze: {r['rows']}, Kolumny: {r['n_cols']}")
        if "date_min" in r:
            print(f"  Zakres dat: {r['date_min']} -> {r['date_max']}")
        print(f"  Wiersze z >=1 NaN: {r['nan_rows']}")
        if r["labels"]:
            print("  Rozklad etykiet:")
            for col, counts in r["labels"].items():
                print(f"    {col}: {counts}")
        print("-" * 40)


def analyze_vwap_dist(
    df: pd.DataFrame,
    model_path: str = r"data\models\vwap_model.pt",
) -> tuple[np.ndarray, np.ndarray] | None:
    from training_weights import load_vwap_model, predict as model_predict

    df = df.copy()
    df.replace(["-", np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"[DEBUG] Poczatkowa liczba rekordow: {len(df)}")

    original_length = len(df)
    mask_extreme = (
        (df["close"] > df["vwap_plus_3_sigma"])
        | (df["close"] < df["vwap_minus_3_sigma"])
    )
    mask_cooldown = apply_cooldown(mask_extreme, seq_len=10, cooldown=30)
    mask_final = mask_extreme & mask_cooldown
    df = df.loc[mask_final]
    print(
        f"[DEBUG] Po filtracji mask_final: {len(df)} / "
        f"{original_length} obserwacji"
    )

    if df.empty:
        print("[DEBUG] Brak danych po filtracji")
        return None

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].to_numpy()

    model = load_vwap_model(model_path)
    y_pred = model_predict(model, X)
    y_class = np.argmax(y_pred, axis=1)

    mask = y_class == 1
    df_sel = df.loc[mask]
    print(f"[DEBUG] Po predykcji, klasa 1: {len(df_sel)} rekordow")

    if df_sel.empty:
        print("[DEBUG] Brak obserwacji z klasa 1")
        return None

    delta_pct = (
        (df_sel["vwap"].to_numpy() - df_sel["close"].to_numpy())
        / df_sel["close"].to_numpy()
        * 100
    )

    below = delta_pct[
        df_sel["close"].to_numpy() < df_sel["vwap"].to_numpy()
    ]
    above = delta_pct[
        df_sel["close"].to_numpy() > df_sel["vwap"].to_numpy()
    ]

    print("[DEBUG] Statystyki globalne delta_pct:")
    print(
        pd.Series(delta_pct).describe(
            percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
        )
    )

    if len(below) > 0:
        print("[DEBUG] Statystyki swiec ponizej VWAP:")
        print(
            pd.Series(below).describe(
                percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
            )
        )
    if len(above) > 0:
        print("[DEBUG] Statystyki swiec powyzej VWAP:")
        print(
            pd.Series(above).describe(
                percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]
            )
        )

    qs = [0.25, 0.5, 0.75]

    if len(below) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(below, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title("Swiece ponizej VWAP (klasa 1)")
        plt.xlabel("Procent brakujacy do VWAP")
        plt.ylabel("Licznosc")
        plt.xlim(-4, 4)
        q_vals = np.quantile(below, qs)
        for q, v in zip(qs, q_vals):
            plt.axvline(v, color="red", linestyle="--")
            plt.text(
                v, plt.ylim()[1] * 0.9,
                f"{int(q * 100)}%={v:.2f}%", rotation=90, color="red",
            )
        plt.show()

    if len(above) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(
            above, bins=50, color="lightgreen", edgecolor="black", alpha=0.7
        )
        plt.title("Swiece powyzej VWAP (klasa 1)")
        plt.xlabel("Procent brakujacy do VWAP")
        plt.ylabel("Licznosc")
        plt.xlim(-4, 4)
        q_vals = np.quantile(above, qs)
        for q, v in zip(qs, q_vals):
            plt.axvline(v, color="red", linestyle="--")
            plt.text(
                v, plt.ylim()[1] * 0.9,
                f"{int(q * 100)}%={v:.2f}%", rotation=90, color="red",
            )
        plt.show()

    return below, above
