import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict
import time

# def calc_indicators(df) -> Dict[str, any]:
#     df["timestamp"] = pd.to_datetime(df["timestamp"])
#     hl2 = (df["high"] + df["low"]) / 2
#
#     # VWAP
#     grouped_date = df["timestamp"].dt.date
#     vwapsum   = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
#     volumesum = df["volume"].groupby(grouped_date).cumsum()
#     vwap = vwapsum / volumesum
#
#     # BB/KC
#     bb = ta.bbands(df['close'], length=20, std=2.0)
#     kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
#
#     # EMA / ATR
#     ema20 = ta.ema(df["close"], length=20)
#     ema50 = ta.ema(df["close"], length=50)
#     atr_abs = ta.atr(df["high"], df["low"], df["close"], length=14)
#
#
#     # RSI / WILLR / CCI / OBV
#     rsi7 = ta.rsi(df["close"], length=7)
#     willr14 = -ta.willr(df["high"], df["low"], df["close"], length=14)
#     cci20 = ta.cci(df["high"], df["low"], df["close"], length=20)
#     obv = ta.obv(df["close"], df["volume"])
#
#     out = {}
#
#     # --- FEATURES ---
#     out["vwap_dev"] = (df['close'] - vwap) / vwap
#     out["bb_pos"] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
#     out["kc_pos"] = (df['close'] - kc.iloc[:, 0]) / (kc.iloc[:, 2] - kc.iloc[:, 0])
#     out["rsi7"] = rsi7
#     out["willr14"] = willr14
#     out["vwap_slope5"] = vwap.pct_change(5)
#     out["cci20"] = cci20
#     out["vol20"] = np.log1p(df["volume"]).diff(20)
#     out["atr_rel"] = atr_abs / (df["close"])
#
#     # EMA cross (dystans / ratio)
#     spread = ema20 - ema50
#     out["ema_cross_dist"] = spread / (ema50 + 1e-12)
#     out["ema_ratio_slope5"] = (ema20 / (ema50 + 1e-12)).diff(5)
#
#     # VWAP – reversion / krzywizna / streak
#     out["vwap_dev_atr"] = (df["close"] - vwap) / (atr_abs + 1e-12)
#     out["vwap_slope_acc10"] = out["vwap_slope5"].diff(5)
#     side = (df["close"] > vwap).astype(int)
#     grp = side.ne(side.shift()).cumsum()
#     streak = side.groupby(grp).cumcount() + 1
#     out["vwap_side_streak"] = streak.where(side == 1, -streak)
#
#     # RSI – kwantylowa pozycja
#     out["rsi_pct100"] = rsi7.rolling(100).rank(pct=True)
#
#     # OBV / CCI – zscore
#     out["obv_zscore100"] = (obv - obv.rolling(100).mean()) / (obv.rolling(100).std() + 1e-12)
#     out["cci_zscore100"] = (cci20 - cci20.rolling(100).mean()) / (cci20.rolling(100).std() + 1e-12)
#
#     # Interakcja VWAP–RSI
#     out["vwap_rsi_inter"] = out["vwap_dev"] * (rsi7 - 50.0)
#
#     return out

def calc_vwap(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hl2 = (df["high"] + df["low"]) / 2

    grouped_date = df["timestamp"].dt.date
    vwapsum   = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()

    vwap = vwapsum / volumesum
    return pd.Series(vwap, index=df.index, name="vwap")

def calc_vwap_sigma(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])   # KONWERSJA

    hl2 = (df["high"] + df["low"]) / 2
    grouped_date = df["timestamp"].dt.date
    vwapsum   = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()
    v2sum     = (df["volume"] * hl2**2).groupby(grouped_date).cumsum()
    vwap = vwapsum / volumesum
    variance = v2sum / volumesum - vwap**2
    return variance.clip(lower=0) ** 0.5

def calc_indicators(df: pd.DataFrame, eps: float = 1e-12, log: bool = True) -> Dict[str, pd.Series]:
    """
    Diagnostyczna wersja calc_indicators:
    - log=True włącza raport na końcu
    - zbiera statystyki mean/std/nan/inf
    - dodatkowo wykrywa 'środkowe' NaN/inf
    - loguje czas całkowity obliczeń
    """
    from utils import make_logprint
    logprint = make_logprint(log)
    t0 = time.time()

    assert "timestamp" in df.columns, "Wymagam kolumny 'timestamp' (ms lub datetime)."
    ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce") if np.issubdtype(df["timestamp"].dtype, np.integer) else pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.copy()

    # === DIAGNOSTYKA WOLUMENU ===
    vol_zero_mask = df["volume"] == 0
    n_zeros = int(vol_zero_mask.sum())
    middle_zeros = 0
    median_zero_date = None

    if n_zeros > 0:
        not_zero = ~vol_zero_mask
        if not_zero.any():
            first_valid = np.argmax(not_zero)  # pierwszy niezerowy
            last_valid = len(df) - np.argmax(not_zero[::-1]) - 1  # ostatni niezerowy

            # sprawdź, czy między nimi są zera
            inner_mask = vol_zero_mask.iloc[first_valid:last_valid + 1]
            middle_zeros = int(inner_mask.sum())

            # medianowa data zerowych wolumenów
            if middle_zeros > 0:
                zero_dates = ts[vol_zero_mask]
                median_zero_date = zero_dates.median()

    # 0.05-kwantyl z wolumenu (dla niezerowych wartości)
    vol_q005 = float(df.loc[df["volume"] > 0, "volume"].quantile(0.05))

    logprint(f"[INFO] Wykryto {n_zeros:,} zerowych wolumenów (0.05-kwantyl: {vol_q005:.6f})")
    if middle_zeros > 0:
        logprint(f"[INFO] 🔸 Zera pośrodku danych: {middle_zeros:,} | Mediana dat zer: {median_zero_date}")
    else:
        logprint("[INFO] Brak zerowych wolumenów pośrodku danych.")

    df["zero_run"] = (df["volume"] == 0).astype(int).groupby((df["volume"] != 0).cumsum()).cumsum()
    long_zeros = df["zero_run"].max()
    logprint(f"[INFO] Najdłuższa ciągła sekwencja zerowych wolumenów: {long_zeros}")


    df["timestamp"] = ts
    idx = df.index

    out, stats = {}, {}

    # === PODSTAWY ===
    vwap = calc_vwap(df)
    bb = ta.bbands(df['close'], length=20, std=2.0)
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
    bb_lower, bb_mid, bb_upper = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
    kc_lower, kc_mid, kc_upper = kc.iloc[:, 0], kc.iloc[:, 1], kc.iloc[:, 2]
    ema20, ema50 = ta.ema(df["close"], length=20), ta.ema(df["close"], length=50)
    atr_abs = ta.atr(df["high"], df["low"], df["close"], length=14)
    rsi7 = ta.rsi(df["close"], length=7)
    willr14 = -ta.willr(df["high"], df["low"], df["close"], length=14)
    cci20 = ta.cci(df["high"], df["low"], df["close"], length=20)
    obv = ta.obv(df["close"], df["volume"])

    # === FEATUREY ===
    out["vwap_dev"] = (df['close'] - vwap) / (np.abs(vwap) + eps)
    out["bb_pos"] = (df['close'] - bb_lower) / (np.abs(bb_upper - bb_lower) + eps)
    out["kc_pos"] = (df['close'] - kc_lower) / (np.abs(kc_upper - kc_lower) + eps)
    out["rsi7"] = rsi7
    out["willr14"] = willr14
    out["cci20"] = cci20

    vwap_shift5 = vwap.shift(5)
    out["vwap_slope5"] = (vwap - vwap_shift5) / (np.abs(vwap_shift5) + eps)
    out["vol20"] = np.log1p(df["volume"]).diff(20)
    out["atr_rel"] = atr_abs / (np.abs(df["close"]) + eps)
    out["ema_cross_dist"] = (ema20 - ema50) / (np.abs(ema50) + eps)
    ema_ratio = ema20 / (np.abs(ema50) + eps)
    out["ema_ratio_slope5"] = ema_ratio.diff(5)
    out["vwap_dev_atr"] = (df["close"] - vwap) / (np.abs(atr_abs) + eps)
    out["vwap_slope_acc10"] = out["vwap_slope5"].diff(5)

    side = (df["close"] > vwap).astype("float64")
    grp = side.ne(side.shift()).cumsum()
    streak = side.groupby(grp).cumcount() + 1
    out["vwap_side_streak"] = streak.where(side == 1, -streak)

    rsi_roll = rsi7.rolling(100)
    out["rsi_pct100"] = rsi_roll.rank(pct=True)

    obv_mean, obv_std = obv.rolling(100).mean(), obv.rolling(100).std()
    out["obv_zscore100"] = (obv - obv_mean) / (np.abs(obv_std) + eps)
    cci_mean, cci_std = cci20.rolling(100).mean(), cci20.rolling(100).std()
    out["cci_zscore100"] = (cci20 - cci_mean) / (np.abs(cci_std) + eps)

    out["vwap_rsi_inter"] = ((df["close"] - vwap) / (np.abs(vwap) + eps)) * (rsi7 - 50.0)

    if not log:
        return out

    # === ZBIERANIE STATYSTYK ===
    for k, s in out.items():
        s_ser = pd.Series(s, index=idx)
        is_nan = s_ser.isna()
        is_inf = np.isinf(s_ser.to_numpy(dtype="float64"))
        n_nan, n_inf = int(is_nan.sum()), int(is_inf.sum())

        # analiza pozycji błędów (czy tylko na końcach)
        bad_idx = np.where(is_nan | is_inf)[0]
        middle_problem = False

        if len(bad_idx) > 0:
            # znajdź indeks pierwszego i ostatniego NIE-NaN
            not_bad = ~(is_nan | is_inf)
            if not_bad.any():
                first_valid = np.argmax(not_bad)  # pierwszy poprawny
                last_valid = len(s_ser) - np.argmax(not_bad[::-1]) - 1  # ostatni poprawny

                # sprawdź, czy pomiędzy nimi są NaN
                inner_mask = (is_nan | is_inf).iloc[first_valid:last_valid + 1]
                if inner_mask.any():
                    middle_problem = True

        stats[k] = {
            "nan": n_nan,
            "inf": n_inf,
            "mean": float(s_ser.mean(skipna=True)),
            "std": float(s_ser.std(skipna=True)),
            "middle_problem": middle_problem,
        }

    # === DIAGNOSTYKA DANYCH WEJŚCIOWYCH ===
    for col in ["volume", "close"]:
        s = df[col]
        stats[col] = {
            "nan": int(s.isna().sum()),
            "zeros": int((s == 0).sum()),
            "min": float(s.min(skipna=True)),
            "max": float(s.max(skipna=True)),
            "mean": float(s.mean(skipna=True)),
            "std": float(s.std(skipna=True))
        }

    # === RAPORT KOŃCOWY ===
    if log:
        dt = time.time() - t0
        logprint(f"[SUMMARY] Liczenie featurów zakończone w {dt:.3f} s")
        logprint("[SUMMARY] Statystyki featurów:")
        for k, v in stats.items():
            if "middle_problem" in v:  # featury
                msg = f"{k:20s} nan={v['nan']:4d}, inf={v['inf']:3d}, mean={v['mean']:+.6f}, std={v['std']:.6f}"
                if v["middle_problem"]:
                    msg += "  ⚠️  PROBLEM: NaN/inf w środku danych!"
            else:  # dane wejściowe
                msg = (
                    f"{k:20s} nan={v['nan']:4d}, zeros={v['zeros']:4d}, "
                    f"min={v['min']:+.6f}, max={v['max']:+.6f}, "
                    f"mean={v['mean']:+.6f}, std={v['std']:.6f}"
                )
            logprint(msg)

    return out

def calc_vwap(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hl2 = (df["high"] + df["low"]) / 2

    grouped_date = df["timestamp"].dt.date
    vwapsum   = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()

    vwap = vwapsum / volumesum
    return pd.Series(vwap, index=df.index, name="vwap")

def calc_vwap_sigma(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])   # KONWERSJA

    hl2 = (df["high"] + df["low"]) / 2
    grouped_date = df["timestamp"].dt.date
    vwapsum   = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()
    v2sum     = (df["volume"] * hl2**2).groupby(grouped_date).cumsum()
    vwap = vwapsum / volumesum
    variance = v2sum / volumesum - vwap**2
    return variance.clip(lower=0) ** 0.5