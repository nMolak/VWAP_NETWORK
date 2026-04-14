import time
from typing import Dict

import numpy as np
import pandas as pd
import pandas_ta as ta


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hl2 = (df["high"] + df["low"]) / 2

    grouped_date = df["timestamp"].dt.date
    vwapsum = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()

    vwap = vwapsum / volumesum
    return pd.Series(vwap, index=df.index, name="vwap")


def calc_vwap_sigma(df: pd.DataFrame) -> pd.Series:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    hl2 = (df["high"] + df["low"]) / 2
    grouped_date = df["timestamp"].dt.date
    vwapsum = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()
    v2sum = (df["volume"] * hl2 ** 2).groupby(grouped_date).cumsum()
    vwap = vwapsum / volumesum
    variance = v2sum / volumesum - vwap ** 2
    return variance.clip(lower=0) ** 0.5


def calc_indicators(
    df: pd.DataFrame, eps: float = 1e-12, log: bool = True
) -> Dict[str, pd.Series]:
    """
    Diagnostyczna wersja calc_indicators:
    - log=True wlacza raport na koncu
    - zbiera statystyki mean/std/nan/inf
    - dodatkowo wykrywa 'srodkowe' NaN/inf
    - loguje czas calkowity obliczen
    """
    from utils import make_logprint

    logprint = make_logprint(log)
    t0 = time.time()

    assert "timestamp" in df.columns, "Wymagam kolumny 'timestamp' (ms lub datetime)."
    ts = (
        pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        if np.issubdtype(df["timestamp"].dtype, np.integer)
        else pd.to_datetime(df["timestamp"], errors="coerce")
    )
    df = df.copy()

    # === DIAGNOSTYKA WOLUMENU ===
    vol_zero_mask = df["volume"] == 0
    n_zeros = int(vol_zero_mask.sum())
    middle_zeros = 0
    median_zero_date = None

    if n_zeros > 0:
        not_zero = ~vol_zero_mask
        if not_zero.any():
            first_valid = np.argmax(not_zero)
            last_valid = len(df) - np.argmax(not_zero[::-1]) - 1

            inner_mask = vol_zero_mask.iloc[first_valid : last_valid + 1]
            middle_zeros = int(inner_mask.sum())

            if middle_zeros > 0:
                zero_dates = ts[vol_zero_mask]
                median_zero_date = zero_dates.median()

    vol_q005 = float(df.loc[df["volume"] > 0, "volume"].quantile(0.05))

    logprint(
        f"[INFO] Wykryto {n_zeros:,} zerowych wolumenow (0.05-kwantyl: {vol_q005:.6f})"
    )
    if middle_zeros > 0:
        logprint(
            f"[INFO] Zera posrodku danych: {middle_zeros:,} "
            f"| Mediana dat zer: {median_zero_date}"
        )
    else:
        logprint("[INFO] Brak zerowych wolumenow posrodku danych.")

    df["zero_run"] = (
        (df["volume"] == 0)
        .astype(int)
        .groupby((df["volume"] != 0).cumsum())
        .cumsum()
    )
    long_zeros = df["zero_run"].max()
    logprint(
        f"[INFO] Najdluzsza ciagla sekwencja zerowych wolumenow: {long_zeros}"
    )

    df["timestamp"] = ts
    idx = df.index

    out, stats = {}, {}

    # === PODSTAWY ===
    vwap = calc_vwap(df)
    bb = ta.bbands(df["close"], length=20, std=2.0)
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)
    bb_lower, bb_mid, bb_upper = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
    kc_lower, kc_mid, kc_upper = kc.iloc[:, 0], kc.iloc[:, 1], kc.iloc[:, 2]
    ema20 = ta.ema(df["close"], length=20)
    ema50 = ta.ema(df["close"], length=50)
    atr_abs = ta.atr(df["high"], df["low"], df["close"], length=14)
    rsi7 = ta.rsi(df["close"], length=7)
    willr14 = -ta.willr(df["high"], df["low"], df["close"], length=14)
    cci20 = ta.cci(df["high"], df["low"], df["close"], length=20)
    obv = ta.obv(df["close"], df["volume"])

    # === FEATURY ===
    out["vwap_dev"] = (df["close"] - vwap) / (np.abs(vwap) + eps)
    out["bb_pos"] = (df["close"] - bb_lower) / (np.abs(bb_upper - bb_lower) + eps)
    out["kc_pos"] = (df["close"] - kc_lower) / (np.abs(kc_upper - kc_lower) + eps)
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

    obv_mean = obv.rolling(100).mean()
    obv_std = obv.rolling(100).std()
    out["obv_zscore100"] = (obv - obv_mean) / (np.abs(obv_std) + eps)
    cci_mean = cci20.rolling(100).mean()
    cci_std = cci20.rolling(100).std()
    out["cci_zscore100"] = (cci20 - cci_mean) / (np.abs(cci_std) + eps)

    out["vwap_rsi_inter"] = (
        (df["close"] - vwap) / (np.abs(vwap) + eps)
    ) * (rsi7 - 50.0)

    if not log:
        return out

    # === ZBIERANIE STATYSTYK ===
    for k, s in out.items():
        s_ser = pd.Series(s, index=idx)
        is_nan = s_ser.isna()
        is_inf = np.isinf(s_ser.to_numpy(dtype="float64"))
        n_nan, n_inf = int(is_nan.sum()), int(is_inf.sum())

        bad_idx = np.where(is_nan | is_inf)[0]
        middle_problem = False

        if len(bad_idx) > 0:
            not_bad = ~(is_nan | is_inf)
            if not_bad.any():
                first_valid = np.argmax(not_bad)
                last_valid = len(s_ser) - np.argmax(not_bad[::-1]) - 1

                inner_mask = (is_nan | is_inf).iloc[first_valid : last_valid + 1]
                if inner_mask.any():
                    middle_problem = True

        stats[k] = {
            "nan": n_nan,
            "inf": n_inf,
            "mean": float(s_ser.mean(skipna=True)),
            "std": float(s_ser.std(skipna=True)),
            "middle_problem": middle_problem,
        }

    # === DIAGNOSTYKA DANYCH WEJSCIOWYCH ===
    for col in ["volume", "close"]:
        s = df[col]
        stats[col] = {
            "nan": int(s.isna().sum()),
            "zeros": int((s == 0).sum()),
            "min": float(s.min(skipna=True)),
            "max": float(s.max(skipna=True)),
            "mean": float(s.mean(skipna=True)),
            "std": float(s.std(skipna=True)),
        }

    # === RAPORT KONCOWY ===
    if log:
        dt = time.time() - t0
        logprint(f"[SUMMARY] Liczenie featurow zakonczone w {dt:.3f} s")
        logprint("[SUMMARY] Statystyki featurow:")
        for k, v in stats.items():
            if "middle_problem" in v:
                msg = (
                    f"{k:20s} nan={v['nan']:4d}, inf={v['inf']:3d}, "
                    f"mean={v['mean']:+.6f}, std={v['std']:.6f}"
                )
                if v["middle_problem"]:
                    msg += "  PROBLEM: NaN/inf w srodku danych!"
            else:
                msg = (
                    f"{k:20s} nan={v['nan']:4d}, zeros={v['zeros']:4d}, "
                    f"min={v['min']:+.6f}, max={v['max']:+.6f}, "
                    f"mean={v['mean']:+.6f}, std={v['std']:.6f}"
                )
            logprint(msg)

    return out
