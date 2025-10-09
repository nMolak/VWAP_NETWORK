import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
from typing import Dict

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

# Ustaw bazowy logger (lub wkomponuj w swój)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("features-diag")

def _log_stats(s: pd.Series, name: str, idx: pd.Index, max_show: int = 5):
    """Zbiorcza diagnostyka serii/liczbowej kolumny."""
    if not isinstance(s, pd.Series):
        s = pd.Series(s, index=idx)

    arr = s.to_numpy(dtype="float64")
    is_nan = np.isnan(arr)
    is_posinf = np.isposinf(arr)
    is_neginf = np.isneginf(arr)

    n = len(arr)
    n_nan = int(is_nan.sum())
    n_posinf = int(is_posinf.sum())
    n_neginf = int(is_neginf.sum())

    # Indeksy problematyczne
    nan_idx = s.index[is_nan][:max_show].tolist() if n_nan else []
    pinf_idx = s.index[is_posinf][:max_show].tolist() if n_posinf else []
    ninf_idx = s.index[is_neginf][:max_show].tolist() if n_neginf else []

    log.info(f"[{name}] n={n} | NaN={n_nan} (+inf={n_posinf}, -inf={n_neginf})")
    if nan_idx:
        log.info(f"    -> first NaN idx: {nan_idx}")
    if pinf_idx:
        log.info(f"    -> first +inf idx: {pinf_idx}")
    if ninf_idx:
        log.info(f"    -> first -inf idx: {ninf_idx}")

def calc_indicators(df: pd.DataFrame, eps: float = 1e-12) -> Dict[str, pd.Series]:
    """
    Alternatywna (diagnostyczna) wersja calc_indicators:
    - pełne logi po każdym etapie
    - wykrywa źródła NaN/±inf/zerowych mianowników
    """
    assert "timestamp" in df.columns, "Wymagam kolumny 'timestamp' (ms lub datetime)."
    # Ujednolicenie timestamp -> datetime (dla groupby po dniu)
    ts = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce") if np.issubdtype(df["timestamp"].dtype, np.integer) else pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.copy()
    df["timestamp"] = ts
    idx = df.index

    # Podstawy
    hl2 = (df["high"] + df["low"]) / 2.0
    _log_stats(hl2, "hl2", idx)

    # === VWAP ===
    #vwap = vwapsum / volumesum_safe
    vwap = calc_vwap(df)
    _log_stats(vwap, "vwap(raw)", idx)

    # === BB / KC ===
    bb = ta.bbands(df['close'], length=20, std=2.0)
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)

    bb_lower, bb_mid, bb_upper = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
    kc_lower, kc_mid, kc_upper = kc.iloc[:, 0], kc.iloc[:, 1], kc.iloc[:, 2]

    bb_width = (bb_upper - bb_lower)
    kc_width = (kc_upper - kc_lower)
    _log_stats(bb_width, "bb_width(upper-lower)", idx)
    _log_stats(kc_width, "kc_width(upper-lower)", idx)

    zero_bb = (bb_width.abs() <= 0)
    zero_kc = (kc_width.abs() <= 0)
    if zero_bb.any():
        log.info(f"[bb_width] zeros={int(zero_bb.sum())}, first idx={idx[zero_bb][:5].tolist()}")
    if zero_kc.any():
        log.info(f"[kc_width] zeros={int(zero_kc.sum())}, first idx={idx[zero_kc][:5].tolist()}")

    # === EMA / ATR ===
    ema20 = ta.ema(df["close"], length=20)
    ema50 = ta.ema(df["close"], length=50)
    atr_abs = ta.atr(df["high"], df["low"], df["close"], length=14)

    _log_stats(ema20, "ema20", idx)
    _log_stats(ema50, "ema50", idx)
    _log_stats(atr_abs, "atr_abs", idx)

    # === RSI / WILLR / CCI / OBV ===
    rsi7 = ta.rsi(df["close"], length=7)
    willr14 = -ta.willr(df["high"], df["low"], df["close"], length=14)
    cci20 = ta.cci(df["high"], df["low"], df["close"], length=20)
    obv = ta.obv(df["close"], df["volume"])

    _log_stats(rsi7, "rsi7", idx)
    _log_stats(willr14, "willr14", idx)
    _log_stats(cci20, "cci20", idx)
    _log_stats(obv, "obv", idx)

    out = {}

    # --- vwap_dev ---
    denom_vwap = (np.abs(vwap) + eps)
    out["vwap_dev"] = (df['close'] - vwap) / denom_vwap
    _log_stats(out["vwap_dev"], "vwap_dev", idx)
    if (np.abs(vwap) <= 0).sum():
        log.info(f"[vwap_dev] |vwap|==0 cases={int((np.abs(vwap) <= 0).sum())}")

    # --- bb_pos / kc_pos ---
    out["bb_pos"] = (df['close'] - bb_lower) / (np.abs(bb_width) + eps)
    out["kc_pos"] = (df['close'] - kc_lower) / (np.abs(kc_width) + eps)
    _log_stats(out["bb_pos"], "bb_pos", idx)
    _log_stats(out["kc_pos"], "kc_pos", idx)

    # --- rsi7 / willr14 / cci20 ---
    out["rsi7"] = rsi7
    out["willr14"] = willr14
    out["cci20"] = cci20

    # --- vwap_slope5 (pct_change może dać inf przy bazie=0) ---
    # liczymy ręcznie, aby móc zdiagnozować mianownik
    vwap_shift5 = vwap.shift(5)
    denom_vwap5 = (np.abs(vwap_shift5) + eps)
    out["vwap_slope5"] = (vwap - vwap_shift5) / denom_vwap5
    _log_stats(out["vwap_slope5"], "vwap_slope5(manual)", idx)
    zero_base = (np.abs(vwap_shift5) <= 0).sum()
    if zero_base:
        log.info(f"[vwap_slope5] base==0 cases={int(zero_base)}")

    # --- vol20 ---
    out["vol20"] = np.log1p(df["volume"]).diff(20)
    _log_stats(out["vol20"], "vol20(log1p.diff20)", idx)

    # --- atr_rel ---
    out["atr_rel"] = atr_abs / (np.abs(df["close"]) + eps)
    _log_stats(out["atr_rel"], "atr_rel", idx)

    # --- ema_cross_dist / ema_ratio_slope5 ---
    spread = ema20 - ema50
    out["ema_cross_dist"] = spread / (np.abs(ema50) + eps)
    _log_stats(out["ema_cross_dist"], "ema_cross_dist", idx)

    ema_ratio = ema20 / (np.abs(ema50) + eps)
    out["ema_ratio_slope5"] = ema_ratio.diff(5)
    _log_stats(out["ema_ratio_slope5"], "ema_ratio_slope5", idx)

    # --- vwap_dev_atr / vwap_slope_acc10 ---
    out["vwap_dev_atr"] = (df["close"] - vwap) / (np.abs(atr_abs) + eps)
    _log_stats(out["vwap_dev_atr"], "vwap_dev_atr", idx)

    out["vwap_slope_acc10"] = out["vwap_slope5"].diff(5)
    _log_stats(out["vwap_slope_acc10"], "vwap_slope_acc10", idx)

    # --- vwap_side_streak ---
    side = (df["close"] > vwap).astype("float64")
    # gdy vwap NaN → side NaN → wykryjemy w logu
    _log_stats(side, "side(close>vwap)", idx)
    grp = side.ne(side.shift()).cumsum()
    streak = side.groupby(grp).cumcount() + 1
    out["vwap_side_streak"] = streak.where(side == 1, -streak)
    _log_stats(out["vwap_side_streak"], "vwap_side_streak", idx)

    # --- rsi_pct100 ---
    rsi_roll = rsi7.rolling(100)
    rsi_rank = rsi_roll.rank(pct=True)
    out["rsi_pct100"] = rsi_rank
    _log_stats(out["rsi_pct100"], "rsi_pct100(rank,win=100)", idx)

    # --- z-score obv/cci (std==0 -> potencjalne inf; +eps) ---
    obv_mean = obv.rolling(100).mean()
    obv_std  = obv.rolling(100).std()
    out["obv_zscore100"] = (obv - obv_mean) / (np.abs(obv_std) + eps)
    _log_stats(out["obv_zscore100"], "obv_zscore100", idx)

    cci_mean = cci20.rolling(100).mean()
    cci_std  = cci20.rolling(100).std()
    out["cci_zscore100"] = (cci20 - cci_mean) / (np.abs(cci_std) + eps)
    _log_stats(out["cci_zscore100"], "cci_zscore100", idx)

    # --- interakcja VWAP–RSI ---
    out["vwap_rsi_inter"] = ((df["close"] - vwap) / (np.abs(vwap) + eps)) * (rsi7 - 50.0)
    _log_stats(out["vwap_rsi_inter"], "vwap_rsi_inter", idx)

    # Podsumowanie kolumn, które mają problem
    problem_cols = []
    for k, s in out.items():
        s_ser = pd.Series(s, index=idx) if not isinstance(s, pd.Series) else s
        if s_ser.isna().any() or np.isinf(s_ser.to_numpy(dtype="float64")).any():
            problem_cols.append(k)
    if problem_cols:
        log.info(f"[SUMMARY] Kolumny z NaN/±inf: {sorted(problem_cols)}")
    else:
        log.info("[SUMMARY] Brak NaN/±inf w kolumnach wyjściowych.")

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