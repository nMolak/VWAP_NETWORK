import pandas_ta as ta
import pandas as pd
import numpy as np

from typing import Dict, Any

def calc_indicators(df) -> Dict[str, any]:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    hl2 = (df["high"] + df["low"]) / 2

    # VWAP
    grouped_date = df["timestamp"].dt.date
    vwapsum   = (hl2 * df["volume"]).groupby(grouped_date).cumsum()
    volumesum = df["volume"].groupby(grouped_date).cumsum()
    vwap = vwapsum / volumesum

    # BB/KC
    bb = ta.bbands(df['close'], length=20, std=2.0)
    kc = ta.kc(df["high"], df["low"], df["close"], length=20, scalar=1.5)

    # EMA / ATR
    ema20 = ta.ema(df["close"], length=20)
    ema50 = ta.ema(df["close"], length=50)
    atr_abs = ta.atr(df["high"], df["low"], df["close"], length=14)


    # RSI / WILLR / CCI / OBV
    rsi7 = ta.rsi(df["close"], length=7)
    willr14 = -ta.willr(df["high"], df["low"], df["close"], length=14)
    cci20 = ta.cci(df["high"], df["low"], df["close"], length=20)
    obv = ta.obv(df["close"], df["volume"])

    out = {}

    # --- FEATURES ---
    out["vwap_dev"] = (df['close'] - vwap) / vwap
    out["bb_pos"] = (df['close'] - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
    out["kc_pos"] = (df['close'] - kc.iloc[:, 0]) / (kc.iloc[:, 2] - kc.iloc[:, 0])
    out["rsi7"] = rsi7
    out["willr14"] = willr14
    out["vwap_slope5"] = vwap.pct_change(5)
    out["cci20"] = cci20
    out["vol20"] = np.log1p(df["volume"]).diff(20)
    out["atr_rel"] = atr_abs / (df["close"])

    # EMA cross (dystans / ratio)
    spread = ema20 - ema50
    out["ema_cross_dist"] = spread / (ema50 + 1e-12)
    out["ema_ratio_slope5"] = (ema20 / (ema50 + 1e-12)).diff(5)

    # VWAP – reversion / krzywizna / streak
    out["vwap_dev_atr"] = (df["close"] - vwap) / (atr_abs + 1e-12)
    out["vwap_slope_acc10"] = out["vwap_slope5"].diff(5)
    side = (df["close"] > vwap).astype(int)
    grp = side.ne(side.shift()).cumsum()
    streak = side.groupby(grp).cumcount() + 1
    out["vwap_side_streak"] = streak.where(side == 1, -streak)

    # RSI – kwantylowa pozycja
    out["rsi_pct100"] = rsi7.rolling(100).rank(pct=True)

    # OBV / CCI – zscore
    out["obv_zscore100"] = (obv - obv.rolling(100).mean()) / (obv.rolling(100).std() + 1e-12)
    out["cci_zscore100"] = (cci20 - cci20.rolling(100).mean()) / (cci20.rolling(100).std() + 1e-12)

    # Interakcja VWAP–RSI
    out["vwap_rsi_inter"] = out["vwap_dev"] * (rsi7 - 50.0)

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