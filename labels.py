from functools import partial

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from calculations import calc_vwap, calc_vwap_sigma


def calc_label1(df: pd.DataFrame, N: int, phi: float = 0.00075) -> pd.Series:
    close = df["close"].to_numpy(dtype=float)
    labels = np.full(len(close), np.nan)

    for t in range(len(close) - N):
        if close[t + N] > close[t] * (1 + phi):
            labels[t] = 1
        elif close[t + N] < close[t] * (1 - phi):
            labels[t] = -1
        else:
            labels[t] = 0

    return pd.Series(labels, index=df.index, name=f"label1_N{N}")


def calc_label2(
    df: pd.DataFrame, N: int, phi: float = 0.01, psi: float = 0.3
) -> pd.Series:
    close = df["close"].to_numpy(dtype=float)

    if "vwap" in df.columns:
        vwap = df["vwap"].to_numpy(dtype=float)
    else:
        vwap = calc_vwap(df).to_numpy(dtype=float)

    labels = np.full(len(close), np.nan)

    for t in range(len(close) - N):
        dev_t = abs(close[t] - vwap[t]) / vwap[t]
        window_close = close[t + 1 : t + N + 1]
        window_dev = abs(window_close - vwap[t]) / vwap[t]

        if np.min(window_dev) <= phi:
            labels[t] = 1
        elif np.max(window_dev) >= dev_t * (1 + psi):
            labels[t] = 0
        else:
            labels[t] = 2

    return pd.Series(labels, index=df.index, name=f"label2_N{N}")


def calc_label3(
    df: pd.DataFrame, N: int, psi: float = 0.3, k_sigma: int = 2
) -> pd.Series:
    close = df["close"].to_numpy(dtype=float)

    if "vwap" in df.columns:
        vwap = df["vwap"].to_numpy(dtype=float)
    else:
        vwap = calc_vwap(df).to_numpy(dtype=float)

    sigma = calc_vwap_sigma(df).to_numpy(dtype=float)
    labels = np.full(len(close), np.nan)

    for t in range(len(close) - N):
        dev_t = abs(close[t] - vwap[t]) / vwap[t]
        window_close = close[t + 1 : t + N + 1]

        lower = vwap[t] - k_sigma * sigma[t]
        upper = vwap[t] + k_sigma * sigma[t]

        if np.any((window_close >= lower) & (window_close <= upper)):
            labels[t] = 1
        elif np.max(abs(window_close - vwap[t]) / vwap[t]) >= dev_t * (1 + psi):
            labels[t] = 0
        else:
            labels[t] = 2

    return pd.Series(labels, index=df.index, name=f"label3_N{N}")


# --------- Pomocnicze ---------


def _first_hit_in_window(w, tp, sl, prefer="conservative", eps=0.0):
    for _, row in w.iterrows():
        hi = row["high"]
        lo = row["low"]
        hit_tp = (hi >= tp - eps) if tp is not None else False
        hit_sl = (lo <= sl + eps) if sl is not None else False
        if hit_tp and not hit_sl:
            return True
        if hit_sl and not hit_tp:
            return False
        if hit_tp and hit_sl:
            return True if prefer == "tp" else False
    return None


def _new_trigger(df_row, k, side):
    """Zwraca ('long'|'short'|None) w zaleznosci od z-score i zadanej strony."""
    from parameters import sigma_val

    z = (df_row["close"] - df_row["vwap"]) / df_row[f"sigma_{sigma_val}"]
    if np.isnan(z):
        return None
    if side in ("both", "short") and z >= k:
        return "short"
    if side in ("both", "long") and z <= -k:
        return "long"
    return None


def _get_horizon(df: pd.DataFrame, t0, T: int) -> pd.DataFrame:
    """Zwraca pod-DataFrame od razu po t0 do konca horyzontu."""
    if isinstance(df.index, pd.DatetimeIndex):
        horizon_end = t0 + pd.Timedelta(minutes=T)
        w = df.loc[(df.index > t0) & (df.index <= horizon_end)]
    else:
        pos = df.index.get_loc(t0)
        w = df.iloc[pos + 1 : pos + T + 1]
    return w


def _horizon_slice(df: pd.DataFrame, t0, T: int) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        t1 = t0 + pd.Timedelta(minutes=T)
        return df.loc[(df.index > t0) & (df.index <= t1)]
    pos = df.index.get_loc(t0)
    end = min(pos + T, len(df.index) - 1)
    return df.iloc[pos + 1 : end + 1]


# --------- Label 4: TP-przed-SL (RR na ulamku drogi do VWAP) ---------


def calc_label4(
    df: pd.DataFrame,
    k: float = 3.0,
    r: float = 0.5,
    lam: float = 0.5,
    T: int = 60,
    cooldown: int = 30,
    side: str = "both",
    prefer: str = "conservative",
    eps: float = 0.0,
) -> pd.Series:
    labels = pd.Series(
        pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index
    )

    next_allowed = df.index[0]
    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        trig = _new_trigger(row, k=k, side=side)
        if trig is None:
            continue

        d = abs(row["close"] - row["vwap"])
        if d == 0 or np.isnan(d):
            continue

        if trig == "long":
            tp = row["close"] + r * d
            sl = row["close"] - lam * d
        else:
            tp = row["close"] - r * d
            sl = row["close"] + lam * d

        w = _get_horizon(df, t0, T)
        win_first = _first_hit_in_window(w, tp, sl, prefer=prefer, eps=eps)

        labels.at[t0] = 1 if win_first else 0

        if isinstance(df.index, pd.DatetimeIndex):
            next_allowed = t0 + pd.Timedelta(minutes=cooldown)
        else:
            pos = df.index.get_loc(t0)
            if pos + cooldown < len(df.index):
                next_allowed = df.index[pos + cooldown]
            else:
                next_allowed = df.index[-1]

    return labels


# --------- Label 5: Halfway first-passage ---------


def calc_label5(
    df: pd.DataFrame,
    k: float = 3.0,
    half: float = 0.5,
    lam: float = 0.3,
    T: int = 45,
    cooldown: int = 30,
    side: str = "both",
    prefer: str = "conservative",
    eps: float = 0.0,
) -> pd.Series:
    labels = pd.Series(
        pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index
    )
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        trig = _new_trigger(row, k=k, side=side)
        if trig is None:
            continue

        d = abs(row["close"] - row["vwap"])
        if d == 0 or np.isnan(d):
            continue

        if trig == "long":
            tp = row["close"] + half * d
            sl = row["close"] - lam * d
        else:
            tp = row["close"] - half * d
            sl = row["close"] + lam * d

        w = _get_horizon(df, t0, T)
        win_first = _first_hit_in_window(w, tp, sl, prefer=prefer, eps=eps)

        labels.at[t0] = 1 if win_first else 0

        if isinstance(df.index, pd.DatetimeIndex):
            next_allowed = t0 + pd.Timedelta(minutes=cooldown)
        else:
            pos = df.index.get_loc(t0)
            next_allowed = df.index[min(pos + cooldown, len(df.index) - 1)]

    return labels


# --------- Label 6: Edge-ratio MFE vs MAE ---------


def calc_label6(
    df: pd.DataFrame,
    k: float = 3.0,
    rho: float = 1.5,
    T: int = 60,
    cooldown: int = 30,
    side: str = "both",
) -> pd.Series:
    """
    Label 6 (0/1/NA):
      - 1 = odbicie (MFE >= rho * MAE w horyzoncie T)
      - 0 = brak odbicia
      - NA = brak triggera
    """
    labels = pd.Series(
        pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index
    )
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        trig = _new_trigger(row, k=k, side=side)
        if trig is None:
            continue

        entry = row["close"]
        w = _get_horizon(df, t0, T)[["high", "low"]]

        if w.empty:
            labels.at[t0] = 0
        else:
            if trig == "long":
                mfe = (w["high"] - entry).clip(lower=0).max()
                mae = (entry - w["low"]).clip(lower=0).max()
            else:
                mfe = (entry - w["low"]).clip(lower=0).max()
                mae = (w["high"] - entry).clip(lower=0).max()

            if mae == 0:
                label = 1 if mfe > 0 else 0
            else:
                label = 1 if (mfe / mae) >= rho else 0

            labels.at[t0] = np.int8(label)

        if isinstance(df.index, pd.DatetimeIndex):
            next_allowed = t0 + pd.Timedelta(minutes=cooldown)
        else:
            pos = df.index.get_loc(t0)
            next_allowed = df.index[min(pos + cooldown, len(df.index) - 1)]

    return labels


def calc_label7(
    df: pd.DataFrame,
    k: float = 3.0,
    r: float = 0.5,
    lam: float = 0.5,
    T: int = 60,
    cooldown: int = 30,
    prefer: str = "conservative",
    side: str = "both",
) -> pd.Series:
    """
    1: TP (reversion ku VWAP) przed SL (kontynuacja), albo przy braku
    trafienia blizej VWAP na koncu; inaczej 0. Zawsze 0/1 (Int8), bez NA.
    """
    if "vwap" not in df.columns:
        raise ValueError("calc_label7 wymaga kolumny 'vwap'.")
    close = df["close"].to_numpy(float)
    vwap = df["vwap"].to_numpy(float)
    d0 = np.abs(close - vwap)

    labels = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        z = (row["close"] - row["vwap"]) / (abs(row["vwap"]) + 1e-12)
        trig = None
        if side in ("both", "short") and z >= k:
            trig = "short"
        if side in ("both", "long") and z <= -k:
            trig = "long"

        if trig is None:
            w = _horizon_slice(df, t0, T)
            if w.empty:
                labels.at[t0] = 0
            else:
                d_start = abs(row["close"] - row["vwap"])
                d_min = abs(w["close"] - row["vwap"]).min()
                labels.at[t0] = np.int8(
                    1 if (d_start - d_min) >= r * max(d_start, 1e-12) else 0
                )
            next_allowed = (
                (t0 + pd.Timedelta(minutes=cooldown))
                if isinstance(df.index, pd.DatetimeIndex)
                else df.index[
                    min(df.index.get_loc(t0) + cooldown, len(df.index) - 1)
                ]
            )
            continue

        d = abs(row["close"] - row["vwap"])
        if d <= 0:
            labels.at[t0] = 0
            continue

        if trig == "long":
            tp = row["close"] + r * d
            sl = row["close"] - lam * d
        else:
            tp = row["close"] - r * d
            sl = row["close"] + lam * d

        w = _horizon_slice(df, t0, T)
        if w.empty:
            labels.at[t0] = 0
        else:
            hit = None
            for _, r_ in w.iterrows():
                hi, lo = r_["high"], r_["low"]
                tp_hit = (hi >= tp) if trig == "long" else (lo <= tp)
                sl_hit = (lo <= sl) if trig == "long" else (hi >= sl)
                if tp_hit and not sl_hit:
                    hit = "tp"
                    break
                if sl_hit and not tp_hit:
                    hit = "sl"
                    break
                if tp_hit and sl_hit:
                    hit = "tp" if prefer == "tp" else "sl"
                    break
            if hit is None:
                end_close = w["close"].iloc[-1]
                label = 1 if abs(end_close - row["vwap"]) < d else 0
            else:
                label = 1 if hit == "tp" else 0
            labels.at[t0] = np.int8(label)

        next_allowed = (
            (t0 + pd.Timedelta(minutes=cooldown))
            if isinstance(df.index, pd.DatetimeIndex)
            else df.index[
                min(df.index.get_loc(t0) + cooldown, len(df.index) - 1)
            ]
        )
    return labels


def calc_label8(
    df: pd.DataFrame,
    k: float = 3.0,
    T: int = 60,
    tau: float = 0.6,
    cooldown: int = 30,
    side: str = "both",
) -> pd.Series:
    """
    1: w oknie T odleglosc |close - vwap| maleje w >= tau czesci swiec;
    inaczej 0. Zawsze 0/1.
    """
    if "vwap" not in df.columns:
        raise ValueError("calc_label8 wymaga kolumny 'vwap'.")
    labels = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        z = (row["close"] - row["vwap"]) / (abs(row["vwap"]) + 1e-12)
        if not (
            (side in ("both", "short") and z >= k)
            or (side in ("both", "long") and z <= -k)
            or (k <= 0)
        ):
            continue

        w = _horizon_slice(df, t0, T)
        if w.empty:
            labels.at[t0] = 0
        else:
            d0 = abs(row["close"] - row["vwap"])
            d_series = (w["close"] - row["vwap"]).abs().values
            dec = (np.diff(np.r_[d0, d_series]) < 0).sum()
            tot = len(d_series)
            labels.at[t0] = np.int8(1 if (dec / max(tot, 1)) >= tau else 0)

        next_allowed = (
            (t0 + pd.Timedelta(minutes=cooldown))
            if isinstance(df.index, pd.DatetimeIndex)
            else df.index[
                min(df.index.get_loc(t0) + cooldown, len(df.index) - 1)
            ]
        )
    return labels


def calc_label9(
    df: pd.DataFrame,
    T: int = 15,
    alpha: float = 0.2,
    use_atr: bool = False,
) -> pd.Series:
    """
    Wynik: Series 0/1, identyczny logicznie z wersja petlowa.
    -1 oznacza brak danych (numpy nie przechowa None).
    """
    if "vwap" not in df.columns:
        raise ValueError("calc_label9 wymaga kolumny 'vwap'.")

    closes = df["close"].to_numpy(float)

    if use_atr:
        if "feature_atr_rel" in df.columns:
            scale = df["feature_atr_rel"].to_numpy(float) + 1e-12
        else:
            print("Brak feature_atr_rel - licze scale alternatywnie!")
            scale = (df["close"] - df["vwap"]).abs().to_numpy(float) + 1e-12
    else:
        scale = (df["close"] - df["vwap"]).abs().to_numpy(float)

    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    labels[:] = -1

    windows = sliding_window_view(closes, T)

    entry = windows[:, 0][:, None]
    ret_paths = (windows - entry) / entry

    thr = alpha * (scale[: n - T + 1] / np.maximum(entry[:, 0], 1e-12))

    up_hits = ret_paths >= thr[:, None]
    dn_hits = ret_paths <= -thr[:, None]

    any_up = up_hits.any(axis=1)
    any_dn = dn_hits.any(axis=1)

    both = any_up & any_dn
    first_up = np.argmax(up_hits[both], axis=1)
    first_dn = np.argmax(dn_hits[both], axis=1)

    labels[: n - T + 1][any_up & ~any_dn] = 1
    labels[: n - T + 1][~any_up & any_dn] = 0
    labels[: n - T + 1][both] = (first_up < first_dn).astype(np.int8)

    no_hits = ~(any_up | any_dn)
    labels[: n - T + 1][no_hits] = (ret_paths[no_hits, -1] > 0).astype(np.int8)

    return pd.Series(labels, index=df.index)


def calc_label10(
    df: pd.DataFrame,
    T: int = 15,
    alpha: float = 0.2,
    use_atr: bool = False,
) -> pd.Series:
    """
    Wektorowa wersja etykiety z trzema klasami:
      0 = ruch spadkowy
      1 = ruch wzrostowy
      2 = stagnacja (brak trafienia progu w zadna strone w horyzoncie T)
    """
    if "vwap" not in df.columns:
        raise ValueError("calc_label10 wymaga kolumny 'vwap'.")

    closes = df["close"].to_numpy(float)

    if use_atr and "feature_atr_rel" in df.columns:
        scale = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale = np.abs(df["close"] - df["vwap"]).to_numpy(float) + 1e-12

    n = len(df)
    labels = np.full(n, 2, dtype=np.int8)

    if n < T:
        return pd.Series(labels, index=df.index, dtype=np.int8)

    windows = sliding_window_view(closes, T)
    entry = windows[:, 0][:, None]
    ret_paths = (windows - entry) / entry

    thr = alpha * (scale[: n - T + 1] / np.maximum(entry[:, 0], 1e-12))

    up_hits = ret_paths >= thr[:, None]
    dn_hits = ret_paths <= -thr[:, None]

    any_up = up_hits.any(axis=1)
    any_dn = dn_hits.any(axis=1)

    both = any_up & any_dn

    first_up = np.argmax(up_hits[both], axis=1) if both.any() else np.array([])
    first_dn = np.argmax(dn_hits[both], axis=1) if both.any() else np.array([])

    labels[: n - T + 1][any_up & ~any_dn] = 1
    labels[: n - T + 1][~any_up & any_dn] = 0
    if both.any():
        labels[: n - T + 1][both] = (first_up < first_dn).astype(np.int8)

    no_hits = ~(any_up | any_dn)
    labels[: n - T + 1][no_hits] = 2

    return pd.Series(labels, index=df.index, dtype=np.int8)


all_labels = [
    partial(
        calc_label7,
        k=2.0,
        r=0.2,
        lam=0.3,
        T=120,
        cooldown=30,
        side="both",
    ),
    partial(calc_label8, k=3.0, T=60, tau=0.6, cooldown=30, side="both"),
    partial(calc_label9, T=60, alpha=0.5, use_atr=False),
]
