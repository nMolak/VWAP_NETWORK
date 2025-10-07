import numpy as np
import pandas as pd
from calculations import calc_vwap, calc_vwap_sigma


def calc_label1(df, N, phi=0.00075):
    close = df["close"].to_numpy(dtype=float)
    labels = np.full(len(close), np.nan)

    for t in range(len(close) - N):
        if close[t+N] > close[t] * (1 + phi):
            labels[t] = 1
        elif close[t+N] < close[t] * (1 - phi):
            labels[t] = -1
        else:
            labels[t] = 0

    return pd.Series(labels, index=df.index, name=f"label1_N{N}")


def calc_label2(df, N, phi=0.01, psi=0.3):
    close = df["close"].to_numpy(dtype=float)

    if "vwap" in df.columns:
        vwap = df["vwap"].to_numpy(dtype=float)
    else:
        vwap = calc_vwap(df).to_numpy(dtype=float)

    labels = np.full(len(close), np.nan)

    for t in range(len(close) - N):
        dev_t = abs(close[t] - vwap[t]) / vwap[t]
        window_close = close[t+1:t+N+1]
        window_dev = abs(window_close - vwap[t]) / vwap[t]

        if np.min(window_dev) <= phi:
            labels[t] = 1
        elif np.max(window_dev) >= dev_t * (1+psi):
            labels[t] = 0
        else:
            labels[t] = 2

    return pd.Series(labels, index=df.index, name=f"label2_N{N}")


def calc_label3(df, N, psi=0.3, k_sigma=2):
    close = df["close"].to_numpy(dtype=float)

    if "vwap" in df.columns:
        vwap = df["vwap"].to_numpy(dtype=float)
    else:
        vwap = calc_vwap(df).to_numpy(dtype=float)

    sigma = calc_vwap_sigma(df).to_numpy(dtype=float)
    labels = np.full(len(close), np.nan)

    for t in range(len(close) - N):
        dev_t = abs(close[t] - vwap[t]) / vwap[t]
        window_close = close[t+1:t+N+1]

        lower = vwap[t] - k_sigma * sigma[t]
        upper = vwap[t] + k_sigma * sigma[t]

        if np.any((window_close >= lower) & (window_close <= upper)):
            labels[t] = 1
        elif np.max(abs(window_close - vwap[t]) / vwap[t]) >= dev_t * (1+psi):
            labels[t] = 0
        else:
            labels[t] = 2

    return pd.Series(labels, index=df.index, name=f"label3_N{N}")

# --------- Pomocnicze ---------

def _first_hit_in_window(w, tp, sl, prefer='conservative', eps=0.0):
    for _, row in w.iterrows():
        hi = row['high']
        lo = row['low']
        hit_tp = (hi >= tp - eps) if tp is not None else False
        hit_sl = (lo <= sl + eps) if sl is not None else False
        if hit_tp and not hit_sl:
            return True
        if hit_sl and not hit_tp:
            return False
        if hit_tp and hit_sl:
            return True if prefer == 'tp' else False
    return None

def _new_trigger(df_row, k, side):
    from parameters import sigma_val
    """Zwraca ('long'|'short'|None) w zależności od z-score i żądanej strony."""
    z = (df_row['close'] - df_row['vwap']) / df_row[f'sigma_{sigma_val}']
    if np.isnan(z):
        return None
    if side in ('both', 'short') and z >= k:
        return 'short'
    if side in ('both', 'long') and z <= -k:
        return 'long'
    return None

def _get_horizon(df, t0, T):
    """Zwraca pod-DataFrame od razu po t0 do końca horyzontu.
    Obsługuje zarówno DatetimeIndex, jak i Int64Index.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        horizon_end = t0 + pd.Timedelta(minutes=T)
        w = df.loc[(df.index > t0) & (df.index <= horizon_end)]
    else:
        # zakładamy że index jest monotoniczny i unikalny
        pos = df.index.get_loc(t0)
        w = df.iloc[pos+1 : pos+T+1]
    return w

def _horizon_slice(df: pd.DataFrame, t0, T: int) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        t1 = t0 + pd.Timedelta(minutes=T)
        return df.loc[(df.index > t0) & (df.index <= t1)]
    # indeks liczbowy
    pos = df.index.get_loc(t0)
    end = min(pos + T, len(df.index) - 1)
    return df.iloc[pos+1:end+1]


# --------- Label 4: TP-przed-SL (RR na ułamku drogi do VWAP) ---------

def calc_label4(df: pd.DataFrame,
                k: float = 3.0,
                r: float = 0.5,
                lam: float = 0.5,
                T: int = 60,
                cooldown: int = 30,
                side: str = 'both',
                prefer: str = 'conservative',
                eps: float = 0.0) -> pd.Series:

    labels = pd.Series(pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index)

    next_allowed = df.index[0]
    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        trig = _new_trigger(row, k=k, side=side)
        if trig is None:
            continue

        d = abs(row['close'] - row['vwap'])
        if d == 0 or np.isnan(d):
            continue

        if trig == 'long':
            tp = row['close'] + r * d
            sl = row['close'] - lam * d
        else:
            tp = row['close'] - r * d
            sl = row['close'] + lam * d

        w = _get_horizon(df, t0, T)
        win_first = _first_hit_in_window(w, tp, sl, prefer=prefer, eps=eps)

        labels.at[t0] = 1 if win_first else 0
        # cooldown
        if isinstance(df.index, pd.DatetimeIndex):
            next_allowed = t0 + pd.Timedelta(minutes=cooldown)
        else:
            pos = df.index.get_loc(t0)
            if pos + cooldown < len(df.index):
                next_allowed = df.index[pos + cooldown]
            else:
                next_allowed = df.index[-1]

    return labels

# --------- Label 5: Halfway first-passage (połowa drogi vs mała kontynuacja) ---------

def calc_label5(df: pd.DataFrame,
                k: float = 3.0,
                half: float = 0.5,
                lam: float = 0.3,
                T: int = 45,
                cooldown: int = 30,
                side: str = 'both',
                prefer: str = 'conservative',
                eps: float = 0.0) -> pd.Series:

    labels = pd.Series(pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index)
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        trig = _new_trigger(row, k=k, side=side)
        if trig is None:
            continue

        d = abs(row['close'] - row['vwap'])
        if d == 0 or np.isnan(d):
            continue

        if trig == 'long':
            tp = row['close'] + half * d
            sl = row['close'] - lam * d
        else:
            tp = row['close'] - half * d
            sl = row['close'] + lam * d

        w = _get_horizon(df, t0, T)
        win_first = _first_hit_in_window(w, tp, sl, prefer=prefer, eps=eps)

        labels.at[t0] = 1 if win_first else 0

        if isinstance(df.index, pd.DatetimeIndex):
            next_allowed = t0 + pd.Timedelta(minutes=cooldown)
        else:
            pos = df.index.get_loc(t0)
            next_allowed = df.index[min(pos+cooldown, len(df.index)-1)]

    return labels



# --------- Label 6: Edge-ratio MFE vs MAE (w oknie T) ---------

def calc_label6(df: pd.DataFrame,
                k: float = 3.0,
                rho: float = 1.5,
                T: int = 60,
                cooldown: int = 30,
                side: str = 'both') -> pd.Series:
    """
    Label 6 (0/1/NA):
      - 1 = odbicie (MFE >= rho * MAE w horyzoncie T)
      - 0 = brak odbicia (warunek niespełniony)
      - NA = brak triggera
    """

    labels = pd.Series(pd.array([pd.NA] * len(df), dtype="Int8"), index=df.index)
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        trig = _new_trigger(row, k=k, side=side)
        if trig is None:
            continue  # brak triggera => NA

        entry = row['close']
        w = _get_horizon(df, t0, T)[['high', 'low']]

        if w.empty:
            labels.at[t0] = 0
        else:
            if trig == 'long':
                mfe = (w['high'] - entry).clip(lower=0).max()
                mae = (entry - w['low']).clip(lower=0).max()
            else:  # short
                mfe = (entry - w['low']).clip(lower=0).max()
                mae = (w['high'] - entry).clip(lower=0).max()

            if mae == 0:
                label = 1 if mfe > 0 else 0
            else:
                label = 1 if (mfe / mae) >= rho else 0

            labels.at[t0] = np.int8(label)

        # cooldown
        if isinstance(df.index, pd.DatetimeIndex):
            next_allowed = t0 + pd.Timedelta(minutes=cooldown)
        else:
            pos = df.index.get_loc(t0)
            next_allowed = df.index[min(pos+cooldown, len(df.index)-1)]

    return labels

def calc_label7(df: pd.DataFrame,
                k: float = 3.0,      # trigger: |z| >= k
                r: float = 0.5,      # TP = r * |close - vwap| w stronę VWAP
                lam: float = 0.5,    # SL = lam * |close - vwap| w stronę kontynuacji
                T: int = 60,
                cooldown: int = 30,
                prefer: str = "conservative",  # przy kolizji intrabar: 'tp' lub 'conservative' (SL-first)
                side: str = "both") -> pd.Series:
    """
    1: TP (reversion ku VWAP) przed SL (kontynuacja), albo przy braku trafienia bliżej VWAP na końcu; inaczej 0.
    Zawsze zwraca 0/1 (Int8), bez NA.
    """
    # z-score względem bieżącego odchylenia (przy braku kolumny sigma – fallback na bezwzględny dystans)
    if "vwap" not in df.columns:
        raise ValueError("calc_label7 wymaga kolumny 'vwap'.")
    close = df["close"].to_numpy(float)
    vwap  = df["vwap"].to_numpy(float)
    d0    = np.abs(close - vwap)  # baza poziomów

    labels = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        # trigger po z-score (jeśli masz własną kolumnę sigma_x, możesz tu podstawić)
        z = (row["close"] - row["vwap"]) / (abs(row["vwap"]) + 1e-12)
        trig = None
        if side in ("both", "short") and z >= k:
            trig = "short"
        if side in ("both", "long") and z <= -k:
            trig = "long"

        if trig is None:
            # brak triggera też rozstrzygamy binarnie: patrzymy czy w T cena przybliży się do VWAP o >= r*d0
            w = _horizon_slice(df, t0, T)
            if w.empty:
                labels.at[t0] = 0
            else:
                d_start = abs(row["close"] - row["vwap"])
                d_min   = abs(w["close"] - row["vwap"]).min()
                labels.at[t0] = np.int8(1 if (d_start - d_min) >= r * max(d_start, 1e-12) else 0)
            next_allowed = (t0 + pd.Timedelta(minutes=cooldown)) if isinstance(df.index, pd.DatetimeIndex) \
                           else df.index[min(df.index.get_loc(t0)+cooldown, len(df.index)-1)]
            continue

        d = abs(row["close"] - row["vwap"])
        if d <= 0:
            labels.at[t0] = 0
            continue

        if trig == "long":
            tp = row["close"] + r   * d
            sl = row["close"] - lam * d
        else:
            tp = row["close"] - r   * d
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
                    hit = "tp"; break
                if sl_hit and not tp_hit:
                    hit = "sl"; break
                if tp_hit and sl_hit:
                    hit = "tp" if prefer == "tp" else "sl"; break
            if hit is None:
                # fallback: bliżej VWAP na końcu okna?
                end_close = w["close"].iloc[-1]
                label = 1 if abs(end_close - row["vwap"]) < d else 0
            else:
                label = 1 if hit == "tp" else 0
            labels.at[t0] = np.int8(label)

        next_allowed = (t0 + pd.Timedelta(minutes=cooldown)) if isinstance(df.index, pd.DatetimeIndex) \
                       else df.index[min(df.index.get_loc(t0)+cooldown, len(df.index)-1)]
    return labels

def calc_label8(df: pd.DataFrame,
                k: float = 3.0,   # opcjonalny trigger: |z| >= k, jeśli nie chcesz triggera ustaw np. k=0
                T: int = 60,
                tau: float = 0.6, # próg udziału "minut zbliżania się do VWAP"
                cooldown: int = 30,
                side: str = "both") -> pd.Series:
    """
    1: w oknie T odległość |close - vwap| maleje w >= tau części świec; inaczej 0. Zawsze 0/1.
    """
    if "vwap" not in df.columns:
        raise ValueError("calc_label8 wymaga kolumny 'vwap'.")
    labels = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    next_allowed = df.index[0]

    for t0, row in df.iterrows():
        if t0 < next_allowed:
            continue

        z = (row["close"] - row["vwap"]) / (abs(row["vwap"]) + 1e-12)
        if not ((side in ("both","short") and z >= k) or (side in ("both","long") and z <= -k) or (k <= 0)):
            continue

        w = _horizon_slice(df, t0, T)
        if w.empty:
            labels.at[t0] = 0
        else:
            d0 = abs(row["close"] - row["vwap"])
            d_series = (w["close"] - row["vwap"]).abs().values
            # policz, ile kroków zmniejsza dystans względem poprzedniej świecy
            dec = (np.diff(np.r_[d0, d_series]) < 0).sum()
            tot = len(d_series)
            labels.at[t0] = np.int8(1 if (dec / max(tot,1)) >= tau else 0)

        next_allowed = (t0 + pd.Timedelta(minutes=cooldown)) if isinstance(df.index, pd.DatetimeIndex) \
                       else df.index[min(df.index.get_loc(t0)+cooldown, len(df.index)-1)]
    return labels

from filters import filter_clean

# def calc_label9(df: pd.DataFrame,
#                 T: int = 15,
#                 alpha: float = 0.2,   # próg w jednostkach d = |close - vwap|; można podmienić na ATR_rel
#                 use_atr: bool = False) -> pd.Series:
#     """
#     1: w T zwrot >= +alpha*d (lub dodatni na końcu, gdy brak progu),
#     0: w T zwrot <= -alpha*d (lub ujemny na końcu, gdy brak progu). Zawsze 0/1.
#     """
#     if "vwap" not in df.columns:
#         raise ValueError("calc_label9 wymaga kolumny 'vwap'.") # add_VWAP z modułu load_data to powinen dodać przed calc_label
#     labels = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
#     idx = df.index
#
#     if use_atr: # feature_atr_rel powinno być liczone w calc_indicators w module calculations
#         if "feature_atr_rel" in df.columns:
#             scale = df["feature_atr_rel"].to_numpy(float) + 1e-12
#         else:
#             print("Brak feature_atr_rel - liczę scale alternatywnie!")
#             scale = (df["close"] - df["vwap"]).abs().to_numpy(float) + 1e-12
#     else:
#         scale = (df["close"] - df["vwap"]).abs().to_numpy(float) + 1e-12
#
#
#     closes = df["close"].to_numpy(float)
#
#     for i, t0 in enumerate(idx):
#         w = _horizon_slice(df, t0, T)
#         if w.empty:
#             labels.iat[i] = 0
#             continue
#         entry = closes[i]
#         ret_path = (w["close"].to_numpy(float) - entry) / entry #wektor zmian procentowych względem wejścia
#         thr = alpha * (scale[i] / max(entry, 1e-12))  # atr normaliz ujemy względem ceny - ret_path tak samo
#
#         up_hit = (ret_path >= thr).any()
#         dn_hit = (ret_path <= -thr).any()
#
#         if up_hit and not dn_hit:
#             labels.iat[i] = 1
#         elif dn_hit and not up_hit:
#             labels.iat[i] = 0
#         elif up_hit and dn_hit:
#             # tiebreak: kto pierwszy
#             j_up = np.argmax(ret_path >= thr)
#             j_dn = np.argmax(ret_path <= -thr)
#             labels.iat[i] = np.int8(1 if j_up < j_dn else 0)
#         else:
#             # brak trafienia progu → tiebreak na końcu okna
#             labels.iat[i] = np.int8(1 if ret_path[-1] > 0 else 0)
#     return labels

from numpy.lib.stride_tricks import sliding_window_view

def calc_label9(df: pd.DataFrame,
                     T: int = 15,
                     alpha: float = 0.2,
                     use_atr: bool = False) -> pd.Series:
    """
    Wektorowa wersja calc_label9.
    Wynik: Series 0/1, identyczny logicznie z wersją pętlową.
    """

    if "vwap" not in df.columns:
        raise ValueError("calc_label9 wymaga kolumny 'vwap'.")

    closes = df["close"].to_numpy(float)

    if use_atr:
        if "feature_atr_rel" in df.columns:
            scale = df["feature_atr_rel"].to_numpy(float) + 1e-12
        else:
            print("Brak feature_atr_rel - liczę scale alternatywnie!")
            scale = (df["close"] - df["vwap"]).abs().to_numpy(float) + 1e-12
    else:
        scale = (df["close"] - df["vwap"]).abs().to_numpy(float) + 1e-12

    n = len(df)
    labels = np.zeros(n, dtype=np.int8)

    # okna: shape (n - T + 1, T)
    windows = sliding_window_view(closes, T)

    # entry = pierwszy element w każdym oknie
    entry = windows[:, 0][:, None]
    ret_paths = (windows - entry) / entry  # zmiany %

    # threshold (dopasowany do pozycji startowej okna)
    thr = alpha * (scale[:n - T + 1] / np.maximum(entry[:, 0], 1e-12))

    # warunki w całym oknie
    up_hits = (ret_paths >= thr[:, None])
    dn_hits = (ret_paths <= -thr[:, None])

    any_up = up_hits.any(axis=1)
    any_dn = dn_hits.any(axis=1)

    # tie-break: kto pierwszy
    both = any_up & any_dn
    first_up = np.argmax(up_hits[both], axis=1)
    first_dn = np.argmax(dn_hits[both], axis=1)

    # wpisz etykiety
    labels[:n - T + 1][any_up & ~any_dn] = 1
    labels[:n - T + 1][~any_up & any_dn] = 0
    labels[:n - T + 1][both] = (first_up < first_dn).astype(np.int8)
    # brak hitów → porównanie na końcu okna
    no_hits = ~(any_up | any_dn)
    labels[:n - T + 1][no_hits] = (ret_paths[no_hits, -1] > 0).astype(np.int8)

    # ostatnie T-1 elementów, gdzie nie ma pełnego okna → 0 (jak w oryginale)
    # (albo można np. ustawić na NaN – zależy od preferencji)
    return pd.Series(labels, index=df.index, dtype=np.int8)

def calc_label10(df: pd.DataFrame,
                 T: int = 15,
                 alpha: float = 0.2,
                 use_atr: bool = False) -> pd.Series:
    """
    Wektorowa wersja etykiety z trzema klasami:
      0 = ruch spadkowy (osiągnięty próg w dół)
      1 = ruch wzrostowy (osiągnięty próg w górę)
      2 = stagnacja (brak trafienia progu w żadną stronę w horyzoncie T)

    Logika:
      - identyczna jak calc_label9, ale gdy brak hitów w oknie -> klasa 2.
    """

    if "vwap" not in df.columns:
        raise ValueError("calc_label10 wymaga kolumny 'vwap'.")

    closes = df["close"].to_numpy(float)

    if use_atr and "feature_atr_rel" in df.columns:
        scale = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale = np.abs(df["close"] - df["vwap"]).to_numpy(float) + 1e-12

    n = len(df)
    labels = np.full(n, 2, dtype=np.int8)  # domyślnie 2 = stagnacja

    if n < T:
        return pd.Series(labels, index=df.index, dtype=np.int8)

    # okna: shape (n - T + 1, T)
    windows = sliding_window_view(closes, T)
    entry = windows[:, 0][:, None]
    ret_paths = (windows - entry) / entry  # zmiany %

    thr = alpha * (scale[:n - T + 1] / np.maximum(entry[:, 0], 1e-12))

    up_hits = (ret_paths >= thr[:, None])
    dn_hits = (ret_paths <= -thr[:, None])

    any_up = up_hits.any(axis=1)
    any_dn = dn_hits.any(axis=1)

    both = any_up & any_dn

    # tie-break: który próg osiągnięto pierwszy
    first_up = np.argmax(up_hits[both], axis=1) if both.any() else np.array([])
    first_dn = np.argmax(dn_hits[both], axis=1) if both.any() else np.array([])

    # wpisz etykiety
    labels[:n - T + 1][any_up & ~any_dn] = 1
    labels[:n - T + 1][~any_up & any_dn] = 0
    if both.any():
        labels[:n - T + 1][both] = (first_up < first_dn).astype(np.int8)

    # stagnacja = brak hitów w całym oknie
    no_hits = ~(any_up | any_dn)
    labels[:n - T + 1][no_hits] = 2  # stagnacja

    # brak pełnego okna na końcu -> klasa 2 (również stagnacja)
    return pd.Series(labels, index=df.index, dtype=np.int8)

from functools import partial

# all_labels = [
#     #partial(calc_label1, phi=0.0005),
#     #partial(calc_label2, phi=0.01, psi=0.4),
#     #partial(calc_label3, psi=0.25, k_sigma=3),
#     partial(calc_label4, k=3.0, r=0.5, lam=0.5, T=60, cooldown=30),
#     partial(calc_label5, k=3.0, half=0.5, lam=0.3, T=45, cooldown=30),
#     partial(calc_label6, k=3.0, rho=1.5, T=60, cooldown=30),
# ]

all_labels = [
    partial(calc_label7,
            k=2.0,  # zamiast 3σ → więcej triggerów bliżej średniej
            r=0.2,  # TP tylko 20% drogi do VWAP
            lam=0.3,  # SL tylko 30% w stronę kontynuacji
            T=120,  # 2h horyzont
            cooldown=30,
            side="both"),
    partial(calc_label8, k=3.0, T=60, tau=0.6, cooldown=30, side="both"),
    partial(calc_label9, T=60, alpha=0.5, use_atr=False),
]
