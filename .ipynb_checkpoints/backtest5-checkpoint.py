import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any

import logging
import hashlib

# --- nasz pipeline ---
from load_data import add_VWAP
from filters import filter_clean, filter_hours, filter_around_fomc
from parameters import sigma_val  # JEDYNE źródło prawdy dla sigma


# =========================
# Pomocnicze (logi/TS)
# =========================
logger = logging.getLogger("backtest-debug")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _sig(cols):
    s = ",".join(cols)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def _df_overview(name: str, dfx: pd.DataFrame, show_cols: int = 12):
    logger.info(f"[{name}] len={len(dfx)}, cols={len(dfx.columns)}: {list(dfx.columns)[:show_cols]}")
    if "timestamp" in dfx.columns:
        ts = pd.to_datetime(dfx["timestamp"].astype("int64"), unit="ms", errors="coerce")
        logger.info(
            f"[{name}] ts: nunique={ts.nunique(dropna=True)}, na={ts.isna().sum()}, duplicated={ts.duplicated().sum()}"
        )
        if ts.notna().any():
            logger.info(f"[{name}] ts range: {ts.min()} → {ts.max()}")

def _coerce_features_timestamp(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizuje kolumnę 'timestamp' w features_df do datetime64[ns].
    Zakłada, że w nowych plikach Parquet timestamp już jest datetime64[ns].
    Dodana heurystyka gdyby jednak był numeryczny (epoch ms).
    """
    if "timestamp" not in features_df.columns:
        return features_df

    col = features_df["timestamp"]
    if np.issubdtype(col.dtype, np.datetime64):
        # OK
        return features_df

    # numeryczny → spróbujmy ms, a jak nie — to bez unit
    try:
        ts_try = pd.to_datetime(col, unit="ms", errors="coerce")
        # jeżeli wygląda rozsądnie (rok > 2005), przyjmujemy ms
        if ts_try.notna().any() and ts_try.min().year > 2005:
            features_df = features_df.copy()
            features_df["timestamp"] = ts_try
            return features_df
    except Exception:
        pass

    # fallback
    features_df = features_df.copy()
    features_df["timestamp"] = pd.to_datetime(col, errors="coerce")
    return features_df


# =========================================================
#  BACKTEST – wersja spójna z treningiem i label9
# =========================================================
def backtest(
    df: pd.DataFrame,
    model,
    features_parquet_path: str,          # identycznie jak w treningu: ".../features/featuresXX.parquet"
    scaler_path: str = "scalers/scaler.pkl",
    T: int = 40,
    alpha: float = 0.72,
    use_atr: bool = False,
    leverage: float = 10.0,
    fee_rate: float = 0.00045,
    prob_long_thresh: float = 0.5,
    position_size: float = 0.20
) -> Dict[str, Any]:
    """
    Backtest na świecach ekstremalnych (wg parameters.sigma_val) z decyzją modelu zgodną z label9:
      - pred=1 -> LONG
      - pred=0 -> SHORT
    Wyjście: pierwszy hit +/-thr lub timeout T. Jeden trade na raz (sekwencyjnie).
    Featury: wczytywane z Parquet (tak jak w treningu), kolejność kolumn zachowana.
    Sigma: pobierana wyłącznie z parameters.sigma_val.

    Zwraca:
      {"summary": dict, "trades": DataFrame, "equity_curve": Series}
    """
    # --- 0) kopia i typy ---
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # log parametrów
    logger.info(
        f"PARAMS: sigma_val={sigma_val}, T={T}, alpha={alpha}, use_atr={use_atr}, "
        f"leverage={leverage}, fee_rate_per_side={fee_rate}, prob_long_thresh={prob_long_thresh}, "
        f"position_size={position_size}, scaler_path='{scaler_path}', features_parquet_path='{features_parquet_path}'"
    )

    # --- 1) VWAP + pasma dla sigma_val ---
    logger.info("Dodaję VWAP z sigma_val…")
    df = add_VWAP(df, sigma_mult=sigma_val)
    _df_overview("df_after_add_VWAP", df)

    # --- 2) Wczytaj FEATURY z Parquet (jak w treningu) ---
    logger.info("Wczytuję features z Parquet…")
    features_df = pd.read_parquet(features_parquet_path)
    features_df = _coerce_features_timestamp(features_df)

    # zapewnij prefix 'feature_' dla wszystkich poza 'timestamp'
    non_ts_cols = [c for c in features_df.columns if c != "timestamp"]
    if not all(c.startswith("feature_") for c in non_ts_cols):
        logger.warning("W Parquet wykryto kolumny bez prefixu 'feature_'. Dodaję prefix do wszystkich poza 'timestamp'.")
        rename_map = {c: (c if c == "timestamp" else f"feature_{c}") for c in features_df.columns}
        features_df = features_df.rename(columns=rename_map)

    _df_overview("features_df_raw", features_df)

    len_df, len_feat = len(df), len(features_df)
    logger.info(f"Długości: df={len_df}, features_df={len_feat}")

    # --- 2a) Dopinanie featurów ---
    if len_feat == len_df:
        # concat po indeksie – USUŃ 'timestamp' z featurów, aby nie dublować kolumny
        if "timestamp" in features_df.columns:
            logger.info("Usuwam 'timestamp' z features_df przed concat (unikam duplikatu).")
            features_df = features_df.drop(columns=["timestamp"])

        df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        logger.info(f"[concat] feature_cols n={len(feature_cols)}, sig={_sig(feature_cols)}")

    else:
        # merge po timestamp – pełna diagnostyka
        logger.warning("NIEZGODNOŚĆ długości df vs features_df – próba scalania po 'timestamp'.")
        if "timestamp" not in features_df.columns:
            raise ValueError(
                f"Liczba wierszy features ({len_feat}) != df ({len_df}), a Parquet nie zawiera 'timestamp' – nie można zmergować."
            )

        # sanity i diagnostyka
        ts_df = df["timestamp"]
        ts_feat = features_df["timestamp"]
        logger.info(f"TS unique: df={ts_df.nunique()}, features={ts_feat.nunique()}")
        logger.info(f"TS df range: {ts_df.min()} → {ts_df.max()}")
        logger.info(f"TS feat range: {ts_feat.min()} → {ts_feat.max()}")

        dup_df = ts_df.duplicated().sum()
        dup_feat = ts_feat.duplicated().sum()
        if dup_df or dup_feat:
            logger.warning(f"Duplicated TS: df={dup_df}, features={dup_feat}")

        # merge
        merged = df.merge(features_df, on="timestamp", how="inner", suffixes=("", "_feat"))
        logger.info(f"[merge] wynik inner-merge len={len(merged)} (df_before={len_df}, feat_before={len_feat})")
        if len(merged) == 0:
            raise ValueError("Po inner-merge brak wspólnych wierszy – sprawdź jednostki i strefy czasu.")
        df = merged
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        logger.info(f"[merge] feature_cols n={len(feature_cols)}, sig={_sig(feature_cols)}")

    _df_overview("df_after_features_attach", df)

    # sanity – wymagane kolumny VWAP
    req_vwap_cols = [f"vwap_plus_{sigma_val}_sigma", f"vwap_minus_{sigma_val}_sigma", "vwap", "close"]
    missing_vwap = [c for c in req_vwap_cols if c not in df.columns]
    if missing_vwap:
        raise ValueError(f"Brak krytycznych kolumn VWAP po połączeniu: {missing_vwap}")

    if not feature_cols:
        raise ValueError("feature_cols puste – coś poszło nie tak przy ustalaniu listy cech.")

    # --- 3) Filtry identyczne jak w treningu ---
    logger.info("Buduję maskę allowed (filter_clean, filter_hours(0,8), filter_around_fomc)…")
    idx_clean = filter_clean()(df).index
    idx_hours = filter_hours(0, 8)(df).index
    idx_fomc  = filter_around_fomc()(df).index
    allowed_idx = set(idx_clean).intersection(idx_hours).intersection(idx_fomc)
    allowed_mask = df.index.to_series().isin(allowed_idx).to_numpy()
    logger.info(f"allowed True: {allowed_mask.sum()} / {len(allowed_mask)}")

    # --- 4) Ekstrema względem VWAP (sigma_val) ---
    upper = df["close"].to_numpy() > df[f"vwap_plus_{sigma_val}_sigma"].to_numpy()
    lower = df["close"].to_numpy() < df[f"vwap_minus_{sigma_val}_sigma"].to_numpy()
    is_extreme = upper | lower
    logger.info(f"is_extreme True: {int(is_extreme.sum())} / {len(is_extreme)}")

    # --- 5) Przygotowanie cech -> scaler -> predykcja ---
    logger.info("Czyszczę cechy i skaluję…")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].ffill().bfill()

    X = df[feature_cols].values
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    p = model.predict(X_scaled, verbose=0).ravel()
    pred_cls = (p >= prob_long_thresh).astype(np.int8)
    logger.info(
        f"Pred summary: p∈[{p.min():.4f}, {p.max():.4f}], mean={p.mean():.4f}; "
        f"pred_cls share 1s={pred_cls.mean():.4f}"
    )

    # --- 6) Parametry progów (thr) jak w label9 ---
    closes = df["close"].to_numpy(float)
    vwap   = df["vwap"].to_numpy(float)
    if use_atr and "feature_atr_rel" in df.columns:
        scale_arr = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale_arr = np.abs(closes - vwap) + 1e-12

    # --- 7) Pętla egzekucji (1 trade na raz) ---
    trades = []
    equity = 1.0
    equity_curve = np.full(len(df), np.nan, dtype=float)

    round_trip_fee_on_equity = 2.0 * fee_rate * leverage * position_size

    i = 0
    n = len(df)
    while i < n:
        equity_curve[i] = equity

        # wejście: ekstremum + allowed
        if not (is_extreme[i] and allowed_mask[i]):
            i += 1
            continue

        # Kierunek = WYŁĄCZNIE z predykcji (spójnie z label9)
        direction = +1 if pred_cls[i] == 1 else -1

        entry_idx = i
        entry_time = df["timestamp"].iloc[i]
        entry_price = closes[i]

        thr = alpha * (scale_arr[i] / max(entry_price, 1e-12))

        exit_idx = i
        tp_hit = False
        sl_hit = False

        j = 1
        while j <= T and (i + j) < n:
            exit_idx = i + j
            ret = (closes[exit_idx] - entry_price) / entry_price
            ret_dir = direction * ret

            if ret_dir >= thr:
                tp_hit = True
                break
            if ret_dir <= -thr:
                sl_hit = True
                break
            j += 1

        exit_time = df["timestamp"].iloc[exit_idx]
        exit_price = closes[exit_idx]
        bars_held = exit_idx - entry_idx

        raw_ret = (exit_price - entry_price) / entry_price
        signed_ret = direction * raw_ret

        pnl_on_equity = position_size * leverage * signed_ret - round_trip_fee_on_equity
        equity *= (1.0 + pnl_on_equity)
        equity_curve[exit_idx] = equity

        exit_reason = "TP" if tp_hit else ("SL" if sl_hit else "TIMEOUT")

        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": "LONG" if direction > 0 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "bars_held": int(bars_held),
            "thr": float(thr),
            "tp_hit": bool(tp_hit),
            "sl_hit": bool(sl_hit),
            "exit_reason": exit_reason,
            "prob": float(p[i]),
            "ret_price": float(raw_ret),
            "signed_ret_price": float(signed_ret),
            "pnl_on_equity": float(pnl_on_equity),
            "equity_after": float(equity)
        })

        i = exit_idx + 1  # kolejny trade dopiero po wyjściu

    # uzupełnij equity_curve do końca
    last_equity = equity
    for k in range(len(df)):
        if np.isnan(equity_curve[k]):
            equity_curve[k] = last_equity
        else:
            last_equity = equity_curve[k]

    trades_df = pd.DataFrame(trades)

    # --- 8) Metryki podsumowujące ---
    if not trades_df.empty:
        wins_mask = trades_df["pnl_on_equity"] > 0
        losses_mask = trades_df["pnl_on_equity"] < 0
        breakeven_mask = trades_df["pnl_on_equity"].abs() < 1e-12

        n_trades = len(trades_df)
        n_wins = int(wins_mask.sum())
        n_losses = int(losses_mask.sum())
        n_be = int(breakeven_mask.sum())

        cond_mask = trades_df["exit_reason"].isin(["TP", "SL"])
        timeout_mask = trades_df["exit_reason"].eq("TIMEOUT")
        n_cond = int(cond_mask.sum())
        n_timeout = int(timeout_mask.sum())

        n_tp = int(trades_df["exit_reason"].eq("TP").sum())
        n_sl = int(trades_df["exit_reason"].eq("SL").sum())

        winrate = n_wins / n_trades if n_trades else 0.0
        lossrate = n_losses / n_trades if n_trades else 0.0
        berate = n_be / n_trades if n_trades else 0.0

        cond_rate = n_cond / n_trades if n_trades else 0.0
        timeout_rate = n_timeout / n_trades if n_trades else 0.0
        tp_rate = n_tp / n_trades if n_trades else 0.0
        sl_rate = n_sl / n_trades if n_trades else 0.0

        avg_win = trades_df.loc[wins_mask, "pnl_on_equity"].mean() if n_wins else 0.0
        avg_loss = trades_df.loc[losses_mask, "pnl_on_equity"].mean() if n_losses else 0.0
        expectancy = trades_df["pnl_on_equity"].mean()

        avg_hold = float(trades_df["bars_held"].mean())
        long_share = float((trades_df["direction"] == "LONG").mean())
        final_equity = float(equity_curve[-1])
        dd_series = pd.Series(equity_curve)
        max_dd = float(((dd_series.cummax() - dd_series) / dd_series.cummax()).max())
    else:
        n_trades = n_wins = n_losses = n_be = n_cond = n_timeout = n_tp = n_sl = 0
        winrate = lossrate = berate = cond_rate = timeout_rate = tp_rate = sl_rate = 0.0
        avg_win = avg_loss = expectancy = 0.0
        avg_hold = long_share = max_dd = np.nan
        final_equity = float(equity)

    summary = {
        "n_trades": int(n_trades),
        "wins": int(n_wins),
        "losses": int(n_losses),
        "breakeven": int(n_be),
        "winrate": float(winrate),
        "lossrate": float(lossrate),
        "breakeven_rate": float(berate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "n_closed_condition": int(n_cond),
        "n_closed_timeout": int(n_timeout),
        "cond_rate": float(cond_rate),
        "timeout_rate": float(timeout_rate),
        "n_tp": int(n_tp),
        "n_sl": int(n_sl),
        "tp_rate": float(tp_rate),
        "sl_rate": float(sl_rate),
        "avg_bars_held": float(avg_hold) if not np.isnan(avg_hold) else np.nan,
        "long_share": float(long_share) if not np.isnan(long_share) else np.nan,
        "final_equity": float(final_equity),
        "max_drawdown": float(max_dd) if not np.isnan(max_dd) else np.nan,
        "params": {
            "sigma_mult": float(sigma_val),
            "T": int(T),
            "alpha": float(alpha),
            "use_atr": bool(use_atr),
            "leverage": float(leverage),
            "fee_rate_per_side": float(fee_rate),
            "prob_long_thresh": float(prob_long_thresh),
            "scaler_path": scaler_path,
            "position_size": float(position_size),
        }
    }

    equity_ser = pd.Series(equity_curve, index=df["timestamp"])
    return {"summary": summary, "trades": trades_df, "equity_curve": equity_ser}


# =========================
# PRZYKŁADOWE UŻYCIE
# =========================
if __name__ == "__main__":
    import tensorflow as tf
    import matplotlib.pyplot as plt

    model_path = "models/20251002_184923_calc_label9{T=40;alpha=0.72;use_atr=False}_2.45_256-256-64_0.22-0.17-0.12_relu_binary_crossentropy_0.6271_adam.keras"
    model = tf.keras.models.load_model(model_path)


    crypto = "SOL" 
    csv_path = f"data/training_data/{crypto}_USDT/{crypto}_USDT_1m_data.csv"
    features_parquet_path = "data/training_data/SOL_USDT/features/features01.parquet"

    df = pd.read_csv(csv_path)

    results = backtest(
        df=df,
        model=model,
        features_parquet_path=features_parquet_path,
        scaler_path="scalers/scaler.pkl",
        T=40,
        alpha=0.72,
        use_atr=False,
        leverage=10.0,
        fee_rate=0.0,
        prob_long_thresh=0.5,
        position_size=0.20
    )

    print("\n>>> SUMMARY")
    for k, v in results["summary"].items():
        print(f"{k}: {v}")

    print("\n>>> TRADES (head)")
    print(results["trades"].head())

    # Krzywa kapitału
    results["equity_curve"].to_csv("equity_curve.csv")
    ax = results["equity_curve"].plot(title="Equity Curve")
    ax.set_xlabel("Time"); ax.set_ylabel("Equity")
    plt.show()
