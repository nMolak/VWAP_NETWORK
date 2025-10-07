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

from pathlib import Path
from datetime import datetime

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
        col = dfx["timestamp"]
        if np.issubdtype(col.dtype, np.datetime64):
            ts = col
        else:
            # bezpieczna konwersja z heurystyką ms/s
            s = pd.to_numeric(col, errors="coerce")
            max_abs = np.nanmax(np.abs(s.values)) if len(s) else np.nan
            if np.isfinite(max_abs) and max_abs < 1e11:
                ts = pd.to_datetime(s, unit="s", errors="coerce")
            elif np.isfinite(max_abs) and max_abs < 1e14:
                ts = pd.to_datetime(s, unit="ms", errors="coerce")
            else:
                ts = pd.to_datetime(col, errors="coerce")

        logger.info(f"[{name}] ts: nunique={ts.nunique(dropna=True)}, na={ts.isna().sum()}, duplicated={ts.duplicated().sum()}")
        if ts.notna().any():
            logger.info(f"[{name}] ts range: {ts.min()} → {ts.max()}")
def _coerce_features_timestamp(features_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in features_df.columns:
        return features_df
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], unit="ms")
    ts_col = features_df["timestamp"]

    # Jeśli już datetime64 -> zostaw
    if np.issubdtype(ts_col.dtype, np.datetime64):
        return features_df

    s = pd.to_numeric(ts_col, errors="coerce")

    features_df = features_df.copy()

    # Heurystyka:
    #  - wartości ~1e9 -> sekundy (np. 1609459200 ~ 2021-01-01)
    #  - wartości ~1e12 -> milisekundy (np. 1609459200000)
    #  - wszystko inne: spróbuj bez unit
    max_abs = np.nanmax(np.abs(s.values)) if len(s) else np.nan

    if np.isfinite(max_abs):
        if max_abs < 1e11:
            # SEKUNDY
            ts = pd.to_datetime(s, unit="s", errors="coerce")
        elif max_abs < 1e14:
            # MILISEKUNDY
            ts = pd.to_datetime(s, unit="ms", errors="coerce")
        else:
            # coś nietypowego – spróbuj bez unit
            ts = pd.to_datetime(ts_col, errors="coerce")
    else:
        ts = pd.to_datetime(ts_col, errors="coerce")

    features_df["timestamp"] = ts
    return features_df

def parse_label_from_path(model_path: str):
    """
    Parsuje string ze ścieżką modelu i zwraca (T, alpha, use_atr).
    Obsługuje format z parametrami w {...}, np.:
    ..._calc_label9{T=40;alpha=0.3;use_atr=False}_2.15_128-64-32-1_0.6460
    """
    try:
        # wyciągnij fragment między { }
        import re
        m = re.search(r"\{([^}]*)\}", model_path)
        if not m:
            raise ValueError("Nie znaleziono parametrów w { }")
        param_str = m.group(1)  # np. "T=40;alpha=0.3;use_atr=False"

        kv_pairs = dict(x.split("=") for x in param_str.split(";"))
        T = int(kv_pairs["T"])
        alpha = float(kv_pairs["alpha"])
        use_atr = kv_pairs["use_atr"].lower() == "true"
        return T, alpha, use_atr
    except Exception as e:
        raise ValueError(f"Nie udało się sparsować: {e}")

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
    position_size: float = 0.20,         # pozostawione dla zgodności – NIE używane w PnL
    model_name: str = "unnamed_model",
    ticker: str = "UNKNOWN"
) -> Dict[str, Any]:
    """
    Backtest na świecach ekstremalnych (wg parameters.sigma_val) z decyzją modelu zgodną z label9:
      - pred=1 -> LONG
      - pred=0 -> SHORT
    Wyjście: pierwszy hit +/-thr lub timeout T. Jeden trade na raz (sekwencyjnie).

    FEATURY: wczytywane z Parquet (jak w treningu).
    SIGMA:   wyłącznie z parameters.sigma_val.

    NOWOŚĆ: P&L liczony zawsze od stałej kwoty 1000 USD:
      pnl_on_1000 = 1000 * (leverage * signed_ret - 2 * fee_rate * leverage)
    equity_curve to skumulowany P&L w USD.
    """
    # --- 0) kopia i typy ---
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    logger.info(
        f"PARAMS: sigma_val={sigma_val}, T={T}, alpha={alpha}, use_atr={use_atr}, "
        f"leverage={leverage}, fee_rate_per_side={fee_rate}, prob_long_thresh={prob_long_thresh}, "
        f"position_size={position_size} (IGNORED for PnL), scaler_path='{scaler_path}', "
        f"features_parquet_path='{features_parquet_path}'"
    )

    # --- 1) VWAP + pasma ---
    logger.info("Dodaję VWAP z sigma_val…")
    df = add_VWAP(df, sigma_mult=sigma_val)
    _df_overview("df_after_add_VWAP", df)

    # --- 2) Featury z Parquet ---
    logger.info("Wczytuję features z Parquet…")
    features_df = pd.read_parquet(features_parquet_path)
    features_df = _coerce_features_timestamp(features_df)

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
        if "timestamp" in features_df.columns:
            logger.info("Usuwam 'timestamp' z features_df przed concat (unikam duplikatu).")
            features_df = features_df.drop(columns=["timestamp"])
        df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
        logger.info(f"[concat] feature_cols n={len(feature_cols)}, sig={_sig(feature_cols)}")
    else:
        logger.warning("NIEZGODNOŚĆ długości df vs features_df – próba scalania po 'timestamp'.")
        if "timestamp" not in features_df.columns:
            raise ValueError(
                f"Liczba wierszy features ({len_feat}) != df ({len_df}), a Parquet nie zawiera 'timestamp'."
            )
        ts_df = df["timestamp"]
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], unit="ms")
        ts_feat = features_df["timestamp"]
        logger.info(f"TS unique: df={ts_df.nunique()}, features={ts_feat.nunique()}")
        logger.info(f"TS df range: {ts_df.min()} → {ts_df.max()}")
        logger.info(f"TS feat range: {ts_feat.min()} → {ts_feat.max()}")
        dup_df = ts_df.duplicated().sum()
        dup_feat = ts_feat.duplicated().sum()
        if dup_df or dup_feat:
            logger.warning(f"Duplicated TS: df={dup_df}, features={dup_feat}")
        merged = df.merge(features_df, on="timestamp", how="inner", suffixes=("", "_feat"))
        logger.info(f"[merge] wynik inner-merge len={len(merged)} (df_before={len_df}, feat_before={len_feat})")
        if len(merged) == 0:
            raise ValueError("Po inner-merge brak wspólnych wierszy – sprawdź jednostki/strefy czasu.")
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

    # --- 3) Filtry (jak w treningu) ---
    logger.info("Buduję maskę allowed (filter_clean, filter_hours(0,8), filter_around_fomc)…")
    idx_clean = filter_clean()(df).index
    idx_hours = filter_hours(0, 8)(df).index
    idx_fomc  = filter_around_fomc()(df).index
    allowed_idx = set(idx_clean).intersection(idx_hours).intersection(idx_fomc)
    allowed_mask = df.index.to_series().isin(allowed_idx).to_numpy()
    logger.info(f"allowed True: {allowed_mask.sum()} / {len(allowed_mask)}")

    # --- 4) Ekstrema (sigma_val) ---
    upper = df["close"].to_numpy() > df[f"vwap_plus_{sigma_val}_sigma"].to_numpy()
    lower = df["close"].to_numpy() < df[f"vwap_minus_{sigma_val}_sigma"].to_numpy()
    is_extreme = upper | lower
    logger.info(f"is_extreme True: {int(is_extreme.sum())} / {len(is_extreme)}")

    # --- 5) Featury -> scaler -> predykcja ---
    logger.info("Czyszczę cechy i skaluję…")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].ffill().bfill()
    X = df[feature_cols].values
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    p = model.predict(X_scaled, verbose=0).ravel()
    pred_cls = (p >= prob_long_thresh).astype(np.int8)
    logger.info(
        f"Pred summary: p∈[{p.min():.4f}, {p.max():.4f}], mean={p.mean():.4f}; pred_cls share 1s={pred_cls.mean():.4f}"
    )

    # --- 6) Progi (thr) jak w label9 ---
    closes = df["close"].to_numpy(float)
    vwap   = df["vwap"].to_numpy(float)
    if use_atr and "feature_atr_rel" in df.columns:
        scale_arr = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale_arr = np.abs(closes - vwap) + 1e-12

    def should_enter(i, is_extreme, allowed_mask):
        """Zwraca True/False – czy otwieramy pozycję w barze i."""
        return bool(is_extreme[i] and allowed_mask[i])

    def simulate_trade(i, df, closes, pred_cls, p, alpha, scale_arr, T, leverage, fee_rate, per_trade_notional):
        """Symuluje 1 trade od wejścia do wyjścia, zwraca dict z pełnym logiem (bez equity_after)."""
        entry_idx = i
        entry_time = df["timestamp"].iloc[i]
        entry_price = closes[i]
        direction = +1 if pred_cls[i] == 1 else -1
        prob = float(p[i])

        thr = alpha * (scale_arr[i] / max(entry_price, 1e-12))

        n = len(df)
        tp_hit = sl_hit = False
        exit_idx = i
        for j in range(1, T + 1):
            if (i + j) >= n:
                break
            exit_idx = i + j
            ret = (closes[exit_idx] - entry_price) / entry_price
            ret_dir = direction * ret
            if ret_dir >= thr:
                tp_hit = True
                break
            if ret_dir <= -thr:
                sl_hit = True
                break

        exit_time = df["timestamp"].iloc[exit_idx]
        exit_price = closes[exit_idx]
        bars_held = exit_idx - entry_idx

        raw_ret = (exit_price - entry_price) / entry_price
        signed_ret = direction * raw_ret
        round_trip_fee_frac = 2.0 * fee_rate * leverage
        pnl_on_1000 = per_trade_notional * (leverage * signed_ret - round_trip_fee_frac)
        exit_reason = "TP" if tp_hit else ("SL" if sl_hit else "TIMEOUT")

        return {
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
            "prob": prob,  # predykcja modelu
            "ret_price": float(raw_ret),  # zmiana ceny %
            "signed_ret_price": float(signed_ret),  # zmiana w kierunku pozycji %
            "PNL_on_1000": float(pnl_on_1000),  # USD
            # equity_after dodamy w pętli
        }

    def update_equity_curve(equity_curve, entry_idx, exit_idx, equity_pnl):
        """Uzupełnia equity_curve pomiędzy trade’ami."""
        equity_curve[exit_idx] = equity_pnl
        return equity_curve

    # --- 7) Pętla egzekucji (1 trade na raz) ---
    trades = []
    equity_pnl = 0.0  # skumulowany P&L w USD
    equity_curve = np.full(len(df), np.nan, dtype=float)
    per_trade_notional = 1000.0
    i = 0
    n = len(df)

    while i < n:
        equity_curve[i] = equity_pnl

        # --- Wejście: ekstremum + allowed ---
        if not should_enter(i, is_extreme, allowed_mask):
            i += 1
            continue

        # --- Symulacja pojedynczego trade’a ---
        trade = simulate_trade(
            i, df, closes, pred_cls, p,
            alpha, scale_arr, T, leverage, fee_rate, per_trade_notional
        )

        # --- Aktualizacja equity ---
        equity_pnl += trade["PNL_on_1000"]
        trade["equity_after"] = float(equity_pnl)
        trades.append(trade)

        # --- Aktualizacja krzywej equity ---
        equity_curve = update_equity_curve(
            equity_curve,
            trade["entry_idx"],
            trade["exit_idx"],
            equity_pnl
        )

        # --- Kolejny trade po wyjściu ---
        i = trade["exit_idx"] + 1

        # (opcjonalny log postępu co 100 transakcji)
        if len(trades) % 100 == 0 and len(trades) > 0:
            logger.info(f"[{ticker}] Trades={len(trades)}, equity={equity_pnl:+.2f} USD, progress={i / n:.1%}")

    # --- Uzupełnij equity_curve do końca ---
    last_equity_val = equity_pnl
    for k in range(len(df)):
        if np.isnan(equity_curve[k]):
            equity_curve[k] = last_equity_val
        else:
            last_equity_val = equity_curve[k]

    trades_df = pd.DataFrame(trades)

    # --- 8) Metryki podsumowujące (już w USD) ---
    if not trades_df.empty:
        wins_mask = trades_df["PNL_on_1000"] > 0
        losses_mask = trades_df["PNL_on_1000"] < 0
        breakeven_mask = trades_df["PNL_on_1000"].abs() < 1e-10

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

        avg_win = trades_df.loc[wins_mask, "PNL_on_1000"].mean() if n_wins else 0.0
        avg_loss = trades_df.loc[losses_mask, "PNL_on_1000"].mean() if n_losses else 0.0
        expectancy = trades_df["PNL_on_1000"].mean()

        # DD na krzywej P&L (USD). Opcjonalnie można dodać stałą bazę, np. 0 -> 100k.
        eq_series = pd.Series(equity_curve, dtype=float)
        running_max = eq_series.cummax()
        with np.errstate(divide='ignore', invalid='ignore'):
            dd = (running_max - eq_series) / running_max.replace(0, np.nan)
        max_dd = float(np.nanmax(dd.values)) if len(dd) else np.nan

        avg_hold = float(trades_df["bars_held"].mean())
        long_share = float((trades_df["direction"] == "LONG").mean())
        final_equity = float(equity_curve[-1])  # to skumulowany P&L USD
    else:
        n_trades = n_wins = n_losses = n_be = n_cond = n_timeout = n_tp = n_sl = 0
        winrate = lossrate = berate = cond_rate = timeout_rate = tp_rate = sl_rate = 0.0
        avg_win = avg_loss = expectancy = 0.0
        avg_hold = long_share = max_dd = np.nan
        final_equity = float(equity_pnl)

    summary = {
        "n_trades": int(n_trades),
        "wins": int(n_wins),
        "losses": int(n_losses),
        "breakeven": int(n_be),
        "winrate": float(winrate),
        "lossrate": float(lossrate),
        "breakeven_rate": float(berate),
        "avg_win_usd": float(avg_win),
        "avg_loss_usd": float(avg_loss),
        "expectancy_usd": float(expectancy),
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
        "final_equity_usd": float(final_equity),   # skumulowany P&L USD
        "max_drawdown_rel": float(max_dd) if not np.isnan(max_dd) else np.nan,
        "params": {
            "sigma_mult": float(sigma_val),
            "T": int(T),
            "alpha": float(alpha),
            "use_atr": bool(use_atr),
            "leverage": float(leverage),
            "fee_rate_per_side": float(fee_rate),
            "prob_long_thresh": float(prob_long_thresh),
            "scaler_path": scaler_path,
            "position_size_IGNORED": float(position_size),
            "per_trade_notional_usd": float(per_trade_notional),
        }
    }

    equity_ser = pd.Series(equity_curve, index=df["timestamp"])
    results = {"summary": summary, "trades": trades_df, "equity_curve": equity_ser}

    # === 9) Eksport wyników ===
    output_dir = Path("models") / model_name / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_csv = output_dir / f"{ticker}_trades.csv"
    trades_df.to_csv(trades_csv, index=False)
    logger.info(f"Zapisano wszystkie transakcje: {trades_csv}")

    log_txt = output_dir / f"{ticker}_trades_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(f"# TRADES LOG – pełna lista transakcji\n")
        f.write(f"# Model: {model_name}\n")
        f.write(f"# Ticker: {ticker}\n")
        f.write(
            f"# Parametry: sigma={summary['params']['sigma_mult']}, T={T}, alpha={alpha}, "
            f"use_atr={use_atr}, leverage={leverage}, fee_rate={fee_rate}, "
            f"per_trade_notional={per_trade_notional}\n"
        )
        f.write(f"# Utworzono: {datetime.now().isoformat()}\n\n")
        for _, r in trades_df.iterrows():
            f.write(
                f"ENTRY {r['entry_time']} | {r['direction']} | price={r['entry_price']:.4f} | prob={r['prob']:.3f} | thr={r['thr']:.6f}\n"
                f"EXIT  {r['exit_time']} | reason={r['exit_reason']} | price={r['exit_price']:.4f} | "
                f"Δ%_price={r['ret_price'] * 100:+.4f}% | Δ%_signed={r['signed_ret_price'] * 100:+.4f}% | bars={r['bars_held']}\n"
                f"PNL_on_1000={r['PNL_on_1000']:+.6f} | equity_after={r['equity_after']:+.6f}\n"
                "----\n"
            )
    logger.info(f"Zapisano log transakcji: {log_txt}")

    return results


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_equity_vs_price(results: dict,
                         df: pd.DataFrame,
                         crypto: str,
                         model_name: str,
                         data_mode: str = "test_data",
                         T: int = 40,
                         alpha: float = 0.72,
                         leverage: float = 10.0,
                         fee: float = 0.00045,
                         do_plot: bool = False):
    """
    Rysuje i zapisuje wykres Equity Curve vs Price.
    - Kolor equity: zielony, jeśli końcowe equity > 1.0, inaczej czerwony.
    - Jeśli do_plot=False → nie pokazuje wykresu (tylko zapisuje PNG).
    - Wykres zapisuje się pod: models/{model_name}/{crypto}_USDT/equity_curve.png
    """
    # --- Equity curve ---
    ec = results["equity_curve"].copy()
    ec.name = "Equity"
    equity_color = "green" if float(ec.iloc[-1]) > 1.0 else "red"

    # --- Cena (dopasowanie indeksu) ---
    price = df["close"].copy()
    if "timestamp" in df.columns:
        price.index = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        price.index = pd.to_datetime(df["date"])
    elif df.index.name is None or not np.issubdtype(df.index.dtype, np.datetime64):
        try:
            price.index = pd.to_datetime(df.index)
        except Exception:
            pass

    price = price.reindex(ec.index, method="ffill")
    price.name = f"{crypto}/USDT price"

    # --- Rysowanie ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    (line_equity,) = ax1.plot(ec.index, ec.values, lw=2.0, label="Equity", color=equity_color)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    (line_price,) = ax2.plot(price.index, price.values, lw=1.2, label=f"{crypto}/USDT", color="orange", alpha=0.85)
    ax2.set_ylabel("Price")

    # --- Tytuł ---
    plt.title(f"Equity vs Price — {crypto} | mode={data_mode}, T={T}, α={alpha}, lev={leverage}x, fee={fee:.4f}")

    # --- Legenda ---
    lines = [line_equity, line_price]
    labels = [l.get_label() for l in lines]
    leg = ax1.legend(lines, labels, loc="upper left", frameon=True, framealpha=0.9,
                     borderpad=0.6, title="Legend")
    leg.get_title().set_fontweight("bold")

    plt.tight_layout()

    # --- Zapis wykresu ---
    out_dir = Path("models") / model_name / f"{crypto}_USDT"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "equity_curve.png"

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Wykres zapisano w: {out_path}")

    # --- Pokazuj tylko jeśli do_plot=True ---
    if do_plot:
        plt.show()
    else:
        plt.close(fig)

# =========================
# PRZYKŁADOWE UŻYCIE
# =========================
# if __name__ == "__main__":
#     import tensorflow as tf
#     import matplotlib.pyplot as plt
#
#     for ticker_folder in Path("data/test_data").iterdir():
#         if ticker_folder.is_dir():
#             safe_ticker = ticker_folder.name
#
#
#
#
#     model_name = "20251002_184923_calc_label9{T=40;alpha=0.72;use_atr=False}_2.45_256-256-64_0.22-0.17-0.12_relu_binary_crossentropy_0.6271_adam"
#     model_path = f"models/{model_name}/{model_name}.keras"
#     model = tf.keras.models.load_model(model_path)
#
#
#     crypto = "XLM"
#     safe_ticker = f"{crypto}_USDT"
#     data_mode = "test_data"
#     csv_path = f"data/{data_mode}/{crypto}_USDT/{crypto}_USDT_1m_data.csv"
#     features_parquet_path = f"data/{data_mode}/{crypto}_USDT/features/features00.parquet"
#
#
#     df = pd.read_csv(csv_path)
#
#     T, alpha, use_atr = parse_label_from_path(model_path)
#     leverage = 10.0
#     fee_rate = 0.0
#     prob_long_thresh = 0.5
#     position_size = 0.20
#
#     results = backtest(
#         df=df,
#         model=model,
#         features_parquet_path=features_parquet_path,
#         scaler_path="scalers/scaler.pkl",
#         T=T,
#         alpha=alpha,
#         use_atr=use_atr,
#         leverage=leverage,
#         fee_rate=fee_rate,
#         prob_long_thresh=prob_long_thresh,
#         position_size=position_size,
#         model_name=model_name,
#         ticker=safe_ticker
#     )
#
#     print("\n>>> SUMMARY")
#     for k, v in results["summary"].items():
#         print(f"{k}: {v}")
#
#     print("\n>>> TRADES (head)")
#     print(results["trades"].head())
#
#     plot_equity_vs_price(
#         results=results,
#         df=df,
#         crypto=crypto,
#         model_name=model_name,
#         T=T,
#         alpha=alpha,
#         leverage=leverage,
#         fee=fee_rate,
#         do_plot=False
#     )







# if __name__ == "__main__":
#     import tensorflow as tf
#     import matplotlib.pyplot as plt
#
#     model_name = "20251002_184908_calc_label9{T=45;alpha=0.725;use_atr=False}_2.45_64-256-128-32-16_0.25-0.20-0.15-0.10-0.05_relu_binary_crossentropy_0.6003_adam"
#     model_path = f"models/{model_name}/{model_name}.keras"
#     model = tf.keras.models.load_model(model_path)
#
#
#
#     for ticker_folder in Path("data/test_data").iterdir():
#         if ticker_folder.is_dir():
#             safe_ticker = ticker_folder.name
#
#
#
#             crypto = safe_ticker.split("_")[0]
#             data_mode = "test_data"
#             csv_path = f"data/{data_mode}/{crypto}_USDT/{crypto}_USDT_1m_data.csv"
#             features_parquet_path = f"data/{data_mode}/{crypto}_USDT/features/features00.parquet"
#
#             df = pd.read_csv(csv_path)
#
#             T, alpha, use_atr = parse_label_from_path(model_path)
#             leverage = 10.0
#             fee_rate = 0.0002
#             prob_long_thresh = 0.50
#             position_size = 0.20
#
#             results = backtest(
#                 df=df,
#                 model=model,
#                 features_parquet_path=features_parquet_path,
#                 scaler_path="scalers/scaler.pkl",
#                 T=T,
#                 alpha=alpha,
#                 use_atr=use_atr,
#                 leverage=leverage,
#                 fee_rate=fee_rate,
#                 prob_long_thresh=prob_long_thresh,
#                 position_size=position_size,
#                 model_name=model_name,
#                 ticker=safe_ticker
#             )
#
#             print("\n>>> SUMMARY")
#             for k, v in results["summary"].items():
#                 print(f"{k}: {v}")
#
#             print("\n>>> TRADES (head)")
#             print(results["trades"].head())
#
#             plot_equity_vs_price(
#                 results=results,
#                 df=df,
#                 crypto=crypto,
#                 model_name=model_name,
#                 T=T,
#                 alpha=alpha,
#                 leverage=leverage,
#                 fee=fee_rate,
#                 do_plot=False
#             )





import tensorflow as tf
import pandas as pd
from pathlib import Path

EXCLUDED = {
    "20251002_184908_calc_label9{T=45;alpha=0.725;use_atr=False}_2.45_64-256-128-32-16_0.25-0.20-0.15-0.10-0.05_relu_binary_crossentropy_0.6003_adam",
    "20251002_184923_calc_label9{T=40;alpha=0.72;use_atr=False}_2.45_256-256-64_0.22-0.17-0.12_relu_binary_crossentropy_0.6271_adam"
}

# if __name__ == "__main__":
#
#     models_dir = Path("models/2_class")
#
#     for model_folder in models_dir.iterdir():
#         if not model_folder.is_dir() or model_folder.name in EXCLUDED:
#             continue
#
#         model_name = model_folder.name
#         model_path = model_folder / f"{model_name}.keras"
#
#         if not model_path.exists():
#             print(f"[!] Pomijam {model_name} – brak pliku .keras")
#             continue
#
#         print(f"\n=== Przetwarzam model: {model_name} ===")
#         model = tf.keras.models.load_model(model_path)
#
#         for ticker_folder in Path("data/test_data").iterdir():
#             if not ticker_folder.is_dir():
#                 continue
#
#             safe_ticker = ticker_folder.name
#             crypto = safe_ticker.split("_")[0]
#
#             csv_path = f"data/test_data/{crypto}_USDT/{crypto}_USDT_1m_data.csv"
#             features_parquet_path = f"data/test_data/{crypto}_USDT/features/features00.parquet"
#
#             df = pd.read_csv(csv_path)
#
#             T, alpha, use_atr = parse_label_from_path(str(model_path))
#             leverage = 10.0
#             fee_rate = 0.0002
#             prob_long_thresh = 0.50
#             position_size = 0.20
#
#             results = backtest(
#                 df=df,
#                 model=model,
#                 features_parquet_path=features_parquet_path,
#                 scaler_path="scalers/scaler.pkl",
#                 T=T,
#                 alpha=alpha,
#                 use_atr=use_atr,
#                 leverage=leverage,
#                 fee_rate=fee_rate,
#                 prob_long_thresh=prob_long_thresh,
#                 position_size=position_size,
#                 model_name=model_name,
#                 ticker=safe_ticker
#             )
#
#             print(f"\n>>> SUMMARY ({model_name} – {safe_ticker})")
#             for k, v in results["summary"].items():
#                 print(f"{k}: {v}")
#
#             print(f"\n>>> TRADES (head) – {safe_ticker}")
#             print(results["trades"].head())
#
#             plot_equity_vs_price(
#                 results=results,
#                 df=df,
#                 crypto=crypto,
#                 model_name=model_name,
#                 T=T,
#                 alpha=alpha,
#                 leverage=leverage,
#                 fee=fee_rate,
#                 do_plot=False
#             )

from sklearn.metrics import accuracy_score

# === Funkcja diagnostyczna ===
def evaluate_thresholds_softmax(y_true, probs, thresholds=None):
    """
    Sprawdza accuracy w zależności od progu (dla modelu 2-klasowego softmax).
    probs: macierz (n,2) – prawdopodobieństwa klas 0/1
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)

    results = []
    for thr in thresholds:
        y_pred = np.where(probs[:, 1] > thr, 1, 0)
        acc = accuracy_score(y_true, y_pred)
        results.append((thr, acc))
    df = pd.DataFrame(results, columns=["threshold", "accuracy"])
    return df


# === Główny blok ===
if __name__ == "__main__":

    # --- Ścieżki ---
    model_name_path = Path("models/2_class/20251002_184908_calc_label9{T=45;alpha=0.725;use_atr=False}_2.45_64-256-128-32-16_0.25-0.20-0.15-0.10-0.05_relu_binary_crossentropy_0.6003_adam")
    model_path = model_name_path / Path("20251002_184908_calc_label9{T=45;alpha=0.725;use_atr=False}_2.45_64-256-128-32-16_0.25-0.20-0.15-0.10-0.05_relu_binary_crossentropy_0.6003_adam.keras")
    csv_path = Path("data/test_data/BTC_USDT/BTC_USDT_1m_data.csv")
    features_path = Path("data/test_data/BTC_USDT/features/features00.parquet")
    scaler_path = Path("scalers/scaler.pkl")
    # --- Wczytanie modelu ---
    print(f"📦 Wczytuję model z: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model poprawnie wczytany!")

    # --- Dane ---
    print(f"📊 Wczytuję dane z: {csv_path}")
    df = pd.read_csv(csv_path)
    features_df = pd.read_parquet(features_path)

    # usuń timestamp, jeśli istnieje
    if "timestamp" in features_df.columns:
        features_df = features_df.drop(columns=["timestamp"])

    X = features_df.values

    # --- Label z calc_label9 ---
    from labels import calc_label9

    T, alpha, use_atr = parse_label_from_path(str(model_path))
    from load_data import add_VWAP

    df = add_VWAP(df, sigma_val)
    y_true = calc_label9(df, T=T, alpha=alpha, use_atr=use_atr).to_numpy()

    # --- Skalowanie ---
    print(f"🧮 Wczytuję scaler z: {scaler_path}")
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # --- Predykcje ---
    print("🤖 Generuję predykcje...")
    probs = model.predict(X_scaled, verbose=0).ravel()  # shape (N,)


    # --- Ewaluacja ---
    def evaluate_thresholds_sigmoid(y_true, probs, thresholds=np.linspace(0, 1, 21)):
        """
        Ewaluacja binary-class modelu (sigmoid) dla różnych progów decyzyjnych.
        """
        results = []
        for thr in thresholds:
            y_pred = (probs > thr).astype(int)
            acc = (y_pred == y_true).mean()
            results.append((thr, acc))
        df = pd.DataFrame(results, columns=["threshold", "accuracy"])
        print("\n=== 📈 Accuracy vs Threshold (sigmoid) ===")
        print(df.to_string(index=False))
        return df


    df_eval = evaluate_thresholds_sigmoid(y_true, probs)

    # --- Najlepszy próg ---
    best_thr = df_eval.loc[df_eval["accuracy"].idxmax()]
    print(f"\n🏆 Najlepszy próg: {best_thr['threshold']:.2f} → accuracy={best_thr['accuracy']:.4f}")