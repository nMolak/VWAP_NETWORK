"""
Backtest v8 — symulacja handlu na swiecach ekstremalnych z decyzja modelu MLP (PyTorch).

Logika:
    pred=1 → LONG, pred=0 → SHORT.
    Wyjscie: pierwszy hit +/-thr lub timeout T.
    Jeden trade na raz (sekwencyjnie).
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from filters import filter_around_fomc, filter_clean, filter_hours
from load_data import add_VWAP
from parameters import sigma_val
from training_weights import VWAPModel, load_vwap_model, predict


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

logger = logging.getLogger("backtest-debug")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _sig(cols: list[str]) -> str:
    s = ",".join(cols)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def _df_overview(name: str, dfx: pd.DataFrame, show_cols: int = 12) -> None:
    logger.info(f"[{name}] len={len(dfx)}, cols={len(dfx.columns)}: {list(dfx.columns)[:show_cols]}")
    if "timestamp" not in dfx.columns:
        return
    col = dfx["timestamp"]
    if np.issubdtype(col.dtype, np.datetime64):
        ts = col
    else:
        s = pd.to_numeric(col, errors="coerce")
        max_abs = np.nanmax(np.abs(s.values)) if len(s) else np.nan
        if np.isfinite(max_abs) and max_abs < 1e11:
            ts = pd.to_datetime(s, unit="s", errors="coerce")
        elif np.isfinite(max_abs) and max_abs < 1e14:
            ts = pd.to_datetime(s, unit="ms", errors="coerce")
        else:
            ts = pd.to_datetime(col, errors="coerce")
    logger.info(f"[{name}] ts: nunique={ts.nunique(dropna=True)}, na={ts.isna().sum()}, dup={ts.duplicated().sum()}")
    if ts.notna().any():
        logger.info(f"[{name}] ts range: {ts.min()} -> {ts.max()}")


def _coerce_features_timestamp(features_df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in features_df.columns:
        return features_df
    features_df = features_df.copy()
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], unit="ms")
    ts_col = features_df["timestamp"]
    if np.issubdtype(ts_col.dtype, np.datetime64):
        return features_df
    s = pd.to_numeric(ts_col, errors="coerce")
    max_abs = np.nanmax(np.abs(s.values)) if len(s) else np.nan
    if np.isfinite(max_abs):
        if max_abs < 1e11:
            ts = pd.to_datetime(s, unit="s", errors="coerce")
        elif max_abs < 1e14:
            ts = pd.to_datetime(s, unit="ms", errors="coerce")
        else:
            ts = pd.to_datetime(ts_col, errors="coerce")
    else:
        ts = pd.to_datetime(ts_col, errors="coerce")
    features_df["timestamp"] = ts
    return features_df


def parse_label_from_path(model_path: str) -> tuple[int, float, bool]:
    """Parsuje (T, alpha, use_atr) z nazwy modelu (fragment ``{...}``)."""
    m = re.search(r"\{([^}]*)\}", model_path)
    if not m:
        raise ValueError("Nie znaleziono parametrow w { }")
    kv_pairs = dict(x.split("=") for x in m.group(1).split(";"))
    return int(kv_pairs["T"]), float(kv_pairs["alpha"]), kv_pairs["use_atr"].lower() == "true"


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(
    df: pd.DataFrame,
    model: VWAPModel,
    features_parquet_path: str,
    scaler_path: str = "scalers/scaler.pkl",
    T: int = 40,
    alpha: float = 0.72,
    use_atr: bool = False,
    leverage: float = 10.0,
    fee_rate: float = 0.00045,
    prob_long_thresh: float = 0.5,
    position_size: float = 0.20,
    model_name: str = "unnamed_model",
    ticker: str = "UNKNOWN",
) -> Dict[str, Any]:
    """Backtest na swiecach ekstremalnych z decyzja modelu.

    P&L liczony od stalej kwoty 1000 USD:
        pnl_on_1000 = 1000 * (leverage * signed_ret - 2 * fee_rate * leverage)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    logger.info(
        f"PARAMS: sigma_val={sigma_val}, T={T}, alpha={alpha}, use_atr={use_atr}, "
        f"leverage={leverage}, fee_rate={fee_rate}, prob_long_thresh={prob_long_thresh}"
    )

    # 1. VWAP + pasma
    df = add_VWAP(df, sigma_mult=sigma_val)
    _df_overview("df_after_add_VWAP", df)

    # 2. Featury z Parquet
    logger.info("Wczytuje features z Parquet...")
    features_df = pd.read_parquet(features_parquet_path)
    features_df = _coerce_features_timestamp(features_df)

    non_ts_cols = [c for c in features_df.columns if c != "timestamp"]
    if not all(c.startswith("feature_") for c in non_ts_cols):
        rename_map = {c: (c if c == "timestamp" else f"feature_{c}") for c in features_df.columns}
        features_df = features_df.rename(columns=rename_map)

    _df_overview("features_df_raw", features_df)

    len_df, len_feat = len(df), len(features_df)
    logger.info(f"Dlugosci: df={len_df}, features_df={len_feat}")

    # 2a. Dolaczenie featurow
    if len_feat == len_df:
        if "timestamp" in features_df.columns:
            features_df = features_df.drop(columns=["timestamp"])
        df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
        feature_cols = [c for c in features_df.columns if c.startswith("feature_")]
    else:
        logger.warning("NIEZGODNOSC dlugosci df vs features_df — proba scalania po timestamp.")
        if "timestamp" not in features_df.columns:
            raise ValueError(f"Niezgodnosc dlugosci ({len_feat} vs {len_df}) i brak timestamp w Parquet.")
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], unit="ms")
        merged = df.merge(features_df, on="timestamp", how="inner", suffixes=("", "_feat"))
        if len(merged) == 0:
            raise ValueError("Po inner-merge brak wspolnych wierszy.")
        df = merged
        feature_cols = [c for c in df.columns if c.startswith("feature_")]

    # Walidacja kolumn
    req_vwap_cols = [f"vwap_plus_{sigma_val}_sigma", f"vwap_minus_{sigma_val}_sigma", "vwap", "close"]
    missing = [c for c in req_vwap_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brak krytycznych kolumn VWAP: {missing}")
    if not feature_cols:
        raise ValueError("feature_cols puste — cos poszlo nie tak przy ustalaniu listy cech.")

    # 3. Filtry
    idx_clean = filter_clean()(df).index
    idx_hours = filter_hours(0, 8)(df).index
    idx_fomc = filter_around_fomc()(df).index
    allowed_idx = set(idx_clean) & set(idx_hours) & set(idx_fomc)
    allowed_mask = df.index.to_series().isin(allowed_idx).to_numpy()

    # 4. Ekstrema
    upper = df["close"].to_numpy() > df[f"vwap_plus_{sigma_val}_sigma"].to_numpy()
    lower = df["close"].to_numpy() < df[f"vwap_minus_{sigma_val}_sigma"].to_numpy()
    is_extreme = upper | lower

    # 5. Predykcja
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].ffill().bfill()
    X = df[feature_cols].values
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    p = predict(model, X_scaled)
    pred_cls = (p >= prob_long_thresh).astype(np.int8)

    # 6. Progi (jak w label9)
    closes = df["close"].to_numpy(float)
    vwap = df["vwap"].to_numpy(float)
    if use_atr and "feature_atr_rel" in df.columns:
        scale_arr = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale_arr = np.abs(closes - vwap) + 1e-12

    # 7. Petla egzekucji
    trades: list[dict] = []
    equity_pnl = 0.0
    equity_curve = np.full(len(df), np.nan, dtype=float)
    per_trade_notional = 1000.0
    i = 0
    n = len(df)

    while i < n:
        equity_curve[i] = equity_pnl

        if not (is_extreme[i] and allowed_mask[i]):
            i += 1
            continue

        # Symulacja trade'a
        entry_price = closes[i]
        direction = +1 if pred_cls[i] == 1 else -1
        thr = alpha * (scale_arr[i] / max(entry_price, 1e-12))

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

        raw_ret = (closes[exit_idx] - entry_price) / entry_price
        signed_ret = direction * raw_ret
        round_trip_fee_frac = 2.0 * fee_rate * leverage
        pnl_on_1000 = per_trade_notional * (leverage * signed_ret - round_trip_fee_frac)
        exit_reason = "TP" if tp_hit else ("SL" if sl_hit else "TIMEOUT")

        equity_pnl += pnl_on_1000
        trade = {
            "entry_idx": i,
            "exit_idx": exit_idx,
            "entry_time": df["timestamp"].iloc[i],
            "exit_time": df["timestamp"].iloc[exit_idx],
            "direction": "LONG" if direction > 0 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(closes[exit_idx]),
            "bars_held": int(exit_idx - i),
            "thr": float(thr),
            "tp_hit": bool(tp_hit),
            "sl_hit": bool(sl_hit),
            "exit_reason": exit_reason,
            "prob": float(p[i]),
            "ret_price": float(raw_ret),
            "signed_ret_price": float(signed_ret),
            "PNL_on_1000": float(pnl_on_1000),
            "equity_after": float(equity_pnl),
        }
        trades.append(trade)
        equity_curve[exit_idx] = equity_pnl
        i = exit_idx + 1

        if len(trades) % 100 == 0:
            logger.info(f"[{ticker}] Trades={len(trades)}, equity={equity_pnl:+.2f} USD, progress={i / n:.1%}")

    # Uzupelnienie equity_curve
    last_val = equity_pnl
    for k in range(len(df)):
        if np.isnan(equity_curve[k]):
            equity_curve[k] = last_val
        else:
            last_val = equity_curve[k]

    trades_df = pd.DataFrame(trades)

    # 8. Metryki
    if not trades_df.empty:
        wins_mask = trades_df["PNL_on_1000"] > 0
        losses_mask = trades_df["PNL_on_1000"] < 0
        n_trades = len(trades_df)
        n_wins = int(wins_mask.sum())
        n_losses = int(losses_mask.sum())
        n_be = int((trades_df["PNL_on_1000"].abs() < 1e-10).sum())
        n_tp = int((trades_df["exit_reason"] == "TP").sum())
        n_sl = int((trades_df["exit_reason"] == "SL").sum())
        n_timeout = int((trades_df["exit_reason"] == "TIMEOUT").sum())

        winrate = n_wins / n_trades
        avg_win = trades_df.loc[wins_mask, "PNL_on_1000"].mean() if n_wins else 0.0
        avg_loss = trades_df.loc[losses_mask, "PNL_on_1000"].mean() if n_losses else 0.0
        expectancy = trades_df["PNL_on_1000"].mean()

        eq_series = pd.Series(equity_curve, dtype=float)
        running_max = eq_series.cummax()
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = (running_max - eq_series) / running_max.replace(0, np.nan)
        max_dd = float(np.nanmax(dd.values)) if len(dd) else np.nan
    else:
        n_trades = n_wins = n_losses = n_be = n_tp = n_sl = n_timeout = 0
        winrate = avg_win = avg_loss = expectancy = 0.0
        max_dd = np.nan

    summary = {
        "n_trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "breakeven": n_be,
        "winrate": float(winrate),
        "avg_win_usd": float(avg_win),
        "avg_loss_usd": float(avg_loss),
        "expectancy_usd": float(expectancy),
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_timeout": n_timeout,
        "tp_rate": float(n_tp / n_trades) if n_trades else 0.0,
        "sl_rate": float(n_sl / n_trades) if n_trades else 0.0,
        "timeout_rate": float(n_timeout / n_trades) if n_trades else 0.0,
        "avg_bars_held": float(trades_df["bars_held"].mean()) if not trades_df.empty else np.nan,
        "long_share": float((trades_df["direction"] == "LONG").mean()) if not trades_df.empty else np.nan,
        "final_equity_usd": float(equity_curve[-1]),
        "max_drawdown_rel": float(max_dd) if not np.isnan(max_dd) else np.nan,
        "params": {
            "sigma_mult": float(sigma_val),
            "T": T, "alpha": alpha, "use_atr": use_atr,
            "leverage": leverage, "fee_rate_per_side": fee_rate,
            "prob_long_thresh": prob_long_thresh,
            "per_trade_notional_usd": per_trade_notional,
        },
    }

    equity_ser = pd.Series(equity_curve, index=df["timestamp"])
    results = {"summary": summary, "trades": trades_df, "equity_curve": equity_ser}

    # 9. Eksport
    output_dir = Path("models") / model_name / ticker
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_csv = output_dir / f"{ticker}_trades.csv"
    trades_df.to_csv(trades_csv, index=False)

    log_txt = output_dir / f"{ticker}_trades_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_txt, "w", encoding="utf-8") as f:
        f.write(f"# TRADES LOG\n# Model: {model_name}\n# Ticker: {ticker}\n")
        f.write(f"# Params: sigma={sigma_val}, T={T}, alpha={alpha}, leverage={leverage}, fee={fee_rate}\n")
        f.write(f"# Created: {datetime.now().isoformat()}\n\n")
        for _, r in trades_df.iterrows():
            f.write(
                f"ENTRY {r['entry_time']} | {r['direction']} | price={r['entry_price']:.4f} | "
                f"prob={r['prob']:.3f} | thr={r['thr']:.6f}\n"
                f"EXIT  {r['exit_time']} | reason={r['exit_reason']} | price={r['exit_price']:.4f} | "
                f"Δ%_price={r['ret_price'] * 100:+.4f}% | bars={r['bars_held']}\n"
                f"PNL_on_1000={r['PNL_on_1000']:+.6f} | equity_after={r['equity_after']:+.6f}\n"
                "----\n"
            )

    return results


# ---------------------------------------------------------------------------
# Wykresy
# ---------------------------------------------------------------------------

def plot_equity_vs_price(
    results: dict,
    df: pd.DataFrame,
    crypto: str,
    model_name: str,
    data_mode: str = "test_data",
    T: int = 40,
    alpha: float = 0.72,
    leverage: float = 10.0,
    fee: float = 0.00045,
    do_plot: bool = False,
) -> None:
    """Rysuje i zapisuje wykres Equity Curve vs Price."""
    ec = results["equity_curve"].copy()
    ec.name = "Equity"
    equity_color = "green" if float(ec.iloc[-1]) > 1.0 else "red"

    price = df["close"].copy()
    if "timestamp" in df.columns:
        price.index = pd.to_datetime(df["timestamp"])
    price = price.reindex(ec.index, method="ffill")
    price.name = f"{crypto}/USDT price"

    fig, ax1 = plt.subplots(figsize=(10, 5))
    line_equity, = ax1.plot(ec.index, ec.values, lw=2.0, label="Equity", color=equity_color)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Equity")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    line_price, = ax2.plot(price.index, price.values, lw=1.2, label=f"{crypto}/USDT", color="orange", alpha=0.85)
    ax2.set_ylabel("Price")
    plt.title(f"Equity vs Price — {crypto} | mode={data_mode}, T={T}, α={alpha}, lev={leverage}x, fee={fee:.4f}")

    ax1.legend([line_equity, line_price], ["Equity", f"{crypto}/USDT"], loc="upper left")
    plt.tight_layout()

    out_dir = Path("models") / model_name / f"{crypto}_USDT"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "equity_curve.png", dpi=300, bbox_inches="tight")

    if do_plot:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostyka progow
# ---------------------------------------------------------------------------

def evaluate_thresholds_sigmoid(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Ewaluacja binary-class modelu (sigmoid) dla roznych progow."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    rows = []
    for thr in thresholds:
        y_pred = (probs > thr).astype(int)
        acc = (y_pred == y_true).mean()
        rows.append((thr, acc))
    return pd.DataFrame(rows, columns=["threshold", "accuracy"])


def evaluate_thresholds_softmax(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Accuracy w zaleznosci od progu (model 2-klasowy softmax)."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9)
    rows = []
    for thr in thresholds:
        y_pred = np.where(probs[:, 1] > thr, 1, 0)
        acc = accuracy_score(y_true, y_pred)
        rows.append((thr, acc))
    return pd.DataFrame(rows, columns=["threshold", "accuracy"])


# ---------------------------------------------------------------------------
# Skrypt
# ---------------------------------------------------------------------------

EXCLUDED: set[str] = set()

if __name__ == "__main__":
    model_name_str = (
        "20251002_184908_calc_label9{T=45;alpha=0.725;use_atr=False}_2.45_"
        "64-256-128-32-16_0.25-0.20-0.15-0.10-0.05_relu_bce_with_logits_0.6003_adam"
    )
    model_path = Path("models/2_class") / model_name_str / f"{model_name_str}.pt"
    csv_path = Path("data/test_data/BTC_USDT/BTC_USDT_1m_data.csv")
    features_path = Path("data/test_data/BTC_USDT/features/features00.parquet")
    scaler_path = Path("scalers/scaler.pkl")

    print(f"Wczytuje model z: {model_path}")
    model = load_vwap_model(model_path)

    print(f"Wczytuje dane z: {csv_path}")
    df = pd.read_csv(csv_path)

    from labels import calc_label9

    T, alpha, use_atr = parse_label_from_path(str(model_path))
    df = add_VWAP(df, sigma_val)
    y_true = calc_label9(df, T=T, alpha=alpha, use_atr=use_atr).to_numpy()

    scaler = joblib.load(scaler_path)
    features_df = pd.read_parquet(features_path)
    if "timestamp" in features_df.columns:
        features_df = features_df.drop(columns=["timestamp"])
    X_scaled = scaler.transform(features_df.values)

    probs = predict(model, X_scaled)

    df_eval = evaluate_thresholds_sigmoid(y_true, probs)
    print("\n=== Accuracy vs Threshold ===")
    print(df_eval.to_string(index=False))

    best = df_eval.loc[df_eval["accuracy"].idxmax()]
    print(f"\nNajlepszy prog: {best['threshold']:.2f} -> accuracy={best['accuracy']:.4f}")
