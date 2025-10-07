import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any
from datetime import datetime

from load_data import add_VWAP, add_indicators
from filters import filter_clean, filter_hours, filter_remove_long_series
from calculations import calc_indicators

import tensorflow as tf
import matplotlib.pyplot as plt


def backtest(df: pd.DataFrame,
             model,
             sigma_mult: float = 2.15,
             T: int = 15,
             alpha: float = 0.2,
             use_atr: bool = False,
             leverage: float = 10.0,
             fee_rate: float = 0.00045,      # 0.045% taker per side
             prob_long_thresh: float = 0.5,  # próg dla LONG (p>=...)
             scaler_path: str = "scalers/scaler.pkl",
             position_size: float = 0.20     # udział kapitału na trade (20%)
             ) -> Dict[str, Any]:
    """
    Backtest wejść na świecach ekstremalnych z decyzją sieci i wyjściem zgodnym z label9
    (pierwszy hit +thr/-thr albo timeout T). Jeden trade na raz (sekwencyjnie).
    Każdy trade angażuje tylko `position_size` kapitału (reszta w gotówce).
    """

    df = df.copy()

    # --- 1) VWAP + pasma + featury ---
    df = add_VWAP(df, sigma_mult=sigma_mult)
    df = add_indicators(df)  # dopisuje kolumny feature_*
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # --- 2) Maski filtrów jako WARUNKI WEJŚCIA (bez wycinania ramek) ---
    idx_clean  = filter_clean()(df).index
    idx_hours  = filter_hours(0, 8)(df).index
    idx_series = filter_remove_long_series(8, sigma_mult)(df).index
    allowed_idx = set(idx_clean).intersection(idx_hours).intersection(idx_series)
    allowed_mask = df.index.to_series().isin(allowed_idx).to_numpy()

    # --- 3) Ekstrema względem pasm VWAP ---
    upper = df["close"].to_numpy() > df[f"vwap_plus_{sigma_mult}_sigma"].to_numpy()
    lower = df["close"].to_numpy() < df[f"vwap_minus_{sigma_mult}_sigma"].to_numpy()
    is_extreme = upper | lower

    # --- 4) Przygotowanie cech + predykcja sieci ---
    # spójny zestaw featurów jak przy treningu: bierzemy klucze z calc_indicators
    ind_keys = list(calc_indicators(df).keys())
    feature_cols = [f"feature_{k}" for k in ind_keys]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Brakuje kolumn w df: {missing}")

    # czyszczenie featurów przed skalowaniem
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(method="ffill").fillna(method="bfill")

    X = df[feature_cols].values
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # p = P(y=1|x) -> 1: long, 0: short
    p = model.predict(X_scaled, verbose=0).ravel()
    pred_cls = (p >= prob_long_thresh).astype(np.int8)

    # --- 5) Parametry progów jak w label9 ---
    closes = df["close"].to_numpy(float)
    vwap = df["vwap"].to_numpy(float)

    if use_atr and "feature_atr_rel" in df.columns:
        scale_arr = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale_arr = np.abs(closes - vwap) + 1e-12

    # --- 6) Pętla po świecach: wejścia i wyjścia ---
    trades = []
    equity = 1.0
    equity_curve = np.full(len(df), np.nan, dtype=float)

    # stałe fee dla pojedynczego trade'u (wejście+wyjście) na udziale pozycji
    # notional = equity * position_size * leverage  -> fee na equity = 2*fee_rate*leverage*position_size
    round_trip_fee_on_equity = 2.0 * fee_rate * leverage * position_size

    i = 0
    n = len(df)
    while i < n:
        equity_curve[i] = equity

        # warunek wejścia: świeca ekstremalna + maski filtrów
        if not (is_extreme[i] and allowed_mask[i]):
            i += 1
            continue

        # - górne przebicie i pred=1 -> LONG
        # - dolne przebicie i pred=0 -> SHORT
        # if upper[i] and pred_cls[i] == 1:
        #     direction = +1
        # elif lower[i] and pred_cls[i] == 0:
        #     direction = -1
        # else:
        #     i += 1
        #     continue

        if is_extreme[i] and allowed_mask[i]:
            direction = +1 if pred_cls[i] == 1 else -1
        else:
            i += 1; continue

        entry_idx = i
        entry_time = df["timestamp"].iloc[i]
        entry_price = closes[i]

        thr = alpha * (scale_arr[i] / max(entry_price, 1e-12))

        # szukamy wyjścia: pierwszy hit TP/SL lub timeout
        exit_idx = i
        tp_hit = False
        sl_hit = False

        j = 1
        while j <= T and (i + j) < n:
            exit_idx = i + j
            ret = (closes[exit_idx] - entry_price) / entry_price
            ret_dir = direction * ret  # zwrot w kierunku pozycji

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

        # P&L tylko na zaangażowanej części kapitału:
        pnl_on_equity = position_size * leverage * signed_ret - round_trip_fee_on_equity
        equity *= (1.0 + pnl_on_equity)
        equity_curve[exit_idx] = equity

        # powód wyjścia
        if tp_hit:
            exit_reason = "TP"     # trafienie progu w stronę pozycji
        elif sl_hit:
            exit_reason = "SL"     # trafienie progu przeciw pozycji
        else:
            exit_reason = "TIMEOUT"  # minął horyzont T

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
            "ret_price": float(raw_ret),          # surowy zwrot ceny (bez znaku pozycji)
            "signed_ret_price": float(signed_ret),# zwrot w kierunku pozycji
            "pnl_on_equity": float(pnl_on_equity),
            "equity_after": float(equity)
        })

        # po wyjściu przechodzimy za trade (jeden trade naraz)
        i = exit_idx + 1

    # uzupełnij equity_curve do końca ostatnią wartością
    last_equity = equity
    for k in range(len(df)):
        if np.isnan(equity_curve[k]):
            equity_curve[k] = last_equity
        else:
            last_equity = equity_curve[k]

    trades_df = pd.DataFrame(trades)

    # --- 7) Metryki podsumowujące ---
    if not trades_df.empty:
        wins_mask = trades_df["pnl_on_equity"] > 0
        losses_mask = trades_df["pnl_on_equity"] < 0
        breakeven_mask = trades_df["pnl_on_equity"].abs() < 1e-12

        n_trades = len(trades_df)
        n_wins = int(wins_mask.sum())
        n_losses = int(losses_mask.sum())
        n_be = int(breakeven_mask.sum())

        # domknięcia z warunku (TP/SL) vs z czasu
        cond_mask = trades_df["exit_reason"].isin(["TP", "SL"])
        timeout_mask = trades_df["exit_reason"].eq("TIMEOUT")
        n_cond = int(cond_mask.sum())
        n_timeout = int(timeout_mask.sum())

        n_tp = int(trades_df["exit_reason"].eq("TP").sum())
        n_sl = int(trades_df["exit_reason"].eq("SL").sum())

        winrate = n_wins / n_trades
        lossrate = n_losses / n_trades
        berate = n_be / n_trades

        cond_rate = n_cond / n_trades
        timeout_rate = n_timeout / n_trades
        tp_rate = n_tp / n_trades
        sl_rate = n_sl / n_trades

        avg_win = trades_df.loc[wins_mask, "pnl_on_equity"].mean() if n_wins else 0.0
        avg_loss = trades_df.loc[losses_mask, "pnl_on_equity"].mean() if n_losses else 0.0
        expectancy = trades_df["pnl_on_equity"].mean()

        avg_hold = trades_df["bars_held"].mean()
        long_share = (trades_df["direction"] == "LONG").mean()
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
            "sigma_mult": sigma_mult,
            "T": T,
            "alpha": alpha,
            "use_atr": use_atr,
            "leverage": leverage,
            "fee_rate_per_side": fee_rate,
            "prob_long_thresh": prob_long_thresh,
            "scaler_path": scaler_path,
            "position_size": position_size,
        }
    }

    equity_ser = pd.Series(equity_curve, index=df["timestamp"])
    return {"summary": summary, "trades": trades_df, "equity_curve": equity_ser}


# ===================== URUCHOMIENIE =====================
#
# model_path = r"C:\Users\norbe\PycharmProjects\VWAP_NETWORK\models\20250925_160627_T=20;alpha=0.5;use_atr=True_2.15_128-64-32-1_0.6631"
# model = tf.keras.models.load_model(model_path)
# model.summary()
#
# csv_path = r"data/training_data/BTC_USDT_1m_data.csv"
# df = pd.read_csv(csv_path)
#
# results = backtest(
#     df, model,
#     sigma_mult=2.15,
#     T=20,
#     alpha=0.5,
#     use_atr=True,
#     leverage=10.0,
#     fee_rate=0.000,          # zgodnie z Twoim wywołaniem
#     position_size=0.20,      # 20% kapitału na trade
#     prob_long_thresh=0.5,    # zgodnie z Twoim wywołaniem
#     scaler_path="scalers/scaler.pkl",
# )
#
# # --- wydruk podsumowania do terminala ---
# print("\n>>> SUMMARY")
# for k, v in results["summary"].items():
#     print(f"{k}: {v}")
#
# trades = results["trades"]
# print("\n>>> TRADES (head)")
# print(trades.head())
#
# if not trades.empty:
#     print("\n>>> STATYSTYKI")
#     s = results["summary"]
#     print(f"Transakcji: {s['n_trades']}")
#     print(f"Wygrane: {s['wins']} ({s['winrate']:.2%}) | Przegrane: {s['losses']} ({s['lossrate']:.2%}) | BE: {s['breakeven']} ({s['breakeven_rate']:.2%})")
#     print(f"Zamknięcia z WARUNKU (TP/SL): {s['n_closed_condition']} ({s['cond_rate']:.2%})  -> TP: {s['n_tp']} ({s['tp_rate']:.2%}), SL: {s['n_sl']} ({s['sl_rate']:.2%})")
#     print(f"Zamknięcia z CZASU (TIMEOUT T): {s['n_closed_timeout']} ({s['timeout_rate']:.2%})")
#     print(f"Średnia wygrana (na equity): {s['avg_win']:.4f}")
#     print(f"Średnia przegrana (na equity): {s['avg_loss']:.4f}")
#     print(f"Expectancy (średni P&L na trade): {s['expectancy']:.4f}")
#     print(f"Max DD: {s['max_drawdown']:.2%}")
#     print(f"Final equity: {s['final_equity']:.4f}")
#
# # zapis krzywej kapitału + wykres
# results["equity_curve"].to_csv("equity_curve.csv")
# results["equity_curve"].plot(title="Equity Curve")
# plt.xlabel("Time")
# plt.ylabel("Equity")
# plt.show()
#
# # --- LOG DO TXT dla LLM (pełna lista trade'ów) ---
# if not trades.empty:
#     log_name = f"trades_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
#     with open(log_name, "w", encoding="utf-8") as f:
#         f.write("# TRADES LOG – pełna lista transakcji do analizy LLM\n")
#         f.write(f"# Utworzono: {datetime.now().isoformat()}\n")
#         f.write(f"# Parametry: sigma_mult={results['summary']['params']['sigma_mult']}, "
#                 f"T={results['summary']['params']['T']}, alpha={results['summary']['params']['alpha']}, "
#                 f"use_atr={results['summary']['params']['use_atr']}, leverage={results['summary']['params']['leverage']}, "
#                 f"fee_rate_per_side={results['summary']['params']['fee_rate_per_side']}, "
#                 f"prob_long_thresh={results['summary']['params']['prob_long_thresh']}, "
#                 f"position_size={results['summary']['params']['position_size']}\n\n")
#         for _, r in trades.iterrows():
#             f.write(
#                 "ENTRY {time} | {dir} | price={ep:.6f} | prob={pr:.3f} | thr={thr:.6f}\n"
#                 "EXIT  {time2} | reason={reason} | price={xp:.6f} | Δ%_price={d1:+.4f}% | Δ%_signed={d2:+.4f}% | bars={bars}\n"
#                 "P&L_on_equity={pnl:+.6f} | equity_after={eq:.6f}\n"
#                 "----\n".format(
#                     time=r['entry_time'],
#                     dir=r['direction'],
#                     ep=r['entry_price'],
#                     pr=r['prob'],
#                     thr=r['thr'],
#                     time2=r['exit_time'],
#                     reason=r['exit_reason'],
#                     xp=r['exit_price'],
#                     d1=100.0 * r['ret_price'],
#                     d2=100.0 * r['signed_ret_price'],
#                     bars=r['bars_held'],
#                     pnl=r['pnl_on_equity'],
#                     eq=r['equity_after'],
#                 )
#             )
#     print(f"\nZapisano pełny log transakcji do pliku: {log_name}")


from numpy.lib.stride_tricks import sliding_window_view
from filters import filters

csv_path = r"data/training_data/SOL_USDT/SOL_USDT_1m_data.csv"
df = pd.read_csv(csv_path)


def label_return_distribution(df: pd.DataFrame, label_partial) -> dict:
    """
    Przyjmuje df i label_partial (np. partial(calc_label9, T=40, alpha=0.75, use_atr=False)).
    Zwraca rozkład zwrotów procentowych oddzielnie dla klas 0 i 1.
    """
    # --- 1) zastosuj filtry ---
    df = df.copy()
    from load_data import add_VWAP
    df = add_VWAP(df)
    
    for f in filters:
        before = len(df)
        df = f(df)
        after = len(df)
        print(f"[filter] {f.__name__}: {before} -> {after}")

    # --- 2) parametry z label_partial ---
    T = label_partial.keywords.get("T", 15)
    alpha = label_partial.keywords.get("alpha", 0.2)
    use_atr = label_partial.keywords.get("use_atr", False)

    closes = df["close"].to_numpy(float)

    if use_atr and "feature_atr_rel" in df.columns:
        scale = df["feature_atr_rel"].to_numpy(float) + 1e-12
    else:
        scale = (df["close"] - df["vwap"]).abs().to_numpy(float) + 1e-12

    n = len(df)
    labels = np.zeros(n, dtype=np.int8)
    realized_returns = np.zeros(n, dtype=np.float32)

    # --- 3) oblicz okna ---
    windows = sliding_window_view(closes, T)
    entry = windows[:, 0][:, None]
    ret_paths = (windows - entry) / entry
    thr = alpha * (scale[:n - T + 1] / np.maximum(entry[:, 0], 1e-12))

    up_hits = (ret_paths >= thr[:, None])
    dn_hits = (ret_paths <= -thr[:, None])
    any_up = up_hits.any(axis=1)
    any_dn = dn_hits.any(axis=1)

    # przypadek: tylko up
    mask_up = any_up & ~any_dn
    first_up = np.argmax(up_hits[mask_up], axis=1)
    realized_returns[:n - T + 1][mask_up] = ret_paths[mask_up, first_up]
    labels[:n - T + 1][mask_up] = 1

    # przypadek: tylko dn
    mask_dn = ~any_up & any_dn
    first_dn = np.argmax(dn_hits[mask_dn], axis=1)
    realized_returns[:n - T + 1][mask_dn] = ret_paths[mask_dn, first_dn]
    labels[:n - T + 1][mask_dn] = 0

    # przypadek: oba
    mask_both = any_up & any_dn
    first_up = np.argmax(up_hits[mask_both], axis=1)
    first_dn = np.argmax(dn_hits[mask_both], axis=1)
    choose_up = first_up < first_dn
    realized_returns[:n - T + 1][mask_both] = np.where(
        choose_up,
        ret_paths[mask_both, first_up],
        ret_paths[mask_both, first_dn]
    )
    labels[:n - T + 1][mask_both] = choose_up.astype(np.int8)

    # przypadek: brak hitów – bierzemy koniec okna
    mask_none = ~(any_up | any_dn)
    realized_returns[:n - T + 1][mask_none] = ret_paths[mask_none, -1]
    labels[:n - T + 1][mask_none] = (ret_paths[mask_none, -1] > 0).astype(np.int8)

    # ostatnie T-1 elementów → brak okna, zostają 0
    realized_returns[n - T + 1:] = np.nan

    # --- 4) podział na klasy ---
    dist_class1 = realized_returns[labels == 1]
    dist_class0 = realized_returns[labels == 0]

    return {
        "class_1_returns": dist_class1[~np.isnan(dist_class1)],
        "class_0_returns": dist_class0[~np.isnan(dist_class0)]
    }

from functools import partial
from labels import calc_label9

out = label_return_distribution(df, partial(calc_label9, T=40, alpha=0.72, use_atr=False))

plt.hist(out["class_1_returns"], bins=100, alpha=0.5, label="class 1")
plt.hist(out["class_0_returns"], bins=100, alpha=0.5, label="class 0")
plt.legend()
plt.show()





import tensorflow as tf
import pandas as pd

# 1) Wczytaj model
model_path = "models/20251002_184923_calc_label9{T=40;alpha=0.72;use_atr=False}_2.45_256-256-64_0.22-0.17-0.12_relu_binary_crossentropy_0.6271_adam.keras"
model = tf.keras.models.load_model(model_path)

# 2) Wczytaj dane
csv_path = r"data/training_data/SOL_USDT/SOL_USDT_1m_data.csv"
df = pd.read_csv(csv_path)

# 3) Uruchom backtest
results = backtest(
    df, model,
    sigma_mult=2.45,
    T=40,
    alpha=0.72,
    use_atr=False,
    leverage=10.0,
    fee_rate=0.00000,
    position_size=0.20,
    prob_long_thresh=0.5,
    scaler_path="scalers/scaler.pkl",
)

# 4) Wydruk podsumowania
print("\n>>> SUMMARY")
for k, v in results["summary"].items():
    print(f"{k}: {v}")

# 5) Podgląd transakcji
trades = results["trades"]
print("\n>>> TRADES (head)")
print(trades.head())

# 6) Wykres equity curve
results["equity_curve"].plot(title="Equity Curve")
import matplotlib.pyplot as plt
plt.xlabel("Time")
plt.ylabel("Equity")
plt.show()

