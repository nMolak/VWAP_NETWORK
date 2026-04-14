import re
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_profitability_margin(
    log_path: str, metric_name: str = "M"
) -> dict:
    """
    Analizuje plik z logiem transakcji.

    Metryka: Profitability Margin = P&L_on_equity / |delta%_price|
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    pnl_values = re.findall(r"PNL_on_1000\s*=\s*([+-]?\d+(?:\.\d+)?)", text)
    delta_values = re.findall(r"Δ%_price=([+-]?\d+\.\d+)%", text)

    pnl = np.array([float(x) for x in pnl_values], dtype=float)
    delta = np.array([abs(float(x)) for x in delta_values], dtype=float)

    n = min(len(pnl), len(delta))
    if n == 0:
        raise ValueError("Brak danych w logu lub nie znaleziono metryk.")

    pnl = pnl[:n]
    delta = delta[:n]

    mask = delta > 0
    pm = pnl[mask] / delta[mask]

    return {
        "metric": metric_name,
        f"mean_{metric_name}": float(np.mean(pm)),
        f"Q1_{metric_name}": float(np.percentile(pm, 25)),
        f"Q2_{metric_name}": float(np.percentile(pm, 50)),
        f"Q3_{metric_name}": float(np.percentile(pm, 75)),
    }


def analyze_reward_to_fee_ratio(
    log_path: str,
    fee_rate: float = 0.00045,
    leverage: float = 10.0,
    per_trade_notional: float = 1000.0,
    metric_name: str = "R_f",
) -> dict:
    """
    Liczy metryce Reward-to-Fee Ratio (R_f) z logu backtestu
    ze stalym nominalem 1000 USD.

    R_f = PNL_on_1000 / fee_usd
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    pnl_values = re.findall(r"PNL_on_1000\s*=\s*([+-]?\d+(?:\.\d+)?)", text)
    pnl = np.array([float(x) for x in pnl_values], dtype=float)

    if len(pnl) == 0:
        raise ValueError(f"Brak danych w logu: {log_path}")

    fee_usd = per_trade_notional * (2.0 * fee_rate * leverage)
    if fee_usd <= 0:
        raise ValueError(
            "fee_rate i leverage musza dawac dodatni koszt oplat w USD."
        )

    rf = pnl / fee_usd

    return {
        "metric": metric_name,
        f"mean_{metric_name}": float(np.mean(rf)),
        f"Q1_{metric_name}": float(np.percentile(rf, 25)),
        f"Q2_{metric_name}": float(np.percentile(rf, 50)),
        f"Q3_{metric_name}": float(np.percentile(rf, 75)),
    }


def analyze_basic_stats(log_path: str, metric_name: str = "BASIC") -> dict:
    """
    Analizuje log transakcji i zwraca podstawowe metryki:
    win_rate, timeout_rate, sharpe_ratio, max_drawdown.
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    pnl_values = re.findall(r"PNL_on_1000=([+-]?\d+\.\d+)", text)
    reasons = re.findall(r"reason=(\w+)", text)
    equity_values = re.findall(r"equity_after=([+-]?\d+\.\d+)", text)

    if len(pnl_values) == 0:
        raise ValueError(f"Brak transakcji w logu: {log_path}")

    pnl = np.array([float(x) for x in pnl_values], dtype=float)
    equity = (
        np.array([float(x) for x in equity_values], dtype=float)
        if equity_values
        else None
    )

    n_trades = len(pnl)
    win_rate = float((pnl > 0).sum() / n_trades)
    timeout_rate = float(sum(r == "TIMEOUT" for r in reasons) / n_trades)

    if np.std(pnl) > 0:
        sharpe = float(np.mean(pnl) / np.std(pnl))
    else:
        sharpe = np.nan

    if equity is not None and len(equity) > 1:
        eq_series = np.array(equity)
        running_max = np.maximum.accumulate(eq_series)
        dd = (running_max - eq_series) / running_max
        max_dd = float(np.max(dd))
    else:
        max_dd = np.nan

    return {
        "metric": metric_name,
        f"win_rate_{metric_name}": win_rate,
        f"timeout_rate_{metric_name}": timeout_rate,
        f"sharpe_{metric_name}": sharpe,
        f"max_dd_{metric_name}": max_dd,
    }


def aggregate_model_metrics(
    model_name: str,
    fee_rate: float = 0.00045,
    leverage: float = 10.0,
    position_size: float = 0.20,
) -> pd.DataFrame:
    """
    Iteruje po folderach w models/{model_name}, wyszukuje najnowszy plik logu,
    analizuje go trzema funkcjami i zapisuje zbiorcze metryki do CSV.
    """
    base_path = Path(f"models/{model_name}")
    if not base_path.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu: {base_path}")

    all_results = {}

    for ticker_dir in base_path.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        txt_files = list(ticker_dir.glob("*_trades_log_*.txt"))
        if not txt_files:
            print(f"[!] Brak logow dla {ticker}")
            continue
        latest_file = max(txt_files, key=lambda f: f.stat().st_mtime)

        try:
            res_m = analyze_profitability_margin(latest_file, metric_name="M")
            res_rf = analyze_reward_to_fee_ratio(
                latest_file,
                fee_rate=fee_rate,
                leverage=leverage,
                metric_name="R_f",
            )
            res_basic = analyze_basic_stats(latest_file, metric_name="BASIC")

            combined = {**res_m, **res_rf, **res_basic}
            combined.pop("metric", None)

            all_results[ticker] = combined
            print(f"[OK] {ticker}: przetworzono {latest_file.name}")

        except Exception as e:
            print(f"[ERR] {ticker}: {e}")
            continue

    if not all_results:
        raise ValueError("Nie udalo sie przetworzyc zadnych logow.")

    df = pd.DataFrame(all_results)
    df.index.name = "metric"

    out_path = base_path / "summary_metrics.csv"
    df.to_csv(out_path, index=True, float_format="%.6f")
    print(f"\nZapisano zbiorczy raport: {out_path}")

    return df


models_path = Path("models")

for file in models_path.iterdir():
    if file.is_file():
        continue

    model_name = file.name
    try:
        print(f"\n[RUN] Przetwarzam model: {model_name}")
        df = aggregate_model_metrics(
            model_name=model_name,
            fee_rate=0.0002,
            leverage=10,
            position_size=0.20,
        )
        print(f"[OK] Zapisano summary_metrics.csv dla: {model_name}")
    except Exception as e:
        print(f"[ERR] {model_name}: {e}")
