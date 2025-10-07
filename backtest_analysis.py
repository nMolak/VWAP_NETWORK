import re
import numpy as np

import pandas as pd
from pathlib import Path
import numpy as np

def analyze_profitability_margin(log_path: str, metric_name: str = "M") -> dict:
    """
    Analizuje plik z logiem transakcji i zwraca słownik:
    {
        'metric': 'M',
        'mean_M': ...,
        'Q1_M': ...,
        'Q2_M': ...,
        'Q3_M': ...
    }

    Metryka: Profitability Margin = P&L_on_equity / |Δ%_price|
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

    mean = float(np.mean(pm))
    q1 = float(np.percentile(pm, 25))
    q2 = float(np.percentile(pm, 50))
    q3 = float(np.percentile(pm, 75))

    return {
        "metric": metric_name,
        f"mean_{metric_name}": mean,
        f"Q1_{metric_name}": q1,
        f"Q2_{metric_name}": q2,
        f"Q3_{metric_name}": q3
    }

import re
import numpy as np

def analyze_reward_to_fee_ratio(log_path: str,
                                fee_rate: float = 0.00045,
                                leverage: float = 10.0,
                                per_trade_notional: float = 1000.0,
                                metric_name: str = "R_f") -> dict:
    """
    Liczy metrykę Reward-to-Fee Ratio (R_f) z logu backtestu ze STAŁYM nominałem 1000 USD.

    Definicja (spójna z backtestem):
      - w logu mamy wartości PNL_on_1000 (PnL w USD dla nominału 1000 USD, już po odjęciu opłat)
      - koszt opłat (round-trip) dla pozycji 1000 USD to:
            fee_usd = 1000 * (2 * fee_rate * leverage)

      R_f = PNL_on_1000 / fee_usd

    Interpretacja:
      R_f > 1  → zysk netto z tej transakcji jest większy niż koszt prowizji (wejście+wyjście)
      R_f ≈ 1  → zysk ~ koszt prowizji
      R_f < 0  → transakcja stratna netto

    Zwraca:
      {
        'metric': 'R_f',
        'mean_R_f': ...,
        'Q1_R_f': ...,
        'Q2_R_f': ...,
        'Q3_R_f': ...
      }
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # PnL w USD zapisany w logu jako "PNL_on_1000=..."
    pnl_values = re.findall(r"PNL_on_1000\s*=\s*([+-]?\d+(?:\.\d+)?)", text)
    pnl = np.array([float(x) for x in pnl_values], dtype=float)

    if len(pnl) == 0:
        raise ValueError(f"Brak danych w logu: {log_path}")

    fee_usd = per_trade_notional * (2.0 * fee_rate * leverage)
    if fee_usd <= 0:
        raise ValueError("fee_rate i leverage muszą dawać dodatni koszt opłat w USD.")

    Rf = pnl / fee_usd  # Reward-to-Fee Ratio w poprawnej skali (USD/USD)

    mean = float(np.mean(Rf))
    q1 = float(np.percentile(Rf, 25))
    q2 = float(np.percentile(Rf, 50))
    q3 = float(np.percentile(Rf, 75))

    return {
        "metric": metric_name,
        f"mean_{metric_name}": mean,
        f"Q1_{metric_name}": q1,
        f"Q2_{metric_name}": q2,
        f"Q3_{metric_name}": q3
    }

def analyze_basic_stats(log_path: str, metric_name: str = "BASIC") -> dict:
    """
    Analizuje log transakcji i zwraca podstawowe metryki:
      - win_rate: udział transakcji z dodatnim P&L_on_equity
      - timeout_rate: udział transakcji zakończonych reason=TIMEOUT
      - sharpe_ratio: średni zwrot / odchylenie standardowe zwrotów (bez risk-free)
      - max_drawdown: maksymalne obsunięcie equity

    Zwraca słownik z kluczami np.:
      {
        'metric': 'BASIC',
        'win_rate_BASIC': ...,
        'timeout_rate_BASIC': ...,
        'sharpe_BASIC': ...,
        'max_dd_BASIC': ...
      }
    """

    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # --- Parsowanie ---
    pnl_values = re.findall(r"PNL_on_1000=([+-]?\d+\.\d+)", text)
    reasons = re.findall(r"reason=(\w+)", text)
    equity_values = re.findall(r"equity_after=([+-]?\d+\.\d+)", text)

    if len(pnl_values) == 0:
        raise ValueError(f"Brak transakcji w logu: {log_path}")

    pnl = np.array([float(x) for x in pnl_values], dtype=float)
    equity = np.array([float(x) for x in equity_values], dtype=float) if equity_values else None

    # --- Miary ---
    n_trades = len(pnl)
    win_rate = float((pnl > 0).sum() / n_trades)
    timeout_rate = float(sum(r == "TIMEOUT" for r in reasons) / n_trades)

    # Sharpe ratio (risk-free = 0)
    if np.std(pnl) > 0:
        sharpe = float(np.mean(pnl) / np.std(pnl))
    else:
        sharpe = np.nan

    # Maksymalne obsunięcie kapitału (DD)
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
        f"max_dd_{metric_name}": max_dd
    }


def aggregate_model_metrics(model_name: str,
                            fee_rate: float = 0.00045,
                            leverage: float = 10.0,
                            position_size: float = 0.20) -> pd.DataFrame:
    """
    Iteruje po folderach w models/{model_name}, wyszukuje najnowszy plik logu (.txt),
    analizuje go trzema funkcjami i zapisuje zbiorcze metryki do CSV.

    Output:
    models/{model_name}/summary_metrics.csv
    """

    base_path = Path(f"models/{model_name}")
    if not base_path.exists():
        raise FileNotFoundError(f"Nie znaleziono folderu: {base_path}")

    all_results = {}

    for ticker_dir in base_path.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name

        # znajdź najnowszy plik logów .txt
        txt_files = list(ticker_dir.glob("*_trades_log_*.txt"))
        if not txt_files:
            print(f"[!] Brak logów dla {ticker}")
            continue
        latest_file = max(txt_files, key=lambda f: f.stat().st_mtime)

        try:
            res_M = analyze_profitability_margin(latest_file, metric_name="M")
            res_Rf = analyze_reward_to_fee_ratio(
                latest_file,
                fee_rate=fee_rate,
                leverage=leverage,
                metric_name="R_f"
            )
            res_BASIC = analyze_basic_stats(latest_file, metric_name="BASIC")

            # połączenie wszystkich metryk w jeden słownik
            combined = {**res_M, **res_Rf, **res_BASIC}

            # usuwamy klucze "metric" (niepotrzebne w tabeli)
            combined.pop("metric", None)

            all_results[ticker] = combined

            print(f"[OK] {ticker}: przetworzono {latest_file.name}")

        except Exception as e:
            print(f"[ERR] {ticker}: {e}")
            continue

    if not all_results:
        raise ValueError("Nie udało się przetworzyć żadnych logów.")

    # konwersja do DataFrame
    df = pd.DataFrame(all_results)
    df.index.name = "metric"

    # zapis do CSV
    out_path = base_path / "summary_metrics.csv"
    df.to_csv(out_path, index=True, float_format="%.6f")
    print(f"\nZapisano zbiorczy raport: {out_path}")

    return df


# model_name = "20251002_184908_calc_label9{T=45;alpha=0.725;use_atr=False}_2.45_64-256-128-32-16_0.25-0.20-0.15-0.10-0.05_relu_binary_crossentropy_0.6003_adam"
# model_path = f"models/{model_name}"
#
# df = aggregate_model_metrics(model_name, fee_rate=0.0002, leverage=10, position_size=0.20)

models_path = Path("models")

for file in models_path.iterdir():
    if file.is_file():
        continue  # pomijamy pliki .keras itp.

    model_name = file.name
    try:
        print(f"\n[RUN] Przetwarzam model: {model_name}")
        df = aggregate_model_metrics(
            model_name=model_name,
            fee_rate=0.0002,
            leverage=10,
            position_size=0.20
        )
        print(f"[OK] Zapisano summary_metrics.csv dla: {model_name}")
    except Exception as e:
        print(f"[ERR] {model_name}: {e}")
