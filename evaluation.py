import pandas as pd

from pathlib import Path
from load_data import add_VWAP

from functools import partial
from labels import calc_label1, calc_label2, calc_label3

import numpy as np

def get_max_class_percent(df: pd.DataFrame, filters: list, N: int, label) -> float:
    """
    Zwraca największy udział (w [0,1]) pojedynczej klasy po policzeniu labela
    i zastosowaniu filtrów na df.

    :param df: ramka danych z danymi rynkowymi
    :param filters: lista funkcji-filtrów (drugi rząd)
    :param N: horyzont labela
    :param label: funkcja obliczająca etykietę (df, N) -> Series
    :return: float, maksymalny udział klasy (0..1)
    """
    # 1. Policz label (na pełnym df)
    series = label(df, N=N)

    # 2. Zastosuj filtry do df
    filtered_df = df.copy()
    for f in filters:
        filtered_df = f(filtered_df)

    # 3. Ogranicz serię do indeksu przefiltrowanego df
    series = series.reindex(filtered_df.index)
    arr = series.dropna().to_numpy()

    if arr.size == 0:
        return np.nan  # brak danych

    # 4. Policz rozkład klas
    _, counts = np.unique(arr, return_counts=True)
    max_share = counts.max() / counts.sum()

    return float(max_share)



def get_quantile_from_files(folder_path: str, sigma: float, filters: list, N: int, label, q: float = 0.75):
    path = Path(folder_path)
    max_class_percent_list = []
    for csv in path.iterdir():
        df = pd.read_csv(csv)
        df = add_VWAP(df, sigma)
        result = get_max_class_percent(df, filters, N, label)
        max_class_percent_list.append(float(result))
    return np.quantile(max_class_percent_list, q)

def scan_labels_in_folder(folder_path: str,
                          label_param_grid: dict,
                          filters: list,
                          threshold: float = 0.4) -> pd.DataFrame:
    """
    Iteruje po plikach CSV i po wszystkich labelach w label_param_grid.
    Oblicza max_class_share dla każdej kombinacji.
    Zwraca DataFrame z wynikami i wyświetla tylko te poniżej threshold.

    Kolumny w wyniku:
    file, label_name, params, N, max_share
    """
    results = []
    path = Path(folder_path)

    for csv in path.glob("*.csv"):
        print(f"\n>>> Przetwarzam plik: {csv.name}")
        df = pd.read_csv(csv)

        for label_name, funcs in label_param_grid.items():
            for func in funcs:
                params = func.keywords.copy()
                N = params.get("N", None)

                try:
                    max_share = get_max_class_percent(df, filters, N, func)
                except Exception as e:
                    print(f"⚠️ Błąd przy {label_name}, {params}: {e}")
                    continue

                row = {
                    "file": csv.name,
                    "label_name": label_name,
                    "params": params,
                    "N": N,
                    "max_share": max_share
                }
                results.append(row)

                # wyświetl tylko jeśli poniżej progu
                if max_share < threshold:
                    print(f"  OK: {label_name}, params={params}, max_share={max_share:.3f}")

    results_df = pd.DataFrame(results)
    return results_df


#
#
# # ===============================
# # Label1 grid (zmiana ceny ΔP)
# # ===============================
# N_grid = [10, 20, 30, 40, 50]
#
# # stałe progi w % (0.05% – 0.4%)
# phi_static = [0.0005, 0.00075, 0.001, 0.0015, 0.0025, 0.004]
#
# # dynamiczne progi skalowane przez sqrt(N/30)
# phi_base = [0.0005, 0.00075, 0.001, 0.0015]
# phi_scaled = {N: [phi * np.sqrt(N/30) for phi in phi_base] for N in N_grid}
#
# label1_grid = []
# for N in N_grid:
#     for phi in phi_static + phi_scaled[N]:
#         label1_grid.append(partial(calc_label1, phi=phi, N=N))
#
#
# # ===============================
# # Label2 grid (VWAP ± φ, kontynuacja psi)
# # ===============================
# phi2_grid = [0.001, 0.002, 0.003, 0.005, 0.0075]  # 0.1%–0.75%
# psi2_grid = [0.10, 0.20, 0.30, 0.40, 0.50]
#
# label2_grid = []
# for N in N_grid:
#     for phi in phi2_grid:
#         for psi in psi2_grid:
#             label2_grid.append(partial(calc_label2, phi=phi, psi=psi, N=N))
#
#
# # ===============================
# # Label3 grid (VWAP ± k·σ, kontynuacja psi)
# # ===============================
# k_sigma_grid = [0.5, 0.75, 1.0, 1.5, 2.0]
# psi3_grid = [0.10, 0.20, 0.30, 0.40]
#
# label3_grid = []
# for N in N_grid:
#     for k in k_sigma_grid:
#         for psi in psi3_grid:
#             label3_grid.append(partial(calc_label3, k_sigma=k, psi=psi, N=N))
#
#
# # ===============================
# # Łączymy w jeden słownik
# # ===============================
# label_param_grid = {
#     "label1": label1_grid,
#     "label2": label2_grid,
#     "label3": label3_grid,
# }
#
#
#
#
