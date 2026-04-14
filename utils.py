import inspect
import logging
from pathlib import Path
from typing import List

import pandas as pd

from parameters import LOG_ENABLED

logging.basicConfig(level=logging.INFO)


def logprint(*args, **kwargs):
    if LOG_ENABLED:
        logging.info(" ".join(str(a) for a in args))


def get_func_name():
    return inspect.currentframe().f_back.f_code.co_name


def display_dataframe(df: pd.DataFrame, title: str, precision: int = 6):
    """
    Wyświetla ramkę danych Pandas jako sformatowaną tabelę tekstową.

    Funkcja dynamicznie oblicza szerokość każdej kolumny, aby zapewnić
    czytelne i spójne wyrównanie bez użycia zewnętrznych bibliotek.
    """
    if df.empty:
        print(f"--- {title} ---")
        print("Ramka danych jest pusta.")
        return

    # --- Krok 1: Przygotowanie danych i obliczenie szerokości kolumn ---

    # Konwertujemy wszystkie dane na sformatowane ciągi znaków
    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    for r_idx, row in df.iterrows():
        for c_idx, value in row.items():
            if pd.isna(value):
                formatted_df.loc[r_idx, c_idx] = "NaN"
            else:
                formatted_df.loc[r_idx, c_idx] = f"{value:.{precision}f}"

    headers: List[str] = ["Feature"] + list(df.columns.astype(str))

    column_widths = {
        "Feature": max(len(str(h)) for h in df.index.tolist() + ["Feature"])
    }
    for col in df.columns:
        max_data_width = formatted_df[col].str.len().max()
        header_width = len(str(col))
        column_widths[str(col)] = max(max_data_width, header_width)

    # --- Krok 2: Drukowanie tabeli ---

    def print_row(data: List[str], widths: dict, align_data: str = "right"):
        row_str = f" {data[0].ljust(widths['Feature'])} "
        for i, item in enumerate(data[1:]):
            col_name = headers[i + 1]
            if align_data == "right":
                row_str += f"| {item.rjust(widths[col_name])} "
            else:
                row_str += f"| {item.ljust(widths[col_name])} "
        print(row_str)

    total_width = sum(column_widths.values()) + len(column_widths) * 3 + 1

    print(f"\n{title.center(total_width)}\n")
    print_row(headers, column_widths, align_data="right")

    separator = "-".join("=" * (w + 2) for w in column_widths.values())
    separator = separator.replace("=", "-", column_widths["Feature"] + 2)
    separator = separator.replace("-", "+", len(column_widths) - 1)
    print(separator)

    for index, row in formatted_df.iterrows():
        row_to_print = [str(index)] + row.tolist()
        print_row(row_to_print, column_widths)

    print("-" * total_width)


def make_logprint(enabled=True):
    caller = inspect.currentframe().f_back.f_code.co_name

    if enabled:
        def _logprint(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            print(f"[{caller}] -> {msg}")
        return _logprint

    def _silent(*args, **kwargs):
        pass
    return _silent


def check_feature_consistency(data_path: str):
    data_path = Path(data_path)
    feature_sets = {}

    for folder in sorted(
        p for p in data_path.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        csv = next(folder.glob("*.csv"), None)
        if csv is None:
            print(f"Brak pliku CSV w {folder.name}")
            continue

        df = pd.read_csv(csv, nrows=5)
        features = sorted(c for c in df.columns if c.startswith("feature_"))
        feature_sets[folder.name] = features

    base = list(feature_sets.values())[0]
    for name, feats in feature_sets.items():
        if feats != base:
            diff1 = set(base) - set(feats)
            diff2 = set(feats) - set(base)
            print(f"[ERROR] {name}: roznice w featurach!")
            if diff1:
                print(f"  Brakuje: {diff1}")
            if diff2:
                print(f"  Nadmiarowe: {diff2}")
            raise ValueError(
                "Zbiory featurow roznia sie miedzy plikami, program przerywa prace!"
            )
    print("Featury we wszystkich plikach sa zgodne!")

    return feature_sets
