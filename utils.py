import logging
import inspect
import pandas as pd

from typing import List
from pathlib import Path


from parameters import LOG_ENABLED

logging.basicConfig(level=logging.INFO)

#TODO: Zmodyfikować tak, aby była funkcja która na początku innej funkcji zwróci różną wersję logprintu
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
    # To pozwoli nam zmierzyć ich rzeczywistą długość
    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    for r_idx, row in df.iterrows():
        for c_idx, value in row.items():
            if pd.isna(value):
                formatted_df.loc[r_idx, c_idx] = "NaN"
            else:
                formatted_df.loc[r_idx, c_idx] = f"{value:.{precision}f}"

    # Nazwy kolumn (nagłówki)
    headers: List[str] = ["Feature"] + list(df.columns.astype(str))

    # Obliczamy maksymalną szerokość dla każdej kolumny
    # Bierzemy pod uwagę szerokość nagłówka oraz wszystkich danych w kolumnie
    column_widths = {
        "Feature": max(len(str(h)) for h in df.index.tolist() + ["Feature"])
    }
    for col in df.columns:
        # Maksymalna szerokość to najdłuższy element w kolumnie LUB nagłówek
        max_data_width = formatted_df[col].str.len().max()
        header_width = len(str(col))
        column_widths[str(col)] = max(max_data_width, header_width)

    # --- Krok 2: Drukowanie tabeli ---

    def print_row(data: List[str], widths: dict, align_data: str = 'right'):
        """Pomocnicza funkcja do drukowania pojedynczego wiersza."""
        # Pierwsza kolumna (Feature) zawsze do lewej
        row_str = f" {data[0].ljust(widths['Feature'])} "

        # Pozostałe kolumny z danymi
        for i, item in enumerate(data[1:]):
            col_name = headers[i + 1]
            if align_data == 'right':
                row_str += f"| {item.rjust(widths[col_name])} "
            else:
                row_str += f"| {item.ljust(widths[col_name])} "
        print(row_str)

    # Obliczamy całkowitą szerokość tabeli do wyśrodkowania tytułu i linii
    total_width = sum(column_widths.values()) + len(column_widths) * 3 + 1

    # Drukujemy tytuł
    print(f"\n{title.center(total_width)}\n")

    # Drukujemy nagłówek
    print_row(headers, column_widths, align_data='right')

    # Drukujemy linię oddzielającą nagłówek od danych
    separator = "-".join("=" * (w + 2) for w in column_widths.values())
    separator = separator.replace("=", "-", column_widths['Feature'] + 2)
    separator = separator.replace("-", "+", len(column_widths) - 1)
    print(separator)

    # Drukujemy wiersze z danymi
    for index, row in formatted_df.iterrows():
        row_to_print = [str(index)] + row.tolist()
        print_row(row_to_print, column_widths)

    print("-" * total_width)

def make_logprint(enabled=True):

    caller = inspect.currentframe().f_back.f_code.co_name

    if enabled:
        def logprint(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            print(f"[{caller}] -> {msg}")
        return logprint
    def silent_logprint(*args, **kwargs):
        pass
    return silent_logprint

def check_feature_consistency(data_path: str):
    data_path = Path(data_path)
    feature_sets = {}

    for folder in sorted([p for p in data_path.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        csv = next(folder.glob("*.csv"), None)
        if csv is None:
            print(f"⚠️ Brak pliku CSV w {folder.name}")
            continue

        df = pd.read_csv(csv, nrows=5)  # wystarczy kilka wierszy
        features = sorted([c for c in df.columns if c.startswith("feature_")])
        feature_sets[folder.name] = features

    # --- porównanie ---
    base = list(feature_sets.values())[0]
    for name, feats in feature_sets.items():
        if feats != base:
            diff1 = set(base) - set(feats)
            diff2 = set(feats) - set(base)
            print(f"❌ {name}: różnice w featurach!")
            if diff1: print(f"  Brakuje: {diff1}")
            if diff2: print(f"  Nadmiarowe: {diff2}")
            raise ValueError("Zbiory featurów różnią się w między plikami, program przerywa pracę!")
    print("Featury we wszystkich plikach są zgodne!")

    return feature_sets