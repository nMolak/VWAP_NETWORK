"""
filters.py

Moduł ten zawiera funkcje filtrujące dane.
Są to funkcje drugiego rzędu, aby ułatwić modyfikację i optymalizowanie parametrów.

Obecnie znajdujące się funkcje zostały zaprojektowane z myślą o wskaźniku VWAP.

"""

import pandas as pd
import numpy as np

def filter_extreme_values(sigma_mult: float):
    """
    Zwraca funkcję-filtr, która zostawia tylko świece ekstremalne,
    tzn. takie gdzie close > vwap_plus_{sigma_mult}_sigma
    lub close < vwap_minus_{sigma_mult}_sigma.
    Zakładamy, że te kolumny są już w ramce
    Uzyskamy je funkcją add_VWAP z modułu load_data.py
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper = df[f"vwap_plus_{sigma_mult}_sigma"]
        lower = df[f"vwap_minus_{sigma_mult}_sigma"]
        mask = (df["close"] > upper) | (df["close"] < lower)
        return df.loc[mask]

    _filter.__name__ = f"filter_extreme_values_{sigma_mult}"
    return _filter

def filter_hours(start_hour: int, end_hour: int):
    """
    Zwraca funkcję-filtr, która usuwa wiersze df
    znajdujące się w podanym przedziale godzin [start_hour, end_hour)
    Zostało to zaprojektowane z myślą o wskaźniku VWAP który jest niestabilny
    w początkowych godzinach sesji
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        # upewniamy się, że timestamp jest datetime
        ts = pd.to_datetime(df["timestamp"])
        mask = (ts.dt.hour < start_hour) | (ts.dt.hour >= end_hour)
        return df.loc[mask]
    _filter.__name__ = f"filter_hours_{start_hour}_{end_hour}"
    return _filter

def filter_clean():
    """
    Zwraca funkcję-filtr czyszczącą ramkę na wszelkie możliwe sposoby
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.replace(["-", np.inf, -np.inf, None], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df
    _filter.__name__ = "filter_clean"
    return _filter

def filter_unique_extremes_in_window(N: int, extreme_col: str = "is_extreme"):
    """
    Filtr zostawia świecę, jeśli w oknie N tylko ona jest ekstremalna.
    Jeśli w oknie pierwsza i inna świeca są ekstremalne,
    usuwa TYLKO pierwszą i przesuwa okno dalej o 1.
    Funkcja zakłada istnienie maski "is_extreme"
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask_keep = np.zeros(len(df), dtype=bool)

        i = 0
        while i < len(df) - N + 1:
            window = df.iloc[i:i+N]
            extremes = window[extreme_col].values

            if extremes[0] and extremes.sum() == 1:
                # tylko pierwsza ekstremalna
                mask_keep[i] = True
                i += N  # przeskocz całe okno
            elif extremes[0] and extremes.sum() > 1:
                # pierwsza + inna ekstremalna → usuń pierwszą
                # (czyli nic nie dodajemy do mask_keep)
                i += 1  # przesuwamy okno dalej o 1
            else:
                i += 1  # brak ekstremum na pierwszej pozycji
        return df.loc[mask_keep]

    _filter.__name__ = f"filter_unique_extremes_in_{N}"
    return _filter

def filter_remove_long_series(max_len: int, sigma_mult: float):
    """
    Funkcja ta usuwa serię świec ekstremalnych gdzie jest ich chociaż max_len pod rząd
    Funkcja zakłada istnienie kolumn vwap_plus/minus_{sigma_mult}_sigma - robi to add_VWAP
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        upper = df[f"vwap_plus_{sigma_mult}_sigma"]
        lower = df[f"vwap_minus_{sigma_mult}_sigma"]
        is_ext = ((df["close"] > upper) | (df["close"] < lower)).to_numpy()

        mask_keep = np.ones(len(df), dtype=bool)

        i = 0
        while i < len(is_ext):
            if is_ext[i]:
                j = i
                while j < len(is_ext) and is_ext[j]:
                    j += 1
                series_len = j - i
                if series_len > max_len:
                    mask_keep[i:j] = False
                i = j
            else:
                i += 1
        return df.loc[mask_keep]

    _filter.__name__ = f"filter_remove_long_series_{max_len}_{sigma_mult}"
    return _filter


def apply_filters(df, filters, log=False):

    orig_len = len(df)
    current_df = df.copy()

    if log:
        print(f">>> Start: {orig_len} rekordów")
        for i, f in enumerate(filters, 1):
            before = len(current_df)
            current_df = f(current_df)   # zakładam, że filtr zwraca ramkę
            after = len(current_df)

            removed = before - after
            pct_step = (removed / before * 100) if before > 0 else 0
            pct_total = (after / orig_len * 100) if orig_len > 0 else 0

            print(
                f"Filtr {i}: {f.__name__:<25} "
                f"usunął {removed} rekordów ({pct_step:.2f}% tego kroku). "
                f"Pozostało {after} rekordów = {pct_total:.2f}% oryginału."
            )

        print(f">>> Koniec: {len(current_df)} rekordów (pozostało {len(current_df) / orig_len:.2%} z oryginału)")
        return current_df
    else:
        for f in filters:
            current_df = f(current_df)
        return current_df

FOMC_DATES = [
    # 2020
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",

    # 2021
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",

    # 2022
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",

    # 2023
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",

    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",

    # 2025 (wg kalendarza tentatywnego) :contentReference[oaicite:1]{index=1}
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-10"
]

def filter_around_fomc():
    """
    Filtr usuwa wiersze ±1 dzień wokół dat FOMC.
    Zakłada, że w df istnieje kolumna 'timestamp' (datetime).
    """
    fomc_days = pd.to_datetime(FOMC_DATES).normalize()
    all_excluded = set(fomc_days) | set(fomc_days - pd.Timedelta(days=1)) | set(fomc_days + pd.Timedelta(days=1))

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "timestamp" not in df.columns:
            raise ValueError("Brak kolumny 'timestamp' w dataframe!")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["day"] = df["timestamp"].dt.normalize()

        mask_drop = df["day"].isin(all_excluded)
        return df.loc[~mask_drop].drop(columns="day")

    _filter.__name__ = "filter_around_fomc"
    return _filter

filters = [filter_clean(), filter_hours(0, 8), filter_around_fomc()] #filter_remove_long_series(40, sigma_val)]