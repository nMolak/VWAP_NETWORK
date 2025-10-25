# TODO: Wszystkie funkcjonalności przerzucić do tego pliku

# TODO: Zrobić może osobny scaler w każdym modelu

# TODO: Upewnić się, że nowa wersja backtestu działa poprawnie (thresholdy zostawić na później)

#=== Ściągnięcie danych OHLCV ===

# from load_data import iterate_over_folder_and_save
# tickers = ["DOT/USDT", "AAVE/USDT", "VET/USDT", ]
# data_path = "data/test_data"
# interval = "1m"
# year, month, day = 2021, 6, 1
# iterate_over_folder_and_save(tickers, interval, year, month, day, data_path, log=True)


#=== Iteracja po modelach i zapisanie ich nazw do .txt w celu dalszej analizy ===

# from load_data import save_model_names
#
# class2_models_path = "models/2_class"
# output_file_name = "model_names2.txt"
# n_classes = 2
#
# save_model_names(class2_models_path, output_file_name, n_classes)





# import pandas as pd
# from parameters import sigma_val as sigma
# from load_data import add_VWAP, add_indicators
# dot_path = r"C:\Users\norbe\PycharmProjects\VWAP_NETWORK\data\test_data\DOT_USDT\DOT_USDT_1m_data.csv"
# csv = pd.read_csv(dot_path)
# csv = add_VWAP(csv, sigma)
# csv = add_indicators(csv)
#
# csv.to_csv(dot_path)



#=== Diagnostyka modułów / wersji pythona ===

# from load_data import modules_diagnostics
# modules_diagnostics()

#=== Wczytanie featurów do istniejących ramek danych ===

# from load_data import iterate_over_folder_and_save_features
# from calculations import calc_indicators
#
# data_path = 'data/test_data'
# iterate_over_folder_and_save_features(data_path, calc_indicators)

#=== Analiza średniej i odchylenia w plikach df ===

# from load_data import diagnose_mean_and_variance
# from utils import display_dataframe
#
# data_path = "data/training_data"
# df_mean, df_std = diagnose_mean_and_variance(data_path, log=False)
# data_path = "data/test_data"
# df_mean, df_std = diagnose_mean_and_variance(data_path, log=False)


#=== Sprawdzenie czy featury są poprawnie liczone ===

from pathlib import Path
from load_data import add_VWAP
from parameters import sigma_val as sigma

# data_path = Path("data/training_data")
#
# for folder in sorted(data_path.iterdir()):
#     if folder.is_dir():
#         csv_files = list(folder.glob("*.csv"))
#         if not csv_files:
#             print(f"[POMINIĘTO] brak CSV w {folder}")
#             continue
#         file = csv_files[0]
#         df = pd.read_csv(file)
#         df = add_VWAP(df, sigma)
#         df["is_extreme"] = (df["close"] > df[f"vwap_plus_{sigma}_sigma"]) | \
#                            (df["close"] < df[f"vwap_minus_{sigma}_sigma"])
#
#         print(f"[OK] Wczytano {file} ({df.shape[0]} wierszy)")
#         calc_indicators(df, log=True)
#










