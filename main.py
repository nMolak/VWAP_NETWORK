# TODO: Wszystkie funkcjonalności przerzucić do tego pliku

# TODO: Zrobić może osobny scaler w każdym modelu

# TODO: Upewnić się, że nowa wersja backtestu działa poprawnie (thresholdy zostawić na później)



#=== Iteracja po modelach i zapisanie ich nazw do .txt w celu dalszej analizy ===

# from load_data import save_model_names
#
# class3_models_path = "models/3_class"
# output_file_name = "model_names.txt"
# n_classes = 3
#
# save_model_names(class3_models_path, output_file_name, n_classes)


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

from load_data import diagnose_mean_and_variance
from utils import display_dataframe

data_path = "data/training_data"
df_mean, df_std = diagnose_mean_and_variance(data_path, log=False)
data_path = "data/test_data"
df_mean, df_std = diagnose_mean_and_variance(data_path, log=False)



