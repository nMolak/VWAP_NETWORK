# TODO: Wszystkie funkcjonalności przerzucić do tego pliku

# TODO: Zrobić może osobny scaler w każdym modelu

# TODO: Upewnić się, że nowa wersja backtestu działa poprawnie (thresholdy zostawić na później)



#=== Iteracja po modelach i zapisanie ich nazw do .txt w celu dalszej analizy ===

from load_data import save_model_names

class3_models_path = "models/3_class"
output_file_name = "model_names.txt"
n_classes = 3

save_model_names(class3_models_path, output_file_name, n_classes)



