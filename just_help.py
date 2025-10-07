# # import os
# # from pathlib import Path
# # import shutil
# #
# # base_dir = Path(r"C:\Users\norbe\PycharmProjects\VWAP_NETWORK\models")
# #
# # for file in base_dir.glob("*.keras"):
# #     model_name = file.stem  # nazwa bez rozszerzenia
# #     target_dir = base_dir / model_name
# #     target_dir.mkdir(exist_ok=True)
# #
# #     target_path = target_dir / file.name
# #     shutil.move(str(file), str(target_path))
# #     print(f"[OK] Przeniesiono: {file.name} → {target_dir}")
#
# import pandas as pd
# from pathlib import Path
#
# models_path = Path("models")
# stat_main = "mean_R_f"       # główna statystyka do sortowania
# stat_sharpe = "sharpe_BASIC" # dodatkowa statystyka do wyświetlenia
#
# model_scores = []  # (model_name, mean_Rf, mean_Sharpe)
#
# for file in models_path.iterdir():
#     if file.is_file():
#         continue  # tylko foldery
#
#     summary_path = file / "summary_metrics.csv"
#     if not summary_path.exists():
#         print(f"[SKIP] Brak summary_metrics.csv dla {file.name}")
#         continue
#
#     try:
#         df = pd.read_csv(summary_path, index_col=0)
#
#         if stat_main not in df.index or stat_sharpe not in df.index:
#             print(f"[WARN] {file.name}: brak jednej z metryk ({stat_main} lub {stat_sharpe})")
#             continue
#
#         row_main = pd.to_numeric(df.loc[stat_main], errors="coerce")
#         row_sharpe = pd.to_numeric(df.loc[stat_sharpe], errors="coerce")
#
#         mean_main = row_main.mean(skipna=True)
#         mean_sharpe = row_sharpe.mean(skipna=True)
#
#         model_scores.append((file.name, mean_main, mean_sharpe))
#         print(f"[OK] {file.name}: {stat_main}={mean_main:.4f}, {stat_sharpe}={mean_sharpe:.4f}")
#
#     except Exception as e:
#         print(f"[ERR] {file.name}: {e}")
#
# # sortowanie malejąco po mean_R_f
# model_scores.sort(key=lambda x: x[1], reverse=True)
#
# print("\n=== MODELE POSORTOWANE WG ŚREDNIEJ mean_R_f (oraz średni Sharpe) ===")
# for name, val_main, val_sharpe in model_scores:
#     print(f"{val_main:>10.4f}  |  Sharpe={val_sharpe:>8.4f}  |  {name}")

# import pandas as pd
#
# # === 1. Wczytanie pliku ===
# df = pd.read_csv("label_summary.csv")
#
# # === 2. Znalezienie 5 najlepszych labeli wg entropii ===
# top_labels = (
#     df.groupby("label")["entropy"]
#     .mean()
#     .sort_values(ascending=False)
#     .head(5)
#     .index
# )
#
# print("=== 🏆 5 najlepszych labeli (wg entropii) ===")
# for name in top_labels:
#     print(" -", name)
#
# # === 3. Wyciągnięcie pełnych statystyk dla tych labeli ===
# top_df = df[df["label"].isin(top_labels)].copy()
#
# stats = (
#     top_df.groupby("label")[
#         [
#             "entropy", "dominant_share",
#             "Q_2_delta_0", "Q_2_delta_1", "Q_2_delta_2",
#             "mean_delta_0", "mean_delta_1", "mean_delta_2"
#         ]
#     ]
#     .mean()
#     .sort_values("entropy", ascending=False)
# )
#
# # === 4. Czytelny wydruk wyników ===
# print("\n=== 📊 Szczegółowe statystyki top 5 labeli ===")
# print(stats.to_string(float_format=lambda x: f"{x:.4f}"))









# import pandas as pd
# import numpy as np
# from pathlib import Path
#
# base_path = Path("data/training_data")
# report_rows = []
#
# print(f"🔍 Start diagnostyki plików features w {base_path.resolve()}")
#
# for ticker_dir in base_path.iterdir():
#     if not ticker_dir.is_dir() or ticker_dir.name.startswith("."):
#         continue
#
#     features_path = ticker_dir / "features" / "features00.parquet"
#     if not features_path.exists():
#         print(f"⚠️ Brak pliku {features_path}")
#         continue
#
#     print(f"📁 Sprawdzam {ticker_dir.name} ...")
#
#     try:
#         df = pd.read_parquet(features_path)
#         for col in df.columns:
#             s = df[col]
#
#             n_nan = s.isna().sum()
#             n_inf = np.isinf(s).sum() if np.issubdtype(s.dtype, np.number) else 0
#             n_none = (s == None).sum() if s.dtype == "object" else 0
#
#             report_rows.append({
#                 "ticker": ticker_dir.name,
#                 "column": col,
#                 "nan": int(n_nan),
#                 "inf": int(n_inf),
#                 "none": int(n_none),
#                 "total": len(s),
#                 "nan_%": round(100 * n_nan / len(s), 4),
#                 "inf_%": round(100 * n_inf / len(s), 4),
#             })
#
#     except Exception as e:
#         print(f"❌ Błąd przy {features_path}: {e}")
#
# # --- zapis raportu ---
# report_df = pd.DataFrame(report_rows)
# output_path = Path("features_diagnostics.csv")
# report_df.to_csv(output_path, index=False, float_format="%.6f")
#
# print(f"\n✅ Raport zapisany: {output_path.resolve()}")
# print(f"Znaleziono {len(report_rows)} kolumn w {report_df['ticker'].nunique()} plikach.")
# print(report_df.groupby('ticker')[['nan', 'inf']].sum())








