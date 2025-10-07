import os
import keras

src = "models"
dst = os.path.join(src, "h5")
os.makedirs(dst, exist_ok=True)

ok = bad = 0
for fname in os.listdir(src):
    if not fname.endswith(".keras"):
        continue
    fpath = os.path.join(src, fname)
    try:
        print("[INFO] Ładuję", fname)
        model = keras.models.load_model(fpath, compile=False)
        outpath = os.path.join(dst, os.path.splitext(fname)[0] + ".h5")
        model.save(outpath, save_format="h5")
        print("  -> Zapisano jako", outpath)
        ok += 1
    except Exception as e:
        print("[BŁĄD]", fname, ":", e)
        bad += 1

print(f"\n✅ Konwersja zakończona: {ok} OK, {bad} błędów.")
