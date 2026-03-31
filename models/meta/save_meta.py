import pickle
from models.meta import MetaRecommender
from util import root_path
import subprocess

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
SAVE_DIR = root_path() / "models/saved_models"

def save_meta_ensemble():
    svd_model = MetaRecommender(db_path=TRAIN_DB)
    with open(SAVE_DIR / "meta.pkl", "wb") as f:
        pickle.dump(svd_model, f)
    print(f"[Meta] Model saved to {SAVE_DIR}/meta.pkl")

def compress_meta_pkl():
    subprocess.call(["7z", "a", "meta.7z", SAVE_DIR/"meta.pkl"])

def main():
    save_meta_ensemble()
    compress_meta_pkl()



if __name__ == "__main__":
    main()