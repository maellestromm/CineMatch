import pickle
from models.auto_rec import AutoRecRecommender
from util import root_path

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
SAVE_DIR = root_path() / "models/saved_models"

autorec_model = AutoRecRecommender(db_path=TRAIN_DB)
with open(SAVE_DIR / "auto_rec.pkl", "wb") as f:
    pickle.dump(autorec_model, f)
print(f"[SVD] Model saved to {SAVE_DIR}/auto_rec.pkl")