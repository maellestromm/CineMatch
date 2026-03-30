import pickle
from models.svd import SVDRecommender
from util import root_path

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
SAVE_DIR = root_path() / "models/saved_models"

best_k = 50 
svd_model = SVDRecommender(db_path=TRAIN_DB, k_factors=best_k)
with open(SAVE_DIR / "svd.pkl", "wb") as f:
    pickle.dump(svd_model, f)
print(f"[SVD] Model with k={best_k} saved to {SAVE_DIR}/svd.pkl")