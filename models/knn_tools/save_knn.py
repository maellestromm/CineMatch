import pickle
from util import root_path
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.user_knn import UserBasedRecommender

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
SAVE_DIR = root_path() / "models/saved_models"

# User-KNN
user_best_k = 10
user_knn = UserBasedRecommender(db_path=TRAIN_DB, k_neighbors=user_best_k)
with open(SAVE_DIR / "user_knn.pkl", "wb") as f:
    pickle.dump(user_knn, f)
print("[User-KNN] Model with k={user_best_k} saved to {SAVE_DIR}/user_knn.pkl")

# Item-KNN
item_best_k = 8
item_knn = ItemBasedRecommender(db_path=True, k_neighbors=item_best_k)
with open(SAVE_DIR / "item_knn.pkl", "wb") as f:
    pickle.dump(item_knn, f)
print("[Item-KNN] Model with k={item_best_k} saved to {SAVE_DIR}/item_knn.pkl")

# Content-KNN
content_knn = ContentBasedRecommender(db_path=root_path() / "data/user_first_cut3_clear.db")
with open(SAVE_DIR / "content_knn.pkl", "wb") as f:
    pickle.dump(content_knn, f)
print("[Content-KNN] Model saved to {SAVE_DIR}/content_knn.pkl")