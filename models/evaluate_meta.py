import math
import random
import numpy as np
from util import load_test_datas, root_path

from .meta import MetaRecommender

# --- Configuration ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
HIDE_RATIO = 0.2

def run_meta_rmse_evaluation():
    print("--- Meta-Ensemble RMSE Benchmark (1-5 Star Accuracy) ---\n")

    print("[Eval] Initializing MetaRecommender...")
    meta_model = MetaRecommender(db_path=TRAIN_DB)

    metrics = []

    print("\n[Eval] Loading test subjects from Test DB...")
    test_reviews, test_users = load_test_datas(TEST_DB)
    valid_users = 0

    for i, user in enumerate(test_users, 1):
        user_data = test_reviews[test_reviews['user_username'] == user]

        if len(user_data) < 5:
            continue  # skip users with too few ratings

        all_movies = user_data['movie_slug'].tolist()
        hidden_count = max(1, int(len(all_movies) * HIDE_RATIO))

        # select movies to hide
        test_set_slugs = random.sample(all_movies, hidden_count)
        train_data = user_data[~user_data['movie_slug'].isin(test_set_slugs)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))
        user_avg = train_data['rating'].mean() if not train_data.empty else 3.0

        # get meta predictions
        raw_recs = meta_model.get_recommendations(train_profile, top_n=3334)
        pred_dict = {rec['slug']: rec['score'] for rec in raw_recs}

        # compute squared errors
        for hidden_slug in test_set_slugs:
            actual_rating = float(user_data[user_data['movie_slug'] == hidden_slug]['rating'].values[0])
            pred_rating = pred_dict.get(hidden_slug, user_avg)
            metrics.append((pred_rating - actual_rating) ** 2)

        valid_users += 1
        if valid_users % 50 == 0:
            print(f"[{valid_users}] Users evaluated...")

    # --- Print final RMSE ---
    final_rmse = math.sqrt(np.mean(metrics))
    print("\n" + "=" * 55)
    print("Meta-Ensemble RMSE (Lower is Better)")
    print(f"Hidden ratings evaluated per model: {len(metrics)}")
    print("=" * 55)
    print(f"Meta-Ensemble RMSE: {final_rmse:.4f}")
    print("=" * 55)

if __name__ == "__main__":
    run_meta_rmse_evaluation()