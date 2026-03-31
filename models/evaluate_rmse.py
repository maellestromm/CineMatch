import math
import random

import numpy as np

from models.load_models import load_model
from util import load_test_datas, root_path
from .meta import MetaRecommender

# --- Configuration ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
HIDE_RATIO = 0.2


def run_rmse_evaluation():
    print("--- Agnostic Multi-Model RMSE Benchmark (1-5 Star Accuracy) ---\n")

    # ==========================================
    # 1. Model Registry (Black-box mode)
    # Test script relies solely on the get_recommendations interface.
    # ==========================================
    print("[Eval] Initializing models (Black-box mode)...")
    models = {
        "User-KNN": load_model("user_knn"),
        "Deep-AutoRec": load_model("auto_rec"),
        "Item-KNN": load_model("item_knn"),
        "SVD-50": load_model("svd"),
        "Content-KNN": load_model("content_knn"),
        "Meta": load_model("meta")
    }

    metrics = {name: [] for name in models}

    print("\n[Eval] Loading test subjects from Test DB...")
    test_reviews, test_users = load_test_datas(TEST_DB)
    valid_users = 0

    for i, user in enumerate(test_users, 1):
        user_data = test_reviews[test_reviews['user_username'] == user]

        if len(user_data) < 5:
            continue

        all_movies = user_data['movie_slug'].tolist()
        hidden_count = max(1, int(len(all_movies) * HIDE_RATIO))

        test_set_slugs = random.sample(all_movies, hidden_count)
        train_data = user_data[~user_data['movie_slug'].isin(test_set_slugs)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))
        user_avg = train_data['rating'].mean() if not train_data.empty else 3.0

        # ==========================================
        # 2. Core Evaluation Loop
        # ==========================================
        for model_name, model in models.items():
            # Request large top_n to get predictions for all remaining movies
            raw_recs = model.get_recommendations(train_profile, top_n=3334)

            # Convert to O(1) lookup dictionary: {slug: score}
            pred_dict = {rec['slug']: rec['score'] for rec in raw_recs}

            # Grade predictions against hidden movies
            for hidden_slug in test_set_slugs:
                actual_rating = float(user_data[user_data['movie_slug'] == hidden_slug]['rating'].values[0])

                # Fallback to user's average rating if model lacks prediction (cold start)
                pred_rating = pred_dict.get(hidden_slug, user_avg)

                # Record squared error
                metrics[model_name].append((pred_rating - actual_rating) ** 2)

        valid_users += 1
        if valid_users % 50 == 0:
            print(f"[{valid_users}] Users evaluated...")

    # ==========================================
    # 3. Print Final Leaderboard
    # ==========================================
    print("\n" + "=" * 55)
    print("RMSE ACCURACY LEADERBOARD (Lower is Better)")
    print(f"Hidden ratings evaluated per model: {len(metrics[list(models.keys())[0]])}")
    print("=" * 55)
    print(f"{'Model Name':<18} | {'RMSE Score':<15}")
    print("-" * 45)

    final_rmses = {name: math.sqrt(np.mean(errors)) for name, errors in metrics.items() if len(errors) > 0}
    sorted_rmses = sorted(final_rmses.items(), key=lambda x: x[1])

    for rank, (name, rmse) in enumerate(sorted_rmses):
        print(f" {name:<17} | {rmse:<15.4f}")

    print("=" * 55)


if __name__ == "__main__":
    run_rmse_evaluation()
