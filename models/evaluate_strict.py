import random
import time

from auto_rec import AutoRecRecommender
from content_knn import ContentBasedRecommender
from item_knn import ItemBasedRecommender
from util import load_test_datas, root_path
from svd import SVDRecommender
from user_knn import UserBasedRecommender

# --- Configuration ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

HIDE_RATIO = 0.2
TOP_N_RECS = 10


def run_strict_evaluation():
    print("--- Agnostic Multi-Model Hit Rate & Precision Benchmark ---\n")

    print("[Eval] Initializing models (Black-box mode)...")
    models = {
        "User-KNN": UserBasedRecommender(db_path=TRAIN_DB),
        "Content-KNN": ContentBasedRecommender(db_path=TRAIN_DB),
        "Deep AutoRec": AutoRecRecommender(db_path=TRAIN_DB),
        "Item-KNN": ItemBasedRecommender(db_path=TRAIN_DB),
        "SVD": SVDRecommender(db_path=TRAIN_DB),
    }

    metrics = {name: {"hits": 0, "precision_sum": 0.0, "time": 0.0} for name in models}

    print("\n[Eval] Loading test subjects from Test DB...")
    test_reviews, test_users = load_test_datas(TEST_DB)
    valid_evaluations = 0

    for i, user in enumerate(test_users, 1):
        user_data = test_reviews[test_reviews['user_username'] == user]
        liked_movies = user_data[user_data['rating'] >= 4.0]['movie_slug'].tolist()

        if len(liked_movies) < 5:
            continue

        hidden_count = max(1, int(len(liked_movies) * HIDE_RATIO))
        test_set = random.sample(liked_movies, hidden_count)

        train_data = user_data[~user_data['movie_slug'].isin(test_set)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

        # ==========================================
        # Core Evaluation Loop
        # ==========================================
        for model_name, model in models.items():
            start_time = time.time()

            # Standardized invocation
            recommendations = model.get_recommendations(train_profile, top_n=TOP_N_RECS)
            rec_slugs = [rec['slug'] for rec in recommendations]

            metrics[model_name]["time"] += (time.time() - start_time)

            hits = len([slug for slug in test_set if slug in rec_slugs])
            if hits > 0:
                metrics[model_name]["hits"] += 1
            metrics[model_name]["precision_sum"] += (hits / TOP_N_RECS)

        valid_evaluations += 1
        if valid_evaluations % 50 == 0:
            print(f"[{valid_evaluations}] Users evaluated...")

    if valid_evaluations == 0:
        print("\nEvaluation failed: No valid users fit the criteria.")
        return

    # ==========================================
    # Print Leaderboard
    # ==========================================
    print("\n" + "=" * 65)
    print(f"HIT RATE & PRECISION LEADERBOARD (Top-{TOP_N_RECS})")
    print(f"Total Valid Users Evaluated: {valid_evaluations}")
    print("=" * 65)
    print(f"{'Model Name':<18} | {'Hit Rate (@10)':<15} | {'Precision (@10)':<15} | {'Avg Latency'}")
    print("-" * 65)

    sorted_models = sorted(metrics.items(), key=lambda x: x[1]["hits"], reverse=True)

    for name, data in sorted_models:
        hit_rate = data["hits"] / valid_evaluations
        precision = data["precision_sum"] / valid_evaluations
        avg_time = (data["time"] / valid_evaluations) * 1000
        print(f" {name:<17} | {hit_rate:>10.2%}      | {precision:>10.2%}      | {avg_time:>6.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    run_strict_evaluation()
