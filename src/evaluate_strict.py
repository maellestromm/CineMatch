import sqlite3
import pandas as pd
import random
import time

# from user_knn_gpu import UserBasedRecommender
from user_knn import UserBasedRecommender
from content_knn import ContentBasedRecommender
from infer_autorec import AutoRecRecommender

# --- Configuration ---
TRAIN_DB = "train_model.db"
TEST_DB = "test_eval.db"

HIDE_RATIO = 0.2
TOP_N_RECS = 10


def run_strict_evaluation():
    print("📊 --- Agnostic Multi-Model Hit Rate & Precision Benchmark --- 📊\n")

    print("[Eval] Initializing models (Black-box mode)...")
    models = {
        "User-KNN": UserBasedRecommender(db_path=TRAIN_DB),
        # "Content-KNN": ContentBasedRecommender(db_path=TRAIN_DB),
        # "Deep AutoRec": AutoRecRecommender(db_path=TRAIN_DB)
    }

    metrics = {name: {"hits": 0, "precision_sum": 0.0, "time": 0.0} for name in models}

    print("\n[Eval] Loading test subjects from Test DB...")
    conn = sqlite3.connect(TEST_DB)
    df_test_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'",
                                        conn)
    df_test_reviews['rating'] = pd.to_numeric(df_test_reviews['rating'], errors='coerce').dropna()
    conn.close()

    test_users = df_test_reviews['user_username'].unique()
    valid_evaluations = 0

    for i, user in enumerate(test_users, 1):
        user_data = df_test_reviews[df_test_reviews['user_username'] == user]
        liked_movies = user_data[user_data['rating'] >= 4.0]['movie_slug'].tolist()

        if len(liked_movies) < 5:
            continue

        hidden_count = max(1, int(len(liked_movies) * HIDE_RATIO))
        test_set = random.sample(liked_movies, hidden_count)

        train_data = user_data[~user_data['movie_slug'].isin(test_set)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

        # ==========================================
        # 核心：纯接口调用，极其优雅的轮询
        # ==========================================
        for model_name, model in models.items():
            start_time = time.time()

            # 标准化调用
            recommendations = model.get_recommendations(train_profile, top_n=TOP_N_RECS)
            rec_slugs = [rec['slug'] for rec in recommendations]

            metrics[model_name]["time"] += (time.time() - start_time)

            hits = len([slug for slug in test_set if slug in rec_slugs])
            if hits > 0:
                metrics[model_name]["hits"] += 1
            metrics[model_name]["precision_sum"] += (hits / TOP_N_RECS)

        valid_evaluations += 1
        if valid_evaluations % 50 == 0:
            print(f"[{valid_evaluations}] Users evaluated... (Simultaneous battle for {len(models)} models)")

    if valid_evaluations == 0:
        print("\nEvaluation failed: No valid users fit the criteria.")
        return

    # ==========================================
    # 打印学术级排行榜
    # ==========================================
    print("\n" + "=" * 65)
    print(f"🏆 HIT RATE & PRECISION LEADERBOARD (Top-{TOP_N_RECS}) 🏆")
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
