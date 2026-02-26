import sqlite3
import pandas as pd
import random
from content_knn import ContentBasedRecommender
from user_knn import UserBasedRecommender

# --- Configuration ---
TRAIN_DB = "train_model.db"
TEST_DB = "test_eval.db"

HIDE_RATIO = 0.2  # Mask 20% of the test user's liked movies
TOP_N_RECS = 10  # Recommend Top-10


def run_strict_evaluation():
    print("📊 --- Starting Strict Evaluation (Train/Test Isolated) --- 📊")

    # 1. Initialize the model using ONLY the Train DB
    print("[Eval] Initializing KNN model with Train DB...")
    recommender = UserBasedRecommender(db_path=TRAIN_DB)
    # recommender = ContentBasedRecommender(db_path=TRAIN_DB)

    # 2. Load the Test users from the Test DB
    print("[Eval] Loading test subjects from Test DB...")
    conn = sqlite3.connect(TEST_DB)
    df_test_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'",
                                        conn)
    df_test_reviews['rating'] = pd.to_numeric(df_test_reviews['rating'], errors='coerce').dropna()
    conn.close()

    test_users = df_test_reviews['user_username'].unique()
    print(f"[Eval] Found {len(test_users)} test users.")

    total_hits = 0
    total_precision = 0.0
    valid_evaluations = 0

    # 3. Evaluation Loop
    for i, user in enumerate(test_users, 1):
        user_data = df_test_reviews[df_test_reviews['user_username'] == user]

        # Ground Truth: Movies the user liked (>= 4.0)
        liked_movies = user_data[user_data['rating'] >= 4.0]['movie_slug'].tolist()

        if len(liked_movies) < 5:
            continue  # Skip users with too few high ratings to split meaningfully

        # Hide a portion of liked movies (Test Set)
        hidden_count = max(1, int(len(liked_movies) * HIDE_RATIO))
        test_set = random.sample(liked_movies, hidden_count)

        # The remaining movies become the "input profile" (Train Set)
        train_data = user_data[~user_data['movie_slug'].isin(test_set)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

        # Get recommendations from the model
        recommendations = recommender.get_recommendations(train_profile, top_n=TOP_N_RECS)
        rec_slugs = [rec['slug'] for rec in recommendations]

        # Check for hits
        hits_for_this_user = len([slug for slug in test_set if slug in rec_slugs])

        if hits_for_this_user > 0:
            total_hits += 1

        precision = hits_for_this_user / TOP_N_RECS
        total_precision += precision
        valid_evaluations += 1

        if i % 50 == 0:
            print(f"[{i}/{len(test_users)}] Evaluated... Current Precision: {total_precision / valid_evaluations:.2%}")

    if valid_evaluations == 0:
        print("\nEvaluation failed: No valid users fit the criteria.")
        return

    final_hit_rate = total_hits / valid_evaluations
    final_precision = total_precision / valid_evaluations

    # 4. Print Academic Results
    print("\n" + "=" * 50)
    print("📈 STRICT EVALUATION METRICS (ZERO LEAKAGE) 📈")
    print("=" * 50)
    print(f"Total Users Evaluated : {valid_evaluations}")
    print(f"Recommendations       : Top-{TOP_N_RECS}")
    print(f"Test Set Ratio        : {HIDE_RATIO * 100}% of user's liked movies")
    print("-" * 50)
    print(f"Hit Rate @ {TOP_N_RECS}        : {final_hit_rate:.2%} (Users with >= 1 correct guess)")
    print(f"Precision @ {TOP_N_RECS}       : {final_precision:.2%} (Percentage of accurate recommendations)")
    print("=" * 50)


if __name__ == "__main__":
    run_strict_evaluation()