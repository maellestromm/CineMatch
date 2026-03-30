import numpy as np
import random

from util import load_test_datas, root_path
from models.meta import MetaRecommender

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

NUM_USERS_SAMPLE = 500
HIDE_RATIO = 0.2
NUM_WEIGHT_SAMPLES = 30   # number of random weight configs to test


# --------------------------------------------------
# Generate random normalized weights
# --------------------------------------------------
def sample_weights():
    weights = np.random.rand(5)  # 5 models
    weights /= weights.sum()

    return {
        "SVD-50": weights[0],
        "User-KNN": weights[1],
        "Item-KNN": weights[2],
        "Deep-AutoRec": weights[3],
        "Content-KNN": weights[4]
    }


# --------------------------------------------------
# Evaluation (same as your tune_svd)
# --------------------------------------------------
def evaluate_model(model, test_reviews, test_users):
    errors = []
    total_users = 0
    hit_count = 0
    precision_total = 0

    for user in test_users:
        user_data = test_reviews[test_reviews['user_username'] == user]

        if len(user_data) < 5:
            continue

        all_movies = user_data['movie_slug'].tolist()
        hidden_count = max(1, int(len(all_movies) * HIDE_RATIO))

        hidden = set(random.sample(all_movies, hidden_count))
        train_data = user_data[~user_data['movie_slug'].isin(hidden)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

        user_avg = train_data['rating'].mean() if not train_data.empty else 3.0

        recs = model.get_recommendations(train_profile, top_n=200)
        pred_dict = {r['slug']: r['score'] for r in recs}

        # --- RMSE ---
        for slug in hidden:
            actual = float(user_data[user_data['movie_slug'] == slug]['rating'].values[0])
            pred = pred_dict.get(slug, user_avg)
            errors.append((pred - actual) ** 2)

        # --- Hit + Precision ---
        top_10 = recs[:10]
        top_10_slugs = set([r['slug'] for r in top_10])

        hits = len(top_10_slugs & hidden)

        if hits > 0:
            hit_count += 1

        precision_total += hits / 10.0
        total_users += 1

    rmse = np.sqrt(np.mean(errors)) if errors else float('nan')
    hit_rate = hit_count / total_users if total_users > 0 else 0.0
    precision = precision_total / total_users if total_users > 0 else 0.0

    return rmse, hit_rate, precision


# --------------------------------------------------
# Tuning loop
# --------------------------------------------------
def tune_meta(results, test_reviews, sampled_users):
    print("\n=== Tuning Meta Ensemble Weights ===")

    for i in range(NUM_WEIGHT_SAMPLES):
        print(f"\n[Meta] Testing config {i+1}/{NUM_WEIGHT_SAMPLES}")

        weights = sample_weights()

        model = MetaRecommender(TRAIN_DB)
        model.weights = weights

        rmse, hit_rate, precision = evaluate_model(model, test_reviews, sampled_users)

        print(f"Weights: {weights}")
        print(f"[Meta] RMSE: {rmse:.4f} | Hit@10: {hit_rate:.2%} | Prec@10: {precision:.2%}")

        results.append((weights, rmse, hit_rate, precision))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("Loading test data...")
    test_reviews, test_users = load_test_datas(TEST_DB)

    sampled_users = random.sample(list(test_users), NUM_USERS_SAMPLE)
    print(f"Using {len(sampled_users)} users for tuning\n")

    results = []
    tune_meta(results, test_reviews, sampled_users)

    # --------------------------------------------
    # Print results sorted by Precision@10
    # --------------------------------------------
    print("\n" + "=" * 80)
    print("META ENSEMBLE TUNING RESULTS")
    print("=" * 80)

    results_sorted = sorted(results, key=lambda x: x[3], reverse=True)

    for weights, rmse, hit_rate, precision in results_sorted[:10]:
        print(f"Weights: {weights}")
        print(f"RMSE: {rmse:.4f} | Hit@10: {hit_rate:.2%} | Prec@10: {precision:.2%}")
        print("-" * 80)


if __name__ == "__main__":
    main()