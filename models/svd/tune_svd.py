import numpy as np
from util import load_test_datas, root_path
import random
from models.svd import SVDRecommender

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

NUM_USERS_SAMPLE = 500   # use subset for speed
HIDE_RATIO = 0.2

#K_VALUES = [20, 30, 40, 50, 60, 70]
K_VALUES = [50, 55, 60, 65, 70]

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

        recs = model.get_recommendations(train_profile, top_n=3334)
        pred_dict = {r['slug']: r['score'] for r in recs}

        # RMSE:
        for slug in hidden:
            actual = float(user_data[user_data['movie_slug'] == slug]['rating'].values[0])
            pred = pred_dict.get(slug, user_avg)
            errors.append((pred - actual) ** 2)

        # Hit Rate and Precision @10
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

def tune_svd(results, test_reviews, sampled_users):
    print("\n=== Tuning SVD ===")
    for k in K_VALUES:
        print(f"\n[SVD] Testing k={k}")

        model = SVDRecommender(db_path=TRAIN_DB, k_factors=k)

        rmse, hit_rate, precision = evaluate_model(model, test_reviews, sampled_users)

        print(f"[SVD k={k}] RMSE: {rmse:.4f} | Hit@10: {hit_rate:.2%} | Prec@10: {precision:.2%}")

        results.append(("SVD", k, rmse, hit_rate, precision))

def main():
    print("Loading test data...")
    test_reviews, test_users = load_test_datas(TEST_DB)

    # get random sample of users 
    #sampled_users = random.sample(list(test_users), NUM_USERS_SAMPLE)

    print(f"Using {len(test_users)} users for tuning\n")

    results = []
    tune_svd(results, test_reviews, test_users)

    # print final results
    print("\n" + "="*70)
    print("SVD TUNING RESULTS")
    print("="*70)
    print(f"{'Model':<10} | {'k':<3} | {'RMSE':<8} | {'Hit@10':<10} | {'Prec@10'}")
    print("-"*70)

    # sort by Precision@10
    results_sorted = sorted(results, key=lambda x: x[4], reverse=True)
    for model, k, rmse, hit_rate, precision in results_sorted:
        print(f"{model:<10} | {k:<3} | {rmse:<8.4f} | {hit_rate:<10.2%} | {precision:.2%}")

if __name__ == "__main__":
    main()