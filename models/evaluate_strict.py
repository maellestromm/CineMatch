import random
import time

from models.evaluate_rmse import test_models

from util import root_path, load_test_datas

# --- Configuration ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

HIDE_RATIO = 0.2
TOP_N_RECS = 10


def get_strict_test_profiles(test_db_path=TEST_DB, hide_ratio=HIDE_RATIO, seed=42):
    random.seed(seed)
    df_test, test_users = load_test_datas(test_db_path)
    profiles = []

    for user in test_users:
        user_data = df_test[df_test['user_username'] == user]

        liked_movies = user_data[user_data['rating'] >= 4.0]['movie_slug'].tolist()

        if len(liked_movies) < 5:
            continue

        hidden_count = max(1, int(len(liked_movies) * hide_ratio))
        test_set = random.sample(liked_movies, hidden_count)

        train_data = user_data[~user_data['movie_slug'].isin(test_set)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

        profiles.append({
            'user': user,
            'train_profile': train_profile,
            'test_set': test_set
        })

    return profiles


def evaluate_model_strict(model, test_profiles, top_n=TOP_N_RECS):
    hits = 0
    precision_sum = 0.0
    total_time = 0.0
    valid_evaluations = len(test_profiles)

    if valid_evaluations == 0:
        return 0.0, 0.0, 0.0

    for profile in test_profiles:
        start_time = time.time()

        recommendations = model.get_recommendations(profile['train_profile'], top_n=top_n)
        rec_slugs = [rec['slug'] for rec in recommendations]

        total_time += (time.time() - start_time)

        user_hits = len([slug for slug in profile['test_set'] if slug in rec_slugs])

        if user_hits > 0:
            hits += 1
        else:
            print(f"MISS | profile size: {len(profile['train_profile'])} "
                  f"| test_set size: {len(profile['test_set'])} "
                  f"| test movies: {profile['test_set']}")
        precision_sum += (user_hits / top_n)

    hit_rate = hits / valid_evaluations
    precision = precision_sum / valid_evaluations
    avg_latency = (total_time / valid_evaluations) * 1000

    return hit_rate, precision, avg_latency


def run_strict_evaluation():
    print("--- Agnostic Multi-Model Hit Rate & Precision Benchmark ---\n")

    print("[Eval] Loading test subjects from Test DB...")
    test_profiles = get_strict_test_profiles(TEST_DB, HIDE_RATIO)
    valid_evaluations = len(test_profiles)

    if valid_evaluations == 0:
        print("\nEvaluation failed: No valid users fit the criteria.")
        return

    print(f"[Eval] Generated {valid_evaluations} valid test profiles.")

    print("[Eval] Initializing models (Black-box mode)...")
    models = test_models(TEST_DB)
    print("\n" + "=" * 65)
    print(f"HIT RATE & PRECISION LEADERBOARD (Top-{TOP_N_RECS})")
    print(f"Total Valid Users Evaluated: {valid_evaluations}")
    print("=" * 65)
    print(f"{'Model Name':<18} | {'Hit Rate (@10)':<15} | {'Precision (@10)':<15} | {'Avg Latency'}")
    print("-" * 65)

    results = []

    for name, model in models.items():
        hit_rate, precision, avg_latency = evaluate_model_strict(model, test_profiles, TOP_N_RECS)
        results.append((name, hit_rate, precision, avg_latency))

    results.sort(key=lambda x: x[1], reverse=True)

    for name, hit_rate, precision, avg_latency in results:
        print(f" {name:<17} | {hit_rate:>10.2%}      | {precision:>10.2%}      | {avg_latency:>6.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    run_strict_evaluation()
