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
    """
    生成用于 Hit Rate 和 Precision 测试的用户画像。
    只隐藏用户打了高分（>= 4.0）的电影作为测试集，因为推荐系统只在乎能否猜中用户喜欢的电影。
    """
    random.seed(seed)
    df_test, test_users = load_test_datas(test_db_path)
    profiles = []

    for user in test_users:
        user_data = df_test[df_test['user_username'] == user]

        # 只把用户喜欢的电影拿出来做隐藏测试
        liked_movies = user_data[user_data['rating'] >= 4.0]['movie_slug'].tolist()

        if len(liked_movies) < 5:
            continue

        hidden_count = max(1, int(len(liked_movies) * hide_ratio))
        test_set = random.sample(liked_movies, hidden_count)

        # 训练集：除了被隐藏的喜欢的电影，其他的都作为已知信息喂给模型
        train_data = user_data[~user_data['movie_slug'].isin(test_set)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

        profiles.append({
            'user': user,
            'train_profile': train_profile,
            'test_set': test_set  # 这里的 test_set 就是模型需要去努力命中的 target
        })

    return profiles


def evaluate_model_strict(model, test_profiles, top_n=TOP_N_RECS):
    """
    统一的模型评测接口，计算 Hit Rate, Precision 和 Latency
    """
    hits = 0
    precision_sum = 0.0
    total_time = 0.0
    valid_evaluations = len(test_profiles)

    if valid_evaluations == 0:
        return 0.0, 0.0, 0.0

    for profile in test_profiles:
        start_time = time.time()

        # 统一的模型推理调用
        recommendations = model.get_recommendations(profile['train_profile'], top_n=top_n)
        rec_slugs = [rec['slug'] for rec in recommendations]

        total_time += (time.time() - start_time)

        # 计算命中数
        user_hits = len([slug for slug in profile['test_set'] if slug in rec_slugs])

        if user_hits > 0:
            hits += 1
        else:
            print(f"MISS | profile size: {len(profile['train_profile'])} "
                  f"| test_set size: {len(profile['test_set'])} "
                  f"| test movies: {profile['test_set']}")
        precision_sum += (user_hits / top_n)

    # 聚合指标
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

    # 核心评测循环，极其干净
    for name, model in models.items():
        hit_rate, precision, avg_latency = evaluate_model_strict(model, test_profiles, TOP_N_RECS)
        results.append((name, hit_rate, precision, avg_latency))

    # 按 Hit Rate 降序排列榜单
    results.sort(key=lambda x: x[1], reverse=True)

    for name, hit_rate, precision, avg_latency in results:
        print(f" {name:<17} | {hit_rate:>10.2%}      | {precision:>10.2%}      | {avg_latency:>6.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    run_strict_evaluation()
