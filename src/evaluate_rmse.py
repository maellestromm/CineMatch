import math
import random
import sqlite3

import numpy as np
import pandas as pd

from content_knn import ContentBasedRecommender
from infer_autorec import AutoRecRecommender
from user_knn_gpu import UserBasedRecommender

# --- 配置 ---
TRAIN_DB = "train_model.db"
TEST_DB = "test_eval.db"
HIDE_RATIO = 0.2


def run_rmse_evaluation():
    print("🥊 --- Agnostic Multi-Model RMSE Benchmark (1-5 Star Accuracy) --- 🥊\n")

    # ==========================================
    # 1. 模型注册表 (黑盒模式)
    # 测试脚本不关心底层实现，只认 get_recommendations 接口！
    # ==========================================
    print("[Eval] Initializing models (Black-box mode)...")
    models = {
        "User-KNN": UserBasedRecommender(db_path=TRAIN_DB),
        "Content-KNN": ContentBasedRecommender(db_path=TRAIN_DB),
        "Deep AutoRec": AutoRecRecommender(db_path=TRAIN_DB)
        # 未来加新模型直接写在这里
    }

    metrics = {name: [] for name in models}

    print("\n[Eval] Loading test subjects from Test DB...")
    conn = sqlite3.connect(TEST_DB)
    df_test = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_test['rating'] = pd.to_numeric(df_test['rating'], errors='coerce').dropna()
    conn.close()

    test_users = df_test['user_username'].unique()
    valid_users = 0

    for i, user in enumerate(test_users, 1):
        user_data = df_test[df_test['user_username'] == user]

        if len(user_data) < 5:
            continue

        all_movies = user_data['movie_slug'].tolist()
        hidden_count = max(1, int(len(all_movies) * HIDE_RATIO))

        test_set_slugs = random.sample(all_movies, hidden_count)
        train_data = user_data[~user_data['movie_slug'].isin(test_set_slugs)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))
        user_avg = train_data['rating'].mean() if not train_data.empty else 3.0

        # ==========================================
        # 2. 核心评估循环：纯接口调用，无任何硬编码
        # ==========================================
        for model_name, model in models.items():
            # 请求极大的 top_n，相当于让模型交出它对所有剩余电影的“预测卷子”
            raw_recs = model.get_recommendations(train_profile, top_n=3334)

            # 将卷子转为 O(1) 查询的字典: {slug: score}
            pred_dict = {rec['slug']: rec['score'] for rec in raw_recs}

            # 开始对隐藏电影进行批改
            for hidden_slug in test_set_slugs:
                actual_rating = float(user_data[user_data['movie_slug'] == hidden_slug]['rating'].values[0])

                # 如果模型预测了这部电影，拿分数；如果模型根本找不到（冷启动），用该用户的历史均分兜底
                pred_rating = pred_dict.get(hidden_slug, user_avg)

                # 记录误差平方
                metrics[model_name].append((pred_rating - actual_rating) ** 2)

        valid_users += 1
        if valid_users % 50 == 0:
            print(f"[{valid_users}] Users evaluated... (Tracking RMSE for {len(models)} models)")

    # ==========================================
    # 3. 打印最终学术成绩单
    # ==========================================
    print("\n" + "=" * 55)
    print("🏆 RMSE ACCURACY LEADERBOARD (Lower is Better) 🏆")
    print(f"Hidden ratings evaluated per model: {len(metrics[list(models.keys())[0]])}")
    print("=" * 55)
    print(f"{'Model Name':<18} | {'RMSE Score':<15} | {'Status'}")
    print("-" * 55)

    final_rmses = {name: math.sqrt(np.mean(errors)) for name, errors in metrics.items() if len(errors) > 0}
    sorted_rmses = sorted(final_rmses.items(), key=lambda x: x[1])

    for rank, (name, rmse) in enumerate(sorted_rmses):
        status = "👑 Champion" if rank == 0 else "💪 Runner-up" if rank == 1 else ""
        print(f" {name:<17} | {rmse:<15.4f} | {status}")

    print("=" * 55)


if __name__ == "__main__":
    run_rmse_evaluation()