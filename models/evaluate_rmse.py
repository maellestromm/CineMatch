import math
import random
import numpy as np
import pandas as pd
import sqlite3

from models.auto_rec import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.user_knn import UserBasedRecommender
from models.svd import SVDRecommender

from util import root_path

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
HIDE_RATIO = 0.2


def get_test_profiles(test_db_path=TEST_DB, hide_ratio=HIDE_RATIO, seed=42):
    """
    [重构抽取]：为所有测试集用户生成一致的 80/20 验证特征。
    固定 seed 确保调参时不同模型面对的是完全同一套“考卷”。
    """
    conn_test = sqlite3.connect(test_db_path)
    df_test = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'",
                                conn_test)
    df_test['rating'] = pd.to_numeric(df_test['rating'], errors='coerce').dropna()
    conn_test.close()

    random.seed(seed)
    test_users = df_test['user_username'].unique()
    profiles = []

    for user in test_users:
        user_data = df_test[df_test['user_username'] == user]
        if len(user_data) < 5:
            continue

        all_movies = user_data['movie_slug'].tolist()
        hidden_count = max(1, int(len(all_movies) * hide_ratio))
        test_set_slugs = random.sample(all_movies, hidden_count)

        train_data = user_data[~user_data['movie_slug'].isin(test_set_slugs)]
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))
        user_avg = train_data['rating'].mean() if not train_data.empty else 3.0

        targets = []
        for hidden_slug in test_set_slugs:
            actual_rating = float(user_data[user_data['movie_slug'] == hidden_slug]['rating'].values[0])
            targets.append({'slug': hidden_slug, 'actual': actual_rating})

        profiles.append({
            'user': user,
            'train_profile': train_profile,
            'user_avg': user_avg,
            'targets': targets
        })

    return profiles


def evaluate_model(model, test_profiles):
    """
    [重构抽取]：接收任意一个实现了 get_recommendations() 的推荐系统对象实例，
    进行黑盒预测并返回 RMSE。
    """
    errors = []
    for profile in test_profiles:
        # 绝对防御：强制调用原版模型的真实推理逻辑，杜绝实现重复
        raw_recs = model.get_recommendations(profile['train_profile'], top_n=3334)
        pred_dict = {rec['slug']: rec['score'] for rec in raw_recs}

        for target in profile['targets']:
            pred_rating = pred_dict.get(target['slug'], profile['user_avg'])
            errors.append((pred_rating - target['actual']) ** 2)

    if not errors:
        return float('inf')
    return math.sqrt(np.mean(errors))


def run_rmse_evaluation():
    """原本的主入口：用于一次性评估全部基座模型"""
    print("--- Agnostic Multi-Model RMSE Benchmark ---\n")

    print("[Eval] 正在生成一致性测试特征 (80/20 split)...")
    test_profiles = get_test_profiles(TEST_DB, HIDE_RATIO)

    print("[Eval] 初始化基座模型 (Black-box mode)...")
    models = {
        "User-KNN": UserBasedRecommender(db_path=TRAIN_DB),
        "Content-KNN": ContentBasedRecommender(db_path=TRAIN_DB),
        "Deep AutoRec": AutoRecRecommender(db_path=TRAIN_DB),
        "Item-KNN": ItemBasedRecommender(db_path=TRAIN_DB),
        "SVD-50": SVDRecommender(db_path=TRAIN_DB),
    }

    print("\n" + "=" * 55)
    print("RMSE ACCURACY LEADERBOARD")
    print("=" * 55)

    for name, model in models.items():
        rmse = evaluate_model(model, test_profiles)
        print(f"{name:<18} | {rmse:.4f}")


if __name__ == "__main__":
    run_rmse_evaluation()
