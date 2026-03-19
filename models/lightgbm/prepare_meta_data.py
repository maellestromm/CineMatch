import os
import random
import shutil
import sqlite3

import pandas as pd
from sklearn.model_selection import KFold

from models.auto_rec.oof_autorec_wrapper import get_oof_autorec
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender
from util import root_path

# --- 配置 ---
TRAIN_DB = root_path() / "data/train_model.db"
META_DB = root_path() / "data/meta_dataset.db"
TEMP_DB = root_path() / "data/temp_fold_train.db"  # 动态生成的临时数据库
HIDE_RATIO = 0.2
N_SPLITS = 5


def build_oof_meta_dataset():
    print(f"[OOF Data Prep] 启动 {N_SPLITS}-Fold Out-of-Fold 交叉预测引擎...")

    # 1. 提取全局上下文特征 (必须基于全量的 TRAIN_DB)
    print("\n[OOF Data Prep] 提取全局电影统计特征...")
    conn_train = sqlite3.connect(TRAIN_DB)
    df_train = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'",
                                 conn_train)
    df_train['rating'] = pd.to_numeric(df_train['rating'], errors='coerce').dropna()

    movie_stats = df_train.groupby('movie_slug').agg(
        movie_avg=('rating', 'mean'),
        movie_count=('rating', 'count'),
        movie_std=('rating', 'std')
    ).fillna(0.0).to_dict('index')

    df_movies = pd.read_sql_query("SELECT slug, year FROM movies", conn_train)
    df_movies['year'] = pd.to_numeric(df_movies['year'], errors='coerce').fillna(2000)
    movie_years = df_movies.set_index('slug')['year'].to_dict()
    conn_train.close()

    # 2. 获取所有训练集用户，并进行 K-Fold 划分
    all_users = df_train['user_username'].unique()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    X_features = []

    # 3. 开启 5 折交叉循环
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_users), 1):
        print(f"\n{'=' * 60}")
        print(f"🚀 正在处理 FOLD {fold} / {N_SPLITS} (训练集: {len(train_idx)}人, 验证集: {len(val_idx)}人)")
        print(f"{'=' * 60}")

        val_users = set(all_users[val_idx])

        # [核心物理隔离]：复制出一个临时的数据库，并彻底删掉验证集用户的打分
        if os.path.exists(TEMP_DB):
            os.remove(TEMP_DB)
        shutil.copyfile(TRAIN_DB, TEMP_DB)

        conn_temp = sqlite3.connect(TEMP_DB)
        # 使用临时表安全删除大批量用户，防止 SQLite SQL 语句过长
        conn_temp.execute("CREATE TEMPORARY TABLE temp_val_users (username TEXT)")
        conn_temp.executemany("INSERT INTO temp_val_users VALUES (?)", [(u,) for u in val_users])
        conn_temp.execute("DELETE FROM reviews WHERE user_username IN (SELECT username FROM temp_val_users)")
        conn_temp.commit()
        conn_temp.close()

        # 4. 初始化并在临时数据库上强制重训 5 大基座模型
        # 注意：这里的 AutoRec 会被触发重训，它的权重(如 autorec.pth)会被反复覆盖，这是正常且符合预期的。
        print(f"[Fold {fold}] 正在加载并重训 5 大基座模型 (基于 4/5 临时数据)...")
        models = {
            "AutoRec": get_oof_autorec(db_path=TEMP_DB),
            "UserKNN": UserBasedRecommender(db_path=TEMP_DB),
            "ItemKNN": ItemBasedRecommender(db_path=TEMP_DB),
            "SVD": SVDRecommender(db_path=TEMP_DB),
            "ContentKNN": ContentBasedRecommender(db_path=TEMP_DB)
        }

        # 5. 对被扣留的 1/5 验证集用户进行零样本推理 (Zero-Shot Inference)
        print(f"[Fold {fold}] 正在对被隔离的验证集用户进行纯净特征预测...")
        fold_processed = 0

        for user in val_users:
            user_data = df_train[df_train['user_username'] == user]
            if len(user_data) < 5:
                continue

            all_movies = user_data['movie_slug'].tolist()
            hidden_count = max(1, int(len(all_movies) * HIDE_RATIO))
            test_set_slugs = random.sample(all_movies, hidden_count)

            # 抽出 80% 作为输入特征
            train_data = user_data[~user_data['movie_slug'].isin(test_set_slugs)]
            train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))

            user_rating_count = len(train_profile)
            user_avg = train_data['rating'].mean() if not train_data.empty else 3.0
            user_std = train_data['rating'].std() if user_rating_count > 1 else 0.0
            if pd.isna(user_std):
                user_std = 0.0

            # 让基座模型盲猜
            model_predictions = {}
            for name, model in models.items():
                raw_recs = model.get_recommendations(train_profile, top_n=3334)
                model_predictions[name] = {rec['slug']: rec['score'] for rec in raw_recs}

            # 组装 12 维特征
            for hidden_slug in test_set_slugs:
                actual_rating = float(user_data[user_data['movie_slug'] == hidden_slug]['rating'].values[0])
                m_stats = movie_stats.get(hidden_slug, {'movie_avg': 3.0, 'movie_count': 0, 'movie_std': 0.0})
                m_year = movie_years.get(hidden_slug, 2000)

                row = {
                    "User_Rating_Count": user_rating_count,
                    "User_Avg": user_avg,
                    "User_Std": user_std,
                    "Movie_Rating_Count": m_stats['movie_count'],
                    "Movie_Avg": m_stats['movie_avg'],
                    "Movie_Std": m_stats['movie_std'],
                    "Release_Year": m_year,
                    "AutoRec_Score": model_predictions["AutoRec"].get(hidden_slug, user_avg),
                    "UserKNN_Score": model_predictions["UserKNN"].get(hidden_slug, user_avg),
                    "ItemKNN_Score": model_predictions["ItemKNN"].get(hidden_slug, user_avg),
                    "SVD_Score": model_predictions["SVD"].get(hidden_slug, user_avg),
                    "ContentKNN_Score": model_predictions["ContentKNN"].get(hidden_slug, user_avg),
                    "Actual_Rating": actual_rating
                }
                X_features.append(row)

            fold_processed += 1
            if fold_processed % 200 == 0:
                print(f"  -> Fold {fold}: 已处理 {fold_processed} 名用户...")

        # 释放当前 Fold 的模型内存
        del models

    # 6. 保存最终的全量 Meta 数据集
    df_meta = pd.DataFrame(X_features)
    print(f"\n[OOF Data Prep] 历劫完成！共生成 {len(df_meta)} 条极其纯净的训练数据，准备保存至 {META_DB}...")

    conn_meta = sqlite3.connect(META_DB)
    df_meta.to_sql("meta_train", conn_meta, if_exists="replace", index=False)
    conn_meta.close()

    # 清理临时战场
    if os.path.exists(TEMP_DB):
        os.remove(TEMP_DB)

    print("[OOF Data Prep] 工业级 5-Fold 特征抽取完毕！你可以开始训练 LightGBM 了。")


if __name__ == "__main__":
    build_oof_meta_dataset()