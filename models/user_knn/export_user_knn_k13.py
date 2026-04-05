import sqlite3
import json
import numpy as np
import pandas as pd
from util import root_path


def export_user_knn_profiles(db_path, output_json=root_path() / "webui/user_knn_k13.json"):
    print("[Export] Loading User-KNN Data...")
    conn = sqlite3.connect(db_path)

    # 加载真实打分
    df_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce')
    df_reviews = df_reviews.dropna()

    # 获取全库所有 slug，对齐前端字典
    df_movies = pd.read_sql_query("SELECT slug FROM movies", conn)
    df_movies.drop_duplicates(subset=['slug'], keep='first', inplace=True)
    all_slugs = df_movies['slug'].tolist()

    conn.close()

    print("[Export] Building Sparse User Profiles...")
    pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating').fillna(0)

    # 补齐未看过的电影
    missing_cols = set(all_slugs) - set(pivot_df.columns)
    for col in missing_cols:
        pivot_df[col] = 0.0
    pivot_df = pivot_df[all_slugs]

    matrix = pivot_df.values
    num_users = matrix.shape[0]

    users_data = []

    for i in range(num_users):
        user_ratings = matrix[i]
        # 预先计算该用户向量的 L2 范数，方便前端直接用于余弦相似度计算！
        norm = float(np.linalg.norm(user_ratings))

        # 提取该用户的非零打分记录
        non_zeros = {}
        for j in range(len(user_ratings)):
            if user_ratings[j] > 0:
                # 保留两位小数，压缩 JSON
                non_zeros[j] = round(float(user_ratings[j]), 2)

        # 只有真正打过分的用户才会被导出
        if norm > 0:
            users_data.append({
                "norm": round(norm, 5),
                "ratings": non_zeros
            })

    print(f"[Export] Extracted {len(users_data)} valid users.")
    with open(output_json, 'w') as f:
        # 使用 separators 极限压缩 JSON 体积
        json.dump(users_data, f, separators=(',', ':'))

    print(f"[Export] 导出完毕！全量稀疏用户画像已保存至: {output_json}")


if __name__ == "__main__":
    export_user_knn_profiles(root_path() / "data/train_model.db")