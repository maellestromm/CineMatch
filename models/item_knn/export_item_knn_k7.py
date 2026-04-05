import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from util import root_path


def export_item_knn_k7(db_path, output_json=root_path() / "webui/item_knn_k7.json", k=7):
    print("[Export] Loading Item-KNN Data...")
    conn = sqlite3.connect(db_path)

    # 1. 加载有效的用户打分记录
    df_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce')
    df_reviews = df_reviews.dropna()

    # 2. 获取所有的电影 slug，保证和字典 JSON 的顺序绝对一致！
    df_movies = pd.read_sql_query("SELECT slug FROM movies", conn)
    df_movies.drop_duplicates(subset=['slug'], keep='first', inplace=True)
    all_slugs = df_movies['slug'].tolist()

    conn.close()

    print("[Export] Building Item-User Pivot Matrix...")
    # 构建透视表 (Users x Items)
    pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating',
                                      aggfunc='mean').fillna(0)

    # 🚨 关键防御：有些极度冷门的电影可能没有任何用户打分，它不会出现在 pivot_df 的列里。
    # 我们必须把缺失的列补上 0，并严格按照 all_slugs 的顺序重排，否则索引会错乱！
    missing_cols = set(all_slugs) - set(pivot_df.columns)
    for col in missing_cols:
        pivot_df[col] = 0.0
    pivot_df = pivot_df[all_slugs]

    # 转置成 (Items x Users) 矩阵，用于计算物品间相似度
    item_user_matrix = pivot_df.values.T

    print("[Export] Calculating Cosine Similarity...")
    sim_matrix = cosine_similarity(item_user_matrix)
    np.fill_diagonal(sim_matrix, 0)  # 自己和自己的相似度清零

    print(f"[Export] Extracting Top-{k} neighbors...")
    sparse_sim = {}

    for i in range(sim_matrix.shape[0]):
        # 升序排列，取最后 k 个
        sorted_indices = np.argsort(sim_matrix[i])
        top_k_indices = sorted_indices[-k:]

        neighbors = []
        for j in top_k_indices:
            val = float(sim_matrix[i][j])
            if val > 0:
                # 每个邻居存为一个两元组 [邻居索引, 相似度]
                neighbors.append([int(j), round(val, 5)])

        if neighbors:
            sparse_sim[int(i)] = neighbors

    with open(output_json, 'w') as f:
        json.dump(sparse_sim, f)

    print(f"[Export] 导出完毕！K=7 极速版 JSON 已保存至: {output_json}")


if __name__ == "__main__":
    export_item_knn_k7(root_path() / "data/train_model.db", k=7)
