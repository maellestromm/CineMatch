import sqlite3
import json
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from util import root_path


def export_svd_k39(db_path, output_json=root_path() / "webui/svd_k39_vt.json", k=39):
    print(f"[Export] Loading SVD Data...")
    conn = sqlite3.connect(db_path)

    # 1. 加载有效的用户打分记录
    df_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce')
    df_reviews = df_reviews.dropna()

    # 2. 获取所有的电影 slug，保证和前端字典 JSON 的顺序绝对一致！
    df_movies = pd.read_sql_query("SELECT slug FROM movies", conn)
    df_movies.drop_duplicates(subset=['slug'], keep='first', inplace=True)
    all_slugs = df_movies['slug'].tolist()

    conn.close()

    print("[Export] Building Pivot Matrix...")
    pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating')

    # 🚨 关键防御：补齐没有被任何人打分的冷门电影，并严格按照字典顺序排列
    missing_cols = set(all_slugs) - set(pivot_df.columns)
    for col in missing_cols:
        pivot_df[col] = np.nan
    pivot_df = pivot_df[all_slugs]

    print("[Export] Centering matrix and computing SVD...")
    # 计算用户均值并减去均值，缺失值填 0
    user_means = pivot_df.mean(axis=1)
    matrix_centered = pivot_df.sub(user_means, axis=0).fillna(0).values

    # 进行奇异值分解
    actual_k = min(k, min(matrix_centered.shape) - 1)
    U, sigma, Vt = svds(matrix_centered, k=actual_k)

    print(f"[Export] SVD completed. Vt shape: {Vt.shape}")

    # 将 numpy 数组转为普通 Python 列表，为了减小 JSON 体积，保留 5 位小数
    vt_list = np.round(Vt, decimals=5).tolist()

    print(f"[Export] Saving Vt matrix to JSON...")
    with open(output_json, 'w') as f:
        # 使用 separators 移除多余的空格以极限压缩文件体积
        json.dump(vt_list, f, separators=(',', ':'))

    print(f"[Export] 导出完毕！SVD k=39 模型权重已保存至: {output_json}")


if __name__ == "__main__":
    export_svd_k39(root_path() / "data/train_model.db", k=39)