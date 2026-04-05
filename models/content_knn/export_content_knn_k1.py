import sqlite3
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from util import root_path


def _create_soup(row):
    director = str(row['director']) if pd.notnull(row['director']) else ''
    genres = str(row['genres']) if pd.notnull(row['genres']) else ''
    cast = str(row['cast']) if pd.notnull(row['cast']) else ''
    description = str(row['description']) if pd.notnull(row['description']) else ''

    director = director.replace(',', ' ')
    genres = genres.replace(',', ' ')
    cast = cast.replace(',', ' ')

    return f"{director} {director} {genres} {genres} {cast} {description}"


def export_content_knn_k1(db_path, output_json=root_path() / "webui/content_knn_k1.json"):
    print("[Export] Loading Data...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM movies WHERE is_enriched = 1", conn)
    conn.close()

    print("[Export] Building TF-IDF...")
    df['soup'] = df.apply(_create_soup, axis=1)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['soup'])

    print("[Export] Calculating Cosine Similarity...")
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # 把自己跟自己的相似度清零，防止最相似的邻居是自己
    np.fill_diagonal(cosine_sim_matrix, 0)

    print("[Export] Extracting Top-1 neighbor for ultimate compression...")
    sparse_sim = {}

    for i in range(cosine_sim_matrix.shape[0]):
        # 取最大值的索引 (Top-1)
        best_j = int(np.argmax(cosine_sim_matrix[i]))
        best_sim = float(cosine_sim_matrix[i][best_j])

        if best_sim > 0:
            # 🚀 极致压缩：结构变成了 "当前电影索引": [唯一邻居索引, 相似度]
            sparse_sim[int(i)] = [best_j, round(best_sim, 5)]

    with open(output_json, 'w') as f:
        json.dump(sparse_sim, f)

    print(f"[Export] 导出完毕！K=1 极速版 JSON 已保存至: {output_json}")


if __name__ == "__main__":
    export_content_knn_k1(root_path() / "data/train_model.db")
