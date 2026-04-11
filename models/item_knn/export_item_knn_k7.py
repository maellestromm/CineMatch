import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from util import root_path, load_export_datas


def export_item_knn_k7(db_path, output_json=root_path() / "webui/item_knn_k7.json", k=7):
    print("[Export] Loading Item-KNN Data...")
    df_reviews, all_slugs = load_export_datas(db_path)

    print("[Export] Building Item-User Pivot Matrix...")

    pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating',
                                      aggfunc='mean').fillna(0)

    missing_cols = set(all_slugs) - set(pivot_df.columns)
    for col in missing_cols:
        pivot_df[col] = 0.0
    pivot_df = pivot_df[all_slugs]

    item_user_matrix = pivot_df.values.T

    print("[Export] Calculating Cosine Similarity...")
    sim_matrix = cosine_similarity(item_user_matrix)
    np.fill_diagonal(sim_matrix, 0)

    print(f"[Export] Extracting Top-{k} neighbors...")
    sparse_sim = {}

    for i in range(sim_matrix.shape[0]):

        sorted_indices = np.argsort(sim_matrix[i])
        top_k_indices = sorted_indices[-k:]

        neighbors = []
        for j in top_k_indices:
            val = float(sim_matrix[i][j])
            if val > 0:
                neighbors.append([int(j), round(val, 5)])

        if neighbors:
            sparse_sim[int(i)] = neighbors

    with open(output_json, 'w') as f:
        json.dump(sparse_sim, f)

    print(f"[Export] The json has been saved to: {output_json}")


if __name__ == "__main__":
    export_item_knn_k7(root_path() / "data/train_model.db", k=7)
