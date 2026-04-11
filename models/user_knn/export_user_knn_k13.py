import json

import numpy as np

from util import root_path, load_export_datas


def export_user_knn_profiles(db_path, output_json=root_path() / "webui/user_knn_k13.json"):
    print("[Export] Loading User-KNN Data...")
    df_reviews,all_slugs = load_export_datas(db_path)

    print("[Export] Building Sparse User Profiles...")
    pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating').fillna(0)

    missing_cols = set(all_slugs) - set(pivot_df.columns)
    for col in missing_cols:
        pivot_df[col] = 0.0
    pivot_df = pivot_df[all_slugs]

    matrix = pivot_df.values
    num_users = matrix.shape[0]

    users_data = []

    for i in range(num_users):
        user_ratings = matrix[i]
        norm = float(np.linalg.norm(user_ratings))

        non_zeros = {}
        for j in range(len(user_ratings)):
            if user_ratings[j] > 0:
                non_zeros[j] = round(float(user_ratings[j]), 2)

        if norm > 0:
            users_data.append({
                "norm": round(norm, 5),
                "ratings": non_zeros
            })

    print(f"[Export] Extracted {len(users_data)} valid users.")
    with open(output_json, 'w') as f:
        json.dump(users_data, f, separators=(',', ':'))

    print(f"[Export] The json has been saved to: {output_json}")


if __name__ == "__main__":
    export_user_knn_profiles(root_path() / "data/train_model.db")