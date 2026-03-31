import numpy as np
from scipy.sparse.linalg import svds

from util import load_review_datas, root_path


class SVDRecommender:
    def __init__(self, db_path, k_factors=39):
        self.db_path = db_path
        self.k_factors = k_factors
        self.df_movies = None
        self.movie_slugs = []
        self.vt = None

        self._load_data_and_train()

    def _load_data_and_train(self):
        print(f"[SVD] Initializing Truncated SVD Recommender (k={self.k_factors})...")
        df_reviews, self.df_movies = load_review_datas(self.db_path)

        print("[SVD] Building dense matrix and computing user means...")
        pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating')
        self.movie_slugs = pivot_df.columns.tolist()

        user_means = pivot_df.mean(axis=1)
        matrix_centered = pivot_df.sub(user_means, axis=0).fillna(0).values

        print("[SVD] Performing Singular Value Decomposition...")
        actual_k = min(self.k_factors, min(matrix_centered.shape) - 1)

        U, sigma, Vt = svds(matrix_centered, k=actual_k)

        self.vt = Vt
        print("[SVD] Model training complete!\n")

    def get_recommendations(self, user_profile, top_n=10):
        if self.vt is None:
            return []

        target_vector = np.zeros(len(self.movie_slugs))
        watched_indices = []

        for slug, rating in user_profile.items():
            if slug in self.movie_slugs:
                idx = self.movie_slugs.index(slug)
                target_vector[idx] = float(rating)
                watched_indices.append(idx)

        if np.sum(target_vector) == 0:
            return []

        user_ratings = list(user_profile.values())
        user_mean = sum(user_ratings) / len(user_ratings)

        target_centered = np.zeros(len(self.movie_slugs))
        for idx in watched_indices:
            target_centered[idx] = target_vector[idx] - user_mean

        reconstructed_centered = target_centered @ self.vt.T @ self.vt

        final_scores = reconstructed_centered + user_mean

        final_scores[watched_indices] = -999.0

        top_n_indices = np.argsort(final_scores)[::-1][:top_n]

        results = []
        for idx in top_n_indices:
            score = final_scores[idx]
            if score <= 0:
                continue

            slug = self.movie_slugs[idx]
            if slug in self.df_movies.index:
                movie_data = self.df_movies.loc[slug]
                results.append({
                    'slug': slug,
                    'title': movie_data['title'],
                    'year': movie_data['year'],
                    'director': movie_data.get('director', ''),
                    'poster_url': movie_data.get('poster_url', ''),
                    'score': float(score)
                })

        return results


if __name__ == "__main__":
    recommender = SVDRecommender(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    import time

    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)
    latency = (time.time() - start_time) * 1000

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']})")
    print(f"Latency: {latency:.2f} ms")