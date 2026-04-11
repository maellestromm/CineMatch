import sqlite3
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from util import root_path


def _create_soup(row):
    """Create a 'Feature Soup' by mixing director, genres, cast, and description."""
    director = str(row['director']) if pd.notnull(row['director']) else ''
    genres = str(row['genres']) if pd.notnull(row['genres']) else ''
    cast = str(row['cast']) if pd.notnull(row['cast']) else ''
    description = str(row['description']) if pd.notnull(row['description']) else ''

    director = director.replace(',', ' ')
    genres = genres.replace(',', ' ')
    cast = cast.replace(',', ' ')

    return f"{director} {director} {genres} {genres} {cast} {description}"


class ContentBasedRecommender:
    def __init__(self, db_path, k_neighbors=871):
        self.k_neighbors = k_neighbors
        self.db_path = db_path

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.df = None
        self.indices = None
        self.movie_to_idx = {}

        self.sim_matrix_tensor = None

        self.fast_slugs = []
        self.fast_titles = []
        self.fast_years = []
        self.fast_directors = []
        self.fast_posters = []

        self._load_data()
        self._build_model()

    def _load_data(self):
        print(f"[Content-KNN] Initializing Engine on: {self.device}")
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM movies WHERE is_enriched = 1"
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"[Content-KNN] Loaded {len(self.df)} movies.")

    def _build_model(self):
        if self.df.empty:
            return

        print("[Content-KNN] Building TF-IDF matrix (CPU)...")
        self.df['soup'] = self.df.apply(_create_soup, axis=1)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['soup'])

        print("[Content-KNN] Calculating Cosine Similarities...")
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        np.fill_diagonal(cosine_sim_matrix, 0)

        for i in range(cosine_sim_matrix.shape[0]):
            sorted_indices = np.argsort(cosine_sim_matrix[i])
            cutoff_indices = sorted_indices[:-self.k_neighbors]
            cosine_sim_matrix[i][cutoff_indices] = 0.0

        print(f"[Content-KNN] Transferring Similarity Matrix to VRAM ({self.device})...")
        self.sim_matrix_tensor = torch.tensor(cosine_sim_matrix, dtype=torch.float32, device=self.device)

        self.indices = pd.Series(self.df.index, index=self.df['slug']).drop_duplicates()
        self.movie_to_idx = {slug: idx for idx, slug in enumerate(self.df['slug'])}

        self.fast_slugs = self.df['slug'].tolist()
        self.fast_titles = self.df['title'].tolist()
        self.fast_years = self.df['year'].tolist()
        self.fast_directors = self.df['director'].tolist()
        self.fast_posters = self.df.get('poster_url', pd.Series([''] * len(self.df))).tolist()

        print("[Content-KNN] Model building complete!\n")

    def get_recommendations(self, user_profile, top_n=10):
        if self.df.empty or not user_profile:
            return []

        num_movies = len(self.df)
        target_array = np.zeros(num_movies, dtype=np.float32)
        watched_indices = []

        for slug, rating in user_profile.items():
            if slug in self.movie_to_idx:
                idx = self.movie_to_idx[slug]
                target_array[idx] = rating
                watched_indices.append(idx)

        if not watched_indices:
            return []

        target_tensor = torch.tensor(target_array, device=self.device)
        has_rated_mask = (target_tensor > 0).float()

        recommendation_scores = torch.matmul(self.sim_matrix_tensor, target_tensor)
        similarity_sums = torch.matmul(self.sim_matrix_tensor, has_rated_mask)

        user_ratings_count = has_rated_mask.sum()
        if user_ratings_count > 0:
            prior_mean = target_tensor.sum() / user_ratings_count
        else:
            prior_mean = torch.tensor(3.0, dtype=torch.float32, device=self.device)

        damping = 0.1

        final_scores = (recommendation_scores + damping * prior_mean) / (similarity_sums + damping)

        final_scores[similarity_sums < 0.01] = 0.0

        final_scores[watched_indices] = -1.0

        actual_top_n = min(top_n, num_movies)
        top_n_scores, top_n_movie_indices = torch.topk(final_scores, actual_top_n)

        top_n_scores = top_n_scores.cpu().numpy()
        top_n_movie_indices = top_n_movie_indices.cpu().numpy()

        results = []
        for score, idx in zip(top_n_scores, top_n_movie_indices):

            if score <= 0:
                continue

            results.append({
                'slug': self.fast_slugs[idx],
                'title': self.fast_titles[idx],
                'year': self.fast_years[idx],
                'director': self.fast_directors[idx],
                'poster_url': self.fast_posters[idx],
                'score': float(score)
            })

        return results


if __name__ == "__main__":
    recommender = ContentBasedRecommender(root_path() / "data/user_first_cut3_clear.db", k_neighbors=10)

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 1.0
    }

    import time

    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=100)
    latency = (time.time() - start_time) * 1000

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']})")
    print(f"Latency: {latency:.2f} ms")
