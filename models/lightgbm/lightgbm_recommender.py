import sqlite3
import pandas as pd
import numpy as np
import joblib

from util import root_path
from models.auto_rec import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender


class LightGBMMetaRecommender:
    def __init__(self, db_path, model_path=root_path() / "data/lgbm_meta_model.pkl"):
        self.db_path = db_path
        self.df_movies = None
        self.movie_stats = {}
        self.movie_years = {}
        self.movie_slugs = []

        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            raise Exception(f"Model file {model_path} not found. Please run train_meta_learner.py first.")

        print("[LGBM Ranker] Initializing 5 Base Models for Inference...")
        self.base_models = {
            "AutoRec": AutoRecRecommender(db_path=self.db_path),
            "UserKNN": UserBasedRecommender(db_path=self.db_path),
            "ItemKNN": ItemBasedRecommender(db_path=self.db_path),
            "SVD": SVDRecommender(db_path=self.db_path),
            "ContentKNN": ContentBasedRecommender(db_path=self.db_path)
        }

        self._load_global_stats()

    def _load_global_stats(self):
        print("[LGBM Ranker] Loading global movie statistics...")
        conn = sqlite3.connect(self.db_path)

        df_reviews = pd.read_sql_query("SELECT movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
        df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce').dropna()
        self.movie_stats = df_reviews.groupby('movie_slug').agg(
            movie_avg=('rating', 'mean'),
            movie_count=('rating', 'count'),
            movie_std=('rating', 'std')
        ).fillna(0.0).to_dict('index')

        query_movies = "SELECT slug, title, year, director, poster_url FROM movies"
        self.df_movies = pd.read_sql_query(query_movies, conn).set_index('slug')
        self.df_movies['year'] = pd.to_numeric(self.df_movies['year'], errors='coerce').fillna(2000)
        self.movie_years = self.df_movies['year'].to_dict()
        self.movie_slugs = self.df_movies.index.tolist()

        conn.close()

    def get_recommendations(self, user_profile, top_n=10):
        if not user_profile:
            return []

        # user context
        user_ratings = list(user_profile.values())
        user_rating_count = len(user_ratings)
        user_avg = np.mean(user_ratings)
        user_std = np.std(user_ratings, ddof=1) if user_rating_count > 1 else 0.0
        if pd.isna(user_std):
            user_std = 0.0

        # score from other model
        model_predictions = {}
        for name, model in self.base_models.items():
            raw_recs = model.get_recommendations(user_profile, top_n=len(self.movie_slugs))
            model_predictions[name] = {rec['slug']: rec['score'] for rec in raw_recs}

        inference_features = []
        candidate_slugs = []

        for slug in self.movie_slugs:
            if slug in user_profile:
                continue

            m_stats = self.movie_stats.get(slug, {'movie_avg': 3.0, 'movie_count': 0, 'movie_std': 0.0})
            m_year = self.movie_years.get(slug, 2000)

            row = [
                user_rating_count,
                user_avg,
                user_std,
                m_stats['movie_count'],
                m_stats['movie_avg'],
                m_stats['movie_std'],
                m_year,
                model_predictions["AutoRec"].get(slug, user_avg),
                model_predictions["UserKNN"].get(slug, user_avg),
                model_predictions["ItemKNN"].get(slug, user_avg),
                model_predictions["SVD"].get(slug, user_avg),
                model_predictions["ContentKNN"].get(slug, user_avg)
            ]
            inference_features.append(row)
            candidate_slugs.append(slug)

        if not inference_features:
            return []

        X_infer = pd.DataFrame(inference_features)
        final_scores = self.model.predict(X_infer)

        top_indices = np.argsort(final_scores)[::-1][:top_n]
        results = []

        for idx in top_indices:
            slug = candidate_slugs[idx]
            movie_data = self.df_movies.loc[slug]
            results.append({
                'slug': slug,
                'title': movie_data['title'],
                'year': movie_data['year'],
                'director': movie_data.get('director', ''),
                'poster_url': movie_data.get('poster_url', ''),
                'score': float(final_scores[idx])
            })

        return results


if __name__ == "__main__":
    recommender = LightGBMMetaRecommender(root_path() / "data/train_model.db")
    demo_profile = {"inception": 5.0, "interstellar": 4.5, "the-dark-knight": 5.0}

    import time

    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)
    print(f"Latency: {(time.time() - start_time) * 1000:.2f} ms")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']})")