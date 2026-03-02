import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from util import load_review_datas


class UserKNNCPUBackend:
    def __init__(self, db_path):
        self.db_path = db_path
        self.user_movie_matrix = None
        self.df_movies = None

        self._load_data()

    def _load_data(self):
        print("[User-KNN-CPU] Loading data from database...")
        df_reviews, self.query_movies = load_review_datas(self.db_path)

        print("[User-KNN-CPU] Building User-Movie Matrix...")
        self.user_movie_matrix = df_reviews.pivot_table(
            index='user_username',
            columns='movie_slug',
            values='rating',
            aggfunc='mean'
        ).fillna(0)

        print(
            f"[User-KNN-CPU] Matrix built! {self.user_movie_matrix.shape[0]} users, {self.user_movie_matrix.shape[1]} movies.\n")

    def get_recommendations(self, user_profile, top_n=10, k_neighbors=15):
        if self.user_movie_matrix is None or self.user_movie_matrix.empty:
            return []

        target_vector = pd.Series(0.0, index=self.user_movie_matrix.columns)

        for slug, rating in user_profile.items():
            if slug in target_vector.index:
                target_vector[slug] = rating
            else:
                print(f"[User-KNN-CPU] Warning: '{slug}' not found in matrix.")

        if target_vector.sum() == 0:
            print("[User-KNN-CPU] Warning: None of the input movies are in the database.")
            return []

        similarities = cosine_similarity([target_vector.values], self.user_movie_matrix.values)[0]
        sim_series = pd.Series(similarities, index=self.user_movie_matrix.index)

        top_k_neighbors = sim_series.nlargest(k_neighbors)

        recommendation_scores = {}
        similarity_sums = {}

        for neighbor_name, sim_score in top_k_neighbors.items():
            if sim_score <= 0:
                continue

            neighbor_ratings = self.user_movie_matrix.loc[neighbor_name]

            for slug, rating in neighbor_ratings.items():
                if rating > 0 and target_vector[slug] == 0:
                    if slug not in recommendation_scores:
                        recommendation_scores[slug] = 0.0
                        similarity_sums[slug] = 0.0

                    recommendation_scores[slug] += rating * sim_score
                    similarity_sums[slug] += sim_score

        # Bayesian Smoothing implementation
        user_ratings = list(user_profile.values())
        prior_mean = sum(user_ratings) / len(user_ratings) if user_ratings else 3.0
        damping = 3.0

        final_scores = {}
        for slug in recommendation_scores:
            final_scores[slug] = (recommendation_scores[slug] + damping * prior_mean) / (
                        similarity_sums[slug] + damping)

        sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        results = []

        for slug, score in sorted_recs[:top_n]:
            if slug in self.df_movies.index:
                movie_data = self.df_movies.loc[slug]
                results.append({
                    'slug': slug,
                    'title': movie_data['title'],
                    'year': movie_data['year'],
                    'director': movie_data.get('director', ''),
                    'poster_url': movie_data.get('poster_url', ''),
                    'score': score
                })

        return results