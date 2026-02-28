import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        self.user_movie_matrix = None
        self.df_movies = None

        # Initialize the model upon creation
        self._load_data()

    def _load_data(self):
        """Load reviews and build the basic dense matrix."""
        print("[User-KNN] Loading data from database...")
        conn = sqlite3.connect(self.db_path)

        # Read reviews and convert ratings to numeric
        query_reviews = "SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'"
        df_reviews = pd.read_sql_query(query_reviews, conn)
        df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce')
        df_reviews = df_reviews.dropna(subset=['rating'])

        # Read movie metadata for formatting output
        query_movies = "SELECT slug, title, year, director, poster_url FROM movies"
        self.df_movies = pd.read_sql_query(query_movies, conn).set_index('slug')
        conn.close()

        print("[User-KNN] Building User-Movie Matrix...")
        # Build dense matrix: Rows = Users, Columns = Movies
        self.user_movie_matrix = df_reviews.pivot_table(
            index='user_username',
            columns='movie_slug',
            values='rating',
            aggfunc='mean'
        ).fillna(0)

        print(
            f"[User-KNN] Matrix built! {self.user_movie_matrix.shape[0]} users, {self.user_movie_matrix.shape[1]} movies.\n")

    def get_recommendations(self, user_profile, top_n=10, k_neighbors=15):
        """
        Get recommendations based on a user profile using basic User KNN.
        :param user_profile: dict, e.g., {'movie-slug': 5.0, 'another-slug': 4.0}
        :param top_n: int, number of recommendations to return
        :param k_neighbors: int, number of similar users to consider
        :return: list of dicts representing recommended movies
        """
        if self.user_movie_matrix is None or self.user_movie_matrix.empty:
            return []

        # 1. Align target user vector with our matrix columns
        target_vector = pd.Series(0.0, index=self.user_movie_matrix.columns)

        for slug, rating in user_profile.items():
            if slug in target_vector.index:
                target_vector[slug] = rating
            else:
                print(f"[User-KNN] Warning: '{slug}' not found in matrix.")

        if target_vector.sum() == 0:
            print("[User-KNN] Warning: None of the input movies are in the database.")
            return []

        # 2. Calculate Cosine Similarity with all users
        similarities = cosine_similarity([target_vector.values], self.user_movie_matrix.values)[0]
        sim_series = pd.Series(similarities, index=self.user_movie_matrix.index)

        # 3. Find top K neighbors
        top_k_neighbors = sim_series.nlargest(k_neighbors)

        # # 4. Collect and weight movies from neighbors
        # recommendation_scores = {}
        #
        # for neighbor_name, sim_score in top_k_neighbors.items():
        #     if sim_score <= 0:
        #         continue
        #
        #     neighbor_ratings = self.user_movie_matrix.loc[neighbor_name]
        #
        #     for slug, rating in neighbor_ratings.items():
        #         # Filter: neighbor watched it (>0) AND target user hasn't seen it (==0)
        #         if rating > 0 and target_vector[slug] == 0:
        #             if slug not in recommendation_scores:
        #                 recommendation_scores[slug] = 0.0
        #
        #             # Weight score by neighbor's rating and similarity
        #             recommendation_scores[slug] += rating * sim_score


        # 4. Collect and weight movies from neighbors
        recommendation_scores = {}
        similarity_sums = {}  # 🚀 新增：用来记录分母！

        for neighbor_name, sim_score in top_k_neighbors.items():
            if sim_score <= 0:
                continue

            neighbor_ratings = self.user_movie_matrix.loc[neighbor_name]

            for slug, rating in neighbor_ratings.items():
                if rating > 0 and target_vector[slug] == 0:
                    if slug not in recommendation_scores:
                        recommendation_scores[slug] = 0.0
                        similarity_sums[slug] = 0.0  # 🚀 初始化分母

                    # 分子：累加 (打分 * 相似度)
                    recommendation_scores[slug] += rating * sim_score
                    # 分母：累加 (相似度)
                    similarity_sums[slug] += sim_score

        # 🚀 新增：计算真正的加权平均分 (1~5星)
        final_scores = {}
        for slug in recommendation_scores:
            final_scores[slug] = recommendation_scores[slug] / similarity_sums[slug]

        # 5. Sort and format results (用真实的平均分来排序！)
        # sorted_recs = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
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


# --- Test Execution ---
if __name__ == "__main__":
    recommender = UserBasedRecommender("../data/user_first_cut3_clear.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    print("--- User-Based Recommendations ---")
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)

    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']}) - Dir: {rec['director']}")
