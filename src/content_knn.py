import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        self.df = None
        self.cosine_sim_matrix = None
        self.indices = None

        # Initialize the model upon creation
        self._load_data()
        self._build_model()

    def _load_data(self):
        """Load enriched movies from the SQLite database."""
        print("[Content-KNN] Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        # Only load movies that have been enriched with details
        query = "SELECT * FROM movies WHERE is_enriched = 1"
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"[Content-KNN] Loaded {len(self.df)} movies.")

    def _create_soup(self, row):
        """Create a 'Feature Soup' by mixing director, genres, cast, and description."""
        director = str(row['director']) if pd.notnull(row['director']) else ''
        genres = str(row['genres']) if pd.notnull(row['genres']) else ''
        cast = str(row['cast']) if pd.notnull(row['cast']) else ''
        description = str(row['description']) if pd.notnull(row['description']) else ''

        director = director.replace(',', ' ')
        genres = genres.replace(',', ' ')
        cast = cast.replace(',', ' ')

        # Repeat director and genres to give them higher TF-IDF weight
        return f"{director} {director} {genres} {genres} {cast} {description}"

    def _build_model(self):
        """Vectorize the text and calculate the Cosine Similarity matrix."""
        if self.df.empty:
            print("[Content-KNN] Error: No movies found.")
            return

        print("[Content-KNN] Building TF-IDF matrix and calculating similarities...")
        self.df['soup'] = self.df.apply(self._create_soup, axis=1)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['soup'])

        self.cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Map movie slugs to their index in the matrix
        self.indices = pd.Series(self.df.index, index=self.df['slug']).drop_duplicates()
        print("[Content-KNN] Model building complete!\n")

    def get_recommendations(self, user_profile, top_n=10):
        """
        Get recommendations based on a user profile.
        :param user_profile: dict, e.g., {'movie-slug': 5.0, 'another-slug': 4.0}
        :param top_n: int, number of recommendations to return
        :return: list of dicts representing recommended movies
        """
        if self.df.empty:
            return []

        # Initialize an array of zeros to hold the aggregated similarity scores
        total_scores = np.zeros(len(self.df))
        valid_slugs = 0

        # Calculate weighted similarity based on user's rated movies
        for slug, rating in user_profile.items():
            if slug in self.indices:
                idx = self.indices[slug]
                # Weight the similarity vector by the user's rating
                total_scores += self.cosine_sim_matrix[idx] * rating
                valid_slugs += 1
            else:
                print(f"[Content-KNN] Warning: '{slug}' not found in database.")

        if valid_slugs == 0:
            print("[Content-KNN] Warning: None of the input movies are in the database.")
            return []

        # Enumerate and sort the scores
        sim_scores = list(enumerate(total_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        watched_slugs = set(user_profile.keys())
        results = []

        # Extract top recommendations, skipping already watched movies
        for idx, score in sim_scores:
            movie_slug = self.df.iloc[idx]['slug']
            if movie_slug not in watched_slugs:
                results.append({
                    'slug': movie_slug,
                    'title': self.df.iloc[idx]['title'],
                    'year': self.df.iloc[idx]['year'],
                    'director': self.df.iloc[idx]['director'],
                    'poster_url': self.df.iloc[idx].get('poster_url', ''),
                    'score': score
                })
                if len(results) >= top_n:
                    break

        return results


# --- Test Execution ---
if __name__ == "__main__":
    recommender = ContentBasedRecommender("../data/user_first_cut2_clear.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    print("--- Content-Based Recommendations ---")
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)

    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']}) - Dir: {rec['director']}")
