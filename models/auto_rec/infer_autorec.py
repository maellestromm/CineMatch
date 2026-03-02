import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import json
import numpy as np

from util import root_path


# ==========================================
# 1. Network Architecture Definition
# ==========================================
class DeepAutoRec(nn.Module):
    def __init__(self, num_movies):
        super(DeepAutoRec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 512),
            nn.Dropout(0.3),  # Automatically disabled during inference (model.eval)
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_movies)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==========================================
# 2. Recommender Wrapper
# ==========================================
class AutoRecRecommender:
    def __init__(self, db_path, dict_path=root_path() / "data/movie_dictionary.json",
                 weights_path=root_path() / "data/autorec_best_weights.pth"):
        self.db_path = db_path
        self.df_movies = None
        self.movie_slugs = []
        self.movie_to_idx = {}
        self.num_movies = 0
        self.model = None

        self._load_data(dict_path, weights_path)

    def _load_data(self, dict_path, weights_path):
        """Load movie metadata, model dictionary, and weights"""
        print("[AutoRec] Loading movie metadata from database...")
        conn = sqlite3.connect(self.db_path)
        query_movies = "SELECT slug, title, year, director, poster_url FROM movies"
        self.df_movies = pd.read_sql_query(query_movies, conn).set_index('slug')
        conn.close()

        print("[AutoRec] Loading model dictionary and weights...")
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                self.movie_slugs = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Dictionary file {dict_path} not found. Please run the training script first.")

        self.num_movies = len(self.movie_slugs)
        self.movie_to_idx = {slug: idx for idx, slug in enumerate(self.movie_slugs)}

        # Initialize model and load weights
        self.model = DeepAutoRec(num_movies=self.num_movies)
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        except FileNotFoundError:
            raise Exception(f"Weights file {weights_path} not found. Please run the training script first.")

        self.model.eval()  # Switch to inference mode
        print(f"[AutoRec] Engine ready! {self.num_movies} movies loaded.\n")

    def get_recommendations(self, user_profile, top_n=10):
        """
        Get recommendations based on a user profile.
        :param user_profile: dict, e.g., {'movie-slug': 5.0, 'another-slug': 4.0}
        :param top_n: int, number of recommendations to return
        :return: list of dicts representing recommended movies
        """
        if self.model is None:
            return []

        # 1. Create a blank tensor for all movies
        target_vector = torch.zeros(self.num_movies)
        watched_indices = []

        # 2. Fill in the user's ratings
        for slug, rating in user_profile.items():
            if slug in self.movie_to_idx:
                idx = self.movie_to_idx[slug]
                target_vector[idx] = float(rating)
                watched_indices.append(idx)
            else:
                print(f"[AutoRec] Warning: '{slug}' not found in model dictionary.")

        if target_vector.sum() == 0:
            print("[AutoRec] Warning: None of the input movies are in the database.")
            return []

        # 3. Add batch dimension: (1, num_movies)
        target_vector = target_vector.unsqueeze(0)

        # 4. Fast inference without gradients
        with torch.no_grad():
            predictions = self.model(target_vector).squeeze(0).numpy()

        # 5. Mask out already watched movies
        predictions[watched_indices] = -999.0

        # 6. Extract top N predicted movie indices
        top_indices = np.argsort(predictions)[::-1][:top_n]

        # 7. Assemble the results
        results = []
        for idx in top_indices:
            slug = self.movie_slugs[idx]
            score = float(predictions[idx])

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
    recommender = AutoRecRecommender(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    print("--- Deep AutoRec Recommendations ---")
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)

    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']}) - Dir: {rec['director']}")
