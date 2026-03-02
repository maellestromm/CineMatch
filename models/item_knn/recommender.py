import torch

from util import load_review_datas, root_path


class ItemBasedRecommender:
    def __init__(self, db_path, k_neighbors=40):
        self.db_path = db_path
        self.k_neighbors = k_neighbors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.df_movies = None
        self.movie_slugs = []
        self.movie_to_idx = {}
        self.sim_matrix = None

        self._load_data()

    def _load_data(self):
        print(f"[Item-KNN] Initializing Engine on: {self.device}")
        df_reviews, self.df_movies = load_review_datas(self.db_path)

        print("[Item-KNN] Building Item-User Matrix...")
        pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating', aggfunc='mean').fillna(0)

        self.movie_slugs = pivot_df.columns.tolist()
        self.movie_to_idx = {slug: idx for idx, slug in enumerate(self.movie_slugs)}

        # Transpose to get (Items x Users) shape for Item-Item similarity
        item_user_tensor = torch.tensor(pivot_df.values, dtype=torch.float32, device=self.device).T

        print("[Item-KNN] Computing Item-Item Cosine Similarity Matrix...")
        # Normalize rows to compute cosine similarity via fast matrix multiplication
        item_norms = item_user_tensor.norm(dim=1, keepdim=True)
        item_norms[item_norms == 0] = 1e-10
        normalized_items = item_user_tensor / item_norms

        # Matrix multiplication: (Items x Users) @ (Users x Items) -> (Items x Items)
        sim_matrix = torch.mm(normalized_items, normalized_items.T)

        # Remove self-similarity (diagonal)
        sim_matrix.fill_diagonal_(0.0)

        print(f"[Item-KNN] Filtering Top-{self.k_neighbors} neighbors to reduce noise...")
        actual_k = min(self.k_neighbors, sim_matrix.shape[1])
        top_k_vals, top_k_indices = torch.topk(sim_matrix, actual_k, dim=1)

        filtered_sim_matrix = torch.zeros_like(sim_matrix)
        filtered_sim_matrix.scatter_(1, top_k_indices, top_k_vals)
        filtered_sim_matrix[filtered_sim_matrix < 0] = 0.0

        self.sim_matrix = filtered_sim_matrix
        print(f"[Item-KNN] Ready! Similarity Matrix VRAM occupied: {self.sim_matrix.nelement() * 4 / 1024 / 1024:.2f} MB\n")

    def get_recommendations(self, user_profile, top_n=10):
        if self.sim_matrix is None:
            return []

        target_tensor = torch.zeros(len(self.movie_slugs), dtype=torch.float32, device=self.device)
        watched_mask = torch.zeros(len(self.movie_slugs), dtype=torch.float32, device=self.device)
        watched_indices = []

        for slug, rating in user_profile.items():
            if slug in self.movie_to_idx:
                idx = self.movie_to_idx[slug]
                target_tensor[idx] = float(rating)
                watched_mask[idx] = 1.0
                watched_indices.append(idx)

        user_ratings_count = watched_mask.sum().item()
        if user_ratings_count == 0:
            return []

        with torch.no_grad():
            # Numerator: Similarity Matrix @ User Ratings Vector
            recommendation_scores = torch.mv(self.sim_matrix, target_tensor)

            # Denominator: Similarity Matrix @ Watched Mask Vector
            similarity_sums = torch.mv(self.sim_matrix, watched_mask)

            # Bayesian Smoothing (identical math to the User-KNN logic)
            prior_mean = target_tensor.sum().item() / user_ratings_count
            damping = 3.0

            final_scores = (recommendation_scores + damping * prior_mean) / (similarity_sums + damping)

            # Mask out already watched movies
            final_scores[watched_indices] = -999.0

            top_n_scores, top_n_movie_indices = torch.topk(final_scores, top_n)

        top_n_scores = top_n_scores.cpu().numpy()
        top_n_movie_indices = top_n_movie_indices.cpu().numpy()

        results = []
        for score, idx in zip(top_n_scores, top_n_movie_indices):
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
    recommender = ItemBasedRecommender(root_path() / "data/train_model.db", k_neighbors=40)

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    import time
    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)
    latency = (time.time() - start_time) * 1000

    print("--- Item-Based Recommendations ---")
    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']}) - Dir: {rec['director']}")
    print(f"Latency: {latency:.2f} ms")