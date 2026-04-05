import torch
import torch.nn.functional as F

from util import load_review_movie_datas


class UserKNNGPUBackend:
    def __init__(self, db_path, k_neighbors):
        self.k_neighbors = k_neighbors
        self.db_path = db_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.df_movies = None
        self.movie_slugs = []
        self.user_usernames = []
        self.matrix_tensor = None
        self.movie_to_idx = {}

        self._load_data()

    def _load_data(self):
        print(f"[User-KNN-GPU] Initializing Engine on: {self.device}")
        df_reviews, self.df_movies = load_review_movie_datas(self.db_path)
        print("[User-KNN-GPU] Building DataFrame...")
        pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating',
                                          aggfunc='mean').fillna(0)

        self.movie_slugs = pivot_df.columns.tolist()
        self.user_usernames = pivot_df.index.tolist()
        self.movie_to_idx = {slug: idx for idx, slug in enumerate(self.movie_slugs)}

        print("[User-KNN-GPU] Locking Matrix in VRAM...")
        self.matrix_tensor = torch.tensor(pivot_df.values, dtype=torch.float32).to(self.device)
        print(f"[User-KNN-GPU] Ready! VRAM occupied: {self.matrix_tensor.nelement() * 4 / 1024 / 1024:.2f} MB\n")

    def get_recommendations(self, user_profile, top_n=10):
        if self.matrix_tensor is None: return []

        target_tensor = torch.zeros(len(self.movie_slugs), dtype=torch.float32, device=self.device)
        watched_indices = []

        for slug, rating in user_profile.items():
            if slug in self.movie_to_idx:
                idx = self.movie_to_idx[slug]
                target_tensor[idx] = float(rating)
                watched_indices.append(idx)

        if target_tensor.sum() == 0: return []

        with torch.no_grad():
            sim_tensor = F.cosine_similarity(target_tensor.unsqueeze(0), self.matrix_tensor, dim=1)

            top_k_sims, top_k_indices = torch.topk(sim_tensor, k=self.k_neighbors)

            valid_mask = top_k_sims > 0
            top_k_sims = top_k_sims[valid_mask]
            top_k_indices = top_k_indices[valid_mask]

            if len(top_k_sims) == 0: return []

            neighbor_matrix = self.matrix_tensor[top_k_indices]
            has_rated_mask = neighbor_matrix > 0

            weighted_ratings = neighbor_matrix * top_k_sims.unsqueeze(1)
            recommendation_scores = weighted_ratings.sum(dim=0)
            similarity_sums = (has_rated_mask.float() * top_k_sims.unsqueeze(1)).sum(dim=0)

            # Bayesian Smoothing implementation
            user_ratings_count = (target_tensor > 0).sum()

            if user_ratings_count > 0:
                prior_mean = target_tensor.sum() / user_ratings_count
            else:
                prior_mean = torch.tensor(3.0, dtype=torch.float32, device=self.device)
            damping = 3.0
            final_scores = (recommendation_scores + damping * prior_mean) / (similarity_sums + damping)

            final_scores[similarity_sums == 0] = 0.0

            final_scores[watched_indices] = -1.0

            top_n_scores, top_n_movie_indices = torch.topk(final_scores, top_n)

        top_n_scores = top_n_scores.cpu().numpy()
        top_n_movie_indices = top_n_movie_indices.cpu().numpy()

        results = []
        for score, idx in zip(top_n_scores, top_n_movie_indices):
            if score <= 0: continue
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
