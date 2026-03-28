import numpy as np
from models.auto_rec import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from util import load_review_datas, root_path
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender

class MetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[Meta] Initializing models...")

        self.models = {
            "User-KNN": UserBasedRecommender(db_path),
            "Deep-AutoRec": AutoRecRecommender(db_path),
            "Item-KNN": ItemBasedRecommender(db_path),
            "SVD-50": SVDRecommender(db_path),
        }

        # defualt - tune later
        self.weights = {
            "SVD-50": 0.3,
            "User-KNN": 0.25,
            "Item-KNN": 0.2,
            "Deep-AutoRec": 0.2
        }

        print("[Meta] Ready.")

    def get_recommendations(self, user_profile, top_n=10):
        model_preds = {}

        # 1. get predictions from each model
        for name, model in self.models.items():
            raw = model.get_recommendations(user_profile, top_n=3334)
            model_preds[name] = {rec['slug']: rec['score'] for rec in raw}

        # 2. collect all movie candidates
        all_movies = set()
        for preds in model_preds.values():
            all_movies.update(preds.keys())

        # 3. normalize predictions
        normalized_preds = {}

        for name, preds in model_preds.items():
            if len(preds) == 0:
                normalized_preds[name] = {}
                continue
            values = np.array(list(preds.values()))

            mean = values.mean()
            std = values.std() + 1e-8

            normalized_preds[name] = {
                k: (v - mean) / std for k, v in preds.items()
            }

        # 4. combine scores
        final_scores = {}
        for movie in all_movies:
            score = 0.0

            for name in self.models:
                pred = normalized_preds[name].get(movie, 0.0)
                score += self.weights[name] * pred

            final_scores[movie] = score

        # scale the predicted ratings between 1 - 5 again for interpretability of results:
        final_scores_array = np.array(list(final_scores.values()))
        min_score = final_scores_array.min()
        max_score = final_scores_array.max()
        for k in final_scores:
            final_scores[k] = 1 + 4 * (final_scores[k] - min_score) / (max_score - min_score)

        # 5. sort and return top_n
        top_movies = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # reuse metadata from ONE model (e.g., SVD)
        svd_preds = self.models["SVD-50"].get_recommendations(user_profile, top_n=3334)
        svd_lookup = {rec['slug']: rec for rec in svd_preds}

        results = []
        for slug, score in top_movies:
            if slug in svd_lookup:
                rec = svd_lookup[slug].copy()
                rec['score'] = score
                results.append(rec)

        return results
    
    def learn_optimal_weights(self, train_reviews):
        """
        Learn linear weights for the base models using known train ratings.
        train_reviews: DataFrame with columns ['user_username', 'movie_slug', 'rating']
        """

        print("[Meta] Learning optimal weights from train data...")
        
        all_users = train_reviews['user_username'].unique()
        y_true = []
        pred_matrix = []

        # precompute predictions only for movies each user rated
        model_cache = {name: {} for name in self.models}
        for name, model in self.models.items():
            print(f"[Meta] Precomputing predictions for {name}...")
            for i, user in enumerate(all_users, 1):
                user_data = train_reviews[train_reviews['user_username'] == user]
                user_profile = dict(zip(user_data['movie_slug'], user_data['rating']))
                # only need predictions for movies this user rated
                raw = model.get_recommendations(user_profile, top_n=len(user_profile))
                model_cache[name][user] = {rec['slug']: rec['score'] for rec in raw}

                if i % 50 == 0:
                    print(f"[Meta] {name}: {i}/{len(all_users)} users processed...")
                    
        # build the prediction matrix
        for i, user in enumerate(all_users, 1):
            user_data = train_reviews[train_reviews['user_username'] == user]

            for idx, row in user_data.iterrows():
                slug = row['movie_slug']
                y_true.append(row['rating'])

                row_preds = []
                for name in self.models:
                    pred = model_cache[name][user].get(slug, 0.0)
                    row_preds.append(pred)
                pred_matrix.append(row_preds)
        y_true = np.array(y_true)
        pred_matrix = np.array(pred_matrix)

        # solve linear regression (least squares)
        weights, _, _, _ = np.linalg.lstsq(pred_matrix, y_true, rcond=None)
        weights = np.clip(weights, 0, None)          # prevent negative weights
        weights /= weights.sum()                     # normalize sum to 1

        # assign to self.weights
        for i, name in enumerate(self.models):
            self.weights[name] = weights[i]

        print("[Meta] Learned weights:", self.weights)
    
if __name__ == "__main__":
    recommender = MetaRecommender(root_path() / "data/train_model.db")

    df_reviews, _ = load_review_datas(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    # Learn optimal weights
    # recommender.learn_optimal_weights(df_reviews)

    recs = recommender.get_recommendations(demo_profile)

    for i, r in enumerate(recs, 1):
        print(f"{i}.\t[{r['score']:.2f}] {r['title']} ({r['year']})")
