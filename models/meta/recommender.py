import numpy as np
from util import load_review_datas, root_path
from models.load_models import load_model, load_all_models

SAVE_DIR = root_path() / "models/saved_models"

class MetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[Meta] Initializing models...")
        
        # self.models = {
        #     "User-KNN": load_model("user_knn"),
        #     "Deep-AutoRec": AutoRecRecommender(db_path),
        #     "Item-KNN": load_model("item_knn"),
        #     "SVD-50": load_model("svd"),
        #     "Content-KNN": load_model("content_knn")
        # }

        self.models = load_all_models()

        # # defualt - tune later
        # self.weights = {
        #     "SVD-50": 0.3,
        #     "User-KNN": 0.25,
        #     "Item-KNN": 0.2,
        #     "Deep-AutoRec": 0.2,
        #     "Content-KNN":0.05
        # }

        # tuned weights
        self.weights = {
            "SVD-50": 0.4532,
            "Item-KNN": 0.3178,
            "Deep-AutoRec": 0.1321,
            "Content-KNN": 0.0957,
            "User-KNN": 0.0012
        }

        print("[Meta] Ready.")

    def get_recommendations(self, user_profile, top_n=10):
        model_preds = {}

        # 1. get predictions from each model
        for name, model in self.models.items():
            raw = model.get_recommendations(user_profile, top_n=3334)
            model_preds[name] = {rec['slug']: rec['score'] for rec in raw}
            
            # store full recs for one model, SVD
            if name == "SVD-50":
                svd_full = {rec['slug']: rec for rec in raw}

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

        # reuse metadata from one model (e.g., SVD)
        svd_lookup = svd_full

        results = []
        for slug, score in top_movies:
            if slug in svd_lookup:
                rec = svd_lookup[slug].copy()
                rec['score'] = score
                results.append(rec)

        return results
    
if __name__ == "__main__":
    recommender = MetaRecommender(root_path() / "data/train_model.db")

    test_reviews, test_users = load_review_datas(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    recs = recommender.get_recommendations(demo_profile)

    for i, r in enumerate(recs, 1):
        print(f"{i}.\t[{r['score']:.2f}] {r['title']} ({r['year']})")

    # rmse, hit_rate, precision = evaluate_model(recommender, test_reviews, test_users)
    # print(f"[Meta] RMSE: {rmse:.4f} | Hit@10: {hit_rate:.2%} | Prec@10: {precision:.2%}")

