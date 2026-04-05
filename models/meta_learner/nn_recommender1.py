import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
from util import root_path

from models.auto_rec.oof_autorec_wrapper import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender

MODEL_LOAD_PATH = root_path() / "data/nn_meta_model.pth"
STD_CLIP_LOWER = 0.1  # 必须和训练时一致
META_DB = root_path() / "data/meta_dataset.db"

MODEL_SCORE_COLS = [
    "AutoRec_Score",
    "UserKNN_RMSE_Score", "UserKNN_Hit_Score",
    "ItemKNN_RMSE_Score",  "ItemKNN_Hit_Score",
    "SVD_Score",
    "ContentKNN_Hit_Score", "ContentKNN_RMSE_Score",
]
INPUT_DIM = 15


class MetaMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


class NNMetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[NN] Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MetaMLP().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=self.device))
        self.model.eval()

        print("[NN] Initializing base models...")
        self.base_models = {
            "AutoRec":         AutoRecRecommender(db_path=self.db_path),
            "UserKNN_RMSE":    UserBasedRecommender(db_path=self.db_path, k_neighbors=168),
            "UserKNN_Hit":     UserBasedRecommender(db_path=self.db_path, k_neighbors=13),
            "ItemKNN_RMSE":    ItemBasedRecommender(db_path=self.db_path, k_neighbors=50),
            "ItemKNN_Hit":     ItemBasedRecommender(db_path=self.db_path, k_neighbors=7),
            "SVD":             SVDRecommender(db_path=self.db_path),
            "ContentKNN_Hit":  ContentBasedRecommender(db_path=self.db_path, k_neighbors=1),
            "ContentKNN_RMSE": ContentBasedRecommender(db_path=self.db_path, k_neighbors=871),
        }

        conn = sqlite3.connect(self.db_path)
        self.df_movies = pd.read_sql_query(
            "SELECT slug, title, year, avg_rating, rating_count, rating_std, release_year "
            "FROM movies", conn
        )
        self.df_movies.set_index('slug', inplace=True)
        self.movie_slugs = self.df_movies.index.tolist()
        conn.close()

    def _normalize_model_scores(self, raw_preds: dict, user_avg: float) -> dict:
        """
        和训练完全一致：
        1. 缺失的 slug 取 0（→ 0 - user_avg = 负数，保留无预测信号）
        2. 减 user_avg
        3. 除以该模型在该用户上的 std，clip(0.1)
        """
        centered = {
            slug: (raw_preds.get(slug, 0.0) - user_avg)
            for slug in self.movie_slugs
        }
        vals = np.array(list(centered.values()), dtype=np.float32)
        std = max(float(vals.std()), STD_CLIP_LOWER)
        return {slug: v / std for slug, v in centered.items()}

    def _get_stat_features(self, slug: str, user_avg: float,
                           user_std: float, user_rating_count: int) -> list:
        """
        和训练完全一致的固定变换。
        """
        movie = self.df_movies.loc[slug]
        return [
            np.log1p(movie.get('rating_count', 0)),   # Movie_Rating_Count
            float(movie.get('avg_rating', 0)),         # Movie_Avg
            float(movie.get('rating_std', 0)),         # Movie_Std
            np.log1p(user_rating_count),               # User_Rating_Count
            user_avg,                                  # User_Avg
            user_std,                                  # User_Std
            (float(movie.get('release_year', 2000)) - 1900) / 100,  # Release_Year
        ]

    def get_recommendations(self, user_profile: dict, top_n: int = 10) -> list:
        if not user_profile:
            return []

        ratings = list(user_profile.values())
        user_avg   = float(np.mean(ratings))
        user_std   = float(np.std(ratings)) if len(ratings) > 1 else 0.0
        user_count = len(ratings)

        # 获取各基础模型全量预测并归一化
        normalized = {}
        for name, model in self.base_models.items():
            raw_recs = model.get_recommendations(user_profile, top_n=len(self.movie_slugs))
            raw_preds = {rec['slug']: rec['score'] for rec in raw_recs}
            normalized[name] = self._normalize_model_scores(raw_preds, user_avg)

        # 组装特征矩阵
        features, candidate_slugs = [], []
        for slug in self.movie_slugs:
            if slug in user_profile:
                continue
            model_feats = [
                normalized["AutoRec"].get(slug, 0.0),
                normalized["UserKNN_RMSE"].get(slug, 0.0),
                normalized["UserKNN_Hit"].get(slug, 0.0),
                normalized["ItemKNN_RMSE"].get(slug, 0.0),
                normalized["ItemKNN_Hit"].get(slug, 0.0),
                normalized["SVD"].get(slug, 0.0),
                normalized["ContentKNN_Hit"].get(slug, 0.0),
                normalized["ContentKNN_RMSE"].get(slug, 0.0),
            ]
            stat_feats = self._get_stat_features(slug, user_avg, user_std, user_count)
            features.append(model_feats + stat_feats)
            candidate_slugs.append(slug)

        if not features:
            return []

        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = self.model(X).squeeze(1).cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            slug = candidate_slugs[idx]
            movie = self.df_movies.loc[slug]
            absolute_score = max(0.5, min(5.0, float(scores[idx]) + user_avg))
            results.append({
                'slug': slug,
                'title': movie['title'],
                'year': movie['year'],
                'score': absolute_score,
            })

        return results


if __name__ == "__main__":
    recommender = NNMetaRecommender(root_path() / "data/train_model.db")
    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0,
    }
    recs = recommender.get_recommendations(demo_profile)
    for r in recs:
        print(f"[{r['score']:.2f}] {r['title']} ({r['year']})")