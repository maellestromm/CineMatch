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


class WideAndDeepMeta(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.wide(x) + self.deep(x)


class NNMetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[NN Recommender] Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WideAndDeepMeta().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=self.device))
        self.model.eval()

        print("[NN Recommender] Initializing base models...")
        self.base_models = {
            "SVD": SVDRecommender(db_path=self.db_path),
            "ItemKNN_Hit": ItemBasedRecommender(db_path=self.db_path, k_neighbors=7),
            "AutoRec": AutoRecRecommender(db_path=self.db_path),
            "ContentKNN_Hit": ContentBasedRecommender(db_path=self.db_path, k_neighbors=1),
            "UserKNN_Hit": UserBasedRecommender(db_path=self.db_path, k_neighbors=13),
        }

        conn = sqlite3.connect(self.db_path)
        self.df_movies = pd.read_sql_query("SELECT slug, title, year FROM movies", conn)
        self.df_movies.set_index('slug', inplace=True)
        self.movie_slugs = self.df_movies.index.tolist()
        conn.close()

    def _normalize_model_scores(self, raw_preds: dict, user_avg: float) -> dict:
        """
        和训练时完全一致的归一化：
        1. 减 user_avg（缺失的 slug 不在 raw_preds 里，取 0 之后减 user_avg 变为负数）
        2. 除以该模型在该用户上预测分的 std（clip 0.1）
        返回 {slug: normalized_score}，缺失 slug 对应负值。
        """
        # Step 1: 减均值，缺失的取 0（→ 0 - user_avg = 负数，和训练一致）
        centered = {
            slug: (raw_preds.get(slug, 0) if raw_preds.get(slug, 0) != 0 else user_avg)
            for slug in self.movie_slugs
        }

        # Step 2: 除以 std，对齐量纲
        vals = np.array(list(centered.values()), dtype=np.float32)
        std = float(vals.std())
        std = max(std, STD_CLIP_LOWER)  # 和训练时 clip(lower=0.1) 一致
        avg = float(vals.mean())
        return {slug: (v - avg) / std for slug, v in centered.items()}

    def get_recommendations(self, user_profile: dict, top_n: int = 10) -> list:
        if not user_profile:
            return []

        user_avg = float(np.mean(list(user_profile.values())))
        user_std = float(np.std(list(user_profile.values())))
        user_std = 1 if user_std == 0 else user_std
        user_std = max(user_std, STD_CLIP_LOWER)

        # 获取各基础模型的全量预测并归一化
        normalized = {}
        raw = {}
        for name, model in self.base_models.items():
            raw_recs = model.get_recommendations(user_profile, top_n=len(self.movie_slugs))
            raw_preds = {rec['slug']: rec['score'] for rec in raw_recs}
            raw[name] = raw_preds
            normalized[name] = self._normalize_model_scores(raw_preds, user_avg)

        # 组装特征矩阵（跳过用户已看过的电影）
        features, candidate_slugs = [], []
        raw_scores = []
        for slug in self.movie_slugs:
            if slug in user_profile:
                continue
            features.append([
                normalized["SVD"].get(slug, 0.0),
                normalized["ItemKNN_Hit"].get(slug, 0.0),
                normalized["AutoRec"].get(slug, 0.0),
                normalized["ContentKNN_Hit"].get(slug, 0.0),
                normalized["UserKNN_Hit"].get(slug, 0.0),
            ])
            candidate_slugs.append(slug)

            raw_scores.append([
                raw["SVD"].get(slug, 0.0),
                raw["ItemKNN_Hit"].get(slug, 0.0),
                raw["AutoRec"].get(slug, 0.0),
                raw["ContentKNN_Hit"].get(slug, 0.0),
                raw["UserKNN_Hit"].get(slug, 0.0)]
            )

        if not features:
            return []

        # 推理
        X = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = self.model(X).squeeze(1).cpu().numpy()

        # Top-N 排序
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            slug = candidate_slugs[idx]
            raw_score = raw_scores[idx]
            movie = self.df_movies.loc[slug]
            # 反变换回绝对评分（仅用于展示，不影响排序）
            absolute_score = (float(scores[idx]) * user_std + user_avg)
            absolute_score = max(0.5, min(5.0, absolute_score))
            results.append({
                'slug': slug,
                'title': movie['title'],
                'year': movie['year'],
                'score': absolute_score,
                'raw_score': raw_score,
                'raw_meta':scores[idx]
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
        print(f"[{r['score']:.2f}] {r['title']} ({r['year']}) meta:{r['raw_meta']} raw: {r['raw_score']}")
