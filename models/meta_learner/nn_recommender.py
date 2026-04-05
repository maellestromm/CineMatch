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


# 🚀 1. 必须在推理文件里定义一模一样的网络结构
class WideAndDeepMeta(nn.Module):
    def __init__(self, input_dim=5):
        super(WideAndDeepMeta, self).__init__()
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
        print("[NN Recommender] Loading PyTorch Wide&Deep Meta-Learner...")

        # 🚀 2. 加载 PyTorch 权重并设为评估模式
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = WideAndDeepMeta().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=self.device))
        self.model.eval()  # 极其重要：关闭 Dropout/BatchNorm 等训练特性

        print("[NN Recommender] Initializing 5 Base Engines...")
        self.base_models = {
            "SVD": SVDRecommender(db_path=self.db_path),
            "ItemKNN_Hit": ItemBasedRecommender(db_path=self.db_path, k_neighbors=7),
            "AutoRec": AutoRecRecommender(db_path=self.db_path),
            "ContentKNN_Hit": ContentBasedRecommender(db_path=self.db_path, k_neighbors=1),
            "UserKNN_Hit": UserBasedRecommender(db_path=self.db_path, k_neighbors=13)
        }

        conn = sqlite3.connect(self.db_path)
        # 注意：确认你的数据库列名是 slug 还是 movie_slug
        self.df_movies = pd.read_sql_query("SELECT slug, title, year FROM movies", conn)
        self.df_movies.set_index('slug', inplace=True)
        self.movie_slugs = self.df_movies.index.tolist()
        conn.close()

    def get_recommendations(self, user_profile, top_n=10):
        if not user_profile:
            return []

        model_z_scores = {}
        for name, model in self.base_models.items():
            raw_recs = model.get_recommendations(user_profile, top_n=len(self.movie_slugs))
            preds = {rec['slug']: rec['score'] for rec in raw_recs}

            if len(preds) > 0:
                values = np.array(list(preds.values()))
                mean = values.mean()
                std = values.std() + 1e-8
                model_z_scores[name] = {k: (v - mean) / std for k, v in preds.items()}
            else:
                model_z_scores[name] = {}

        inference_features = []
        candidate_slugs = []

        for slug in self.movie_slugs:
            if slug in user_profile:
                continue

            row = [
                model_z_scores["SVD"].get(slug, 0.0),
                model_z_scores["ItemKNN_Hit"].get(slug, 0.0),
                model_z_scores["AutoRec"].get(slug, 0.0),
                model_z_scores["ContentKNN_Hit"].get(slug, 0.0),
                model_z_scores["UserKNN_Hit"].get(slug, 0.0)
            ]
            inference_features.append(row)
            candidate_slugs.append(slug)

        if not inference_features:
            return []

        # 🚀 3. 推理前向传播
        X_tensor = torch.tensor(inference_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # 此时网络输出的是“相对偏差分” (Centered Scores)
            centered_preds = self.model(X_tensor).squeeze(1).cpu().numpy()

        # ==========================================================
        # 🚀 死保 Hit Rate：直接用相对偏差分排序！
        # ==========================================================
        top_indices = np.argsort(centered_preds)[::-1][:top_n]
        results = []

        # 获取用户真实的打分均值，如果某个基座模型没能给出分数，用均分作为完美的“中立兜底”
        user_ratings = list(user_profile.values())
        user_avg = np.mean(user_ratings) if user_ratings else 3.5

        # 1. 收集 5 个基座模型的 原始绝对分数 (Raw Scores)
        raw_preds = {}
        for name, model in self.base_models.items():
            raw_recs = model.get_recommendations(user_profile, top_n=len(self.movie_slugs))
            raw_preds[name] = {rec['slug']: rec['score'] for rec in raw_recs}

        inference_features = []
        candidate_slugs = []

        for slug in self.movie_slugs:
            if slug in user_profile:
                continue

            # 直接喂入原始分数，缺失值用 user_avg 填补
            row = [
                raw_preds["SVD"].get(slug, user_avg),
                raw_preds["ItemKNN_Hit"].get(slug, user_avg),
                raw_preds["AutoRec"].get(slug, user_avg),
                raw_preds["ContentKNN_Hit"].get(slug, user_avg),
                raw_preds["UserKNN_Hit"].get(slug, user_avg)
            ]
            inference_features.append(row)
            candidate_slugs.append(slug)

        if not inference_features:
            return []

        # 2. 纯粹的推断前向传播
        X_tensor = torch.tensor(inference_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # 网络吐出的 final_scores 已经是完美契合 1~5 刻度的绝对星级了！
            final_scores = self.model(X_tensor).squeeze(1).cpu().numpy()

        # 3. 直接排序并输出
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        results = []

        for idx in top_indices:
            slug = candidate_slugs[idx]
            movie_data = self.df_movies.loc[slug]

            # 越界保护 (1~5星)，这是唯一需要的后处理
            final_absolute_score = max(0.5, min(5.0, final_scores[idx]))

            results.append({
                'slug': slug,
                'title': movie_data['title'],
                'year': movie_data['year'],
                'score': float(final_absolute_score)
            })

        return results


if __name__ == "__main__":
    recommender = NNMetaRecommender(root_path() / "data/train_model.db")
    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }
    recs = recommender.get_recommendations(demo_profile)
    for r in recs:
        print(f"[{r['score']:.4f}] {r['title']} ({r['year']})")
