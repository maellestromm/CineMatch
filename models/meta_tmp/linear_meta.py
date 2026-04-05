import sqlite3
import numpy as np
import pandas as pd
from models.auto_rec.oof_autorec_wrapper import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender
from util import root_path


class LinearMetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[Linear Meta] Initializing NumPy-Accelerated Engines...")

        self.base_models = {
            "SVD": SVDRecommender(db_path=self.db_path),
            "ItemKNN_Hit": ItemBasedRecommender(db_path=self.db_path, k_neighbors=7),
            "AutoRec": AutoRecRecommender(db_path=self.db_path),
            "ContentKNN_Hit": ContentBasedRecommender(db_path=self.db_path, k_neighbors=1),
            "UserKNN_Hit": UserBasedRecommender(db_path=self.db_path, k_neighbors=13)
        }

        conn = sqlite3.connect(self.db_path)
        self.df_movies = pd.read_sql_query("SELECT slug, title, year FROM movies", conn)
        conn.close()

        # 预存电影 Slug 列表和元数据，避免循环内查询
        self.movie_slugs = self.df_movies['slug'].tolist()
        self.movie_meta = self.df_movies.set_index('slug').to_dict('index')

        # 预设线性权重向量 (包含 Intercept)
        # 顺序: SVD, ItemKNN, AutoRec, ContentKNN, UserKNN
        self.weights = np.array([-0.14803391, 0.05850098, 0.03052794, 0.09194181, 0.69078344])
        self.intercept = 1.25980506

    def get_recommendations(self, user_profile, top_n=10):
        if not user_profile: return []

        user_avg = np.mean(list(user_profile.values()))

        # 1. 批量获取所有基座模型的分数
        # 我们创建一个矩阵 Score_Matrix: (num_movies, 5)
        num_movies = len(self.movie_slugs)
        score_matrix = np.full((num_movies, 5), user_avg)  # 默认用 user_avg 填充

        model_names = ["SVD", "ItemKNN_Hit", "AutoRec", "ContentKNN_Hit", "UserKNN_Hit"]

        for i, name in enumerate(model_names):
            raw_recs = self.base_models[name].get_recommendations(user_profile, top_n=num_movies)
            # 将 list 转为 dict 方便快速映射
            pred_dict = {r['slug']: r['score'] for r in raw_recs}

            # 🚀 向量化填充：将预测分填入对应的列
            for idx, slug in enumerate(self.movie_slugs):
                score_matrix[idx, i] = pred_dict.get(slug, user_avg)

        # 2. 🚀 核心加速：矩阵乘法 (N, 5) @ (5, 1) -> (N, )
        final_scores = (score_matrix @ self.weights) + self.intercept

        for idx, slug in enumerate(self.movie_slugs):
            if slug in user_profile:
                final_scores[idx] = -9999.0

            # 🛡️ 防御一：将可能潜伏的 NaN 毒药全部替换成绝对低分，彻底打入冷宫
        final_scores = np.nan_to_num(final_scores, nan=-9999.0)

        # 🛡️ 防御二：🚀 核心修复！完美复刻 Python 的 Stable Reverse Sort
        # 绝不能用 np.argsort(final_scores)[::-1]！
        # 对分数取负号 (-final_scores) 配合 kind='stable'，
        # 这样既实现了降序排列，又完美保留了同分电影的原始(流行度)顺序！
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if final_scores[idx] <= 0:
                continue
            slug = self.movie_slugs[idx]
            meta = self.movie_meta[slug]

            results.append({
                'slug': slug,
                'title': meta['title'],
                'year': meta['year'],
                'score': float(final_scores[idx])
            })

        return results

if __name__ == "__main__":
    recommender = LinearMetaRecommender(root_path() / "data/train_model.db")
    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }
    recs = recommender.get_recommendations(demo_profile)
    for r in recs:
        print(f"[{r['score']:.4f}] {r['title']} ({r['year']})")