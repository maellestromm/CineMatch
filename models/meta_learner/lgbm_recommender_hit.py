import numpy as np
import pandas as pd
import lightgbm as lgb
import sqlite3
from util import root_path

# 导入基座模型
from models.auto_rec.oof_autorec_wrapper import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender

MODEL_LOAD_PATH = root_path() / "data/lgbm_residual_model.txt"


class ResidualMetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[Residual Recommender] Loading Native LightGBM Booster...")
        # 🚀 加载原生 Booster，它完美支持 init_score 接口
        self.booster = lgb.Booster(model_file=str(MODEL_LOAD_PATH))

        print("[Residual Recommender] Initializing 5 Base Engines...")
        self.base_models = {
            "SVD": SVDRecommender(db_path=self.db_path),
            "ItemKNN_Hit": ItemBasedRecommender(db_path=self.db_path, k_neighbors=7),
            "AutoRec": AutoRecRecommender(db_path=self.db_path),
            "ContentKNN_Hit": ContentBasedRecommender(db_path=self.db_path, k_neighbors=1),
            "UserKNN_Hit": UserBasedRecommender(db_path=self.db_path, k_neighbors=13)
        }

        conn = sqlite3.connect(self.db_path)
        self.df_movies = pd.read_sql_query("SELECT slug, title, year FROM movies", conn)
        self.df_movies.set_index('slug', inplace=True)
        self.movie_slugs = self.df_movies.index.tolist()
        conn.close()

    def get_recommendations(self, user_profile, top_n=10):
        if not user_profile:
            return []

        # 1. 获取所有基准打分并计算局部 Z-Score
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
        svd_base_margins = []
        candidate_slugs = []

        # 2. 构造推理特征
        for slug in self.movie_slugs:
            if slug in user_profile:
                continue

            # 🚀 单独提取 SVD 的 Z-Score 作为这棵树的基准线！
            svd_base = model_z_scores["SVD"].get(slug, 0.0)
            svd_base_margins.append(svd_base)

            # 🚀 严格按照训练集的 4 维顺序提取剩余特征，绝对不能有 SVD！
            row = [
                model_z_scores["ItemKNN_Hit"].get(slug, 0.0),
                model_z_scores["AutoRec"].get(slug, 0.0),
                model_z_scores["ContentKNN_Hit"].get(slug, 0.0),
                model_z_scores["UserKNN_Hit"].get(slug, 0.0)
            ]
            inference_features.append(row)
            candidate_slugs.append(slug)

        if not inference_features:
            return []

        X_infer = pd.DataFrame(inference_features, columns=[
            "ItemKNN_Hit_Score", "AutoRec_Score",
            "ContentKNN_Hit_Score", "UserKNN_Hit_Score"
        ])

        # 🚀 终极降维打击：在预测时注入 init_score！
        # raw_score=True 意味着输出的直接是 (SVD基准 + 树残差) 的对数几率得分
        # 在推荐系统排序中，我们不需要把它转换回概率，直接用 raw_score 排序精度最高！
        # 1. 独立预测：让树模型根据 KNN 等特征，算出一个“补丁分” (Residual)
        tree_residuals = self.booster.predict(X_infer, raw_score=True)

        # 2. 物理合并：把 SVD 的铁血底分 和 树的补丁分 直接相加！
        # 这才是 Residual Learning 在推理阶段的真实做法
        final_scores = np.array(svd_base_margins) + tree_residuals

        # 3. 降序排列返回 Top N
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        results = []

        for idx in top_indices:
            slug = candidate_slugs[idx]
            movie_data = self.df_movies.loc[slug]
            results.append({
                'slug': slug,
                'title': movie_data['title'],
                'year': movie_data['year'],
                'score': float(final_scores[idx])
            })

        return results

if __name__ == "__main__":
    # Test Execution
    recommender = ResidualMetaRecommender(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    import time

    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)
    print(f"Latency: {(time.time() - start_time) * 1000:.2f} ms")

    for i, _rec in enumerate(recommendations, 1):
        print(f"{i}. [{_rec['score']:.2f}] {_rec['title']} ({_rec['year']})")
