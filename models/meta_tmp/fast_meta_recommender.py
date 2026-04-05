import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from util import root_path

# 导入你的 5 个基座模型
from models.auto_rec.oof_autorec_wrapper import AutoRecRecommender
from models.content_knn import ContentBasedRecommender
from models.item_knn import ItemBasedRecommender
from models.svd import SVDRecommender
from models.user_knn import UserBasedRecommender


class FastMetaRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        META_DB = root_path() / "data/meta_dataset.db"
        # 1. 瞬间加载基座模型
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

        conn = sqlite3.connect(META_DB)
        # 2. 初始化时直接读取 43 万条 OOF 训练数据
        print("[Fast Meta] Instantly training constrained linear model...")
        df_train = pd.read_sql_query("SELECT * FROM meta_train", conn)
        conn.close()

        score_cols = ["SVD_Score", "ItemKNN_Hit_Score", "AutoRec_Score", "ContentKNN_Hit_Score", "UserKNN_Hit_Score"]

        X = df_train[score_cols].values
        y = df_train["Actual_Rating"].values

        # 3. 核心：强制所有权重为正数，数学上保证 RMSE 最小的同时绝不破坏基座的原始排序
        self.model = LinearRegression(positive=True)
        self.model.fit(X, y)

        # 打印可以直接写死在前端 JS 里的常量
        print("-" * 40)
        print("Frontend JS Constants (Copy these!):")
        print(f"const INTERCEPT = {self.model.intercept_:.4f};")
        for name, coef in zip(score_cols, self.model.coef_):
            print(f"const WEIGHT_{name.split('_')[0].upper()} = {coef:.4f};")
        print("-" * 40)

    def get_recommendations(self, user_profile, top_n=10):
        if not user_profile:
            return []

        # 缺失值兜底策略：如果基座模型没算出来，用该用户均分代替，最安全
        user_ratings = list(user_profile.values())
        user_avg = np.mean(user_ratings) if user_ratings else 3.5

        raw_preds = {}
        for name, model in self.base_models.items():
            raw_recs = model.get_recommendations(user_profile, top_n=len(self.movie_slugs))
            raw_preds[name] = {rec['slug']: rec['score'] for rec in raw_recs}

        inference_features = []
        candidate_slugs = []

        for slug in self.movie_slugs:
            if slug in user_profile:
                continue

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

        # 直接使用训练好的线性方程计算绝对星级
        X_infer = np.array(inference_features)
        final_scores = self.model.predict(X_infer)

        top_indices = np.argsort(final_scores)[::-1][:top_n]
        results = []

        for idx in top_indices:
            slug = candidate_slugs[idx]
            # 强行截断至 0.5 ~ 5.0，保护 RMSE 不受极端值影响
            score = float(max(0.5, min(5.0, final_scores[idx])))
            results.append({
                'slug': slug,
                'title': self.df_movies.loc[slug, 'title'],
                'year': self.df_movies.loc[slug, 'year'],
                'score': score
            })

        return results

if __name__ == "__main__":
    recommender = FastMetaRecommender(root_path() / "data/train_model.db")
    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }
    recs = recommender.get_recommendations(demo_profile)
    for r in recs:
        print(f"[{r['score']:.4f}] {r['title']} ({r['year']})")