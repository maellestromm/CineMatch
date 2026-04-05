import numpy as np
import pandas as pd
import sqlite3
from util import root_path

# 引入你的 AutoRec 基座模型
from models.auto_rec.oof_autorec_wrapper import AutoRecRecommender


class PopularityBoostedAutoRec:
    def __init__(self, db_path):
        self.db_path = db_path
        print("[Boosted AutoRec] Initializing Base AutoRec Engine...")
        self.autorec = AutoRecRecommender(db_path=self.db_path)

        print("[Boosted AutoRec] Loading and calculating popularity weights...")
        conn = sqlite3.connect(self.db_path)

        # 1. 加载电影基础信息
        self.df_movies = pd.read_sql_query("SELECT slug, title, year FROM movies", conn)
        self.df_movies.set_index('slug', inplace=True)

        # 2. 统计每部电影的真实评分数量
        counts_df = pd.read_sql_query("""
                                      SELECT movie_slug, COUNT(*) as review_count
                                      FROM reviews
                                      GROUP BY movie_slug
                                      """, conn)
        counts_df.set_index('movie_slug', inplace=True)
        conn.close()

        # 3. 数据合并
        self.df_movies = self.df_movies.join(counts_df)
        self.df_movies['review_count'] = self.df_movies['review_count'].fillna(0)

        # 🚀 4. 核心魔法：对数平滑加权 (Logarithmic Smoothing)
        # 公式: 1.0 + (当前评论对数 / 最大评论对数) * Max_Boost
        # 假设我们允许的最大加成比例是 50% (0.5)
        max_log = np.log1p(3806)  # 大约 8.24
        self.df_movies['pop_weight'] = 1.0 + ((np.log1p(self.df_movies['review_count']) / max_log) * 0.5)

    def get_recommendations(self, user_profile, top_n=10):
        if not user_profile:
            return []

        # 1. 获取 AutoRec 对所有电影的原始预测分
        # 注意这里把 top_n 设为电影总数，因为我们要对全盘数据重新排序
        raw_recs = self.autorec.get_recommendations(user_profile, top_n=len(self.df_movies))

        boosted_results = []
        for rec in raw_recs:
            slug = rec['slug']
            raw_score = rec['score']

            # 2. 提取权重（防止有新电影没记录，给个默认兜底 1.0）
            if slug in self.df_movies.index:
                weight = self.df_movies.loc[slug, 'pop_weight']
            else:
                weight = 1.0

            # 3. 分数加权融合
            final_score = raw_score * weight

            boosted_results.append({
                'slug': slug,
                'title': rec['title'],
                'year': rec['year'],
                'score': final_score,  # 排序用的最终融合分
                'raw_autorec': raw_score,  # 留存原始分供 debug
                'review_count': int(self.df_movies.loc[slug, 'review_count'] if slug in self.df_movies.index else 0)
            })

        # 4. 根据加权后的新分数重新降序排列
        boosted_results.sort(key=lambda x: x['score'], reverse=True)

        # 5. 按照项目标准格式返回 Top N
        final_top_n = []
        for r in boosted_results[:top_n]:
            final_top_n.append({
                'slug': r['slug'],
                'title': r['title'],
                'year': r['year'],
                'score': float(r['score'])  # 保持格式纯净，适配评估脚本
            })

        return final_top_n


if __name__ == "__main__":
    recommender = PopularityBoostedAutoRec(root_path() / "data/train_model.db")

    # 用你的高阶打分习惯测试一下
    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    recs = recommender.get_recommendations(demo_profile)
    for r in recs:
        print(f"[{r['score']:.4f}] {r['title']} ({r['year']})")