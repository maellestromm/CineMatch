import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import json
import numpy as np

from util import root_path


# ==========================================
# 1. 网络结构定义 (必须与训练时完全一致)
# ==========================================
class DeepAutoRec(nn.Module):
    def __init__(self, num_movies):
        super(DeepAutoRec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 512),
            nn.Dropout(0.3),  # 推理时(model.eval)会自动失效
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, num_movies)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ==========================================
# 2. 推荐器封装 (完全对齐 UserBasedRecommender)
# ==========================================
class AutoRecRecommender:
    def __init__(self, db_path, dict_path=root_path() / "data/movie_dictionary.json",
                 weights_path=root_path() / "data/autorec_best_weights.pth"):
        self.db_path = db_path
        self.df_movies = None
        self.movie_slugs = []
        self.movie_to_idx = {}
        self.num_movies = 0
        self.model = None

        # 实例化时自动加载数据和模型
        self._load_data(dict_path, weights_path)

    def _load_data(self, dict_path, weights_path):
        """加载电影元数据、模型字典和权重"""
        print("[AutoRec] Loading movie metadata from database...")
        conn = sqlite3.connect(self.db_path)
        query_movies = "SELECT slug, title, year, director, poster_url FROM movies"
        self.df_movies = pd.read_sql_query(query_movies, conn).set_index('slug')
        conn.close()

        print("[AutoRec] Loading model dictionary and weights...")
        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                self.movie_slugs = json.load(f)
        except FileNotFoundError:
            raise Exception(f"❌ 找不到字典文件 {dict_path}，请先运行训练脚本！")

        self.num_movies = len(self.movie_slugs)
        self.movie_to_idx = {slug: idx for idx, slug in enumerate(self.movie_slugs)}

        # 初始化模型并加载权重 (map_location='cpu' 保证没有显卡也能极速跑)
        self.model = DeepAutoRec(num_movies=self.num_movies)
        try:
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        except FileNotFoundError:
            raise Exception(f"❌ 找不到权重文件 {weights_path}，请先运行训练脚本！")

        self.model.eval()  # 🚨 极其重要：切换到推理模式
        print(f"[AutoRec] Engine ready! {self.num_movies} movies loaded.\n")

    def get_recommendations(self, user_profile, top_n=10):
        """
        和 User-KNN 拥有完全一致的签名和返回格式
        :param user_profile: dict, e.g., {'movie-slug': 5.0, 'another-slug': 4.0}
        :param top_n: int, 返回的推荐数量
        :return: list of dicts representing recommended movies
        """
        if self.model is None:
            return []

        # 1. 创建一张 3334 维的空白答题卡
        target_vector = torch.zeros(self.num_movies)
        watched_indices = []

        # 2. 填入教授/用户打的分数
        for slug, rating in user_profile.items():
            if slug in self.movie_to_idx:
                idx = self.movie_to_idx[slug]
                target_vector[idx] = float(rating)
                watched_indices.append(idx)
            else:
                print(f"[AutoRec] Warning: '{slug}' not found in model dictionary.")

        if target_vector.sum() == 0:
            print("[AutoRec] Warning: None of the input movies are in the database.")
            return []

        # 将 1D 数组升维成 (1, 3334)
        target_vector = target_vector.unsqueeze(0)

        # 3. 瞬间推理
        with torch.no_grad():
            predictions = self.model(target_vector).squeeze(0).numpy()

        # 4. 强行屏蔽已经看过的电影
        predictions[watched_indices] = -999.0

        # 5. 提取预测分最高的前 N 部电影索引
        top_indices = np.argsort(predictions)[::-1][:top_n]

        # 6. 组装返回结果 (与 KNN 完全一致的字典格式)
        results = []
        for idx in top_indices:
            slug = self.movie_slugs[idx]
            score = float(predictions[idx])

            if slug in self.df_movies.index:
                movie_data = self.df_movies.loc[slug]
                results.append({
                    'slug': slug,
                    'title': movie_data['title'],
                    'year': movie_data['year'],
                    'director': movie_data.get('director', ''),
                    'poster_url': movie_data.get('poster_url', ''),
                    'score': score
                })

        return results


# --- 极简测试运行 ---
if __name__ == "__main__":
    # 替换为你实际的数据库路径
    recommender = AutoRecRecommender(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    print("--- Deep AutoRec Recommendations ---")
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)

    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']}) - Dir: {rec['director']}")
