import sqlite3

import pandas as pd
import torch
import torch.nn.functional as F


class UserBasedRecommender:
    def __init__(self, db_path):
        self.db_path = db_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.df_movies = None
        self.movie_slugs = []
        self.user_usernames = []
        self.matrix_tensor = None
        self.movie_to_idx = {}

        self._load_data()

    def _load_data(self):
        print(f"[User-KNN-Ultra] ⚡ Initializing Pure-Tensor Engine on: {self.device} ⚡")
        conn = sqlite3.connect(self.db_path)

        df_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'",
                                       conn)
        df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce').dropna()

        query_movies = "SELECT slug, title, year, director, poster_url FROM movies"
        self.df_movies = pd.read_sql_query(query_movies, conn).set_index('slug')
        conn.close()

        print("[User-KNN-Ultra] Building DataFrame...")
        pivot_df = df_reviews.pivot_table(index='user_username', columns='movie_slug', values='rating',
                                          aggfunc='mean').fillna(0)

        # 缓存维度映射
        self.movie_slugs = pivot_df.columns.tolist()
        self.user_usernames = pivot_df.index.tolist()
        self.movie_to_idx = {slug: idx for idx, slug in enumerate(self.movie_slugs)}

        print("[User-KNN-Ultra] Locking Matrix in VRAM...")
        # 把大矩阵直接锁死在显存里
        self.matrix_tensor = torch.tensor(pivot_df.values, dtype=torch.float32).to(self.device)
        print(f"[User-KNN-Ultra] Ready! VRAM occupied: {self.matrix_tensor.nelement() * 4 / 1024 / 1024:.2f} MB\n")

    def get_recommendations(self, user_profile, top_n=10, k_neighbors=15):
        """
        全 GPU 张量化 KNN：0 循环，0 Pandas 查询，极其残暴的速度。
        """
        if self.matrix_tensor is None: return []

        # 1. 在 GPU 上直接初始化考卷张量
        target_tensor = torch.zeros(len(self.movie_slugs), dtype=torch.float32, device=self.device)
        watched_indices = []

        for slug, rating in user_profile.items():
            if slug in self.movie_to_idx:
                idx = self.movie_to_idx[slug]
                target_tensor[idx] = float(rating)
                watched_indices.append(idx)

        if target_tensor.sum() == 0: return []

        # ==========================================
        # 🚀 显卡主场：没有任何 Python 循环的纯数学压制
        # ==========================================
        with torch.no_grad():
            # a. 全局并发余弦相似度
            sim_tensor = F.cosine_similarity(target_tensor.unsqueeze(0), self.matrix_tensor, dim=1)

            # b. 直接在显存里提取 Top-K 个最像的人 (神仙操作 torch.topk)
            top_k_sims, top_k_indices = torch.topk(sim_tensor, k=k_neighbors)

            # c. 砍掉相似度 <= 0 的恶邻居
            valid_mask = top_k_sims > 0
            top_k_sims = top_k_sims[valid_mask]
            top_k_indices = top_k_indices[valid_mask]

            if len(top_k_sims) == 0: return []

            # d. 瞬间提取出这些邻居的打分矩阵 (Shape: K x 3300)
            neighbor_matrix = self.matrix_tensor[top_k_indices]

            # e. 找邻居打过分(>0) 的掩码
            has_rated_mask = neighbor_matrix > 0

            # f. 广播机制：权重相乘 (将 K 维的相似度拍到 K x 3300 的矩阵上)
            weighted_ratings = neighbor_matrix * top_k_sims.unsqueeze(1)

            # g. 沿着邻居维度(dim=0)向下拍扁求和，得到分子！
            recommendation_scores = weighted_ratings.sum(dim=0)

            # h. 同样地，把有打分的相似度求和，得到分母！
            similarity_sums = (has_rated_mask.float() * top_k_sims.unsqueeze(1)).sum(dim=0)

            # i. GPU 内部瞬间除法，完成 1-5星严格归一化 (防除零)
            final_scores = torch.zeros_like(recommendation_scores)
            valid_denoms = similarity_sums > 0
            final_scores[valid_denoms] = recommendation_scores[valid_denoms] / similarity_sums[valid_denoms]

            # j. 强行把看过的电影打分为 -999，彻底屏蔽
            final_scores[watched_indices] = -999.0

            # k. 选出得分最高的前 N 部电影
            top_n_scores, top_n_movie_indices = torch.topk(final_scores, top_n)

        # ==========================================
        # 直到所有纯数学算完了，才回传给 CPU 进行文本组装！
        # ==========================================
        top_n_scores = top_n_scores.cpu().numpy()
        top_n_movie_indices = top_n_movie_indices.cpu().numpy()

        results = []
        for score, idx in zip(top_n_scores, top_n_movie_indices):
            if score <= 0: continue
            slug = self.movie_slugs[idx]
            if slug in self.df_movies.index:
                movie_data = self.df_movies.loc[slug]
                results.append({
                    'slug': slug,
                    'title': movie_data['title'],
                    'year': movie_data['year'],
                    'director': movie_data.get('director', ''),
                    'poster_url': movie_data.get('poster_url', ''),
                    'score': float(score)
                })

        return results


# --- 测试运行 ---
if __name__ == "__main__":
    recommender = UserBasedRecommender("train_model.db")  # 替换为你的数据库
    demo_profile = {"inception": 5.0, "interstellar": 4.5}

    import time

    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)
    latency = (time.time() - start_time) * 1000
    print(f"⏱️ 推荐耗时: {latency:.2f} 毫秒")