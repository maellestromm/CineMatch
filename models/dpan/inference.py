import pickle

import torch

from models.dpan.dpan_model import DynamicAggRecModel
from util import root_path


class Recommender:
    def __init__(self, db_path):
        """
        初始化推荐器，加载元数据和模型参数
        注意: 这里我们只使用 db_path 作为路径参考，实际为了提速读取的是 pkl 和 pth
        """
        model_path = root_path() / "data/dynamic_rec_model.pth"
        meta_path = root_path() / "data/model_meta.pkl"

        # 1. 加载元数据 (词典和特征矩阵)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.slug2idx = meta['slug2idx']
        self.idx2movie = meta['idx2movie']
        self.item_features = meta['item_features_tensor']
        self.num_items = len(self.slug2idx)

        # 2. 初始化并加载模型
        self.model = DynamicAggRecModel(num_items=self.num_items, item_feature_dim=meta['feat_dim'])
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def get_recommendations(self, profile_dict, top_n=10, max_history=5):
        """
        max_history=5: 强制模型只看用户打分最高的 5 部电影，保持向量纯度！
        """
        # 1. 核心修复：按评分从高到低排序，切断低分干扰和长序列平均效应
        sorted_items = sorted(profile_dict.items(), key=lambda x: float(x[1]), reverse=True)
        best_items = sorted_items[:max_history]

        hist_indices = []
        hist_ratings = []

        # 2. 只使用精选的 Best Items 构建用户向量
        for slug, rating in best_items:
            if slug in self.slug2idx:
                hist_indices.append(self.slug2idx[slug])
                hist_ratings.append(float(rating))

        if not hist_indices:
            return []

        with torch.no_grad():
            # 1. 构造输入的 Tensor
            h_idx = torch.tensor(hist_indices, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            h_rat = torch.tensor(hist_ratings, dtype=torch.float).unsqueeze(0)  # (1, seq_len)
            h_feat = self.item_features[h_idx]  # (1, seq_len, feat_dim)

            # 2. 瞬间计算出该用户的动态偏好向量
            user_vec = self.model.compute_user_vector(h_idx, h_feat, h_rat)  # (1, embed_dim)

            # 3. 获取所有电影的特征表征
            all_idx = torch.arange(self.num_items)  # (3334,)
            all_feat = self.item_features  # (3334, feat_dim)
            target_reps = self.model.get_item_rep(all_idx, all_feat)  # (3334, embed_dim)

            # 4. 矩阵拼接，全连接层批量打分
            # 将 (1, embed_dim) 扩张为 (3334, embed_dim)
            user_vec_expanded = user_vec.expand(self.num_items, -1)
            x = torch.cat([user_vec_expanded, target_reps], dim=-1)

            # 瞬间得出 3334 部电影的预测分数
            all_scores = self.model.rating_mlp(x).squeeze()

            # 5. 排序与过滤
        # 将张量转为列表，附加 ID
        scores_list = [(idx, score.item()) for idx, score in enumerate(all_scores)]
        # 过滤掉用户已经看过的电影
        seen_indices = set(hist_indices)
        scores_list = [s for s in scores_list if s[0] not in seen_indices]

        # 降序排列并截取 Top K
        scores_list.sort(key=lambda x: x[1], reverse=True)
        top_items = scores_list[:top_n]

        # 6. 构造返回格式
        results = []
        for idx, score in top_items:
            movie_info = self.idx2movie[idx]
            results.append({
                "slug": movie_info['slug'],
                "title": movie_info['title'],
                "year": movie_info['year'],
                "score": score
            })

        return results


# ================= 测试你的 API =================
if __name__ == "__main__":
    # 完全按照你的要求运作
    recommender = Recommender(root_path() / "data/train_model.db")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    recs = recommender.get_recommendations(demo_profile)
    for r in recs:
        print(f"[{r['score']:.4f}] {r['title']} ({r['year']})")