import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAggRecModel(nn.Module):
    def __init__(self, num_items, item_feature_dim, embed_dim=64):
        super().__init__()
        self.item_embed = nn.Embedding(num_items, embed_dim)
        # 将静态内容特征(分类、年份等)与ID特征融合
        self.item_fusion = nn.Linear(embed_dim + item_feature_dim, embed_dim)

        self.rating_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_item_rep(self, item_indices, item_features):
        emb = self.item_embed(item_indices)
        return F.relu(self.item_fusion(torch.cat([emb, item_features], dim=-1)))

    def compute_user_vector(self, hist_indices, hist_features, hist_ratings):
        # 提取历史电影的表征
        hist_reps = self.get_item_rep(hist_indices, hist_features)

        # 评分中心化 (假设满分5分，以3分为界限，好评加正权重，差评加负权重)
        weights = (hist_ratings - 3.0).unsqueeze(-1)

        # 加权平均得到动态用户向量
        user_vec = torch.sum(hist_reps * weights, dim=1) / (torch.sum(torch.abs(weights), dim=1) + 1e-8)
        return user_vec

    def forward(self, hist_indices, hist_features, hist_ratings, target_indices, target_features):
        user_vec = self.compute_user_vector(hist_indices, hist_features, hist_ratings)
        target_rep = self.get_item_rep(target_indices, target_features)

        x = torch.cat([user_vec, target_rep], dim=-1)
        pred = self.rating_mlp(x)
        return pred.squeeze()