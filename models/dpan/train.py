import pickle
import sqlite3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from models.dpan.dpan_model import DynamicAggRecModel
from util import root_path

DB_PATH = root_path() / "data/train_model.db"
MODEL_PATH = root_path() / "data/dynamic_rec_model.pth"
META_PATH = root_path() / "data/model_meta.pkl"

# ================= 核心修改 1：设备检测 =================
# 自动检测并使用 CUDA (NVIDIA) 或 MPS (Apple Silicon)，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Initializing training on device: {device}")


class RecDataset(Dataset):
    def __init__(self, data, item_features_tensor):
        self.data = data
        self.item_features = item_features_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        hist_idx = torch.tensor(row['hist_items'], dtype=torch.long)
        hist_feat = self.item_features[hist_idx]
        hist_rat = torch.tensor(row['hist_ratings'], dtype=torch.float)

        tgt_idx = torch.tensor(row['target_item'], dtype=torch.long)
        tgt_feat = self.item_features[tgt_idx]
        tgt_rat = torch.tensor(row['target_rating'], dtype=torch.float)

        return hist_idx, hist_feat, hist_rat, tgt_idx, tgt_feat, tgt_rat


def prepare_data():
    conn = sqlite3.connect(DB_PATH)
    movies_df = pd.read_sql("SELECT slug, title, year, genres, rating_average FROM movies", conn)
    reviews_df = pd.read_sql(
        "SELECT user_username, movie_slug, CAST(rating AS REAL) as rating,review_date FROM reviews WHERE rating IS NOT NULL", conn)
    conn.close()

    slug2idx = {slug: idx for idx, slug in enumerate(movies_df['slug'])}
    idx2movie = {idx: row.to_dict() for idx, row in movies_df.iterrows()}

    movies_df['genres'] = movies_df['genres'].fillna('').apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(movies_df['genres'])

    scaler = MinMaxScaler()
    num_features = scaler.fit_transform(
        movies_df[['year', 'rating_average']].fillna(movies_df.median(numeric_only=True)))

    item_features = np.hstack([genres_encoded, num_features])
    item_features_tensor = torch.tensor(item_features, dtype=torch.float32)

    reviews_df['item_idx'] = reviews_df['movie_slug'].map(slug2idx)
    reviews_df = reviews_df.dropna(subset=['item_idx']).sort_values('review_date')

    training_data = []
    for user, group in reviews_df.groupby('user_username'):
        items = group['item_idx'].tolist()
        ratings = group['rating'].tolist()

        if len(items) < 4: continue

        for i in range(3, len(items)):
            hist_items = items[i - 3:i]
            hist_ratings = ratings[i - 3:i]
            target_item = items[i]
            target_rating = ratings[i]

            training_data.append({
                'hist_items': hist_items,
                'hist_ratings': hist_ratings,
                'target_item': target_item,
                'target_rating': target_rating
            })

    return training_data, item_features_tensor, slug2idx, idx2movie, item_features.shape[1]


def train():
    print("Preparing data and features...")
    training_data, item_features_tensor, slug2idx, idx2movie, feat_dim = prepare_data()

    # 增加 num_workers 可以加速数据加载，但由于数据量不大，设为 0 即可
    dataset = RecDataset(training_data, item_features_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    num_items = len(slug2idx)

    # ================= 核心修改 2：将模型推入 GPU =================
    model = DynamicAggRecModel(num_items=num_items, item_feature_dim=feat_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    criterion_mse = nn.MSELoss()
    criterion_rank = nn.MarginRankingLoss(margin=0.5)  # 强制正样本比负样本高 0.5 分
    print(f"Start Training on {len(training_data)} samples...")
    model.train()
    for epoch in range(10):
        total_loss = 0
        for h_idx, h_feat, h_rat, t_idx, t_feat, t_rat in dataloader:
            neg_idx = t_idx[torch.randperm(t_idx.size(0))]
            neg_feat = item_features_tensor[neg_idx]
            # ================= 核心修改 3：将每一个 Batch 的数据推入 GPU =================
            h_idx, h_feat, h_rat = h_idx.to(device), h_feat.to(device), h_rat.to(device)
            t_idx, t_feat, t_rat = t_idx.to(device), t_feat.to(device), t_rat.to(device)
            neg_idx, neg_feat = neg_idx.to(device), neg_feat.to(device)

            optimizer.zero_grad()

            # 1. 正样本预测 (用户真正看的电影)
            pos_preds = model(h_idx, h_feat, h_rat, t_idx, t_feat)

            # 2. 构造负样本 (随机抽一部用户没看的电影)
            # 简单实现：将 t_idx 随机打乱作为负样本


            # 负样本预测
            neg_preds = model(h_idx, h_feat, h_rat, neg_idx, neg_feat)

            # 3. 联合 Loss 优化
            loss_mse = criterion_mse(pos_preds, t_rat)

            # Margin Ranking Loss 要求给出 target 标签：1 表示 pos 应该大于 neg
            target_ones = torch.ones_like(pos_preds)
            loss_rank = criterion_rank(pos_preds, neg_preds, target_ones)

            # 权重叠加：0.2 顾及 MSE，0.8 全力冲刺排序！
            total_loss = 0.2 * loss_mse + 0.8 * loss_rank

            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1:02d} | MSE Loss: {total_loss / len(dataloader):.4f}")

    # 保存模型时不需要指定设备，PyTorch 会自动保存权重
    torch.save(model.state_dict(), MODEL_PATH)
    with open(META_PATH, 'wb') as f:
        pickle.dump({
            'slug2idx': slug2idx,
            'idx2movie': idx2movie,
            'item_features_tensor': item_features_tensor,  # 这个 tensor 依然是 CPU 上的，方便推理
            'feat_dim': feat_dim
        }, f)
    print("✅ Training complete. Model and metadata saved.")


if __name__ == "__main__":
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    train()
