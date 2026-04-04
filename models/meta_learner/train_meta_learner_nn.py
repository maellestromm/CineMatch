import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from util import root_path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

META_DB = root_path() / "data/meta_dataset.db"
# 🚀 PyTorch 模型后缀通常用 .pth
MODEL_SAVE_PATH = root_path() / "data/mlp_meta_model.pth"


# 🚀 1. 定义大厂级 Wide & Deep 架构
class WideAndDeepMeta(nn.Module):
    def __init__(self, input_dim=5):
        super(WideAndDeepMeta, self).__init__()
        # Wide 侧：最纯粹的线性回归 (你的 78.29% 保底线)
        self.wide = nn.Linear(input_dim, 1)

        # Deep 侧：负责捕捉非线性残差，极度精简防止过拟合
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        # 物理合并：线性底分 + 非线性微调
        return self.wide(x) + self.deep(x)


def train_nn_meta_learner():
    print("[Train] Loading 430k dataset from SQLite...")
    conn = sqlite3.connect(META_DB)
    df = pd.read_sql_query("SELECT * FROM meta_train ORDER BY user_username", conn)
    conn.close()

    score_cols = [
        "SVD_Score", "ItemKNN_Hit_Score", "AutoRec_Score",
        "ContentKNN_Hit_Score", "UserKNN_Hit_Score"
    ]

    print("[Train] Dynamically calculating query-level Z-Scores in memory...")
    for col in score_cols:
        df[col] = df.groupby('user_username')[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )

    # 🚀 2. 核心修正：取消 0/1 分类，还原为连续浮点数！
    X = df[score_cols].values.astype(np.float32)
    y = df["Actual_Rating"].values.astype(np.float32)  # 保持 0.5 ~ 5.0 的真实刻度

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为 PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    print("[Train] Training Wide & Deep Meta-Learner (PyTorch)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideAndDeepMeta().to(device)

    # 🚀 3. 损失函数改为 MSE (均方误差)
    criterion = nn.MSELoss()

    # 🚀 4. 分组优化器：对 Deep 侧施加 Weight Decay(L2正则)，逼迫它乖乖听 Wide 侧的话
    optimizer = optim.Adam([
        {'params': model.wide.parameters(), 'weight_decay': 0.0},  # 宽侧不限制
        {'params': model.deep.parameters(), 'weight_decay': 1e-3}  # 深侧加惩罚
    ], lr=0.001)

    # 简易训练循环 (20 Epoch)
    epochs = 200
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证集评估
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{epochs} - Train MSE: {train_loss:.4f} - Val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\n" + "=" * 50)
    print(f" 🏆 Best Val MSE: {best_val_loss:.4f} (Model saved to {MODEL_SAVE_PATH})")
    print("=" * 50)


if __name__ == "__main__":
    train_nn_meta_learner()