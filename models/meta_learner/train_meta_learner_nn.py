import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from util import root_path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

META_DB = root_path() / "data/meta_dataset.db"
MODEL_SAVE_PATH = root_path() / "data/nn_meta_model.pth"


# 🚀 1. 定义大厂级 Wide & Deep 架构
class WideAndDeepMeta(nn.Module):
    def __init__(self, input_dim=5):
        super(WideAndDeepMeta, self).__init__()
        # Wide 侧：最纯粹的线性回归 (你的保底线)
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

    # 🚀 2. 目标去均值化 (Target Mean-Centering)
    # 只减均分，绝对不除以标准差！防止 MSE 梯度爆炸！
    print("[Train] Mean-Centering Targets...")
    df["Actual_Rating_Centered"] = df["Actual_Rating"] - df.groupby('user_username')["Actual_Rating"].transform(
        'mean')

    X = df[score_cols].values.astype(np.float32)
    y = df["Actual_Rating_Centered"].values.astype(np.float32)  # 目标变成了纯粹的偏差值 (如 +0.5, -1.2)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为 PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

    print("[Train] Training Wide & Deep Meta-Learner (PyTorch)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideAndDeepMeta().to(device)

    # 🚀 3. 损失函数改为 MSE
    criterion = nn.MSELoss()

    # 🚀 4. 分组优化器：Deep侧加狗链防过拟合
    optimizer = optim.Adam([
        {'params': model.wide.parameters(), 'weight_decay': 0.0},
        {'params': model.deep.parameters(), 'weight_decay': 1e-3}
    ], lr=0.0001)

    # ==========================================
    # 5. 早停机制与记录器初始化
    # ==========================================
    epochs = 800
    patience = 30
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    train_losses = []
    val_losses = []

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
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1:03d}/{epochs} - Train MSE: {train_loss:.4f} - Val MSE: {val_loss:.4f}")

        # 早停核心逻辑
        if val_loss < best_val_loss - 1e-6:  # 设置微小的宽容度，防止轻微抖动导致的假性重置
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly Stopping triggered! Val loss hasn't improved for {patience} epochs.")
            print(f"Best model remains at Epoch {best_epoch}, Val MSE: {best_val_loss:.4f}")
            break

    print("\n" + "=" * 50)
    print(f" 🏆 Best Val MSE: {best_val_loss:.4f} (Model saved to {MODEL_SAVE_PATH})")
    print("=" * 50)
    deep_weights = model.deep[0].weight.data.cpu().numpy()
    print("Deep 层权重均值:", np.mean(np.abs(deep_weights)))
    print("Deep 层权重最大值:", np.max(np.abs(deep_weights)))
    # ==========================================
    # 6. 生成双轨 Loss 曲线图
    # ==========================================
    plt.figure(figsize=(10, 6))
    actual_epochs = len(train_losses)

    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train MSE', color='#1f77b4', linewidth=2)
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Val MSE', color='#ff7f0e', linewidth=2, linestyle='--')

    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5)

    plt.title('Wide & Deep Meta-Learner: Train vs Val Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss (Z-Score Space)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('meta_learner_loss_curve.png', dpi=300, bbox_inches='tight')
    print("\nLoss curve saved as meta_learner_loss_curve.png")


if __name__ == "__main__":
    train_nn_meta_learner()