import sqlite3
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from util import root_path

META_DB = root_path() / "data/meta_dataset.db"
MODEL_SAVE_PATH = root_path() / "data/nn_meta_model.pth"
NORM_SAVE_PATH = root_path() / "data/nn_meta_norm.json"

SCORE_COLS = [
    "SVD_Score", "ItemKNN_Hit_Score", "AutoRec_Score",
    "ContentKNN_Hit_Score", "UserKNN_Hit_Score"
]


class WideAndDeepMeta(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.wide(x) + self.deep(x)


def normalize(df):
    """
    对每个用户的每列分数做归一化：
    1. 减去用户真实评分均值（消除用户偏置，同时让 0 变成负数作为"无预测"信号）
    2. 除以该用户该列的 std（对齐各模型量纲）
    返回归一化后的 df，以及每列的全局 std（推理时用）
    """
    user_avgs = df.groupby('user_username')["Actual_Rating"].transform('mean')
    user_stds = (df.groupby('user_username')["Actual_Rating"].transform('std')
                 .fillna(1.0).replace(0, 1.0).clip(lower=0.1))  # 防止 std 过小导致数值爆炸
    df["Actual_Rating_Centered"] = (df["Actual_Rating"] - user_avgs) / user_stds

    col_global_stds = {}
    for col in SCORE_COLS:
        mask = (df[col] == 0)
        df.loc[mask, col] = user_avgs[mask]
        # Step 1: 减用户均值（0 → 负数，作为有效的"无预测"信号保留）
        col_avgs = df.groupby('user_username')[col].transform('mean')
        df[col] = df[col] - col_avgs

        # Step 2: 除以该用户该列的 std，对齐各模型的量纲
        col_stds = (
            df.groupby('user_username')[col]
            .transform('std')
            .fillna(1.0)  # 单行用户 std=NaN
            .replace(0, 1.0)  # 全相同值 std=0
            .clip(lower=0.1)  # 防止 std 过小导致数值爆炸
        )
        df[col] = df[col] / col_stds

        # 保存全局 std 供推理时做健全性检查（不用于归一化）
        col_global_stds[col] = float(df[col].std())

    return df, user_avgs, col_global_stds


def train():
    print("[1/5] Loading data...")
    conn = sqlite3.connect(META_DB)
    df = pd.read_sql_query("SELECT * FROM meta_train", conn)
    conn.close()
    print(f"      {len(df):,} rows, {df['user_username'].nunique():,} users")

    print("[2/5] Normalizing...")
    df, _, col_global_stds = normalize(df)

    # 检查归一化结果
    for col in SCORE_COLS:
        nan_count = df[col].isna().sum()
        print(f"      {col}: NaN={nan_count}, "
              f"min={df[col].min():.2f}, max={df[col].max():.2f}")
    if df[SCORE_COLS].isna().any().any():
        raise ValueError("NaN detected after normalization, check data.")

    print("[3/5] Splitting by user (no leakage)...")
    all_users = df['user_username'].unique()
    train_users, val_users = train_test_split(all_users, test_size=0.2, random_state=42)

    train_df = df[df['user_username'].isin(train_users)]
    val_df = df[df['user_username'].isin(val_users)]
    print(f"      Train: {len(train_df):,} rows ({len(train_users):,} users)")
    print(f"      Val:   {len(val_df):,} rows ({len(val_users):,} users)")

    X_train = train_df[SCORE_COLS].values.astype(np.float32)
    y_train = train_df["Actual_Rating_Centered"].values.astype(np.float32)
    X_val = val_df[SCORE_COLS].values.astype(np.float32)
    y_val = val_df["Actual_Rating_Centered"].values.astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
        batch_size=2048, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1)),
        batch_size=2048, shuffle=False
    )

    print("[4/5] Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Device: {device}")

    model = WideAndDeepMeta().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.wide.parameters(), 'weight_decay': 0.0},
        {'params': model.deep.parameters(), 'weight_decay': 1e-3},
    ], lr=1e-3)

    # 验证 loss 连续 5 个 epoch 没改善就降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    epochs = 300
    patience = 20
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bX.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1:03d}/{epochs} | "
              f"Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            print(f"Best: epoch {best_epoch}, Val MSE: {best_val_loss:.4f}")
            break

    print(f"\n[5/5] Saving normalization params to {NORM_SAVE_PATH}...")
    # 推理时需要知道用哪些列、clip 的下限是多少
    norm_params = {
        "score_cols": SCORE_COLS,
        "std_clip_lower": 0.1,
        "col_global_stds": col_global_stds,  # 仅供参考，不用于推理归一化
    }
    with open(NORM_SAVE_PATH, "w") as f:
        json.dump(norm_params, f, indent=2)

    # Loss 曲线
    plt.figure(figsize=(10, 5))
    actual_epochs = len(train_losses)
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train MSE', linewidth=2)
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Val MSE', linewidth=2, linestyle='--')
    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Wide & Deep Meta-Learner')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('meta_learner_loss_curve.png', dpi=150)
    print("Loss curve saved.")


if __name__ == "__main__":
    train()
