import sqlite3
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

# 模型分数列：减 user_avg + 除用户内 std
MODEL_SCORE_COLS = [
    "AutoRec_Score",
    "UserKNN_RMSE_Score", "UserKNN_Hit_Score",
    "ItemKNN_RMSE_Score",  "ItemKNN_Hit_Score",
    "SVD_Score",
    "ContentKNN_Hit_Score", "ContentKNN_RMSE_Score",
]

# 统计特征列：固定变换，不依赖训练数据统计量
STAT_COLS = [
    "Movie_Rating_Count", "Movie_Avg", "Movie_Std",
    "User_Rating_Count",  "User_Avg",  "User_Std",
    "Release_Year",
]

ALL_FEATURE_COLS = MODEL_SCORE_COLS + STAT_COLS
INPUT_DIM = len(ALL_FEATURE_COLS)  # 15


class MetaMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def transform_stat_cols(df):
    """
    固定变换，不依赖任何训练数据统计量。
    训练和推理用完全相同的公式。
    """
    df = df.copy()
    df["Movie_Rating_Count"] = np.log1p(df["Movie_Rating_Count"])
    df["User_Rating_Count"]  = np.log1p(df["User_Rating_Count"])
    df["Release_Year"]       = (df["Release_Year"] - 1900) / 100
    return df


def normalize_model_scores(df):
    """
    模型分数：减 user_avg + 除用户内 std。
    保留 0 → 负数作为无预测信号。
    """
    df = df.copy()
    user_avgs = df.groupby('user_username')["Actual_Rating"].transform('mean')
    df["Actual_Rating_Centered"] = df["Actual_Rating"] - user_avgs

    for col in MODEL_SCORE_COLS:
        df[col] = df[col] - user_avgs
        user_stds = (
            df.groupby('user_username')[col]
            .transform('std')
            .fillna(1.0)
            .replace(0, 1.0)
            .clip(lower=0.1)
        )
        df[col] = df[col] / user_stds

    return df, user_avgs


def train():
    print("[1/5] Loading data...")
    conn = sqlite3.connect(META_DB)
    df = pd.read_sql_query("SELECT * FROM meta_train", conn)
    conn.close()
    print(f"      {len(df):,} rows, {df['user_username'].nunique():,} users")

    print("[2/5] Normalizing...")
    df = transform_stat_cols(df)
    df, _ = normalize_model_scores(df)

    for col in ALL_FEATURE_COLS:
        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()
        print(f"      {col}: NaN={nan_count}, Inf={inf_count}, "
              f"min={df[col].min():.2f}, max={df[col].max():.2f}")

    if df[ALL_FEATURE_COLS].isna().any().any():
        raise ValueError("NaN detected, check data.")

    print("[3/5] Splitting by user...")
    all_users = df['user_username'].unique()
    train_users, val_users = train_test_split(all_users, test_size=0.2, random_state=42)
    train_df = df[df['user_username'].isin(train_users)]
    val_df   = df[df['user_username'].isin(val_users)]
    print(f"      Train: {len(train_df):,} rows ({len(train_users):,} users)")
    print(f"      Val:   {len(val_df):,} rows ({len(val_users):,} users)")

    X_train = train_df[ALL_FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["Actual_Rating_Centered"].values.astype(np.float32)
    X_val   = val_df[ALL_FEATURE_COLS].values.astype(np.float32)
    y_val   = val_df["Actual_Rating_Centered"].values.astype(np.float32)

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

    model = MetaMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(device), by.to(device)
                val_loss += criterion(model(bX), by).item() * bX.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:03d}/{epochs} | "
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
            print(f"\nEarly stopping at epoch {epoch+1}.")
            print(f"Best: epoch {best_epoch}, Val MSE: {best_val_loss:.4f}")
            break

    print(f"\n[5/5] Done. Best Val MSE: {best_val_loss:.4f} (epoch {best_epoch})")

    plt.figure(figsize=(10, 5))
    actual_epochs = len(train_losses)
    plt.plot(range(1, actual_epochs+1), train_losses, label='Train MSE', linewidth=2)
    plt.plot(range(1, actual_epochs+1), val_losses,   label='Val MSE',   linewidth=2, linestyle='--')
    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Meta-Learner MLP: Train vs Val Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('meta_learner_loss_curve.png', dpi=150)
    print("Loss curve saved.")


if __name__ == "__main__":
    train()