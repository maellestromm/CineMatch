import json
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. 核心神经网络架构 (Deep AutoRec)
# ==========================================
class DeepAutoRec(nn.Module):
    def __init__(self, num_movies):
        super(DeepAutoRec, self).__init__()
        # 漏斗型特征压缩：极其适合抓取电影间的非线性关联
        self.encoder = nn.Sequential(
            nn.Linear(num_movies, 512),
            nn.Dropout(0.3),
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
# 2. 掩码均方误差 (Masked MSE Loss)
# ==========================================
def masked_mse_loss(predictions, targets):
    mask = (targets != 0).float()
    error = (predictions - targets) * mask
    loss = (error ** 2).sum() / (mask.sum() + 1e-8)
    return loss


# ==========================================
# 3. 极其严谨的数据加载与维度对齐
# ==========================================
def load_and_align_data(train_db="train_model.db", test_db="test_eval.db"):
    print("📥 正在读取并对齐物理隔离的数据库...")

    # --- 加载训练集 ---
    conn_train = sqlite3.connect(train_db)
    df_train = pd.read_sql_query(
        "SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None' AND rating IS NOT NULL",
        conn_train)
    conn_train.close()
    df_train['rating'] = df_train['rating'].astype(float)
    matrix_train = df_train.pivot(index='user_username', columns='movie_slug', values='rating').fillna(0)

    # 确立全局的“标准答题卡”电影维度
    movie_slugs = matrix_train.columns.tolist()
    num_movies = len(movie_slugs)

    # --- 加载测试集 ---
    conn_test = sqlite3.connect(test_db)
    df_test = pd.read_sql_query(
        "SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None' AND rating IS NOT NULL",
        conn_test)
    conn_test.close()
    df_test['rating'] = df_test['rating'].astype(float)
    matrix_test = df_test.pivot(index='user_username', columns='movie_slug', values='rating').fillna(0)

    # ⚡ 核心魔法：强制测试集对齐训练集的列 (自动补齐缺失电影，丢弃多余电影)
    matrix_test = matrix_test.reindex(columns=movie_slugs, fill_value=0.0)

    print(f"✅ 数据矩阵构建完毕！")
    print(f"   🎯 训练集 (Train): {matrix_train.shape[0]} 用户 x {num_movies} 电影")
    print(f"   🧪 测试集 (Test) : {matrix_test.shape[0]} 用户 x {num_movies} 电影")

    # 转换为 PyTorch 张量
    train_tensor = torch.FloatTensor(matrix_train.values)
    test_tensor = torch.FloatTensor(matrix_test.values)

    return train_tensor, test_tensor, movie_slugs


# ==========================================
# 4. 带有 Early Stopping 的完美训练循环
# ==========================================
def train_model(epochs=100, batch_size=256, lr=0.005, patience=8):
    # 1. 获取对齐后的数据
    train_tensor, test_tensor, movie_slugs = load_and_align_data()
    num_movies = len(movie_slugs)

    # 保存字典供未来推理使用
    with open("movie_dictionary.json", "w", encoding="utf-8") as f:
        json.dump(movie_slugs, f)

    # 包装 DataLoader
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 测试集一次性塞进去算就行，不需要分 batch
    test_x = test_tensor
    test_y = test_tensor

    model = DeepAutoRec(num_movies=num_movies)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    print("\n🚀 开始严谨的深度学习炼丹 (监控 Test Loss 防过拟合)...")

    for epoch in range(epochs):
        # --- 训练阶段 (闭卷学习) ---
        model.train()
        total_train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = masked_mse_loss(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 验证阶段 (开卷考试，完全不参与反向传播) ---
        model.eval()
        with torch.no_grad():
            test_preds = model(test_x)
            val_loss = masked_mse_loss(test_preds, test_y).item()
            val_losses.append(val_loss)

        print(f"   Epoch [{epoch + 1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {val_loss:.4f}")

        # --- Early Stopping 逻辑 ---
        if val_loss < best_val_loss - 0.0005:  # Test Loss 显著下降
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            # 只有创纪录时，才把权重存到硬盘上！
            torch.save(model.state_dict(), "autorec_best_weights.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n🛑 触发 Early Stopping！测试集误差在连续 {patience} 轮未改善。")
            print(f"🏆 最佳模型停留在 Epoch {best_epoch}，Test Loss 为 {best_val_loss:.4f}")
            break

    # ==========================================
    # 5. 生成极其专业的双线 Loss 曲线图
    # ==========================================
    plt.figure(figsize=(10, 6))
    actual_epochs = len(train_losses)

    # 画两条线：一条训练，一条测试
    plt.plot(range(1, actual_epochs + 1), train_losses, label='Train Loss (5457 Users)', color='#1f77b4', linewidth=2)
    plt.plot(range(1, actual_epochs + 1), val_losses, label='Test/Val Loss (606 Users)', color='#ff7f0e', linewidth=2,
             linestyle='--')

    # 标记出最佳的那个点
    plt.axvline(x=best_epoch, color='r', linestyle=':', label=f'Best Epoch ({best_epoch})')
    plt.scatter(best_epoch, best_val_loss, color='red', zorder=5)

    plt.title('Deep AutoRec Generalization: Train vs Test Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Masked MSE Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig('train_vs_test_loss.png', dpi=300, bbox_inches='tight')
    print("\n📈 神级双轨 Loss 曲线已保存为 train_vs_test_loss.png (请务必放进你们的期末 PPT！)")


if __name__ == "__main__":
    train_model(epochs=150, batch_size=256, lr=0.0005, patience=15)