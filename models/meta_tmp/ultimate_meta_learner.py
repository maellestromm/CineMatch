import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from util import root_path


def train_ultimate_meta_learner():
    db_path = root_path() / "data/meta_dataset.db"
    print("📥 Loading OOF validation data...")
    conn = sqlite3.connect(db_path)
    # 读取你辛苦做出来的 43 万条纯净 OOF 数据
    df = pd.read_sql_query("SELECT * FROM meta_train", conn)
    conn.close()

    # 只提取我们需要的 5 个核心基座分数
    score_cols = ["SVD_Score", "AutoRec_Score", "ItemKNN_Hit_Score", "UserKNN_Hit_Score", "ContentKNN_Hit_Score"]
    X = df[score_cols].values
    y_true = df["Actual_Rating"].values

    # 将 4.0 分以上的定义为“用户真正喜欢的神作 (Hit)”
    y_binary = (y_true >= 4.0).astype(int)

    # =========================================================
    # STAGE 1: 暴力寻找排序最优解 (Maximize AUC for Hit Rate)
    # =========================================================
    print(f"🚀 [STAGE 1] Searching for Golden Ranking Weights (10,000 iterations)...")
    best_auc = 0
    best_weights = None

    # 用 Dirichlet 分布瞬间生成 10000 组和为 1 的随机权重
    np.random.seed(42)
    random_weights = np.random.dirichlet(np.ones(5), size=10000)

    # 矩阵乘法瞬间算完 10000 种组合的分数
    all_preds = X @ random_weights.T  # shape: (432740, 10000)

    for i in range(10000):
        # 计算每一组权重的排序能力 (ROC-AUC)
        auc = roc_auc_score(y_binary, all_preds[:, i])
        if auc > best_auc:
            best_auc = auc
            best_weights = random_weights[i]

    print(f"✅ Best Ranking AUC Achieved: {best_auc:.4f}")

    # =========================================================
    # STAGE 2: 线性拉伸校准 RMSE (Minimize MSE without altering Rank)
    # =========================================================
    print(f"📐 [STAGE 2] Calibrating to minimize RMSE...")
    # 使用刚刚找到的最优排序权重，计算出最原始的融合分
    raw_golden_scores = X @ best_weights

    # 用一个一元线性回归（只有一个特征）去拟合真实分数
    calibrator = LinearRegression()
    calibrator.fit(raw_golden_scores.reshape(-1, 1), y_true)

    slope = calibrator.coef_[0]
    intercept = calibrator.intercept_

    # 验证校准后的 RMSE
    final_preds = slope * raw_golden_scores + intercept
    final_rmse = np.sqrt(np.mean((final_preds - y_true) ** 2))
    print(f"✅ Calibrated OOF RMSE: {final_rmse:.4f}")

    # =========================================================
    # 输出前端部署用的终极常量
    # =========================================================
    print("\n" + "=" * 50)
    print("🏆 THE ULTIMATE FRONTEND CONSTANTS 🏆")
    print("=" * 50)
    for name, w in zip(score_cols, best_weights):
        # 我们把 slope 直接乘进权重里，前端连乘法都省了一步！
        final_w = w * slope
        print(f"const WEIGHT_{name.split('_')[0].upper()} = {final_w:.8f};")

    print(f"const INTERCEPT = {intercept:.4f};")
    print("=" * 50)
    print("""
前端 JS 计算公式:
let final_score = (SVD * WEIGHT_SVD) + 
                  (AutoRec * WEIGHT_AUTOREC) + 
                  (ItemKNN * WEIGHT_ITEMKNN) + 
                  (UserKNN * WEIGHT_USERKNN) + 
                  (ContentKNN * WEIGHT_CONTENTKNN) + 
                  INTERCEPT;
final_score = Math.max(0.5, Math.min(5.0, final_score));
""")


if __name__ == "__main__":
    train_ultimate_meta_learner()