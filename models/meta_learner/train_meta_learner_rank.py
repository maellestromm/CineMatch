import sqlite3
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from util import root_path
import numpy as np

META_DB = root_path() / "data/meta_dataset.db"
MODEL_SAVE_PATH = root_path() / "data/lgbm_residual_model.txt"


def train_residual_lightgbm():
    print("[Train] Loading dataset from SQLite...")
    conn = sqlite3.connect(META_DB)
    df = pd.read_sql_query("SELECT * FROM meta_train ORDER BY user_username", conn)
    conn.close()

    if df.empty:
        raise ValueError("Dataset is empty. Run prepare_meta_data.py first.")

    score_cols = [
        "SVD_Score", "ItemKNN_Hit_Score", "AutoRec_Score",
        "ContentKNN_Hit_Score", "UserKNN_Hit_Score"
    ]

    print("[Train] Dynamically calculating query-level Z-Scores in memory...")
    # 动态计算 Z-Score
    for col in score_cols:
        df[col] = df.groupby('user_username')[col].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8)
        )

    # 🚀 目标设定：预测是否为神作 (Hit)
    y = (df["Actual_Rating"] >= 4.0).astype(int)

    # 🚀 核心剥离：把 SVD 单独抽出来作为 Base Margin
    base_margin_all = df["SVD_Score"].values

    # 🚀 特征池：彻底剔除 SVD！只剩下 4 个辅助模型
    features_no_svd = [
        "ItemKNN_Hit_Score", "AutoRec_Score",
        "ContentKNN_Hit_Score", "UserKNN_Hit_Score"
    ]
    X = df[features_no_svd]

    # 切分数据集 (注意这里要把 base_margin 也一起切分对齐)
    X_train, X_val, y_train, y_val, base_train, base_val = train_test_split(
        X, y, base_margin_all, test_size=0.2, random_state=42
    )

    print("[Train] Training LightGBM with SVD Residual Learning...")

    # 初始化二分类树模型 (克制参数，因为只是做微调，防过拟合)
    model = lgb.LGBMClassifier(
        n_estimators=100,  # 树不需要太多
        learning_rate=0.03,  # 步子迈小一点
        max_depth=4,  # 极浅的树，严防过拟合
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='binary',
        metric='auc'
    )

    # 🚀 见证奇迹的时刻：注入 init_score！
    model.fit(
        X_train, y_train,
        init_score=base_train,  # 训练集的 SVD 基座分
        eval_set=[(X_val, y_val)],
        eval_init_score=[base_val],  # 验证集的 SVD 基座分
    )

    # 验证 AUC (注意：使用自带的 predict_proba 时，LightGBM 会自动处理树内部保存的验证集，但为了严谨我们手动测试)
    # 因为 scikit-learn API 的 predict_proba 不太好传 init_score，我们直接用底层 booster 算 raw_score

    # 1. 树模型输出验证集的残差 (纯打补丁分)
    val_tree_residuals = model.booster_.predict(X_val, raw_score=True)

    # 2. 物理合并：验证集的 SVD 基座分 + 残差
    val_raw_predictions = base_val + val_tree_residuals
    # 将 raw_score (对数几率) 转换为概率算 AUC
    val_probs = 1 / (1 + np.exp(-val_raw_predictions))
    auc_score = roc_auc_score(y_val, val_probs)

    print("\n" + "=" * 50)
    print(f" 🏆 Residual LightGBM Validation AUC: {auc_score:.4f}")
    print("=" * 50)

    # 保存原生 Booster (因为 scikit-learn wrapper 对 init_score 的推断支持不够灵活)
    model.booster_.save_model(str(MODEL_SAVE_PATH))
    print(f"[Train] Saved Native Booster model to {MODEL_SAVE_PATH}!")


if __name__ == "__main__":
    train_residual_lightgbm()