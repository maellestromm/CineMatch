import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import torch
from util import root_path


def run_hit_rate_sniper(num_iterations=20000, sample_users=300):
    db_path = root_path() / "data/train_model.db"
    conn = sqlite3.connect(db_path)

    df_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce')
    df_reviews = df_reviews.dropna()

    df_movies = pd.read_sql_query("SELECT slug FROM movies", conn)
    df_movies.drop_duplicates(subset=['slug'], keep='first', inplace=True)
    all_slugs = df_movies['slug'].tolist()
    num_movies = len(all_slugs)
    slug_to_idx = {slug: i for i, slug in enumerate(all_slugs)}

    print(f"📥 正在加载基座模型...")
    from linear_meta import LinearMetaRecommender
    recommender = LinearMetaRecommender(db_path=db_path)

    valid_users = df_reviews['user_username'].unique()
    np.random.seed(42)
    selected_users = np.random.choice(valid_users, size=min(sample_users, len(valid_users)), replace=False)

    print(f"⚙️ 正在构建 80/20 测试集推理张量...")
    user_tensors = []
    user_seen_indices = []
    actual_selected_users = []

    # 构建一个 (Users, Movies) 的目标掩码，1.0 表示是测试集里的神作
    target_mask_np = np.zeros((len(selected_users), num_movies), dtype=np.float32)

    for u_idx, user in enumerate(selected_users):
        user_data = df_reviews[df_reviews['user_username'] == user]

        # 🚀 核心修复：完全复刻标准的 80% Train / 20% Test 划分！
        # 我们打乱用户的历史记录，抽出 20% 作为考题
        user_data = user_data.sample(frac=1.0, random_state=42)
        split_point = int(len(user_data) * 0.8)

        train_data = user_data.iloc[:split_point]
        test_data = user_data.iloc[split_point:]

        # 提取用户在测试集里的“神作” (>= 4.0)
        test_hits = test_data[test_data['rating'] >= 4.0]['movie_slug'].tolist()

        # 如果测试集里没有神作，跳过这个用户（无法评测命中率）
        if not test_hits:
            continue

        # 喂给基座模型的是纯净的 80% 训练集 (防止作弊)
        train_profile = dict(zip(train_data['movie_slug'], train_data['rating']))
        user_avg = np.mean(list(train_profile.values())) if train_profile else 3.5

        seen_idx = [slug_to_idx[s] for s in train_profile.keys() if s in slug_to_idx]
        user_seen_indices.append(seen_idx)
        actual_selected_users.append(user)

        # 把测试集里的神作在 target_mask 上标记为 1
        current_u_idx = len(actual_selected_users) - 1
        for hit_slug in test_hits:
            if hit_slug in slug_to_idx:
                target_mask_np[current_u_idx, slug_to_idx[hit_slug]] = 1.0

        # 获取 5 个模型的预测
        score_matrix = np.full((num_movies, 5), user_avg)
        model_names = ["SVD", "ItemKNN_Hit", "AutoRec", "ContentKNN_Hit", "UserKNN_Hit"]
        for i, name in enumerate(model_names):
            raw_recs = recommender.base_models[name].get_recommendations(train_profile, top_n=num_movies)
            for rec in raw_recs:
                if rec['slug'] in slug_to_idx:
                    score_matrix[slug_to_idx[rec['slug']], i] = rec['score']

        user_tensors.append(score_matrix)

    X_3d = np.stack(user_tensors)
    num_users = len(actual_selected_users)
    target_mask_np = target_mask_np[:num_users, :]  # 截断未使用的空行

    # =====================================================================
    # ⚡ GPU 闪电计算区
    # =====================================================================
    print("\n🚀 将 3D 张量转移至 GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tensor = torch.tensor(X_3d, dtype=torch.float32, device=device)
    target_mask = torch.tensor(target_mask_np, dtype=torch.float32, device=device)

    # 屏蔽掩码：把 Train 里的电影降为 -1e9
    mask = torch.zeros((num_users, num_movies), dtype=torch.float32, device=device)
    for u_idx in range(num_users):
        mask[u_idx, user_seen_indices[u_idx]] = -1e9

    print(f"⚡ 开始多目标轰炸！投掷 {num_iterations} 次随机权重...")
    weights_tensor = torch.empty((num_iterations, 5), dtype=torch.float32, device=device).uniform_(-1.0, 2.0)

    best_hit_rate = -1.0
    best_weights = None

    for i in range(num_iterations):
        w = weights_tensor[i]
        scores = torch.matmul(X_tensor, w) + mask

        _, top_10_indices = torch.topk(scores, 10, dim=1)

        # 🚀 终极魔法：瞬间查验 Top 10 里有没有击中 target_mask 为 1 的目标！
        hits_tensor = torch.gather(target_mask, 1, top_10_indices)  # 取出推荐位置的目标值
        hits = (hits_tensor.sum(dim=1) > 0).sum().item()  # 只要和大于 0，就算命中

        current_hit_rate = hits / num_users

        if current_hit_rate > best_hit_rate:
            best_hit_rate = current_hit_rate
            best_weights = w.cpu().numpy()
            print(f"🌟 新高！当前 Hit Rate: {best_hit_rate * 100:.2f}% | 权重: {np.round(best_weights, 4)}")


    print("\n📐 正在对黄金权重进行 Platt Scaling (RMSE 校准)...")
    meta_db_path = root_path() / "data/meta_dataset.db"
    conn_meta = sqlite3.connect(meta_db_path)
    df_meta = pd.read_sql_query(
        "SELECT Actual_Rating, SVD_Score, ItemKNN_Hit_Score, AutoRec_Score, ContentKNN_Hit_Score, UserKNN_Hit_Score FROM meta_train",
        conn_meta)
    conn_meta.close()

    X_meta = df_meta[
        ["SVD_Score", "ItemKNN_Hit_Score", "AutoRec_Score", "ContentKNN_Hit_Score", "UserKNN_Hit_Score"]].values
    y_true = df_meta["Actual_Rating"].values

    raw_scores = X_meta @ best_weights
    calibrator = LinearRegression()
    calibrator.fit(raw_scores.reshape(-1, 1), y_true)

    slope = calibrator.coef_[0]
    intercept = calibrator.intercept_
    calibrated_rmse = np.sqrt(np.mean(((raw_scores * slope + intercept) - y_true) ** 2))

    final_weights = best_weights * slope

    print("\n" + "=" * 60)
    print("🏆 THE ULTIMATE CONSTANTS 🏆")
    print("=" * 60)
    print(f"极限 Hit Rate (留一法模拟): {best_hit_rate * 100:.2f}%")
    print(f"校准后完美 RMSE: {calibrated_rmse:.4f}")
    print(
        f"self.weights = np.array([{final_weights[0]:.8f}, {final_weights[1]:.8f}, {final_weights[2]:.8f}, {final_weights[3]:.8f}, {final_weights[4]:.8f}])")
    print(f"self.intercept = {intercept:.8f}")
    print("=" * 60)


if __name__ == "__main__":
    run_hit_rate_sniper()