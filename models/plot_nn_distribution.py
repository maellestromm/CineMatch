import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.meta_tmp import FastMetaRecommender, LinearMetaRecommender
from util import root_path

# 导入你刚刚上传的最终推荐器
from models.meta_learner import NNMetaRecommender


def plot_current_recommender_dist(username="Juggernaut323"):
    db_path = root_path() / "data/train_model.db"
    print(f"🔍 正在加载用户 '{username}' 的历史偏好...")

    # 1. 获取该用户的历史打分档案 (Profile)
    conn = sqlite3.connect(db_path)
    df_user = pd.read_sql_query(
        f"SELECT movie_slug, rating FROM reviews WHERE user_username = '{username}' AND rating != 'None'", conn)
    conn.close()

    if df_user.empty:
        print(f"❌ 未找到用户 {username}")
        return
    df_user['rating'] = pd.to_numeric(df_user['rating'], errors='coerce')
    df_user = df_user.dropna()
    user_profile = dict(zip(df_user['movie_slug'], df_user['rating']))
    print(f"✅ 找到 {len(user_profile)} 条评价记录。正在初始化 NN Meta-Learner...")

    # 2. 初始化你的神经网络推荐引擎
    # recommender = LinearMetaRecommender(db_path=db_path)
    recommender = NNMetaRecommender(db_path=db_path)

    # 3. 拦截基座模型的原始输出 (为了画对比图)
    print("⏳ 正在计算 5 个基座引擎的原始推荐分数 (Top 3334)...")
    base_scores = {}
    for name, model in recommender.base_models.items():
        raw_recs = model.get_recommendations(user_profile, top_n=len(recommender.movie_slugs))
        base_scores[name] = {r['slug']: r['score'] for r in raw_recs}

    # 4. 获取神经网络的最终输出
    print("🚀 正在执行 Wide & Deep 神经网络前向传播...")
    # 把 top_n 设为 5000，确保拿到所有未看电影的分数
    meta_recs = recommender.get_recommendations(user_profile, top_n=5000)
    meta_scores = {r['slug']: r['score'] for r in meta_recs}

    # 5. 对齐数据以便 Seaborn 绘图 (合并已知打分和未知预测)
    records = []
    # 获取所有的 slug 并集（已看 + 未看）
    all_slugs = set(meta_scores.keys()).union(set(user_profile.keys()))

    for slug in all_slugs:
        record = {
            "Movie": slug,
            "Actual Rating (History)": user_profile.get(slug, np.nan),
            "NN Meta-Learner (Final)": meta_scores.get(slug, np.nan)
        }
        # 把该电影对应的基座分数也放进来
        for name in base_scores.keys():
            record[name] = base_scores[name].get(slug, np.nan)
        records.append(record)

    df_plot = pd.DataFrame(records)

    # ==========================================
    # 开始绘制专业双拼图表
    # ==========================================
    print("🎨 正在生成高清分布图表...")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # --- 图 1：概率密度图 (KDE) ---
    ax1 = axes[0]

    # 1. 绘制用户真实的打分分布 (黑色粗线)
    if len(user_profile) > 1:
        sns.kdeplot(data=df_plot, x="Actual Rating (History)", ax=ax1, color="black",
                    linewidth=4, label="Actual Rating (History)", fill=True, alpha=0.1)

    # 2. 绘制神经网络的最终输出 (红色粗实线 + 填充)
    sns.kdeplot(data=df_plot, x="NN Meta-Learner (Final)", ax=ax1, color="crimson",
                linewidth=4, label="NN Meta-Learner (Final)", fill=True, alpha=0.15)

    # 3. 绘制 5 个基座模型的分布作为对比 (彩色虚线)
    palette = sns.color_palette("husl", len(base_scores))
    for i, name in enumerate(base_scores.keys()):
        sns.kdeplot(data=df_plot, x=name, ax=ax1, color=palette[i],
                    linewidth=2, linestyle="--", label=name)

    ax1.set_title(f"Dynamic Inference Distribution for User: '{username}'", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)
    ax1.set_xlim(0, 6)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    # --- 图 2：箱线图 (Boxplot) ---
    ax2 = axes[1]

    # 用 dropna 清理掉 NaN 值，以便 Seaborn 正确计算各项指标
    df_melted = df_plot.melt(id_vars=["Movie"], var_name="Model", value_name="Score").dropna(subset=["Score"])

    # 强制排序，把真实打分和最终模型放在最上面
    model_order = ["Actual Rating (History)", "NN Meta-Learner (Final)"] + list(base_scores.keys())
    box_palette = ["black", "crimson"] + list(palette)

    sns.boxplot(data=df_melted, x="Score", y="Model", ax=ax2, palette=box_palette, order=model_order,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"})

    ax2.set_xlim(0, 6)
    ax2.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax2.set_ylabel("")
    ax2.set_title("Score Spread, Median, and Mean Summary", fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_path = f"{username}_nn_inference_dist.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 绘制完成！请查看: {save_path}")

#50ShadesOfHay Juggernaut323 04MCAVOY
if __name__ == "__main__":
    plot_current_recommender_dist("Juggernaut323")