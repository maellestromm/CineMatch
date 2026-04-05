import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from util import root_path


def plot_user_distributions(username="Juggernaut323"):
    print(f"正在读取 {username} 的 OOF 评分数据...")

    # 尝试连接数据库 (适配 meta_dataset.db 或 train_model.db)
    db_path = root_path() / "data/meta_dataset.db"

    conn = sqlite3.connect(db_path)

    # 提取该用户的真实打分和所有基座模型的预测分
    query = (f"SELECT Actual_Rating, SVD_Score, AutoRec_Score, "
             f"ItemKNN_Hit_Score,  ItemKNN_RMSE_Score, "
             f"ContentKNN_Hit_Score,ContentKNN_RMSE_Score, "
             f"UserKNN_Hit_Score, UserKNN_RMSE_Score "
             f"FROM meta_train WHERE user_username = '{username}'")
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(f"❌ 数据库中未找到用户 {username} 的数据！请检查用户名是否正确或数据表名是否为 meta_train。")
        return

    print(f"✅ 成功加载 {len(df)} 条 {username} 的电影评分记录。开始绘制分布图...")

    # 设置 seaborn 样式
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # 提取特征列
    models = [col for col in df.columns if col != "Actual_Rating"]

    # ==========================================
    # 1. 概率密度分布图 (KDE Plot) - 看形状和重合度
    # ==========================================
    ax1 = axes[0]

    # 突出显示用户的真实打分 (红色粗实线 + 阴影填充)
    sns.kdeplot(data=df, x="Actual_Rating", ax=ax1, color="crimson",
                linewidth=3, label="Actual Rating (Ground Truth)", fill=True, alpha=0.15)

    # 绘制各个模型的预测分 (彩色虚线)
    palette = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        sns.kdeplot(data=df, x=model, ax=ax1, color=palette[i],
                    linewidth=2, linestyle="--", label=model.replace("_Score", ""))

    ax1.set_title(f"Score Distribution Density for User: '{username}'", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)
    ax1.set_xlim(0, 6)  # 统一 X 轴刻度，留一点视觉边缘
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    # ==========================================
    # 2. 箱线图 (Boxplot) - 看均值、方差和离群点
    # ==========================================
    ax2 = axes[1]

    # 将 DataFrame 转换为长格式 (Tidy data) 适配 seaborn 的箱线图
    df_melted = df.melt(var_name="Model", value_name="Score")
    # 把名称简化一下让图表更好看
    df_melted["Model"] = df_melted["Model"].str.replace("_Score", "")
    df_melted["Model"] = df_melted["Model"].str.replace("Actual_Rating", "Actual Rating")

    # 设定箱线图颜色，真实打分依然保持红色
    box_palette = ["crimson"] + list(palette)
    sns.boxplot(data=df_melted, x="Score", y="Model", ax=ax2, palette=box_palette,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"})

    ax2.set_title("Score Spread, Median, and Mean Summary", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax2.set_ylabel("")
    ax2.set_xlim(0, 6)

    # 调整布局并保存
    plt.tight_layout()
    save_path = f"{username}_rating_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 绘制完成！图表已保存为: {save_path}")


if __name__ == "__main__":
    plot_user_distributions("Juggernaut323")
