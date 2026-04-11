import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from util import root_path


def plot_user_distributions(username="Juggernaut323"):
    print(f"Reading OOF score data for {username}...")

    db_path = root_path() / "data/meta_dataset.db"

    conn = sqlite3.connect(db_path)

    query = (f"SELECT Actual_Rating, SVD_Score, AutoRec_Score, "
             f"ItemKNN_Hit_Score,  ItemKNN_RMSE_Score, "
             f"ContentKNN_Hit_Score,ContentKNN_RMSE_Score, "
             f"UserKNN_Hit_Score, UserKNN_RMSE_Score "
             f"FROM meta_train WHERE user_username = '{username}'")
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print(f"No data for user {username} was found in the database.")
        return

    print(f"Successfully loaded {len(df)} movie rating records for {username}. Starting to draw the distribution graph...")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    models = [col for col in df.columns if col != "Actual_Rating"]

    ax1 = axes[0]

    sns.kdeplot(data=df, x="Actual_Rating", ax=ax1, color="crimson",
                linewidth=3, label="Actual Rating (Ground Truth)", fill=True, alpha=0.15)

    palette = sns.color_palette("husl", len(models))
    for i, model in enumerate(models):
        sns.kdeplot(data=df, x=model, ax=ax1, color=palette[i],
                    linewidth=2, linestyle="--", label=model.replace("_Score", ""))

    ax1.set_title(f"Score Distribution Density for User: '{username}'", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)
    ax1.set_xlim(0, 6)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    ax2 = axes[1]

    df_melted = df.melt(var_name="Model", value_name="Score")

    df_melted["Model"] = df_melted["Model"].str.replace("_Score", "")
    df_melted["Model"] = df_melted["Model"].str.replace("Actual_Rating", "Actual Rating")

    box_palette = ["crimson"] + list(palette)
    sns.boxplot(data=df_melted, x="Score", y="Model", ax=ax2, palette=box_palette,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"})

    ax2.set_title("Score Spread, Median, and Mean Summary", fontsize=16, fontweight='bold')
    ax2.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax2.set_ylabel("")
    ax2.set_xlim(0, 6)

    plt.tight_layout()
    save_path = f"{username}_rating_distributions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"The chart has been saved as: {save_path}")


if __name__ == "__main__":
    plot_user_distributions("Juggernaut323")
