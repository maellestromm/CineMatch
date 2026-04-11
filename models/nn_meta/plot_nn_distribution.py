import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from util import root_path

from models.nn_meta import NNMetaRecommender


def plot_current_recommender_dist(username="Juggernaut323"):
    db_path = root_path() / "data/train_model.db"
    print(f"Reading OOF score data for {username}...")

    conn = sqlite3.connect(db_path)
    df_user = pd.read_sql_query(
        f"SELECT movie_slug, rating FROM reviews WHERE user_username = '{username}' AND rating != 'None'", conn)
    conn.close()

    if df_user.empty:
        print(f"No data for user {username} was found in the database.")
        return
    df_user['rating'] = pd.to_numeric(df_user['rating'], errors='coerce')
    df_user = df_user.dropna()
    user_profile = dict(zip(df_user['movie_slug'], df_user['rating']))
    print(
        f"Successfully loaded {len(user_profile)} movie rating records for {username}.Initializing NN Meta-Learner...")

    recommender = NNMetaRecommender(db_path=db_path)

    print("Retrieving raw recommendation scores from 5 base model...")
    base_scores = {}
    for name, model in recommender.base_models.items():
        raw_recs = model.get_recommendations(user_profile, top_n=len(recommender.movie_slugs))
        base_scores[name] = {r['slug']: r['score'] for r in raw_recs}

    print("inference...")

    meta_recs = recommender.get_recommendations(user_profile, top_n=5000)
    meta_scores = {r['slug']: r['score'] for r in meta_recs}

    records = []

    all_slugs = set(meta_scores.keys()).union(set(user_profile.keys()))

    for slug in all_slugs:
        record = {
            "Movie": slug,
            "Actual Rating (History)": user_profile.get(slug, np.nan),
            "NN Meta-Learner (Final)": meta_scores.get(slug, np.nan)
        }

        for name in base_scores.keys():
            record[name] = base_scores[name].get(slug, np.nan)
        records.append(record)

    df_plot = pd.DataFrame(records)

    print("Generating distribution chart...")
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    ax1 = axes[0]

    if len(user_profile) > 1:
        sns.kdeplot(data=df_plot, x="Actual Rating (History)", ax=ax1, color="black",
                    linewidth=4, label="Actual Rating (History)", fill=True, alpha=0.1)

    sns.kdeplot(data=df_plot, x="NN Meta-Learner (Final)", ax=ax1, color="crimson",
                linewidth=4, label="NN Meta-Learner (Final)", fill=True, alpha=0.15)

    palette = sns.color_palette("husl", len(base_scores))
    for i, name in enumerate(base_scores.keys()):
        sns.kdeplot(data=df_plot, x=name, ax=ax1, color=palette[i],
                    linewidth=2, linestyle="--", label=name)

    ax1.set_title(f"Dynamic Inference Distribution for User: '{username}'", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Star Rating (0.5 - 5.0)", fontsize=14)
    ax1.set_ylabel("Density", fontsize=14)
    ax1.set_xlim(0, 6)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

    ax2 = axes[1]

    df_melted = df_plot.melt(id_vars=["Movie"], var_name="Model", value_name="Score").dropna(subset=["Score"])

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
    print(f"The chart has been saved as: {save_path}")


# 50ShadesOfHay Juggernaut323 04MCAVOY
if __name__ == "__main__":
    plot_current_recommender_dist("Juggernaut323")
