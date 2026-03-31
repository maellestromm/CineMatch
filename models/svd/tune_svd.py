import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

from util import root_path
from models.svd import SVDRecommender
from models.evaluate_rmse import get_test_profiles, evaluate_model

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
K_RANGE = range(25, 55, 1)


def global_sigma_plot():
    print("[SVD Tuner] gen global sigma plot...")
    conn = sqlite3.connect(TRAIN_DB)
    df = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').dropna()
    conn.close()

    pivot = df.pivot_table(index='user_username', columns='movie_slug', values='rating')
    matrix = pivot.sub(pivot.mean(axis=1), axis=0).fillna(0).values
    _, sigma, _ = svds(matrix, k=min(100, min(matrix.shape) - 1))

    plt.plot(range(1, 101), sigma, marker='.', color='#2ca02c', linewidth=2)
    plt.title('SVD Scree Plot: Information Decay', fontsize=14)
    plt.xlabel('Latent Component (Index)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('global_sigma_plot.png', dpi=300, bbox_inches='tight')


def tune_svd():
    print("[SVD Tuner] get test profiles...")
    test_profiles = get_test_profiles(TEST_DB)

    rmse_results = []
    print(f"[SVD Tuner] testing {len(K_RANGE)} K values...")

    for k in K_RANGE:
        model = SVDRecommender(db_path=TRAIN_DB, k_factors=k)
        rmse = evaluate_model(model, test_profiles)
        rmse_results.append(rmse)
        print(f"  -> k={k:03d} | RMSE: {rmse:.4f}")

    return K_RANGE, rmse_results


def plot_results(k_range, rmse_results):
    best_k = k_range[np.argmin(rmse_results)]
    best_rmse = np.min(rmse_results)

    plt.plot(k_range, rmse_results, marker='D', color='#d62728', linewidth=2)
    plt.axvline(x=best_k, color='black', linestyle='--', alpha=0.7)
    plt.scatter(best_k, best_rmse, color='black', s=100, zorder=5)

    plt.annotate(f'Best $k$={best_k}\nRMSE={best_rmse:.4f}',
                 xy=(best_k, best_rmse), xytext=(best_k + 5, best_rmse + 0.001),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

    plt.title('SVD Tuning: RMSE vs. $k$', fontsize=14)
    plt.xlabel('Number of Latent Factors ($k$)', fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('svd_tuning_k.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    global_sigma_plot()
    k_range, rmse_results = tune_svd()
    plot_results(k_range, rmse_results)
