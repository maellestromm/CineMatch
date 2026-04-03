import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse.linalg import svds

from models.evaluate_rmse import get_rmse_test_profiles, evaluate_model_rmse
from models.svd import SVDRecommender
from util import root_path, plot_results

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
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), sigma[::-1], marker='.', color='#2ca02c', linewidth=2)
    plt.title('SVD Scree Plot: Information Decay', fontsize=14)
    plt.xlabel('Latent Component (Index)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig('global_sigma_plot.png', dpi=300, bbox_inches='tight')


def tune_svd():
    print("[SVD Tuner] get test profiles...")
    test_profiles = get_rmse_test_profiles(TEST_DB)

    rmse_results = []
    print(f"[SVD Tuner] testing {len(K_RANGE)} K values...")

    for k in K_RANGE:
        model = SVDRecommender(db_path=TRAIN_DB, k_factors=k)
        rmse = evaluate_model_rmse(model, test_profiles)
        rmse_results.append(rmse)
        print(f"  -> k={k:03d} | RMSE: {rmse:.4f}")

    return rmse_results


if __name__ == "__main__":
    global_sigma_plot()
    plot_results(tune_svd(), K_RANGE,
                 "SVD Tuning: RMSE vs. $k$",
                 "Number of Latent Factors ($k$)",
                 'Root Mean Squared Error (RMSE)',
                 "svd_tuning_k_rmse.png")
