import sqlite3

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def root_path():
    return Path(__file__).parent


def load_review_datas(db_path):
    conn = sqlite3.connect(db_path)

    df_reviews = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce').dropna()

    query_movies = "SELECT slug, title, year, director, poster_url FROM movies"
    df_movies = pd.read_sql_query(query_movies, conn).set_index('slug')
    conn.close()
    return df_reviews, df_movies


def load_test_datas(db_path):
    conn = sqlite3.connect(db_path)
    df_test = pd.read_sql_query("SELECT user_username, movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_test['rating'] = pd.to_numeric(df_test['rating'], errors='coerce').dropna()
    conn.close()

    test_users = df_test['user_username'].unique()
    return df_test, test_users


def plot_results(rmse_results, k_range, title: str, xlabel: str, filename: str):
    best_i_k = k_range[np.argmin(rmse_results)]
    best_i_rmse = np.min(rmse_results)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, rmse_results, marker='s', color='#ff7f0e', linewidth=2)
    plt.axvline(x=best_i_k, color='red', linestyle='--', alpha=0.7)
    plt.scatter(best_i_k, best_i_rmse, color='red', s=100, zorder=5)
    plt.annotate(f'Best $k$={best_i_k}\nRMSE={best_i_rmse:.4f}',
                 xy=(best_i_k, best_i_rmse), xytext=(best_i_k + 10, best_i_rmse + 0.0002),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
