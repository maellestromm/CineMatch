import os
import sqlite3

from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from matplotlib import pyplot as plt


def root_path():
    return Path(__file__).parent


def load_review_movie_datas(db_path):
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


def plot_results(results, k_range, title: str, xlabel: str, ylabel: str, filename: str, min_best: bool = True):
    best_i_k = k_range[np.argmin(results)] if min_best else k_range[np.argmax(results)]
    best_i_result = np.min(results) if min_best else np.max(results)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, results, marker='s', color='#ff7f0e', linewidth=2)
    plt.axvline(x=best_i_k, color='red', linestyle='--', alpha=0.7)
    plt.scatter(best_i_k, best_i_result, color='red', s=100, zorder=5)
    plt.annotate(f'Best $k$={best_i_k}\n{best_i_result:.4f}',
                 xy=(best_i_k, best_i_result), xytext=(best_i_k + 10, best_i_result + 0.0002),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def merge_onnx_model(onnx_path, data_path):
    print("[ONNX Merger] 启动模型融合程序...")

    if not os.path.exists(onnx_path):
        print(f"{onnx_path} not found!")
        return

    print("[ONNX Merger] 正在读取主模型和外部权重...")
    # onnx.load 会自动把同目录下的 .data 文件里的权重吸入内存
    model = onnx.load(str(onnx_path))

    print("[ONNX Merger] 正在将权重写入主干，生成单文件模型...")
    # onnx.save 默认 save_as_external_data=False，因此会强行打包成一个文件
    onnx.save_model(model, str(onnx_path))

    print("\n" + "=" * 50)
    print(f"融合成功！现在的{onnx_path}已经是包含所有权重的终极单文件了。")
    print("=" * 50)

    # 顺手帮你把那个碍事的 data 文件删掉
    if os.path.exists(data_path):
        os.remove(data_path)
        print(f"已自动清理残留的{data_path}文件。")
