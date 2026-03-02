import sqlite3

from pathlib import Path
import pandas as pd

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
    return df_test,test_users