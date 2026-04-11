import sqlite3
import pandas as pd
import json

from util import root_path


def export_movie_database(db_path=root_path() / "data/user_first_cut3_clear.db",
                          export_path=root_path() / "webui/movie_database.json"):
    print("[Exporter] Merging and exporting the front-end movie database...")
    conn = sqlite3.connect(db_path)

    df_movies = pd.read_sql_query("SELECT slug, title, year, director, poster_url FROM movies", conn)
    df_movies['year'] = pd.to_numeric(df_movies['year'], errors='coerce').fillna(2000).astype(int)

    df_reviews = pd.read_sql_query("SELECT movie_slug, rating FROM reviews WHERE rating != 'None'", conn)
    df_reviews['rating'] = pd.to_numeric(df_reviews['rating'], errors='coerce').dropna()

    movie_stats = df_reviews.groupby('movie_slug').agg(
        count=('rating', 'count'),
        avg=('rating', 'mean'),
        std=('rating', 'std')
    ).fillna(0.0)

    conn.close()

    frontend_db = {}
    for _, row in df_movies.iterrows():
        slug = row['slug']

        stats = movie_stats.loc[slug] if slug in movie_stats.index else {'count': 0, 'avg': 3.0, 'std': 0.0}

        frontend_db[slug] = {
            "title": row['title'],
            "year": int(row['year']),
            "director": row['director'] if pd.notna(row['director']) else "",
            "poster_url": row['poster_url'] if pd.notna(row['poster_url']) else "",
            "count": int(stats['count']),
            "avg": round(float(stats['avg']), 3),
            "std": round(float(stats['std']), 3)
        }

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(frontend_db, f, ensure_ascii=False, separators=(',', ':'))

    print(f"[Exporter] The complete data for {len(frontend_db)} movies has been exported to {export_path}")


if __name__ == "__main__":
    export_movie_database()
