import sqlite3
import os

DB_NAME = "../data/user_first_cut3_clear.db"


def clean_database():
    initial_size = os.path.getsize(DB_NAME) / (1024 * 1024)
    print(f"{initial_size:.2f} MB")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
                   DELETE
                   FROM reviews
                   WHERE movie_slug IN (SELECT slug
                                        FROM movies
                                        WHERE is_enriched = 0)
                   """)
    deleted_reviews = cursor.rowcount
    print(f"delete {deleted_reviews} reviews")

    cursor.execute("DELETE FROM movies WHERE is_enriched = 0")
    deleted_movies = cursor.rowcount
    print(f"delete {deleted_movies} movies")

    cursor.execute("DELETE FROM user_queue")
    deleted_users_queue = cursor.rowcount
    print(f"Cleared {deleted_users_queue} users from user_queue")

    cursor.execute("DELETE FROM movie_queue")
    deleted_movies_queue = cursor.rowcount
    print(f"Cleared {deleted_movies_queue} movies from movie_queue")
    conn.commit()

    print("VACUUM")
    cursor.execute("VACUUM")

    conn.close()

    final_size = os.path.getsize(DB_NAME) / (1024 * 1024)
    print(f"{initial_size - final_size:.2f} MB -> {final_size:.2f} MB ")


if __name__ == "__main__":
    clean_database()