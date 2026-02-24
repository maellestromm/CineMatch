import logging
import random
import sqlite3
import time

from letterboxdpy import user, movie

# --- Configuration ---
DB_NAME = "movie_first.db"
START_USER = "jimothy1989"
MAX_USERS_TO_SCRAPE = 5000
MIN_DELAY = 1
MAX_DELAY = 1.5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_db_connection():
    return sqlite3.connect(DB_NAME, timeout=30)


def random_sleep():
    """Sleep for a random interval to avoid rate limiting."""
    sleep_time = random.uniform(MIN_DELAY, MAX_DELAY)
    logging.info(f"💤 Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)


def list_to_str(lst):
    """Convert list to comma-separated string."""
    if not lst: return ""
    return ",".join([str(x) for x in lst if x])


def parse_date(date_obj):
    """Handle date dictionary or string and convert to YYYY-MM-DD."""
    if not date_obj: return None
    if isinstance(date_obj, dict):
        try:
            y, m, d = date_obj.get('year'), date_obj.get('month'), date_obj.get('day')
            if y and m and d: return f"{y}-{m:02d}-{d:02d}"
        except:
            pass
    return str(date_obj) if not isinstance(date_obj, str) else date_obj


def init_db():
    """Initialize SQLite database tables."""
    conn = get_db_connection()
    c = conn.cursor()

    # Queues
    c.execute('''CREATE TABLE IF NOT EXISTS user_queue
                 (
                     username TEXT PRIMARY KEY,
                     status   TEXT      DEFAULT 'pending',
                     added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS movie_queue
                 (
                     slug     TEXT PRIMARY KEY,
                     status   TEXT      DEFAULT 'pending',
                     added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 )''')

    # Data tables
    c.execute('''CREATE TABLE IF NOT EXISTS movies
                 (
                     slug           TEXT PRIMARY KEY,
                     title          TEXT,
                     year           INTEGER,
                     director       TEXT,
                     genres         TEXT,
                     cast           TEXT,
                     country        TEXT,
                     description    TEXT,
                     rating_average REAL,
                     poster_url     TEXT,
                     is_enriched    INTEGER DEFAULT 0
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (
                     username     TEXT PRIMARY KEY,
                     display_name TEXT
                 )''')

    # Review tables
    c.execute('''CREATE TABLE IF NOT EXISTS reviews
                 (
                     id            TEXT PRIMARY KEY,
                     user_username TEXT,
                     movie_slug    TEXT,
                     rating        TEXT,
                     content       TEXT,
                     review_date   TEXT,
                     FOREIGN KEY (user_username) REFERENCES users (username),
                     FOREIGN KEY (movie_slug) REFERENCES movies (slug),
                     UNIQUE (user_username, movie_slug)
                 )''')

    conn.commit()
    conn.close()


def reset_stuck_tasks():
    """Reset tasks stuck in 'processing' state due to previous crashes."""
    conn = get_db_connection()
    conn.execute("UPDATE user_queue SET status = 'pending' WHERE status = 'processing'")
    conn.execute("UPDATE movie_queue SET status = 'pending' WHERE status = 'processing'")
    conn.commit()
    conn.close()


# --- Queue Operations ---

def add_user_to_queue(conn, username):
    try:
        conn.execute("INSERT OR IGNORE INTO user_queue (username) VALUES (?)", (username,))
    except:
        pass


def add_movie_to_queue(conn, slug):
    try:
        conn.execute("INSERT OR IGNORE INTO movie_queue (slug) VALUES (?)", (slug,))
    except:
        pass


def get_next_user(conn):
    c = conn.cursor()
    c.execute("SELECT username FROM user_queue WHERE status = 'pending' LIMIT 1")
    row = c.fetchone()
    return row[0] if row else None


def get_random_movie_from_queue(conn):
    c = conn.cursor()
    # Pick a random pending movie to avoid getting stuck on one type
    c.execute("SELECT slug FROM movie_queue WHERE status = 'pending' ORDER BY RANDOM() LIMIT 1")
    row = c.fetchone()
    return row[0] if row else None


def mark_status(conn, table, key_col, key_val, status):
    conn.execute(f"UPDATE {table} SET status = ? WHERE {key_col} = ?", (status, key_val))
    conn.commit()


# --- Helper Functions ---

def extract_names_from_list(data_list, key='name'):
    if not data_list or not isinstance(data_list, list): return []
    return [item.get(key) for item in data_list if item.get(key)]


def extract_directors(crew_dict):
    if not crew_dict: return []
    directors = crew_dict.get('director', [])
    return [d.get('name') for d in directors if d.get('name')]


# --- Worker Logic ---

def worker_process_user(conn, current_username):
    logging.info(f"👤 [User Worker] Processing user: {current_username}")
    mark_status(conn, 'user_queue', 'username', current_username, 'processing')

    try:
        lbd_user = user.User(current_username)
        conn.execute("INSERT OR IGNORE INTO users (username, display_name) VALUES (?, ?)",
                     (current_username, lbd_user.username))
        conn.commit()

        random_sleep()

        try:
            reviews_response = lbd_user.get_reviews()
        except Exception as e:
            logging.warning(f"   ⚠️ Failed to fetch reviews: {e}")
            mark_status(conn, 'user_queue', 'username', current_username, 'error')
            return

        if not reviews_response or 'reviews' not in reviews_response:
            logging.info(f"   User has no reviews.")
            mark_status(conn, 'user_queue', 'username', current_username, 'done')
            return

        reviews_dict = reviews_response['reviews']
        movies_found_count = 0

        for review_id in reviews_dict:
            review = reviews_dict[review_id]
            movie_data = review.get('movie')
            if not movie_data: continue

            m_slug = movie_data.get('slug')
            m_title = movie_data.get('title')
            m_year = movie_data.get('year')

            raw_date = review.get('date') or review.get('activity', {}).get('date')
            parsed_date_str = parse_date(raw_date)

            # 1. Save basic movie info
            conn.execute("INSERT OR IGNORE INTO movies (slug, title, year) VALUES (?, ?, ?)",
                         (m_slug, m_title, m_year))

            # 2. Save review
            conn.execute('''INSERT OR IGNORE INTO reviews
                                (id, user_username, movie_slug, rating, content, review_date)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                         (review_id, current_username, m_slug, str(review.get('rating')), review.get('content'),
                          parsed_date_str))

            # 3. Add to the movie queue for enrichment
            add_movie_to_queue(conn, m_slug)
            movies_found_count += 1

        conn.commit()
        mark_status(conn, 'user_queue', 'username', current_username, 'done')
        logging.info(f"   ✅ User {current_username} finished. Added {movies_found_count} related movies.")

    except Exception as e:
        logging.error(f"❌ User Worker Error: {e}")
        mark_status(conn, 'user_queue', 'username', current_username, 'error')


def worker_process_movie(conn, current_movie_slug):
    logging.info(f"🎬 [Movie Worker] Enriching movie data: {current_movie_slug}")
    mark_status(conn, 'movie_queue', 'slug', current_movie_slug, 'processing')

    try:
        lbd_movie_obj = movie.Movie(current_movie_slug)

        # Extract details
        title = getattr(lbd_movie_obj, 'title', "")
        year = getattr(lbd_movie_obj, 'year', None)
        crew_data = getattr(lbd_movie_obj, 'crew', {})
        directors_list = extract_directors(crew_data)

        genres_data = getattr(lbd_movie_obj, 'genres', [])
        genres_list = extract_names_from_list(genres_data)

        cast_data = getattr(lbd_movie_obj, 'cast', [])
        cast_list = extract_names_from_list(cast_data[:10])

        details_data = getattr(lbd_movie_obj, 'details', [])
        country_list = []
        if details_data and isinstance(details_data, list):
            country_list = [d.get('name') for d in details_data if d.get('type') == 'country']

        description = getattr(lbd_movie_obj, 'description', "")
        rating_average = getattr(lbd_movie_obj, 'rating', None)
        poster_url = getattr(lbd_movie_obj, 'poster', "")

        conn.execute('''
                     UPDATE movies
                     SET title=?,
                         year=?,
                         director=?,
                         genres=?,
                         cast=?,
                         country=?,
                         description=?,
                         rating_average=?,
                         poster_url=?,
                         is_enriched=1
                     WHERE slug = ?
                     ''', (
                         title, year,
                         list_to_str(directors_list),
                         list_to_str(genres_list),
                         list_to_str(cast_list),
                         list_to_str(country_list),
                         description,
                         rating_average,
                         poster_url,
                         current_movie_slug
                     ))
        conn.commit()

        random_sleep()

        # Discover new users from popular reviews
        pop_reviews = getattr(lbd_movie_obj, 'popular_reviews', [])
        new_users_count = 0
        if pop_reviews:
            for rev in pop_reviews:
                reviewer = None
                if isinstance(rev, dict):
                    user_part = rev.get('user')
                    if isinstance(user_part, dict):
                        reviewer = user_part.get('username')
                    if not reviewer:
                        reviewer = rev.get('reviewer')

                if reviewer:
                    add_user_to_queue(conn, reviewer)
                    new_users_count += 1

        conn.commit()
        mark_status(conn, 'movie_queue', 'slug', current_movie_slug, 'done')
        logging.info(f"✅ Movie processed. Found {new_users_count} new users.")

    except Exception as e:
        logging.error(f"❌ Movie Worker Error: {e}")
        # traceback.print_exc()
        mark_status(conn, 'movie_queue', 'slug', current_movie_slug, 'error')


def main():
    init_db()
    reset_stuck_tasks()
    conn = get_db_connection()

    # Check seed user
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM user_queue")
    if cursor.fetchone()[0] == 0:
        add_user_to_queue(conn, START_USER)

    total_processed_users = 0

    while total_processed_users < MAX_USERS_TO_SCRAPE:
        # --- Scheduling Logic: Movie First ---
        # 1. Check counts
        cursor.execute("SELECT count(*) FROM movie_queue WHERE status = 'pending'")
        pending_movies_count = cursor.fetchone()[0]

        cursor.execute("SELECT count(*) FROM user_queue WHERE status = 'pending'")
        pending_users_count = cursor.fetchone()[0]

        logging.info(f"📊 Status: Pending Movies: {pending_movies_count} | Pending Users: {pending_users_count}")

        # 2. Prioritize Movies
        if pending_movies_count > 0:
            target_movie = get_random_movie_from_queue(conn)
            if target_movie:
                worker_process_movie(conn, target_movie)
            else:
                pass

        # 3. Only process users if no movies are pending
        else:
            if pending_users_count > 0:
                target_user = get_next_user(conn)
                if target_user:
                    worker_process_user(conn, target_user)
                    total_processed_users += 1
            else:
                logging.info("🏁 All tasks completed: All users and movies have been processed.")
                break

        random_sleep()

    conn.close()


if __name__ == "__main__":
    main()
