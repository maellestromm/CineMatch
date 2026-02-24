import sqlite3
import time
import random
import logging
from letterboxdpy import user, movie

# --- Configuration ---
DB_NAME = "user_first.db"
START_USER = "jimothy1989"
MAX_USERS_TO_SCRAPE = 5000
MIN_DELAY = 1
MAX_DELAY = 1.5
# Limit reviews per user to avoid getting stuck on power users who have logged 10,000+ films
MAX_REVIEWS_PER_USER = 10000

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_db_connection():
    return sqlite3.connect(DB_NAME, timeout=30)


def random_sleep():
    """Sleep for a random interval to mimic human behavior and avoid rate limits."""
    sleep_time = random.uniform(MIN_DELAY, MAX_DELAY)
    logging.info(f"💤 Sleeping for {sleep_time:.2f} seconds...")
    time.sleep(sleep_time)


def list_to_str(lst):
    """Convert a list to a comma-separated string safely."""
    if not lst: return ""
    return ",".join([str(x) for x in lst if x])


def parse_date(date_obj):
    """Parse date from dict or string to YYYY-MM-DD format."""
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
    """Reset tasks stuck in 'processing' state due to previous interruptions."""
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


def get_most_popular_pending_movie(conn):
    """
    Select the pending movie that has the most reviews currently in our database.
    This 'Hub-based' approach guarantees we build a dense User-Movie matrix.
    """
    c = conn.cursor()
    query = """
            SELECT mq.slug, COUNT(r.id) as review_count
            FROM movie_queue mq
                     LEFT JOIN reviews r ON mq.slug = r.movie_slug
            WHERE mq.status = 'pending'
            GROUP BY mq.slug
            ORDER BY review_count DESC
            LIMIT 1 \
            """
    c.execute(query)
    row = c.fetchone()
    if row:
        slug, count = row
        logging.info(f"🏆 Selected most popular pending movie: {slug} ({count} reviews in DB)")
        return slug
    return None


def mark_status(conn, table, key_col, key_val, status):
    conn.execute(f"UPDATE {table} SET status = ? WHERE {key_col} = ?", (status, key_val))
    conn.commit()


# --- Extraction Helpers ---

def extract_names_from_list(data_list, key='name'):
    if not data_list or not isinstance(data_list, list): return []
    return [item.get(key) for item in data_list if item.get(key)]


def extract_directors(crew_dict):
    if not crew_dict: return []
    directors = crew_dict.get('director', [])
    return [d.get('name') for d in directors if d.get('name')]


# --- Worker Logic ---

def worker_process_user(conn, current_username):
    """Scrape a user's reviews, prioritizing movies we already track."""
    logging.info(f"👤 [User Worker] Scraping user: {current_username}")
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
            logging.warning(f"   ⚠️ Failed to fetch reviews for {current_username}: {e}")
            mark_status(conn, 'user_queue', 'username', current_username, 'error')
            return

        if not reviews_response or 'reviews' not in reviews_response:
            logging.info(f"   User has no reviews or diary entries.")
            mark_status(conn, 'user_queue', 'username', current_username, 'done')
            return

        reviews_dict = reviews_response['reviews']

        # Greedy strategy: Map slugs to review IDs to prioritize overlapping movies
        slug_to_review_id = {}
        for rid, rdata in reviews_dict.items():
            if rdata.get('movie'):
                slug = rdata['movie']['slug']
                if slug not in slug_to_review_id:
                    slug_to_review_id[slug] = rid

        all_slugs = list(slug_to_review_id.keys())
        target_review_ids = []

        if len(all_slugs) <= MAX_REVIEWS_PER_USER:
            target_review_ids = list(reviews_dict.keys())
        else:
            # Find which movies this user has seen that are ALREADY in our DB
            placeholders = ','.join(['?'] * len(all_slugs))
            query = f"SELECT slug FROM movies WHERE slug IN ({placeholders})"
            cursor = conn.cursor()
            cursor.execute(query, all_slugs)
            existing_slugs = {row[0] for row in cursor.fetchall()}

            # Prioritize existing movies to boost matrix density
            for slug in existing_slugs:
                if len(target_review_ids) < MAX_REVIEWS_PER_USER:
                    target_review_ids.append(slug_to_review_id[slug])

            # Fill the remaining slots randomly to explore new movies
            if len(target_review_ids) < MAX_REVIEWS_PER_USER:
                remaining_slugs = list(set(all_slugs) - existing_slugs)
                needed = MAX_REVIEWS_PER_USER - len(target_review_ids)
                new_picks = random.sample(remaining_slugs, needed)
                for slug in new_picks:
                    target_review_ids.append(slug_to_review_id[slug])

        movies_found_count = 0
        for review_id in target_review_ids:
            review = reviews_dict[review_id]
            movie_data = review.get('movie')
            if not movie_data: continue

            m_slug = movie_data.get('slug')
            m_title = movie_data.get('title')
            m_year = movie_data.get('year')
            r_date = parse_date(review.get('date') or review.get('activity', {}).get('date'))

            # Save basic movie structure
            conn.execute("INSERT OR IGNORE INTO movies (slug, title, year) VALUES (?, ?, ?)", (m_slug, m_title, m_year))

            # Save review
            conn.execute('''INSERT OR IGNORE INTO reviews (id, user_username, movie_slug, rating, content, review_date)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                         (review_id, current_username, m_slug, str(review.get('rating')), review.get('content'),
                          r_date))

            add_movie_to_queue(conn, m_slug)
            movies_found_count += 1

        conn.commit()
        mark_status(conn, 'user_queue', 'username', current_username, 'done')
        logging.info(f"   ✅ User processed. Saved {movies_found_count} reviews/movies.")

    except Exception as e:
        logging.error(f"❌ User Worker Error ({current_username}): {e}")
        mark_status(conn, 'user_queue', 'username', current_username, 'error')


def worker_process_movie(conn, current_movie_slug):
    """Enrich movie metadata and discover new users from its comment section."""
    logging.info(f"🎬 [Movie Worker] Enriching popular movie: {current_movie_slug}")
    mark_status(conn, 'movie_queue', 'slug', current_movie_slug, 'processing')

    try:
        lbd_movie_obj = movie.Movie(current_movie_slug)

        # --- FIX: Ensure title and year are fetched to prevent data loss ---
        title = getattr(lbd_movie_obj, 'title', "")
        year = getattr(lbd_movie_obj, 'year', None)

        # Extract metadata
        crew_data = getattr(lbd_movie_obj, 'crew', {})
        directors_list = extract_directors(crew_data)

        genres_data = getattr(lbd_movie_obj, 'genres', [])
        genres_list = extract_names_from_list(genres_data)

        cast_data = getattr(lbd_movie_obj, 'cast', [])
        cast_list = extract_names_from_list(cast_data[:5])

        details_data = getattr(lbd_movie_obj, 'details', [])
        country_list = [d.get('name') for d in details_data if d.get('type') == 'country'] if isinstance(details_data,
                                                                                                         list) else []

        description = getattr(lbd_movie_obj, 'description', "")
        rating_average = getattr(lbd_movie_obj, 'rating', None)
        poster_url = getattr(lbd_movie_obj, 'poster', "")

        # 1. Insert or Ignore first (handles cases where the slug was manually injected)
        conn.execute("INSERT OR IGNORE INTO movies (slug, title, year) VALUES (?, ?, ?)",
                     (current_movie_slug, title, year))

        # 2. Update all enriched fields
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
                         title, year, list_to_str(directors_list), list_to_str(genres_list), list_to_str(cast_list),
                         list_to_str(country_list), description, rating_average, poster_url, current_movie_slug
                     ))

        random_sleep()

        # Discover new users from the 'popular_reviews' section of this movie
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
        logging.info(f"   ✅ Movie enriched. Discovered {new_users_count} new users.")

    except Exception as e:
        logging.error(f"❌ Movie Worker Error ({current_movie_slug}): {e}")
        mark_status(conn, 'movie_queue', 'slug', current_movie_slug, 'error')


# ==========================================
# Main Control Flow: Hub-Based BFS
# ==========================================
def main():
    init_db()
    reset_stuck_tasks()
    conn = get_db_connection()

    # Inject seed user if starting fresh
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM user_queue")
    if cursor.fetchone()[0] == 0:
        logging.info(f"🌱 Queue is empty. Injecting seed user: {START_USER}")
        add_user_to_queue(conn, START_USER)

    total_processed_users = 0

    while total_processed_users < MAX_USERS_TO_SCRAPE:

        # Step 1: Exhaust the User Queue first
        # We process all recently discovered users immediately to harvest their movie ratings.
        target_user = get_next_user(conn)
        if target_user:
            worker_process_user(conn, target_user)
            total_processed_users += 1
            random_sleep()
            continue  # Loop back to ensure user queue is completely empty

        # Step 2: Once User Queue is empty, pick the MOST POPULAR movie to discover new users
        # This acts as the "Hub" to ensure high overlap in the resulting dataset.
        best_movie = get_most_popular_pending_movie(conn)
        if best_movie:
            worker_process_movie(conn, best_movie)
            random_sleep()
            # The next iteration will automatically go back to Step 1 to process the newly found users.
        else:
            logging.info("🏁 Crawling finished: Both user and movie queues are exhausted.")
            break

    conn.close()


if __name__ == "__main__":
    main()
