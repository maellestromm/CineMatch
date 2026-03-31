import time
from util import root_path
from models.load_models import load_model
from letterboxdpy import user
import sqlite3

"""
Mock backend that outputs default results: work in progress
Intended to allow for GUI development while models, I/O and scraper
integration are yet to be finalized
"""

# --- Config ---
TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

# placeholder sample output
MOCK_RESULTS = [
    {
        "title": "Batman Begins",
        "year": 2005,
        "score": 5.00,
        "poster_url": ""
    },
    {
        "title": "The Matrix",
        "year": 1999,
        "score": 4.8,
        "poster_url": ""
    },
    {
        "title": "Blade Runner 2049",
        "year": 2017,
        "score": 4.6,
        "poster_url": ""
    },
    {
        "title": "Arrival",
        "year": 2016,
        "score": 4.5,
        "poster_url": ""
    },
    {
        "title": "The Lord of the Rings: The Return of the King",
        "year": 2003,
        "score": 4.34,
        "poster_url": ""
    },
    {
        "title": "War of the Worlds",
        "year": 2025,
        "score": 4.15,
        "poster_url": ""
    },
    {
        "title": "Don't Look Up",
        "year": 2021,
        "score": 4.01,
        "poster_url": ""
    },
    {
        "title": "Ant-Man and the Wasp: Quantumania",
        "year": 2023,
        "score": 3.99,
        "poster_url": ""
    },
    {
        "title": "He's All That",
        "year": 2021,
        "score": 3.91,
        "poster_url": ""
    },
    {
        "title": "The Idea of You",
        "year": 2024,
        "score": 3.65,
        "poster_url": ""
    },
]

# load all movie slugs from DB into a set for lookup
def load_valid_movie_slugs(db_path):
  
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT slug FROM movies")
    slugs = {row[0] for row in cursor.fetchall()}

    conn.close()
    return slugs

# to be called for username-based predictions
def fetch_user_profile(username, db_path=TRAIN_DB, max_movies=500):
    """
    Build user profile from Letterboxd films (in DB and with rating val)

    Returns:
        dict: {movie_slug: rating}
    """

    try:
        print(f"[IO] Fetching films for user: {username}")

        # load valid DB slugs
        valid_slugs = load_valid_movie_slugs(db_path)

        # fetch films
        lbd_user = user.User(username)
        films = lbd_user.get_films()['movies']

        if not films:
            print("[IO] No films found.")
            return {}

        profile = {}
        count = 0

        # filter + build profile
        for slug, data in films.items():
            rating = data.get("rating")

            # skip unrated
            if rating is None:
                continue

            # skip movies not in our DB
            if slug not in valid_slugs:
                continue

            profile[slug] = float(rating)
            count += 1

            if count >= max_movies:
                break

        print(f"[IO] Collected {len(profile)} usable ratings")

        return profile

    except Exception as e:
        print(f"[IO] Error: {e}")
        return {}
    
# mock model initialization
recommender = None

def get_recommender():
    global recommender

    if recommender is None:
        print("[Backend] Loading MetaRecommender...")

        # real version:
        recommender = load_model("meta")

        #recommender = "MOCK"

    return recommender

# work in progress function: to be called from gui.py
def get_recommendations_from_profile(username):
    """
    Input: Letterboxd username (string)
    Output: list of recommendation dicts
    """

    print(f"[Backend] Fetching profile for user: {username}")

    #user_profile = mock_fetch_user_profile(username)
    user_profile = fetch_user_profile(username)

    if not user_profile:
        return []

    recommender = get_recommender()

    # mock mode:
    if recommender == "MOCK":
        return MOCK_RESULTS
    
    # real version:
    recs = recommender.get_recommendations(user_profile, top_n=10)
    return recs

# work in progress function: to be called from gui.py
def get_recommendations_from_movie(movie_title):
    """
    Input: movie title (string)
    Output: list of recommendation dicts
    """

    print(f"[Backend] Searching for movie: {movie_title}")

    if recommender == "MOCK":
        return MOCK_RESULTS
    
# mock function until real implementation with scraper is complete
def mock_fetch_user_profile(username):
    if username.strip() == "":
        return None

    return {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

# helper function to print top10 recs to terminal
def print_recs(recs):

    print("\n" + "=" * 65)
    print("You might enjoy checking out one of these movies:")
    print("=" * 65)
    for i, r in enumerate(recs, 1):
        print(f"{i}.\t[{r['score']:.2f}] {r['title']} ({r['year']})")