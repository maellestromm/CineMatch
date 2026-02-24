import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
DB_NAME = "movie_first.db"


class ContentBasedRecommender:
    def __init__(self, db_path=DB_NAME):
        self.db_path = db_path
        self.df = None
        self.cosine_sim_matrix = None
        self.indices = None

        # Initialize the model upon creation
        self._load_data()
        self._build_model()

    def _load_data(self):
        """Step 1: Load enriched movies from SQLite database."""
        print("Loading data from database...")
        conn = sqlite3.connect(self.db_path)
        # Only load movies that have been enriched by the Movie Worker
        query = """SELECT * FROM movies WHERE is_enriched = 1"""
        self.df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Loaded {len(self.df)} movies for the recommendation engine.")

    def _create_soup(self, row):
        """
        Step 2: Create a 'Feature Soup' for each movie.
        We mix director, genres, cast, and description into one long string.
        Trick: We repeat 'director' and 'genres' to give them higher weight.
        """
        # Handle potential None/NaN values
        director = str(row['director']) if pd.notnull(row['director']) else ''
        genres = str(row['genres']) if pd.notnull(row['genres']) else ''
        cast = str(row['cast']) if pd.notnull(row['cast']) else ''
        description = str(row['description']) if pd.notnull(row['description']) else ''

        # Remove commas to treat them as individual keywords (e.g., "Sci-Fi,Action" -> "Sci-Fi Action")
        director = director.replace(',', ' ')
        genres = genres.replace(',', ' ')
        cast = cast.replace(',', ' ')

        # Combine them. Repeating director and genres makes TF-IDF score them higher.
        soup = f"{director} {director} {genres} {genres} {cast} {description}"
        return soup

    def _build_model(self):
        """Step 3 & 4: Vectorize the text and calculate Cosine Similarity."""
        if self.df.empty:
            print("Error: No enriched movies found. Please run the crawler first.")
            return

        print("Building TF-IDF matrix and calculating similarities...")

        # Apply the soup function to every movie
        self.df['soup'] = self.df.apply(self._create_soup, axis=1)

        # Initialize TF-IDF Vectorizer
        # stop_words='english' removes common words like 'the', 'is', 'and' from descriptions
        tfidf = TfidfVectorizer(stop_words='english')

        # Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(self.df['soup'])

        # Compute the cosine similarity matrix
        # This creates a massive grid (N x N) comparing every movie to every other movie
        self.cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create a reverse mapping of movie titles and slugs to DataFrame indices
        # This helps us find the index of a movie quickly when a user searches for it
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()
        self.slug_indices = pd.Series(self.df.index, index=self.df['slug']).drop_duplicates()

        print("Model building complete! Ready for recommendations.\n")

    def get_recommendations_by_title(self, title, top_k=10):
        """Step 5: Get top K similar movies based on a movie title."""
        if title not in self.indices:
            return f"Movie '{title}' not found in the database. Please try another one."

        # Get the index of the movie that matches the title
        idx = self.indices[title]

        # Handle cases where multiple movies have the exact same title (e.g., remakes)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Get the pairwise similarity scores of all movies with that movie
        # Output format: [(index0, score0), (index1, score1), ...]
        sim_scores = list(enumerate(self.cosine_sim_matrix[idx]))

        # Sort the movies based on the similarity scores in descending order
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top K most similar movies
        # (Skip the first one because it is the movie itself, similarity = 1.0)
        top_k_scores = sim_scores[1:top_k + 1]

        # Get the movie indices
        movie_indices = [i[0] for i in top_k_scores]
        similarity_values = [i[1] for i in top_k_scores]

        # Return the top K most similar movies with their scores and poster URLs
        results = self.df.iloc[movie_indices][['title', 'year', 'director', 'poster_url']].copy()
        results['similarity_score'] = similarity_values

        return results


# --- Test the Model ---
if __name__ == "__main__":
    # 1. Initialize the recommender (This will load DB and train the model instantly)
    recommender = ContentBasedRecommender()

    # 2. Test it out! (Make sure these movies exist in your database)
    # You can change "V for Vendetta" to whatever movie you know your crawler has enriched
    target_movie = "Superman"

    print(f"--- Top 10 Recommendations if you like '{target_movie}' ---")
    recommendations = recommender.get_recommendations_by_title(target_movie, top_k=10)

    if isinstance(recommendations, str):
        print(recommendations)  # Prints error message if movie not found
    else:
        # Format the output nicely
        for i, row in recommendations.iterrows():
            score = row['similarity_score']
            print(f"[{score:.3f}] {row['title']} ({row['year']}) - Dir: {row['director']}")
            # print(f"Poster: {row['poster_url']}") # Uncomment to see poster links