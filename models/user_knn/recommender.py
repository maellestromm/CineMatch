from models.user_knn.cpu_backend import UserKNNCPUBackend
from models.user_knn.gpu_backend import UserKNNGPUBackend
from util import root_path


class UserBasedRecommender:
    def __init__(self, db_path, k_neighbors=10, backend='gpu'):
        """
        Main entry point for User-KNN Recommender.
        :param db_path: Path to the SQLite database.
        :param backend: 'gpu' or 'cpu'. Defaults to 'gpu'.
        """
        self.db_path = db_path
        self.k_neighbors = k_neighbors
        self.backend_type = backend.lower()

        if self.backend_type == 'gpu':
            try:
                self.engine = UserKNNGPUBackend(self.db_path)
            except Exception as e:
                print(f"[User-KNN] GPU backend initialization failed: {e}. Falling back to CPU.")
                self.engine = UserKNNCPUBackend(self.db_path)
        elif self.backend_type == 'cpu':
            self.engine = UserKNNCPUBackend(self.db_path)
        else:
            raise ValueError("Backend must be either 'gpu' or 'cpu'.")

    def get_recommendations(self, user_profile, top_n=10, k_neighbors=15):
        """
        Routes the recommendation request to the selected backend engine.
        """
        return self.engine.get_recommendations(user_profile, top_n, k_neighbors)


# --- Test Execution ---
if __name__ == "__main__":
    recommender = UserBasedRecommender(root_path() / "data/train_model.db", backend="gpu")

    demo_profile = {
        "inception": 5.0,
        "interstellar": 4.5,
        "the-dark-knight": 5.0
    }

    print("--- User-Based Recommendations ---")
    import time

    start_time = time.time()
    recommendations = recommender.get_recommendations(demo_profile, top_n=10)
    latency = (time.time() - start_time) * 1000

    if not recommendations:
        print("No recommendations found.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [{rec['score']:.2f}] {rec['title']} ({rec['year']}) - Dir: {rec['director']}")
    print(f"Latency: {latency:.2f} ms")