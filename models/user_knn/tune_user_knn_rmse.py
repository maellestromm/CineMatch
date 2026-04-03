from models.evaluate_rmse import get_rmse_test_profiles, evaluate_model_rmse
from models.user_knn import UserBasedRecommender
from util import root_path, plot_results

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

K_RANGE = range(160, 185, 1)


def tune_knn_models():
    print("[KNN Tuner] get test profiles...")
    test_profiles = get_rmse_test_profiles(TEST_DB)
    print("\n[KNN Tuner] init KNN...")
    model = UserBasedRecommender(db_path=TRAIN_DB)

    item_results = []
    print(f"[KNN Tuner] testing {len(K_RANGE)} K values...")
    for k in K_RANGE:
        model.engine.k_neighbors = k

        rmse = evaluate_model_rmse(model, test_profiles)
        item_results.append(rmse)
        print(f"  -> User-KNN k={k:03d} | RMSE: {rmse:.4f}")

    return item_results


if __name__ == "__main__":
    plot_results(tune_knn_models(), K_RANGE,
                 "User-KNN Tuning: RMSE vs. $k$",
                 "Number of Neighbors ($k$)",
                 'Root Mean Squared Error (RMSE)',
                 "user_knn_tune_k_rmse.png")
