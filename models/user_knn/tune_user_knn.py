from models.evaluate_rmse import get_test_profiles, evaluate_model
from models.user_knn import UserBasedRecommender
from util import root_path, plot_results

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

ITEM_K_RANGE = range(160, 185, 1)


def tune_knn_models():
    print("[KNN Tuner] get test profiles...")
    test_profiles = get_test_profiles(TEST_DB)
    print("\n[KNN Tuner] init KNN...")
    user_model = UserBasedRecommender(db_path=TRAIN_DB)

    item_rmse_results = []
    print(f"[KNN Tuner] testing {len(ITEM_K_RANGE)} K values...")
    for k in ITEM_K_RANGE:
        user_model.engine.k_neighbors = k

        rmse = evaluate_model(user_model, test_profiles)
        item_rmse_results.append(rmse)
        print(f"  -> User-KNN k={k:03d} | RMSE: {rmse:.4f}")

    return item_rmse_results


if __name__ == "__main__":
    plot_results(tune_knn_models(), ITEM_K_RANGE,
                 "User-KNN Tuning: RMSE vs. $k$",
                 "Number of Neighbors ($k$)",
                 "user_knn_tune_k.png")
