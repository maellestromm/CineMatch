from models.evaluate_rmse import get_rmse_test_profiles, evaluate_model_rmse
from models.item_knn import ItemBasedRecommender
from util import root_path, plot_results

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

K_RANGE = range(35, 65, 1)


def tune_knn_models():
    print("[KNN Tuner] get test profiles...")
    test_profiles = get_rmse_test_profiles(TEST_DB)
    print("\n[KNN Tuner] init KNN...")

    item_rmse_results = []
    print(f"[KNN Tuner] testing {len(K_RANGE)} K values...")
    for k in K_RANGE:
        model = ItemBasedRecommender(db_path=TRAIN_DB, k_neighbors=k)
        rmse = evaluate_model_rmse(model, test_profiles)
        item_rmse_results.append(rmse)
        print(f"  -> Item-KNN k={k:03d} | RMSE: {rmse:.4f}")

    return item_rmse_results


if __name__ == "__main__":
    plot_results(tune_knn_models(), K_RANGE,
                 "Item-KNN Tuning: RMSE vs. $k$",
                 "Number of Neighbors ($k$)",
                 'Root Mean Squared Error (RMSE)',
                 "item_knn_tune_k_rmse.png")
