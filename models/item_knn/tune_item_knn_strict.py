from models.evaluate_rmse import get_rmse_test_profiles, evaluate_model_rmse
from models.evaluate_strict import get_strict_test_profiles, evaluate_model_strict
from models.item_knn import ItemBasedRecommender
from util import root_path, plot_results

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

K_RANGE = range(2, 32, 1)


def tune_knn_models():
    print("[KNN Tuner] get test profiles...")
    test_profiles = get_strict_test_profiles(TEST_DB)
    print("\n[KNN Tuner] init KNN...")

    hit_results = []
    perc_results = []
    print(f"[KNN Tuner] testing {len(K_RANGE)} K values...")
    for k in K_RANGE:
        model = ItemBasedRecommender(db_path=TRAIN_DB, k_neighbors=k)
        hit_rate, precision, avg_latency = evaluate_model_strict(model, test_profiles)
        hit_results.append(hit_rate)
        perc_results.append(precision)
        print(f"  -> Item-KNN k={k:03d} | Hit rate: {hit_rate:.4f} | Precision: {precision:.4f}")

    return hit_results, perc_results


if __name__ == "__main__":
    hit, perc = tune_knn_models()
    plot_results(hit, K_RANGE,
                 "Item-KNN Tuning: Hit Rate vs. $k$",
                 "Number of Neighbors ($k$)",
                 'Hit rate',
                 "item_knn_tune_k_hit.png",
                 min_best=False)
    plot_results(perc, K_RANGE,
                 "Item-KNN Tuning: Precision vs. $k$",
                 "Number of Neighbors ($k$)",
                 'Precision',
                 "item_knn_tune_k_prec.png",
                 min_best=False)
