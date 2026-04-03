from models.evaluate_strict import get_strict_test_profiles, evaluate_model_strict
from models.user_knn import UserBasedRecommender
from util import root_path, plot_results

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"

K_RANGE = range(2, 32, 1)


def tune_knn_models():
    print("[KNN Tuner] get test profiles...")
    test_profiles = get_strict_test_profiles(TEST_DB)
    print("\n[KNN Tuner] init KNN...")
    model = UserBasedRecommender(db_path=TRAIN_DB)

    hit_results = []
    perc_results = []
    print(f"[KNN Tuner] testing {len(K_RANGE)} K values...")
    for k in K_RANGE:
        model.engine.k_neighbors = k

        hit_rate, precision, avg_latency = evaluate_model_strict(model, test_profiles)
        hit_results.append(hit_rate)
        perc_results.append(precision)
        print(f"  -> User-KNN k={k:03d} | Hit rate: {hit_rate:.4f} | Precision: {precision:.4f}")

    return hit_results, perc_results


if __name__ == "__main__":
    hit, perc = tune_knn_models()
    plot_results(hit, K_RANGE,
                 "User-KNN Tuning: Hit Rate vs. $k$",
                 "Number of Neighbors ($k$)",
                 'Hit rate',
                 "user_knn_tune_k_hit.png",
                 min_best=False)
    plot_results(perc, K_RANGE,
                 "User-KNN Tuning: Precision vs. $k$",
                 "Number of Neighbors ($k$)",
                 'Precision',
                 "user_knn_tune_k_prec.png",
                 min_best=False)
