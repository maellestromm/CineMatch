from models.evaluate_strict import get_strict_test_profiles, evaluate_model_strict
from models.svd import SVDRecommender
from util import root_path, plot_results

TRAIN_DB = root_path() / "data/train_model.db"
TEST_DB = root_path() / "data/test_eval.db"
K_RANGE = range(25, 55, 1)


def tune_svd():
    print("[SVD Tuner] get test profiles...")
    test_profiles = get_strict_test_profiles(TEST_DB)

    hit_results = []
    perc_results = []
    print(f"[SVD Tuner] testing {len(K_RANGE)} K values...")

    for k in K_RANGE:
        model = SVDRecommender(db_path=TRAIN_DB, k_factors=k)
        hit_rate, precision, avg_latency = evaluate_model_strict(model, test_profiles)
        hit_results.append(hit_rate)
        perc_results.append(precision)
        print(f"  -> SVD k={k:03d} | Hit rate: {hit_rate:.4f} | Precision: {precision:.4f}")

    return hit_results, perc_results


if __name__ == "__main__":
    hit, perc = tune_svd()
    plot_results(hit, K_RANGE,
                 "SVD Tuning: Hit Rate vs. $k$",
                 "Number of Latent Factors ($k$)",
                 'Hit rate',
                 "svd_tuning_k_hit.png",
                 min_best=False)
    plot_results(perc, K_RANGE,
                 "SVD Tuning: Precision vs. $k$",
                 "Number of Latent Factors ($k$)",
                 'Precision',
                 "svd_tuning_k_prec.png",
                 min_best=False)
