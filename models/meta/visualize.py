import matplotlib.pyplot as plt
import numpy as np
from util import root_path

models = ["SVD-50", "Item-KNN", "Content-KNN", "Deep-AutoRec", "User-KNN"]

results = [
    {# 1.
        "weights": {
            "SVD-50": 0.4532,
            "User-KNN": 0.0012,
            "Item-KNN": 0.3178,
            "Deep-AutoRec": 0.1321,
            "Content-KNN": 0.0957
        },
        "hit": 0.7817,
        "prec": 0.2683
    },
    { # 2.
        "weights": {
            "SVD-50": 0.3016,
            "User-KNN": 0.1381,
            "Item-KNN": 0.2511,
            "Deep-AutoRec": 0.1411,
            "Content-KNN": 0.1682
        },
        "hit": 0.7664,
        "prec": 0.2557
    },
    { # 3.
        "weights": {
            "SVD-50": 0.3065,
            "User-KNN": 0.0389,
            "Item-KNN": 0.1877,
            "Deep-AutoRec": 0.0837,
            "Content-KNN": 0.3832
        },
        "hit": 0.7689,
        "prec": 0.2548
    },
    { # 4.
        "weights": {
            "SVD-50": 0.3145,
            "User-KNN": 0.0232,
            "Item-KNN": 0.2729,
            "Deep-AutoRec": 0.0671,
            "Content-KNN": 0.3222
        },
        "hit": 0.7642,
        "prec": 0.2515
    },
    { # 5.
        "weights": {
            "SVD-50": 0.1771,
            "User-KNN": 0.3816,
            "Item-KNN": 0.3716,
            "Deep-AutoRec": 0.0115,
            "Content-KNN": 0.0582
        },
        "hit": 0.7882,
        "prec": 0.2491
    },
    { # 6.
        "weights": {
            "SVD-50": 0.3694,
            "User-KNN": 0.1891,
            "Item-KNN": 0.0555,
            "Deep-AutoRec": 0.2280,
            "Content-KNN": 0.1581
        },
        "hit": 0.7402,
        "prec": 0.2450
    },
    { # 7.
        "weights": {
            "SVD-50": 0.1974,
            "User-KNN": 0.1553,
            "Item-KNN": 0.2144,
            "Deep-AutoRec": 0.2571,
            "Content-KNN": 0.1758
        },
        "hit": 0.7511,
        "prec": 0.2437
    },
    { # 8.
        "weights": {
            "SVD-50": 0.3467,
            "User-KNN": 0.0419,
            "Item-KNN": 0.1384,
            "Deep-AutoRec": 0.2475,
            "Content-KNN": 0.2254
        },
        "hit": 0.7424,
        "prec": 0.2426
    },
    { # 9.
        "weights": {
            "SVD-50": 0.1772,
            "User-KNN": 0.1910,
            "Item-KNN": 0.2369,
            "Deep-AutoRec": 0.1357,
            "Content-KNN": 0.2594
        },
        "hit": 0.7533,
        "prec": 0.2421
    },
    { # 10.
        "weights": {
            "SVD-50": 0.1682,
            "User-KNN": 0.3882,
            "Item-KNN": 0.2979,
            "Deep-AutoRec": 0.1163,
            "Content-KNN": 0.0295
        },
        "hit": 0.7445,
        "prec": 0.2389
    }
]

def visualize_weight_correlations():
    for model in models:
        weights = np.array([r["weights"][model] for r in results])
        precs = np.array([r["prec"] for r in results])

        corr = np.corrcoef(weights, precs)[0, 1]
        print(f"{model} vs Precision correlation: {corr:.3f}")

    correlations = []

    for model in models:
        weights = np.array([r["weights"][model] for r in results])
        precs = np.array([r["prec"] for r in results])

        corr = np.corrcoef(weights, precs)[0, 1]
        correlations.append(corr)

    colors =['tab:blue', 'tab:green', 'tab:olive', 'tab:orange', 'tab:red'] 
    plt.figure()
    plt.bar(models, correlations, label=models, color=colors)
    plt.ylabel("Correlation with Precision@10")
    plt.title("Model Weight Influence on Recommendation Quality \nwith MetaRecommender")
    plt.xticks([])
    plt.legend()
    plt.grid()
    #plt.show()

    # save figure
    plt.savefig(root_path() / 'visualizations/model_weights_vs_precision.png')

def visualize_grid():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, model in enumerate(models):
        weights = np.array([r["weights"][model] for r in results])
        precs = np.array([r["prec"] for r in results])

        ax = axes[i]
        ax.scatter(weights, precs)

        z = np.polyfit(weights, precs, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(weights)
        ax.plot(x_sorted, p(x_sorted))

        ax.set_title(model)
        ax.set_xlabel("Weight")
        ax.set_ylabel("Precision")

    plt.tight_layout()
    plt.show()

def main():

    #visualize_weight_correlations()
    #visualize_grid()

    import pickle
    print(pickle.__version__)
    

if __name__ == "__main__":
    main()