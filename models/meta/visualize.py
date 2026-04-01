import matplotlib.pyplot as plt
import numpy as np
from util import root_path
from models.meta.visualize_data import models, weight_results, confusion_results

def visualize_weight_correlations():
    for model in models:
        weights = np.array([r["weights"][model] for r in weight_results])
        precs = np.array([r["prec"] for r in weight_results])

        corr = np.corrcoef(weights, precs)[0, 1]
        print(f"{model} vs Precision correlation: {corr:.3f}")

    correlations = []

    for model in models:
        weights = np.array([r["weights"][model] for r in weight_results])
        precs = np.array([r["prec"] for r in weight_results])

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
        weights = np.array([r["weights"][model] for r in weight_results])
        precs = np.array([r["prec"] for r in weight_results])

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

def compute_metrics(confusion_results):
    results = {}

    data = confusion_results[0]

    for model_name, vals in data.items():
        TP = vals["TP"]
        FP = vals["FP"]
        FN = vals["FN"]

        # avoid divide by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results[model_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hitrate":data[model_name]["Hit-Rate"]
        }

    sorted_results = sorted(results.items(), key=lambda x: x[1]["precision"], reverse=False)
    precisions = []
    recalls = []
    f1s = []
    hitrates = []
    sorted_names = []

    for model_name, data in sorted_results:
        precisions.append(data["precision"])
        recalls.append(data["recall"])
        f1s.append(data["f1"])
        hitrates.append(data["hitrate"])
        sorted_names.append(model_name)

    return sorted_results, precisions, recalls, f1s, hitrates, sorted_names

def visualize_confusion():
    results, precisions, recalls, f1s, hitrates, model_names = compute_metrics(confusion_results)

    x = np.arange(len(models))
    width = 0.25
    
    plt.figure()

    plt.bar(x - width, precisions, width, label="Precision@10")
    plt.bar(x, recalls, width, label="Recall@10")
    plt.bar(x + width, f1s, width, label="F1 Score")

    plt.xticks(x, model_names, rotation=30)
    plt.ylabel("Score")
    plt.title("Model Comparison (Top-10 Recommendations)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    #plt.show()
    plt.savefig(root_path() / 'visualizations/model_comparison.png')

def visualize_comparison_scatter():
    results, precisions, recalls, f1s, hitrates, model_names = compute_metrics(confusion_results)

    plt.figure(figsize=(10,6))

    for i, model in enumerate(model_names):
        plt.scatter(precisions[i], hitrates[i])
        plt.text(precisions[i], hitrates[i], model, wrap=True)

    plt.xlabel("Precision@10")
    plt.ylabel("Hit Rate@10")
    plt.title("Precision vs Hit Rate (Top-10 Recommendations)")

    #plt.show()
    plt.savefig(root_path() / 'visualizations/comparison_scatter.png')

def main():

    #visualize_weight_correlations()
    #visualize_grid()
    visualize_confusion()
    #visualize_comparison_scatter()

    

if __name__ == "__main__":
    main()