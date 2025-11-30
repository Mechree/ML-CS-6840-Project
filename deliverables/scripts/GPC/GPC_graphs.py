import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from pathlib import Path

def plot_roc_curve(y_true, probs, split_name, output_dir):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = roc_auc_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(7,6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {auc_score:.3f}")
    ax.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – HAB Detection (GPC) {split_name}")
    ax.legend(loc="lower right")
    ax.grid(True)

    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / f"gpc_roc_curve_{split_name}.png"
    plt.savefig(plot_path, dpi=300)

    plt.close()

def plot_confusion_matrix(y_true, y_pred, split_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred: No HAB","Pred: HAB"],
                yticklabels=["True: No HAB","True: HAB"], ax=ax)

    ax.set_title(f"Confusion Matrix – GPC {split_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / f"gpc_confusion_matrix_{split_name}.png"
    plt.savefig(plot_path, dpi=300)

    plt.close()

def plot_confidence_distribution(confidences, split_name, output_dir):
    fig, ax = plt.subplots(figsize=(8,5))

    sns.histplot(confidences, bins=20, kde=True, color='purple', ax=ax)

    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Number of Predictions")
    ax.set_title(f"Distribution of GPC Model Confidence {split_name}")
    ax.grid(True)

    plt.tight_layout()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plot_path = Path(output_dir) / f"gpc_confidence_distribution_{split_name}.png"
    plt.savefig(plot_path, dpi=300)

    plt.close()

def generate_graphs_for_splits():
    splits = ['Train20:Test80','Train40:Test60','Train80:Test20']

    output_dir = "../../../assets/GPC"

    for split_name in splits:
        csv_path = Path(output_dir) / f"gpc_predictions_{split_name}.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {split_name}")
            continue

        data = pd.read_csv(csv_path)

        y_true = data["True_Label"]
        y_pred = data["Predicted_Label"]
        probs = data["Confidence_HAB"]
        confidences = data["Model_Confidence"]

        plot_roc_curve(y_true, probs, split_name, output_dir)
        plot_confusion_matrix(y_true, y_pred, split_name, output_dir)
        plot_confidence_distribution(confidences, split_name, output_dir)

if __name__ == "__main__":
    generate_graphs_for_splits()
