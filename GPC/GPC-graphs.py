import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Load the results from the GPC
algal_data = pd.read_csv("gpc_predictions_with_confidence.csv")

y_true = algal_data["True_Label"]
y_pred = algal_data["Predicted_Label"]
probabilities = algal_data["Confidence_HAB"]  # Probability for HAB class
confidences = algal_data["Model_Confidence"]

# ROC curve and AUC
false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
auc_score = roc_auc_score(y_true, probabilities)

plt.figure(figsize=(7,6))
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f"AUC = {auc_score:.3f}")
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – HAB Detection (GPC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("gpc_roc_curve.png", dpi=300)
plt.close()

# Confusion matrix heatmap
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Pred: No HAB", "Pred: HAB"],
    yticklabels=["True: No HAB", "True: HAB"]
)
plt.title("Confusion Matrix – GPC")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("gpc_confusion_matrix.png", dpi=300)
plt.close()

# Confidence Distribution Histogram
plt.figure(figsize=(8,5))
sns.histplot(confidences, bins=20, kde=True, color='purple')
plt.xlabel("Model Confidence")
plt.ylabel("Number of Predictions")
plt.title("Distribution of GPC Model Confidence")
plt.grid(True)
plt.tight_layout()
plt.savefig("gpc_confidence_distribution.png", dpi=300)
plt.close()
