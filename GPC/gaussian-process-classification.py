import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Define and load dataset
csv_path = Path("../dataset-harmful-algal-bloom(HAB)/HAB_Artificial_GAN_Dataset.csv")
algal_data = pd.read_csv(csv_path)

# Define target column (HAB_Present)
target_col = "HAB_Present"

# Define features
X = algal_data.drop(columns=[target_col]).values
y = algal_data[target_col].values

# Train and test the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the kernel for the Gaussian Process with bounds
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

# Initialize Gaussian Process Classifier
gpc = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=300, n_restarts_optimizer=5)

# Train the Gaussian Process classifier
gpc.fit(X_train_scaled, y_train)

# Predictions and confidence (probability)
y_pred = gpc.predict(X_test_scaled)
probs = gpc.predict_proba(X_test_scaled)
confidence = probs.max(axis=1)

results_df = pd.DataFrame({
    "True_Label": y_test,
    "Predicted_Label": y_pred,
    "Confidence_HAB": probs[:, 1],
    "Confidence_NotHAB": probs[:, 0],
    "Model_Confidence": confidence
})

# Some cool model statistics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, probs[:, 1])
logloss = log_loss(y_test, probs)

print("\nModel Performance Review:")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Log-Loss: {logloss:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# PCA visualization of test predictions
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

plt.figure(figsize=(8,6))
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=probs[:, 1], cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Predicted HAB Probability")
plt.title("PCA of Test Set Colored by HAB Probability")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.savefig("hab_pca_probability_plot.png", dpi=300)
plt.close()

# Save the result table to construct other graphs
results_df.to_csv("gpc_predictions_with_confidence.csv", index=False)
