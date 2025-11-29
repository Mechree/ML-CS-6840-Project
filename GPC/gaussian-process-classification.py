import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
from sklearn.decomposition import PCA

from typing import Tuple
import dataframe_image as dfi
from playwright.sync_api import sync_playwright

def load_dataset(csv_path: Path, target_column: str)\
-> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    x = df.drop(columns=[target_column]).values
    y = df[target_column].values
    return df, x, y

def split_scale_pca(x: np.ndarray, y: np.ndarray, train_size: float = 0.75, pca_variance: float = 0.95)\
-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, PCA]:

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_size, stratify=y, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    pca_model = PCA(n_components=pca_variance, svd_solver='full')
    x_train_pca = pca_model.fit_transform(x_train_scaled)
    x_test_pca = pca_model.transform(x_test_scaled)
    return x_train_pca, x_test_pca, y_train, y_test, x_train_scaled, x_test_scaled, scaler, pca_model

def train_gpc(x_train: np.ndarray, y_train: np.ndarray, max_iter: int = 300, n_restarts: int = 1)\
-> GaussianProcessClassifier:

    kernel = (ConstantKernel(1.0)
              * Matern(length_scale=1.0, length_scale_bounds=(1e-5,1e1), nu=1.5) +
             WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5,1e1)))

    gpc_model = GaussianProcessClassifier(kernel=kernel, random_state=42, max_iter_predict=max_iter, n_restarts_optimizer=n_restarts)
    gpc_model.fit(x_train, y_train)
    return gpc_model

def evaluate_gpc(gpc_model: GaussianProcessClassifier, x_test: np.ndarray, y_test: np.ndarray)\
-> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    y_pred = gpc_model.predict(x_test)
    probs = gpc_model.predict_proba(x_test)
    confidence = probs.max(axis=1)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("Log Loss:", log_loss(y_test, probs))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    false_positive_idx = np.where((y_test==0) & (y_pred==1))[0]
    false_negative_idx = np.where((y_test==1) & (y_pred==0))[0]
    print("False positives at indices:", false_positive_idx.tolist())
    print("False negatives at indices:", false_negative_idx.tolist())

    return y_pred, probs, confidence

def cross_validate_gpc(gpc_model: GaussianProcessClassifier, x_train: np.ndarray, y_train: np.ndarray, cv_folds: int=5)\
-> np.ndarray:

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gpc_model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)

    print("CV ROC-AUC scores:", cv_scores)
    print("Mean CV ROC-AUC:", cv_scores.mean())

    return cv_scores

def save_results_to_csv(output_dir: Path, y_test: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, confidence: np.ndarray, name_suffix: str="")\
-> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame({
        "True_Label": y_test,
        "Predicted_Label": y_pred,
        "Confidence_HAB": probs[:,1],
        "Confidence_NotHAB": probs[:,0],
        "Model_Confidence": confidence
    })

    csv_path = output_dir / f"gpc_predictions{name_suffix}.csv"

    if csv_path.exists():
        csv_path.unlink()
    df_results.to_csv(csv_path, index=False)

def save_pca_loadings(output_dir: Path, pca_model: PCA, feature_names: list, name_suffix: str="")\
-> None:

    loadings = pd.DataFrame(pca_model.components_.T, index=feature_names, columns=[f'PC{i+1}' for i in range(pca_model.n_components_)])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"pca_loadings{name_suffix}.csv"

    if csv_path.exists():
        csv_path.unlink()
    loadings.to_csv(csv_path)

    table_path = output_dir / f"pca_loadings_table{name_suffix}.png"

    dfi.export(loadings, table_path, dpi=300)

def plot_3d_pca(x_test_scaled: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, output_dir: Path, name_suffix: str="", split_name: str="")\
-> None:

    pca_vis = PCA(n_components=3)

    x_test_pca3 = pca_vis.fit_transform(x_test_scaled)

    false_positive_idx = np.where((y_test==0) & (y_pred==1))[0]
    false_negative_idx = np.where((y_test==1) & (y_pred==0))[0]

    correct_idx = np.where(y_test==y_pred)[0]

    fig = plt.figure(figsize=(10,8))

    ax = fig.add_subplot(111, projection='3d')

    scatter_correct = ax.scatter(x_test_pca3[correct_idx,0], x_test_pca3[correct_idx,1], x_test_pca3[correct_idx,2],
                                 c=probs[correct_idx,1], cmap='viridis', alpha=0.8, label='Correctly Classified')

    ax.scatter(x_test_pca3[false_positive_idx,0], x_test_pca3[false_positive_idx,1], x_test_pca3[false_positive_idx,2],
               c='red', marker='X', s=100, label='False Positive')

    ax.scatter(x_test_pca3[false_negative_idx,0], x_test_pca3[false_negative_idx,1], x_test_pca3[false_negative_idx,2],
               c='red', marker='o', s=100, edgecolor='k', label='False Negative')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA of Test Set ({split_name})")

    cbar = plt.colorbar(scatter_correct, ax=ax, pad=0.1)

    cbar.set_label("Predicted HAB Probability")
    ax.legend(loc='upper left')

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"gpc_test_pca3{name_suffix}.png"

    if plot_path.exists():
        plot_path.unlink()
    plt.savefig(plot_path, dpi=300)

    plt.close()

data_path = Path("..") / "dataset-harmful-algal-bloom(HAB)" / "HAB_Artificial_GAN_Dataset.csv"
output_dir = Path("GPC-figures")

df, x, y = load_dataset(data_path, target_column="HAB_Present")
feature_names = df.drop(columns=["HAB_Present"]).columns.tolist()

splits = {'Train20:Test80':0.2,'Train40:Test60':0.4,'Train80:Test20':0.8}

for split_name, train_size in splits.items():
    print("Running split", split_name)

    x_train_pca, x_test_pca, y_train, y_test, x_train_scaled, x_test_scaled, scaler, pca_model = split_scale_pca(x, y, train_size=train_size)

    gpc_model = train_gpc(x_train_pca, y_train)
    cross_validate_gpc(gpc_model, x_train_pca, y_train)
    y_pred, probs, confidence = evaluate_gpc(gpc_model, x_test_pca, y_test)

    save_results_to_csv(output_dir, y_test, y_pred, probs, confidence, name_suffix="_"+split_name)
    save_pca_loadings(output_dir, pca_model, feature_names, name_suffix="_"+split_name)

    plot_3d_pca(x_test_scaled, y_test, y_pred, probs, output_dir, name_suffix="_"+split_name, split_name=split_name)

print("All splits completed.")
