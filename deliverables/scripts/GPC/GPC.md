# Gaussian Process Classification (GPC)
Gaussian Process Classification (GPC) was selected as a probabilistic, non-parametric machine learning model to evaluate classification performance on the HAB dataset. Unlike deterministic models that output fixed labels, GPC provides predictive probabilities and quantifies uncertainty, which is valuable when real-world decisions rely not just on the prediction but also on the model’s confidence (e.g., identifying uncertain borderline bloom events).

GPC is particularly effective for complex decision boundaries and small-to-medium sized datasets, making it well-suited for this project. By combining GPC with PCA, cross-validation, and multiple evaluation metrics, the model provides a supervised baseline against which other models can be compared to.

Some limitations of GPC include its computational cost. The Gaussian Processes scales poorly with large datasets (O(n<sup>3</sup>)) and become challenging when the number of samples increases significantly. Additionally, performance is heavily influenced by kernel choice and hyperparameter optimization, and poor choices can lead to underfitting or overfitting.

### Hyperparameters

#### kernel
The kernel defines how similarity between input samples is measured. For this project, a composite kernel was used:
- **Matern Kernel:** Chosen for flexibility. This kernel operates well with noisy, real-world biological data as it models less smooth functions.

- **ConstantKernel:** Controls the overall magnitude of the function.

- **WhiteKernel:** Explicitly models observational noise, preventing overconfident predictions.

The combination allows GPC to handle moderate noise levels while still capturing nonlinear patterns.

#### random_state
A random seed used to ensure reproducibility. 0 signifies complete randomness. The number 42 was chosen to
ensure this reproducibility and also as a nod to the number’s popularity among computer scientists.

#### max_iter_predict
The maximum iterations during prediction for numerical stability. Set to 300
to ensure convergence.

#### n restarts optimizer:
The number of times the optimizer is restarted to find the optimal kernel
parameters. A value of 1 keeps the computation time manageable while still allowing the optimizer
to explore alternative starting points for an optimal set of parameters.

## How GPC works
1. A Gaussian process prior is placed over an unobserved latent function \( f(x) \).  
2. A link function (logistic probit) maps \( f(x) \) to class probabilities.  
3. Because the likelihood is non-Gaussian, approximations (Laplace method in scikit-learn) are used.  
4. The marginal likelihood is maximized to determine kernel hyperparameters.  
5. During prediction, the model produces:
   - Class label  
   - Class probability  
   - Model confidence (max probability)  

These predictive distributions allow us to analyze not just correctness but certainty and error boundaries.

## Additional Concepts

### PCA (Principal Component Analysis)

In this project, PCA serves two purposes:

1. **Dimensionality Reduction**:  
   PCA retains **95% of explained variance**, reducing noise and feature redundancy.

2. **Visualization**:  
   A separate 3-component PCA is used for 3D visualization of classification outcomes, highlighting:
   - Correct predictions  
   - False positives  
   - False negatives  
   - Probability gradients  

### Train/Test Splits

Three splits were explored to assess stability:

- **20/80**
- **40/60**
- **80/20**

Each split follows:
1. Train-test split with stratification  
2. Standardization (Z-score scaling)  
3. PCA transformation  
4. Training GPC on PCA-reduced features  

### Cross-Validation (5-fold)

Stratified K-Fold CV is applied using **ROC-AUC** as the scoring metric. This tests the model generalization and confirms stability of kernel hyperparameters.

## Metrics

GPC enables evaluation using both deterministic and probabilistic metrics.

### Accuracy
Measures overall correctness of predictions.

### Precision & Recall
Useful for asymmetric error importance (e.g., missing a real bloom is worse than a false alarm).

### F1 Score
Balances precision and recall.

### ROC-AUC
Used during cross-validation to evaluate separation between classes independent of threshold.

### Log Loss
Penalizes incorrect and overconfident probability predictions.

### Confusion Matrix
Highlights false positives and false negatives, which are crucial for HAB monitoring scenarios.

### Confidence Scores
The model’s maximum predicted probability for a given prediction.

## Training Process

The model was implemented entirely in Scikit-Learn due to its easy-to-use API and its built-in Gaussian Process components.

**Steps:**
1. Load the dataset. 
2. Remove any duplicates.
3. Split data using stratified sampling.  
4. Scale the features using `StandardScaler`.  
5. Apply PCA to preserve 95% variance.  
6. Train the Gaussian Process Classifier using the composite Matern kernel.  
7. Perform 5-fold ROC-AUC cross-validation.  
8. Evaluate on the held-out test set.  
9. Save predictions, PCA loadings, and 3D PCA visualizations.

## Workflow

1. **Import dataset** and remove duplicates.  
2. **Scale** features.  
3. **Apply PCA** to retain 95% variance.  
4. **Train GPC** on PCA-transformed training data.  
5. **Run cross-validation** using ROC-AUC.  
6. **Evaluate test performance** (Accuracy, Precision, Recall, F1, Log Loss).  
7. Generate:
   - CSV results  
   - PCA loading tables  
   - 3D PCA scatterplots highlighting classification behavior  
8. Repeat for each train/test split.  
9. Save aggregated split metrics into summary tables.
10. Run GPC_graphs.py to generate the confidence distribution, confusion matrix, and ROC curve for each split.

## Results

# WRITE HERE

### PCA and Visualization

PCA allowed reduction of feature dimensionality while maintaining most information, followed by 3D PCA visualization of classification outcomes.

**Highlights**
- Probability gradients in correct predictions  
- Spatial distribution of false positives and false negatives  
- Separation (or overlap) between HAB and non-HAB classes  

These plots assist in diagnosing whether errors cluster in specific PCA regions or result from nonlinear separability.

### Cross-Validation

5-fold ROC-AUC values were computed for each train split. High and consistent values indicate good generalization and strong kernel fit to the feature space.

### Evaluation Metrics

For each of the three data splits (20/80, 40/60, 80/20), the following were generated:
- Accuracy  
- Precision  
- Recall  
- F1 score  
- Log Loss  
- Confusion matrix  
- False positive/negative indices  

These metrics allow comparison of how training size affects GPC decision boundaries and reliability.

## Summary

# Resources

# Example resource link

[Scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)