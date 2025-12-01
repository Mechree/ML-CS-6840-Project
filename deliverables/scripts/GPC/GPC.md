# Gaussian Process Classification (GPC)

Gaussian Process Classification (GPC) was selected as a probabilistic, non-parametric machine learning model to evaluate classification performance on the HAB dataset. Unlike deterministic models that output fixed labels, GPC provides predictive probabilities and quantifies uncertainty, which is valuable when real-world decisions rely not just on the prediction but also on the model’s confidence (e.g., identifying uncertain borderline bloom events).

GPC is particularly effective for complex decision boundaries and small-to-medium sized datasets, making it well-suited for this project. By combining GPC with PCA, cross-validation, and multiple evaluation metrics, the model provides a supervised baseline against which other models can be compared to.

Some limitations of GPC include its computational cost. The Gaussian Processes scales poorly with large datasets (O(n<sup>3</sup>) where n is the amount of data points) and becomes challenging when the number of samples increases significantly. Additionally, performance is heavily influenced by kernel choice and hyperparameter optimization, and poor choices can lead to underfitting or overfitting.

### Hyperparameters

#### kernel

The kernel defines how similarity between input samples is measured. For this project, a composite kernel was used:

- **ConstantKernel:** Controls the overall magnitude of the function.

- **Matern Kernel:** Chosen for flexibility. This kernel operates well with noisy, real-world biological data as it models less smooth functions.

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

### PCA and Visualization

PCA allowed reduction of feature dimensionality while maintaining most information, followed by 3D PCA visualization of classification outcomes.

**Highlights**

- Probability gradients in correct predictions
- Spatial distribution of false positives and false negatives
- Separation (or overlap) between HAB and non-HAB classes

These plots assist in diagnosing whether errors cluster in specific PCA regions or result from nonlinear separability.

<figure>
    <img src="../../../assets\GPC\gpc_test_pca3_Train20_Test80.png" alt="Alt text"  width="600"/>
    <figcaption><b>Figure 1</b>: The 3D visualization of the class classifications in PCA space for a 20:80 split.
</figcaption>
</figure>

- Misclassifications increase and cluster near the decision boundary. False negatives occur in darker (low probability) regions; false positives in lighter (high probability) regions.

<figure>
    <img src="../../../assets\GPC\gpc_test_pca3_Train40_Test60.png" alt="Alt text"  width="600"/>
    <figcaption><b>Figure 2</b>: The 3D visualization of the class classifications in PCA space for a 40:60 split.
</figcaption>
</figure>

- Medium amount of misclassifications. The separation trend between high and low probability points remains strong.

<figure>
    <img src="../../../assets\GPC\gpc_test_pca3_Train80_Test20.png" alt="Alt text"  width="600"/>
    <figcaption><b>Figure 3</b>: The 3D visualization of the class classifications in PCA space for a 80:20 split.
</figcaption>
</figure>

- Fewest misclassifications. Correctly classified points show clear separation based on predicted HAB probability.

### Cross-Validation

5-fold ROC-AUC values were computed for each train split. High and consistent values indicate good generalization and strong kernel fit to the feature space.

| Train/Test Split | CV ROC-AUC Scores                 | Mean CV ROC-AUC |
| ---------------- | --------------------------------- | --------------- |
| 20/80            | 1.000, 0.997, 1.000, 0.997, 1.000 | 0.9989          |
| 40/60            | 1.000, 0.994, 1.000, 1.000, 0.999 | 0.9988          |
| 80/20            | 1.000, 0.999, 0.999, 0.995, 1.000 | 0.9990          |

The mean CV ROC-AUC values above 0.998 demonstrate that the GPC is robust across different training sizes, and kernel hyperparameters generalize well.

### Confidence Distributions

The confidence distrubutions provide a visual representation of the predictive certainty of GPC for each train and test split. These histograms highlight how confident the model is in its predictions, and how training data volume impacts certainty.

#### 20:80

- With minimal training data, the Confidence Distribution is the most spread out with a single peak (≈0.85), indicating lower certainty and more ambiguous predictions.
<figure>
    <img src="../../../assets\GPC\gpc_confidence_distribution_Train20_Test80.png" alt="Alt text"  width="600"/>
    <figcaption><b>Figure 4</b>: The confidence distrubution histogram for the 20:80 split
</figcaption>
</figure>

#### 40:60

- With moderate training data, the Confidence Distribution shows two peaks (≈0.85 and ≈0.90), reflecting a more nuanced decision surface and slightly reduced certainty.
<figure>
    <img src="../../../assets\GPC\gpc_confidence_distribution_Train40_Test60.png" alt="Alt text"  width="600"/>
    <figcaption><b>Figure 5</b>: The confidence distrubution histogram for the 40:60 split
</figcaption>
</figure>

#### 80:20

- With the largest training set, the Confidence Distribution peaks at ≈0.93 with very tight clustering, showing the model’s maximum prediction certainty.
<figure>
    <img src="../../../assets\GPC\gpc_confidence_distribution_Train80_Test20.png" alt="Alt text"  width="600"/>
    <figcaption><b>Figure 6</b>: The confidence distrubution histogram for the 80:20 split
</figcaption>
</figure>

### Evaluation Metrics

For each of the three data splits (20/80, 40/60, 80/20), the following metrics were calculated:

| Train/Test Split | Accuracy | Precision | Recall | F1 Score | Log Loss |
| ---------------- | -------- | --------- | ------ | -------- | -------- |
| 20/80            | 0.994    | 0.99      | 0.98   | 0.979    | 0.188    |
| 40/60            | 0.996    | 1.00      | 0.98   | 0.985    | 0.166    |
| 80/20            | 1.000    | 1.00      | 1.00   | 1.000    | 0.117    |

These metrics allow comparison of how training size affects GPC decision boundaries and reliability.

### Confusion Matrices

#### 20:80

|                      | **Predicted: Positive** | **Predicted: Negative** |
| :------------------: | :---------------------: | :---------------------: |
| **Actual: Positive** |          1329           |            5            |
| **Actual: Negative** |            4            |           210           |

- False positives: `[218, 456, 1043, 1292, 1324]`
- False negatives: `[473, 804, 937, 1418]`

#### 40:60

|                      | **Predicted: Positive** | **Predicted: Negative** |
| :------------------: | :---------------------: | :---------------------: |
| **Actual: Positive** |           997           |            3            |
| **Actual: Negative** |            2            |           159           |

- False positives: `[175, 206, 1153]`
- False negatives: `[105, 855]`

#### 80:20

|                      | **Predicted: Positive** | **Predicted: Negative** |
| :------------------: | :---------------------: | :---------------------: |
| **Actual: Positive** |           333           |            0            |
| **Actual: Negative** |           F0            |           54            |

- No misclassifications; perfect discrimination

These metrics allow comparison of how training size affects GPC decision boundaries and reliability. Increasing training data improves both accuracy and confidence, reducing false positives and false negatives. Through the results, it can be seen that the model performed better when given a larger training set.

As a side note, it seems that the misclassifcations occur at a decision boundary where a given datapoint is is not clearly an harmful algal bloom. In other words, the data point is in a gray area in terms of it's true label.

## Summary

GPC effectively distinguishes harmful algal blooms, providing probabilistic predictions alongside high classification accuracy. Across all splits, the model achieved accuracy above 99\%, with perfect discrimination for the 80:20 and 40:60 splits and 99.42\% for the 20:80 split. ROC-AUC scores were near-perfect (0.999–1.000), indicating strong class separability. 3D PCA visualizations reveal a clear separation between HAB and non-HAB classes, with misclassifications occurring near decision boundaries. PCA loadings confirm that the primary components capture most variance, reducing dimensionality while retaining essential features. Overall, GPC demonstrates flexible, non-linear modeling capabilities, making it an effective approach for HAB detection despite its computational cost.

# Resources

[Scikit-learn: GaussianProcessClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html)

[Scikit-learn: Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc)

[Scikit-learn: accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

[Scikit-learn: f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

[Scikit-learn: roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

[Scikit-learn: classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

[Scikit-learn: confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)

[Scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

[Pandas: DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)

[Pandas: DataFrame.to_csv](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)

[Gaussian Process Classification (GPC) on Iris Dataset](https://www.geeksforgeeks.org/machine-learning/gaussian-process-classification-gpc-on-iris-dataset/)

[EGU Prediction of Algal Blooms via Data-drive Machine Learning Models](https://gmd.copernicus.org/articles/16/35/2023/index.html)

[Comparative assessment of artificial intelligence (AI)-based algorithms for detection of harmful bloom-forming algae](https://link.springer.com/article/10.1007/s13201-023-01919-0)
