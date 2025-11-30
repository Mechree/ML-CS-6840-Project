# 3.2 Logistic Regression

Logistic regression was selected as a supervised learning method for binary classification of harmful algal blooms using hyperspectral-derived features. Unlike unsupervised approaches such as K-Means, logistic regression leverages labeled ground truth data to directly learn the decision boundary between bloom and non-bloom classes. This makes it well-suited for problems where explicit labels are available and the goal is predictive accuracy rather than exploratory clustering.

Logistic regression is a method that models the probability of a binary outcome using the logistic function. Its simplicity, interpretability, and computational efficiency make it an excellent simple model for classification tasks. Additionally, logistic regression performs well on linearly separable data and provides probability estimates that can be threshold-adjusted for different operational requirements.

The core of logistic regression is the sigmoid function, which maps any real-valued input to a probability between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = \mathbf{w}^T \mathbf{x} + b$ is the linear combination of features and learned weights. The output represents the predicted probability that a sample belongs to the positive class (HAB present).

Training is accomplished by minimizing the Binary Cross-Entropy (BCE) loss function:

$$L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

where $y_i$ is the true label (0 or 1) and $\hat{y}_i$ is the predicted probability. Model weights are optimized using the Adam optimizer, which combines adaptive learning rates with momentum to efficiently navigate the loss landscape.

## 3.2.1 Hyperparameters

1. **Learning Rate (lr):** Controls the step size during weight updates when optimizing model parameters. A learning rate that is too high causes instability and divergence, while a learning rate that is too low results in slow training. For this project, 0.01 was selected, giving a trade-off between slow results and unsatble learning.

2. **Batch Size:** The number of samples processed before updating model weights. Larger batch sizes reduce gradient noise but may converge to suboptimal solutions, while smaller batches provide noisier gradients that may allow better exploration of the loss landscape. A batch size of 32 was chosen as it is standard.

3. **Epochs:** The maximum number of complete passes through the training dataset. More epochs allow better convergence but increase computation time and risk of overfitting. For this project, 200 epochs was selected as a reasonable upper bound; however, early stopping (see below) typically terminates training before reaching this limit.

4. **Patience (Early Stopping):** The number of consecutive epochs without validation loss improvement before training terminates. Early stopping is a regularization technique that prevents overfitting by halting training when generalization performance plateaus. A patience value of 15 was chosen to provide reasonable tolerance for noisy validation metrics while preventing excessive computation on stalled training.

## 3.2.2 Training Process

### Implementation Framework

PyTorch was selected as the implementation framework for logistic regression due to its automatic differentiation capabilities (autograd) for efficient gradient computation, and flexible tensor operations.


### Train/Validation/Test Split Strategy

Data is partitioned into three sets to enable proper model evaluation:

- **Training set:** Used to update model weights via gradient descent
- **Validation set:** Used during training to monitor performance and trigger early stopping (20% of training data)
- **Test set:** Held-out evaluation set used to assess final model generalization (never seen during training)

Three different train/test split ratios are explored to analyze the bias-variance trade-off:

| Split Ratio | Train Size | Test Size | Expected Behavior |
|------------|-----------|-----------|-------------------|
| 20:80      | 20%       | 80%       | Underfitting - insufficient training data |
| 40:60      | 40%       | 60%       | Balanced - moderate training data |
| 80:20      | 80%       | 20%       | Standard - abundant training data, potential overfitting |

For each split, the training data is further divided into 80% training and 20% validation to enable early stopping during the learning process.

## 3.2.3 Results

### Performance Metrics by Split

**Table 1: Performance Metrics by Split**

| Split  | Accuracy | Precision | Recall   | F1-Score |
|--------|----------|-----------|----------|----------|
| 20:80  | 0.9931   | 0.9808    | 0.9771   | 0.9790   |
| 40:60  | 0.9950   | 0.9801    | 0.9900   | 0.9850   |
| 80:20  | 0.9975   | 1.0000    | 0.9848   | 0.9924   |

All three split ratios achieved excellent classification performance, with accuracy exceeding 99% across all experiments. The 80:20 split demonstrated the highest accuracy (99.75%) and perfect precision (1.0000), indicating no false positive predictions on the test set. The 40:60 split showed balanced performance with strong recall (99.00%), while the 20:80 split, despite having limited training data, still resulted in a 99.31% accuracy.

### Misclassifications Analysis

The model produced very few misclassifications across all splits. The 20:80 split had 22 total misclassifications (16 false positives, 6 false negatives), the 40:60 split had 7 misclassifications (4 false positives, 3 false negatives), and the 80:20 split had only 1 total misclassification (1 false negative). This dramatic reduction in errors with increased training data demonstrates the importance of sufficient training samples for model generalization.

**Figure 1: Misclassifications by Split**
![Misclassifications - 20:80 Split](/assets/logistic-regression/misclassifications_20_80.png)
![Misclassifications - 40:60 Split](/assets/logistic-regression/misclassifications_40_60.png)
![Misclassifications - 80:20 Split](/assets/logistic-regression/misclassifications_20_80.png)

### Learning Curves and Decision Boundaries

Learning curves reveal that logistic regression efficiently learns the HAB classification task with fast convergence and little overfitting. The 80:20 split achieved the best generalization with both training and validation loss converging within the first 10 epochs. The 3D PCA visualizations show that the model successfully identifies decision boundaries in the feature space, with misclassifications occurring predominantly at the boundary regions between bloom and non-bloom classes.

**Figure 2: 3D PCA Visualization - 20:80 Split**
![3D PCA of Test Set with Misclassifications - 20:80 Split](/assets/logistic-regression/pca_misclassifications_20_80.png)

The 3D PCA plot reveals spatial distribution of misclassifications. In the 20:80 split, misclassified samples cluster near the decision boundary in PCA space, indicating that these examples are inherently ambiguous and lie in overlapping regions. This is expected behavior for a linear model and suggests that the core classification task is well-defined.

## 3.2.4 Logistic Regression Summary

Logistic regression proved to be an effective supervised learning approach for binary classification of harmful algal blooms using hyperspectral-derived features. The model achieved excellent performance across all train/test split ratios, with accuracy consistently exceeding 99%. The 80:20 split demonstrated the highest performance (99.75% accuracy, 1.0000 precision), while remarkably, the 20:80 split also achieved strong results (99.31% accuracy) despite severe data limitations.

A key observation is that all three splits exhibited well-behaved learning curves with minimal divergence between training and validation loss, indicating that overfitting is not a concern for this problem. The model successfully learned generalizable decision boundaries across different training set sizes, suggesting that the extracted hyperspectral features are highly discriminative for HAB detection.

It is important to note that this dataset is artificially generated using generative models, which may contribute to the exceptional classification performance observed. Synthetic data often exhibits cleaner class separation and fewer edge cases compared to real-world hyperspectral imagery. The high performance should be interpreted with the understanding that real-world HAB detection scenarios may present additional challenges such as sensor noise, environmental variability, and more ambiguous bloom signatures. However, these results demonstrate that logistic regression serves as a strong baseline model and validates the quality of the feature engineering performed on the hyperspectral data.

Overall, logistic regression provides a computationally efficient, interpretable, and accurate foundation for automated HAB detection. The simplicity and strong performance of this approach suggest it could serve as an effective tool for operational deployment, particularly as a baseline for comparison with more complex supervised learning methods.

---

## Resources

- [Scikit-learn: Logistic Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [Binary Cross-Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
- [Understanding Overfitting and Underfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [Bias-Variance Trade-off](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [Early Stopping Regularization](https://en.wikipedia.org/wiki/Early_stopping)
- [Adam Optimizer](https://arxiv.org/abs/1412.6980)
- [Feature Scaling and Standardization](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-mean-removal-and-variance-scaling)
- [ROC Curves and AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Precision, Recall, and F1-Score](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-and-jaccard-similarity)