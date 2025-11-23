# Logistic Regression

This model was chosen as a supervised learning approach to classify harmful algal blooms. Logistic regression is a simple yet effective binary classification algorithm that works well for linearly separable data. The model outputs a probability that can be compared against ground truth labels to evaluate classification performance. By training on multiple train/test splits, we can analyze overfitting and underfitting behaviors to understand model generalization.

## How Logistic Regression Works

1. Initialize model weights randomly
2. For each training sample, compute predicted probability using sigmoid function
3. Calculate loss using Binary Cross-Entropy (BCE)
4. Compute gradients of loss with respect to weights
5. Update weights using gradient descent (via optimizer)
6. Repeat steps 2-5 for all epochs
7. Early stopping: stop if validation loss stops improving
8. Evaluate final model on test set

## Additional Concepts

### Standardization (Feature Scaling)

Since hyperspectral data contains features with different units and ranges (e.g., algal biomass vs. chlorophyll-a concentration), standardization is required to prevent any single feature from dominating the model. Standardization transforms features to have mean=0 and standard deviation=1.

### Train/Test/Validation Split

Data is divided into three sets:
- **Training set:** Used to train the model (adjust weights)
- **Validation set:** Used during training to monitor performance and enable early stopping
- **Test set:** Used to evaluate final model performance (never seen during training)

Different split ratios reveal different model behaviors:
- **20:80 (20% train):** Demonstrates underfitting with insufficient training data
- **40:60 (40% train):** Balanced approach with moderate training data
- **80:20 (80% train):** Standard practice with abundant training data; may show slight overfitting

### Learning Rate

The learning rate controls the step size when updating model weights during training. A learning rate that is too high causes instability and divergence. A learning rate that is too low results in slow training and may get stuck in local minima. The default value of 0.01 is a reasonable starting point.

### Epochs

Epochs represent the number of times the model iterates through the entire training dataset. More epochs allow the model to learn better but increase computation time. Early stopping prevents unnecessary training by halting when validation loss stops improving.

### Patience (Early Stopping)

Patience is the number of consecutive epochs to wait for validation loss improvement before stopping training. If validation loss does not improve for "patience" epochs, training terminates. This prevents overfitting and saves computation time. The default value of 15 is a reasonable threshold.

### Sigmoid Function

The sigmoid function maps model output to a probability between 0 and 1:

```
σ(z) = 1 / (1 + e^-z)
```

This allows logistic regression to output interpretable probabilities. A probability > 0.5 is classified as HAB present (class 1), while < 0.5 is classified as no HAB (class 0).

### Binary Cross-Entropy Loss

Binary Cross-Entropy (BCE) measures the difference between predicted probabilities and true labels:

```
BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

Where y is the true label and ŷ is the predicted probability. Minimizing BCE during training improves model accuracy.

## Performance Metrics

### Accuracy

The proportion of correct predictions:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Useful as a general metric but can be misleading with imbalanced datasets.

### Precision

Of predicted HAB cases, how many were actually HAB:
```
Precision = TP / (TP + FP)
```

Important when false positives are costly.

### Recall

Of actual HAB cases, how many were correctly identified:
```
Recall = TP / (TP + FN)
```

Important when false negatives are costly (missing blooms is problematic).

### F1-Score

The harmonic mean of precision and recall:
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

Provides a balanced measure when both precision and recall matter.

## Learning Curves

Learning curves plot training loss and validation loss over epochs. They reveal model behavior:

**Underfitting (20:80 split):**
- Both training and validation losses remain high
- Lines are roughly parallel with minimal improvement
- Model is not learning the training data well

**Good Fit (40:60 split):**
- Both losses decrease smoothly and converge
- Train and validation losses remain close
- Model generalizes well to unseen data

**Overfitting (80:20 split):**
- Training loss continues to decrease
- Validation loss plateaus or increases
- Growing gap between training and validation loss
- Model memorizes training data instead of learning generalizable patterns

## Full Workflow

1. Import the dataset and separate features from target label
2. Standardize features using StandardScaler
3. Split data into training, validation, and test sets
4. For each train/test split ratio (20:80, 40:60, 80:20):
   - Create PyTorch DataLoaders for batching
   - Initialize logistic regression model
   - Train model with early stopping
   - Evaluate on test set and calculate metrics
   - Plot learning curves to visualize training dynamics
5. Compare results across splits to analyze overfitting/underfitting
6. Export summary table and visualizations

## Resources

- [Logistic Regression Explained](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Understanding Overfitting and Underfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [Binary Cross-Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
- [Gradient Descent and Optimization](https://www.deeplearningbook.org/contents/optimization.html)
- [ROC and AUC Explained](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Feature Scaling Methods](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-mean-removal-and-variance-scaling)
- [Early Stopping Regularization](https://en.wikipedia.org/wiki/Early_stopping)
