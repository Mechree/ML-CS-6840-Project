# K-Means Clustering

> description and why this model was chosen and limitations

## How K-Means works

1. Define number of clusters
2. Randomly select datapoints (for initial clusters)
3. Measure distance between first initial cluster and the other (Euclidian distance)
4. Assign first point to the nearest cluster centroid

   a. Repeat steps 3 and 4 for each point

5. Calculate mean point of each cluster
6. Repeat steps 3 and 5 using the mean values
7. Repeat step 6 until no further changes occur to the datapoints.

## Additional Concepts

### Scaling

Since the data we use has high variance among the range and unit measurements (e.g., Temperature vs pH), scaling is required to prevent any single feature from having more influence on the data than the others.

### Principal Components Analysis (PCA)

PCA is used to reduce the dimensionality of the data set by reducing the number of variables. This is done by applying a linear transformation to the features of the passed in dataset and creating new features labeled "Components." After this transformation, the output can be analyzed to show the variance returned. According to some sources, 70-80% of variance is an acceptable cut off.

Each number represents how much an original feature contributes to a given PCA component.

- `Magnitude` indicates how strong of an influence a feature has on that component

- Direction of the relationship

  - `Positive` score means the feature increases the component score

  - `Negative` score means the feature decreases the component score

### Silhouette Score

The Silhouette Score evaluates how well each data point fits within its assigned cluster, and how distinctly separated it is from other clusters.

## Full workflow

1. Import the dataset. If data contains labeled data, ensure the ground truth column is ignored or dropped for future processing.
2. Transform data using a scaler.
3. Reduce dimensionality with PCA.
4. Use Elbow Method to identify optimum n_clusters (Optional if desired cluster size is known).
5. Implement K-Means using scaled data and calculate Silhouette Score.
6. Plot data and compare with labeled dataset.

# Resources

[K-Means Clustering Algorithm with Python](https://www.youtube.com/watch?v=iNlZ3IU5Ffw&t=53s)

[Elbow Method to Find Optimal Clusters](https://www.statology.org/elbow-method-in-python/)

[What is Silhouette Score?](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)

[K-Means with PCA Dimensionality Reduction](https://www.stepbystepdatascience.com/k-means-and-pca-in-python-with-sklearn)

[What is PCA?](https://365datascience.com/tutorials/python-tutorials/principal-components-analysis/)
