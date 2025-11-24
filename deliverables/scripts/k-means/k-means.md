# K-Means Clustering

This model was chosen to compare the performance of an unsupervised model with that of a labeled dataset. K-Means is relatively simplistic and efficient compared to other unsupervised models and allows for the segmentation of data into clusters. The resulting clusters work well for our use case, since our ground truth labeled data is binary. Meaning, the labels from the KMeans model can be compared against the ground truth labels and additional metrics can be used to verify their similarity.

Some limitations of this model include difficulties with datasets with many features. However, Principal Components Analysis (PCA) and other dimension reduction methods can be employed to mitigate this issue. Another limitation is that the resulting centroids may be heavily influenced by outliers. It is generally recommended to identify and remove these outliers during data preprocessing to avoid this.

### Hyperparameters

#### random_state

A random seed is used when reproducibility is desired. The value of 0 is used for completely random. When a specific value is chosen ensures deterministic results.For this project, the value 42 was chosen to ensure reproducibility of the results, but also because of the number's famous pop culture reference in the Computer Science domain.

#### n_clusters

Used to determine the number of clusters the model must create. Using the [Elbow Method](#elbow-method) is a common way to identify the optimal number of clusters for the dataset. More detailed information can be found on this in the [Additional Concepts](#additional-concepts) and [Resource](#resources) sections.

#### algorithm

Allows a user to choose k-means algorithm to use. The choices are Lloyd or Elkan. Lloyd is the default, but Elkan uses triangle inequality, thus working better with datasets that have well-defined clusters. For our use case we went with the default as it is adequate for a dataset.

#### n_init

Specifies the number of times the k-means algorithm is run with different centroid seeds and the best result is returned. Modifying this number had little to no effect on the data so the last value used (9) is what was chosen.s

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

### Elbow Method

The Elbow Method is simply the process of creating a plot with the number of clusters on the x-axis and the total sum of squared errors (SSE) on the y-axis. Then identifying where a bend appears in the plot. If multiple bends exist choosing the one with the most inertia (largest SSE drop between points) would be a good candidate to choose.

### Scaling

Since the data we use has high variance among the range and unit measurements (e.g., Temperature vs pH), scaling is required to prevent any single feature from having more influence on the data than the others.

### Principal Components Analysis (PCA)

PCA is used to reduce the dimensionality of the data set by reducing the number of variables. This is done by applying a linear transformation to the features of the passed in dataset and creating new features labeled "Components." After this transformation, the output can be analyzed to show the variance returned. According to some sources, 70-80% of variance is an acceptable cut off.

Each number represents how much an original feature contributes to a given PCA component.

- `Magnitude` indicates how strong of an influence a feature has on that component

- Direction of the relationship

  - `Positive` score means the feature increases the component score

  - `Negative` score means the feature decreases the component score

## Metrics

### Silhouette Score

The Silhouette Score evaluates how well each data point fits within its assigned cluster, and how distinctly separated it is from other clusters. Values range between -1 and 1.
A 1 means the data point fits very well in its own cluster. 0 indicates the data point is between clusters or the clusters are overlapping. -1 means the data point is in the wrong cluster.

### Normalized Mutual Index (NMI)

NMI measures the amount of shared information between the predicted clusters and the true clusters. Ranging from 0 to 1 where 1 is high similarity and 0 is no similarity.

### Adjusted Rand Index (ARI)

ARI computes another similarity metric used to measure two clusterings, adjusted for chance. Useful for determining the agreement of results between two methods or against ground truth data. A value of 1.0 indicates perfect match, a value of 0 indicates agreement equivilent to randomness, and a negative value indicates (up to -1) indicates severe disagreement. Considered more robust the NMI.

## Training Process

For this project, the Scikit-Learn library was chosen over frameworks like PyTorch or TensorFlow. One primary reason is its simplicity, familiarity from prior use, and the availability of all the necessary methods for the chosen model, which prevents the need to create the model from scratch. Instead, passing in the desired hyperparameter values dictates how the model behaves.

## Workflow

1. Import the dataset. If data contains labeled data, ensure the ground truth column is ignored or dropped for future processing.
2. Transform data using a scaler.
3. Use PCA to identify component variance.
4. Use Elbow Method to identify optimum n_clusters (Optional if desired cluster size is known).
5. Use K-Means on the scaled data.
6. Plot KMeans points in PCA Space (for easier visualization with 3D/2D graphs)
7. Perform evaluation through various metrics.

## Results

## Summary

# Resources

[Scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)

[K-Means Clustering Algorithm with Python](https://www.youtube.com/watch?v=iNlZ3IU5Ffw&t=53s)

[Elbow Method to Find Optimal Clusters](https://www.statology.org/elbow-method-in-python/)

[What is Silhouette Score?](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)

[K-Means with PCA Dimensionality Reduction](https://www.stepbystepdatascience.com/k-means-and-pca-in-python-with-sklearn)

[What is PCA?](https://365datascience.com/tutorials/python-tutorials/principal-components-analysis/)

[Clustering Centroids](https://pythonprogramminglanguage.com/kmeans-clustering-centroid/index.html)

[3D Plotting with KMeans](https://stackoverflow.com/questions/64987810/3d-plotting-of-a-dataset-that-uses-k-means)

[Clustering Performance Metrics](https://medium.com/@Sunil_Kumawat/performance-metrics-for-clustering-9badee0b7db8)

[K-Means Pros and Cons](https://developers.google.com/machine-learning/clustering/kmeans/advantages-disadvantages)
