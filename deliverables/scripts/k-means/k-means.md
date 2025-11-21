# K-Means Clustering

This model was chosen to compare the performance of an unsupervised model with that of a labeled dataset. K-Means is relatively simplistic and efficient compared to other unsupervised models and allows for the segmentation of data into clusters. The resulting clusters work well for our use case, since our ground truth labeled data is binary. Meaning, the labels from the KMeans model can be compared against the ground truth labels and additional metrics can be used to verify their similarity.

Some limitations of this model include difficulties with datasets with many features. However, Principal Components Analysis (PCA) and other dimension reduction methods can be employed to mitigate this issue. Another limitation is that the resulting centroids may be heavily influenced by outliers. It is generally recommended to identify and remove these outliers during data preprocessing to avoid this.

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

The Silhouette Score evaluates how well each data point fits within its assigned cluster, and how distinctly separated it is from other clusters. Values range between -1 and 1.
A 1 means the data point fits very well in its own cluster. 0 indicates the data point is between clusters or the clusters are overlapping. -1 means the data point is in the wrong cluster.

### Normalized Mutual Index (NMI)

NMI measures the amount of shared information between the predicted clusters and the true clusters. Ranging from 0 to 1 where 1 is high similarity and 0 is no similarity.

### Adjusted Rand Index (ARI)

ARI computes another similarity metric used to measure two clusterings, adjusted for chance. Useful for determining the agreement of results between two methods or against ground truth data. A value of 1.0 indicates perfect match, a value of 0 indicates agreement equivilent to randomness, and a negative value indicates (up to -1) indicates severe disagreement. Considered more robust the NMI.

## Full workflow

1. Import the dataset. If data contains labeled data, ensure the ground truth column is ignored or dropped for future processing.
2. Transform data using a scaler.
3. Reduce dimensionality with PCA.
4. Use Elbow Method to identify optimum n_clusters (Optional if desired cluster size is known).
5. Implement K-Means using scaled data and calculate Silhouette Score.
6. Plot using PCA and compare against ground truth.

# Resources

[K-Means Clustering Algorithm with Python](https://www.youtube.com/watch?v=iNlZ3IU5Ffw&t=53s)

[Elbow Method to Find Optimal Clusters](https://www.statology.org/elbow-method-in-python/)

[What is Silhouette Score?](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)

[K-Means with PCA Dimensionality Reduction](https://www.stepbystepdatascience.com/k-means-and-pca-in-python-with-sklearn)

[What is PCA?](https://365datascience.com/tutorials/python-tutorials/principal-components-analysis/)

[Clustering Centroids](https://pythonprogramminglanguage.com/kmeans-clustering-centroid/index.html)

[3D Plotting with KMeans](https://stackoverflow.com/questions/64987810/3d-plotting-of-a-dataset-that-uses-k-means)

[Clustering Performance Metrics](https://medium.com/@Sunil_Kumawat/performance-metrics-for-clustering-9badee0b7db8)

[K-Means Pros and Cons](https://developers.google.com/machine-learning/clustering/kmeans/advantages-disadvantages)
