# K-Means Clustering

> description and why this model was chosen and limitations

## Steps

1. Define number of clusters (2)
2. Randomly select datapoints (for initial clusters)
3. Measure distance between first initial cluster and the other (euclidian distance)
4. Assign first point to the nearest cluster centroid
   a. Repeat steps 3 and 4 for each point
5. Calculate mean point of each cluster
6. Repeat steps 3 and 5 using the mean values
7. Repeat step 6 until no further changes occur to the datapoints.

# Resources

[K-Means Clustering Algorithm with Python](https://www.youtube.com/watch?v=iNlZ3IU5Ffw&t=53s)
[Elbow Method to Find Optimal Clusters](https://www.statology.org/elbow-method-in-python/)
[What is Silhouette Score?](https://www.geeksforgeeks.org/machine-learning/what-is-silhouette-score/)
