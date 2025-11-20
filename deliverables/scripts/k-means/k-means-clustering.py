""" 
Michael McCain
CS-6840-01
Assistant Professor Dr. Wen Zhang
11/19/2025
"""

# Libraries
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Functions
def Elbow_Method(data, init_c):
  kmeans_kwargs = {
  "init": "random",
  "n_init": init_c,
  "random_state": 0,
  }
  sse = []
  for k in range(1, init_c + 1):
      kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
      kmeans.fit(data)
      sse.append(kmeans.inertia_)

  plt.plot(range(1, init_c+1), sse)
  plt.xticks(range(1, init_c+1))
  plt.xlabel("Number of Clusters")
  plt.ylabel("Sum of Squared Errors (SSE)")
  plt.show()

# Main
### Import data
data_path = Path(__file__).parent.parent.parent.parent / 'dataset-harmful-algal-bloom(HAB)' / 'HAB_Artificial_GAN_Dataset.csv'
print (data_path)
df = pd.read_csv(data_path)

col_names = df.columns[:-1]
new_col_names = []
for col in col_names:
   new_col_names.append(col + '_T')

### Transform data 
scaler = StandardScaler()
df[new_col_names] = scaler.fit_transform(df[col_names])

print(df)

### Apply PCA and identify influential features
pca = PCA(n_components=0.70)
pca.fit_transform(df[new_col_names])
component_labels = [f"PC{i+1}" for i in range(pca.n_components_)]
print(component_labels)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=df[new_col_names].columns
)
print(loadings)

top_feats = loadings.abs().idxmax()
print(top_feats)

### Identify optimum clusters using elbow method
Elbow_Method(df, 7)

### fit K-Means to scaled data
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, n_init=42)
kmeans.fit(df[new_col_names])
df['KMeans_2'] = kmeans.labels_
print(df['KMeans_2'].value_counts().sort_index())
print(df.groupby(["KMeans_2"])[new_col_names].mean())

sil_val = silhouette_score(df[new_col_names],kmeans.labels_)
print(sil_val)

### Plot over k-means scaled data


### Plot over k-means PCA data