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
from sklearn.datasets import make_blobs
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Functions
def Elbow_Method(data, init_c):
  kmeans_kwargs = {
  "init": "random",
  "n_init": init_c,
  "random_state": 1,
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
data_path = Path(__file__).parent.parent.parent.parent / 'dataset-harmful-algal-bloom(HAB)' / 'HAB_Artificial_GAN_Dataset.csv'
print (data_path)

df = pd.read_csv(data_path)

col_names = df.columns
new_col_names = []
for col in col_names:
   new_col_names.append(col + '_T')

scaler = StandardScaler()
df[new_col_names] = scaler.fit_transform(df[col_names])
print(df)

Elbow_Method(df, 7)

kmeans = KMeans(n_clusters=5)
kmeans.fit(df[new_col_names])
df['KMeans_2'] = kmeans.labels_
print(df)

sil_val = silhouette_score(df[new_col_names],kmeans.fit_predict(df[col_names]))
print(sil_val)