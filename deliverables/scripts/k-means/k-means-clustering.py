""" 
Michael McCain
CS-6840-01
Assistant Professor Dr. Wen Zhang
11/19/2025
"""

# Libraries
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import dataframe_image as dfi
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.patches as mpatches

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
  plt.title('Elbow Method')
  plt.xlabel("Number of Clusters")
  plt.ylabel("Sum of Squared Errors (SSE)")
  plt.show()

# Main
### Import data
asset_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'k-means-figures'
data_path = Path(__file__).parent.parent.parent.parent / 'dataset-harmful-algal-bloom(HAB)' / 'HAB_Artificial_GAN_Dataset.csv'
print(data_path)
df = pd.read_csv(data_path)

hab_count = df['HAB_Present'].value_counts().sort_index()
df_hab = hab_count.to_frame(name='hab-count')
dfi.export(df_hab, filename=f'{asset_path}\\hab-data-count.png',dpi=300)

col_names = df.columns[:-1]
new_col_names = []
for col in col_names:
  new_col_names.append(col + '_T')

### Transform data 
scaler = StandardScaler()
df[new_col_names] = scaler.fit_transform(df[col_names])
print(df[new_col_names])

### Apply PCA, plot, identify influential features, and reduce
pca = PCA()
pca.fit(df[new_col_names])

## PCA Plot
plt.figure(figsize=(12,9))
plt.plot(range(1,len(pca.explained_variance_ratio_.cumsum())+ 1), pca.explained_variance_ratio_.cumsum(),marker='o', linestyle='--')
plt.title('Variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

## PCA reduce
pca_red = PCA(n_components=0.70)
pca_red.fit_transform(df[new_col_names])

component_labels = [f"PC{i+1}" for i in range(pca_red.n_components_)]
print(component_labels)

loadings = pd.DataFrame(
    pca_red.components_.T,
    columns=[f"PC{i+1}" for i in range(pca_red.n_components_)],
    index=df[new_col_names].columns
)
print(loadings)
dfi.export(loadings, filename=f'{asset_path}\\pc-loadings.png', dpi=300)

top_feats = loadings.abs().idxmax()
print(top_feats)

### Identify optimum clusters using elbow method
Elbow_Method(df[new_col_names], 7)

### fit K-Means to scaled data
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, n_init=42).fit(df[new_col_names])
df['KMeans_2'] = kmeans.labels_

cluster_value_count = df['KMeans_2'].value_counts().sort_index()
df_clus = cluster_value_count.to_frame(name='kmeans-count')
dfi.export(df_clus, filename=f'{asset_path}\\cluster-data-count.png',dpi=300)

# print(df.groupby(["KMeans_2"])[new_col_names].mean())
sil_val = silhouette_score(df[new_col_names],kmeans.labels_)
print(sil_val)

### Plot over k-means PCA data
df_pca = PCA(n_components=0.70).set_output(transform="pandas").fit_transform(df[new_col_names])
centers_pca = PCA(n_components=0.70).set_output(transform="pandas").fit(df[new_col_names]).transform(kmeans.cluster_centers_)

fig_3d_scatter = plt.figure()
ax = fig_3d_scatter.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df_pca['pca0'], df_pca['pca1'], df_pca['pca2'],
    c=df['KMeans_2'], cmap='winter', s=50)
ax.scatter(
    centers_pca['pca0'], centers_pca['pca1'], centers_pca['pca2'],
    c='red', s=250, marker='X', label='Cluster Centers', edgecolor='k', zorder=1000)
for i, (x, y, z) in enumerate(centers_pca[['pca0','pca1','pca2']].values):
    ax.text(x, y, z, f'Center {i}', color='red', fontsize=12, fontweight='bold', zorder=1000)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

handles, _ = scatter.legend_elements(prop="colors")
labels = [f'Cluster {i}' for i in sorted(df['KMeans_2'].unique())]
center_patch = mpatches.Patch(color='red', label='Cluster Centers')
ax.legend(handles=handles + [center_patch], labels=labels + ['Cluster Centers'])

plt.title('KMeans on Scaled Data / Plotted in PCA space')
plt.show()

### Compare clustering vs ground truth
ari = adjusted_rand_score(df['HAB_Present'], df['KMeans_2'])
nmi = normalized_mutual_info_score(df['HAB_Present'], df['KMeans_2'])
print("ARI:", ari, "NMI:", nmi)

df_metrics = pd.DataFrame([[sil_val,ari,nmi]], columns=['Silhouette_Score','Adjusted_Rand_Inde','Normalized_Mutual_Information'])
dfi.export(df_metrics, filename=f'{asset_path}\\evaluation-metrics.png',dpi=300)
