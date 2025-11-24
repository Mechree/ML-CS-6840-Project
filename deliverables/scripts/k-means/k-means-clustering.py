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
from sklearn.metrics import calinski_harabasz_score

# Functions
def Elbow_Method(data, init_c):
  kmeans_kwargs = {
  "init": "random",
  "n_init": init_c,
  "random_state": 42,
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
def Metrics(data_frame, df_features, kmeans_labels, truth_label, pred_label, filepath):
    # Compare clustering vs ground truth
    sil_val = silhouette_score(data_frame[df_features], kmeans_labels)
    ari = adjusted_rand_score(data_frame[truth_label], data_frame[pred_label])
    nmi = normalized_mutual_info_score(data_frame[truth_label], data_frame[pred_label])
    print(f'Silhouette Score: {sil_val} \nARI: {ari} \nNMI: {nmi}')

    # Save metrics as image
    df_metrics = pd.DataFrame([[sil_val,ari,nmi]], columns=['Silhouette_Score','Adjusted_Rand_Index','Normalized_Mutual_Information'])
    dfi.export(df_metrics, filepath, dpi=300)
def Plot_Component_Variance(pca_obj, file_path):
    plt.figure(figsize=(12,9))
    plt.plot(range(1, len(pca_obj.explained_variance_ratio_.cumsum()) + 1), pca_obj.explained_variance_ratio_.cumsum(),marker='o', linestyle='--')
    plt.title('Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.savefig(file_path)
    plt.show()
    return
def Cluster_Counts(df, feature, col_label, file_path):
    cluster_value_count = df[feature].value_counts().sort_index()
    df_clus = cluster_value_count.to_frame(name=col_label)
    dfi.export(df_clus, file_path,dpi=300)
def VRC_Comp(num_c_start, rg, dataframe, features, label, filepath):
    num_clusters = num_c_start

    for i in range(rg):
        kmeans = KMeans(n_clusters=num_clusters, n_init=9, random_state=42).fit(dataframe[features])
        dataframe['KMeans-Cluster'] = kmeans.labels_
        vrc = calinski_harabasz_score(dataframe[features], dataframe[label])
        df_vrc = pd.DataFrame([vrc], columns=['Variance Ratio Criterion'])
        file_name = f'vrc-{num_clusters}C.png'
        full_filepath = f'{filepath}\\{file_name}'
        dfi.export(df_vrc, full_filepath, dpi=300)
        num_clusters += 1

# Main
### Import data
asset_path = Path(__file__).parent.parent.parent.parent / 'assets' / 'k-means-figures'
data_path = Path(__file__).parent.parent.parent.parent / 'dataset-harmful-algal-bloom(HAB)' / 'HAB_Artificial_GAN_Dataset.csv'
df = pd.read_csv(data_path)

Cluster_Counts(df, 'HAB_Present', 'HAB-count', f'{asset_path}\\hab-data-count.png')

col_names = df.columns[:-1]
scaled_feats = []
for col in col_names:
  scaled_feats.append(col + '_T')

### Transform data using scaler 
scaler = StandardScaler()
df[scaled_feats] = scaler.fit_transform(df[col_names])

### Apply PCA, plot variance, identify influential features, and reduce
pca = PCA()
pca.fit(df[scaled_feats])
Plot_Component_Variance(pca, f'{asset_path}\\pca-component-variance.png')

## PCA reduce
pca_red = PCA(n_components=0.70)
pca_red.fit_transform(df[scaled_feats])

component_labels = [f"PC{i+1}" for i in range(pca_red.n_components_)]
print(component_labels)

loadings = pd.DataFrame(
    pca_red.components_.T,
    columns=[f"PC{i+1}" for i in range(pca_red.n_components_)],
    index=df[scaled_feats].columns
)
print(loadings)
dfi.export(loadings, filename=f'{asset_path}\\pc-loadings.png', dpi=300)

top_feats = loadings.abs().idxmax()
print(top_feats)

### Identify optimum clusters using elbow method
Elbow_Method(df[scaled_feats], 7)

### Verify optimum clusters using VRC
VRC_Comp(2, 4, df, scaled_feats, 'KMeans-Cluster', asset_path)

### Fit K-Means to scaled data
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, n_init=9, random_state=42).fit(df[scaled_feats])
df['KMeans-Cluster'] = kmeans.labels_
Cluster_Counts(df, 'KMeans-Cluster', 'kmeans-count', f'{asset_path}\\cluster-data-count.png')


### Plot kmeans scaled data in PCA space
df_pca = PCA(n_components=0.70).set_output(transform="pandas").fit_transform(df[scaled_feats])
centers_pca = PCA(n_components=0.70).set_output(transform="pandas").fit(df[scaled_feats]).transform(kmeans.cluster_centers_)

fig_3d_scatter = plt.figure()
ax = fig_3d_scatter.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df_pca['pca0'], df_pca['pca1'], df_pca['pca2'],
    c=df['KMeans-Cluster'], cmap='winter', s=50)
ax.scatter(
    centers_pca['pca0'], centers_pca['pca1'], centers_pca['pca2'],
    c='red', s=250, marker='X', label='Cluster Centers', edgecolor='k', zorder=1000)
for i, (x, y, z) in enumerate(centers_pca[['pca0','pca1','pca2']].values):
    ax.text(x, y, z, f'Center {i}', color='red', fontsize=12, fontweight='bold', zorder=1000)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

handles, _ = scatter.legend_elements(prop="colors")
labels = [f'Cluster {i}' for i in sorted(df['KMeans-Cluster'].unique())]
center_patch = mpatches.Patch(color='red', label='Cluster Centers')
ax.legend(handles=handles + [center_patch], labels=labels + ['Cluster Centers'])
plt.title('KMeans on Scaled Data / Plotted in PCA space')
plt.show()

### Compare clustering vs ground truth and obtain silhouette score
Metrics(df, scaled_feats, kmeans.labels_,'HAB_Present', 'KMeans-Cluster', f'{asset_path}\\evaluation-metrics.png')