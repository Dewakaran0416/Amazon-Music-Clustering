import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print('All libraries imported successfully!')

file_path = r"D:\Python\Project\Project 4\single_genre_artists.csv"
df = pd.read_csv(file_path)
print('Column Names:')
print(df.columns.tolist())
print('Data Types:')
print(df.dtypes)
print('Missing Values:')
print(df.isnull().sum())
# This dataset has NO missing values!
print('Number of duplicate rows:', df.duplicated().sum())
# This dataset has NO duplicates!
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence',
                  'tempo', 'duration_ms']
df[audio_features].describe().round(3)
#Plot distributions of all audio features
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle('Distribution of Audio Features', fontsize=16, fontweight='bold')
axes = axes.flatten()

colors = ['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6',
          '#1abc9c','#e67e22','#34495e','#d35400','#16a085']

for i, feat in enumerate(audio_features):
    axes[i].hist(df[feat], bins=40, color=colors[i], edgecolor='white', alpha=0.85)
    axes[i].set_title(feat, fontsize=11)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: feature_distributions.png')
#Correlation heatmap
plt.figure(figsize=(11, 8))
corr = df[audio_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Audio Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: correlation_heatmap.png')

# These columns are just text/metadata — NOT useful for clustering
columns_to_drop = ['id_songs', 'name_song', 'id_artists', 'release_date',
                   'genres', 'name_artists', 'explicit', 'key', 'mode',
                   'time_signature', 'popularity_songs', 'popularity_artists', 'followers']

df_clean = df.drop(columns=columns_to_drop)

print('Columns kept for clustering:')
print(df_clean.columns.tolist())

# 10 audio features that describe how a song SOUNDS
selected_features = ['danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence',
                     'tempo', 'duration_ms']

X = df[selected_features].copy()

print('Feature matrix shape:', X.shape)
X.head()

# StandardScaler makes mean=0 and std=1 for each feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Scaling done!')
print('Mean of scaled data (should be ~0):', X_scaled.mean(axis=0).round(3))
print('Std  of scaled data (should be ~1):', X_scaled.std(axis=0).round(3))

inertia_values = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertia_values.append(kmeans.inertia_)
    # sample_size=5000 speeds things up for large datasets
    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    silhouette_scores.append(sil)
    print(f'k={k}  Inertia={kmeans.inertia_:.0f}  Silhouette={sil:.4f}')

best_k_idx = silhouette_scores.index(max(silhouette_scores))
print('\nBest k by silhouette:', list(k_range)[best_k_idx])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(k_range), inertia_values, marker='o', color='#3498db', linewidth=2.5, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Optimal k=3')
axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (SSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), silhouette_scores, marker='s', color='#e74c3c', linewidth=2.5, markersize=8)
axes[1].axvline(x=3, color='blue', linestyle='--', alpha=0.7, label='Best k=3')
axes[1].set_title('Silhouette Score vs k', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: elbow_silhouette.png')
print('Conclusion: k=3 gives the best silhouette score (0.2413)')

best_k = 3

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

print(f'K-Means model trained with k={best_k}')
print('\nNumber of songs in each cluster:')
print(pd.Series(cluster_labels).value_counts().sort_index())

# Add cluster column to original dataframe
df['Cluster'] = cluster_labels

print('Cluster column added!')
df[['name_song', 'name_artists', 'Cluster']].head(10)

sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=5000, random_state=42)
db_score  = davies_bouldin_score(X_scaled, cluster_labels)

print(f'Silhouette Score     : {sil_score:.4f}  (range -1 to 1, higher is better)')
print(f'Davies-Bouldin Index : {db_score:.4f}  (lower is better)')
print(f'Inertia (SSE)        : {final_kmeans.inertia_:.0f}')

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print('Explained Variance:')
print(f'  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%')
print(f'  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%')
print(f'  Total: {sum(pca.explained_variance_ratio_)*100:.1f}%')

cluster_colors = ['#e74c3c', '#3498db', '#2ecc71']
cluster_plot_names = [
    'Cluster 0: Speechy / Short Tracks',
    'Cluster 1: Calm / Acoustic Tracks',
    'Cluster 2: Energetic / Loud Tracks'
]

plt.figure(figsize=(10, 7))
for c in range(best_k):
    mask = cluster_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=cluster_colors[c], label=cluster_plot_names[c],
                alpha=0.4, s=10)

plt.title('K-Means Clusters Visualized using PCA (2D)', fontsize=14, fontweight='bold')
plt.xlabel('PC1 (27.1% variance)')
plt.ylabel('PC2 (18.8% variance)')
plt.legend(markerscale=3, fontsize=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('cluster_pca_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cluster_pca_plot.png')

# Average feature values per cluster
cluster_profile = df.groupby('Cluster')[selected_features].mean()
print('Average feature values per cluster:')
cluster_profile.round(3)

# Bar chart of normalized feature values
profile_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())

profile_norm.T.plot(kind='bar', figsize=(14, 6), colormap='Set1', edgecolor='black', width=0.7)
plt.title('Normalized Average Feature Values per Cluster', fontsize=13, fontweight='bold')
plt.xlabel('Audio Feature')
plt.ylabel('Normalized Value (0 to 1)')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Cluster', labels=['Cluster 0','Cluster 1','Cluster 2'])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_feature_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cluster_feature_bar.png')

# Heatmap of cluster profiles
plt.figure(figsize=(13, 4))
sns.heatmap(profile_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, xticklabels=selected_features,
            yticklabels=[f'Cluster {i}' for i in range(best_k)])
plt.title('Heatmap: Cluster Feature Profiles', fontsize=13, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cluster_heatmap.png')

# Assign names based on the actual cluster feature means:
# Cluster 0: speechiness=0.83 (highest), duration_ms=97520 (shortest)  -> Speechy / Short Tracks
# Cluster 1: acousticness=0.75 (highest), energy=0.31 (lowest)         -> Calm / Acoustic Tracks
# Cluster 2: energy=0.69, loudness=-7.6dB (loudest), tempo=124.9 BPM   -> Energetic / Loud Tracks

cluster_names = {
    0: 'Speechy / Short Tracks',
    1: 'Calm / Acoustic Tracks',
    2: 'Energetic / Loud Tracks'
}

df['Cluster_Name'] = df['Cluster'].map(cluster_names)

print('Songs per cluster:')
print(df['Cluster_Name'].value_counts())

for c in range(best_k):
    print(f'\n--- Cluster {c}: {cluster_names[c]} ---')
    sample = df[df['Cluster'] == c][['name_song', 'name_artists', 'genres']].head(5)
    print(sample.to_string(index=False))
    
    df.to_csv('amazon_music_clustered_output.csv', index=False)

print('File saved: amazon_music_clustered_output.csv')
print('Shape:', df.shape)
df[['name_song', 'name_artists', 'danceability', 'energy', 'Cluster', 'Cluster_Name']].head()

print('=' * 62)
print('       AMAZON MUSIC CLUSTERING - FINAL SUMMARY')
print('=' * 62)
print(f'Total songs in dataset        : {len(df):,}')
print(f'Total features used           : {len(selected_features)}')
print(f'Best number of clusters (k)   : {best_k}')
print(f'Silhouette Score              : {sil_score:.4f}')
print(f'Davies-Bouldin Index          : {db_score:.4f}')
print(f'Inertia (SSE)                 : {final_kmeans.inertia_:.0f}')
print()
print('Cluster Breakdown:')
for c in range(best_k):
    count = (df['Cluster'] == c).sum()
    pct = count / len(df) * 100
    print(f'  Cluster {c} - {cluster_names[c]}: {count:,} songs ({pct:.1f}%)')
print('=' * 62)