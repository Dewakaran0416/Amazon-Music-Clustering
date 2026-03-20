# 🎵 Amazon Music Clustering

> Automatically grouping 95,837 Amazon Music songs into meaningful clusters based on audio features — using unsupervised machine learning (K-Means Clustering).

---

## 📌 Project Overview

With millions of songs on streaming platforms, manually categorizing tracks by genre or mood is impossible. This project uses **K-Means Clustering** to automatically group songs based on how they *sound* — analyzing features like energy, tempo, danceability, acousticness, and more — without any prior genre labels.

**Result:** 3 distinct, musically meaningful clusters discovered from 95,837 songs.

---

## 🗂️ Repository Structure

```
amazon-music-clustering/
│
├── Amazon_Music_Clustering.ipynb       # Main Jupyter Notebook (14 steps)
├── single_genre_artists.csv            # Dataset (95,837 songs × 23 features)
├── amazon_music_clustered_output.csv   # Final output with cluster labels
│
├── plots/
│   ├── feature_distributions.png       # Histograms of all audio features
│   ├── correlation_heatmap.png         # Feature correlation matrix
│   ├── elbow_curve.png                 # Elbow method plot
│   ├── silhouette_scores.png           # Silhouette score vs k
│   ├── cluster_pca_plot.png            # 2D cluster visualization (PCA)
│   ├── cluster_feature_bar.png         # Feature averages per cluster
│   └── cluster_heatmap.png             # Cluster vs feature heatmap
│
├── Amazon_Music_Clustering_Report.docx # Full project report
└── README.md                           # This file
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **File** | `single_genre_artists.csv` |
| **Rows** | 95,837 songs |
| **Columns** | 23 (10 audio features used for clustering) |
| **Missing Values** | None |
| **Duplicates** | None |

### Audio Features Used for Clustering

| Feature | Description |
|---|---|
| `danceability` | How suitable a track is for dancing (0.0 – 1.0) |
| `energy` | Perceptual intensity and activity (0.0 – 1.0) |
| `loudness` | Overall loudness in decibels (dB) |
| `speechiness` | Presence of spoken words (0.0 – 1.0) |
| `acousticness` | Confidence the track is acoustic (0.0 – 1.0) |
| `instrumentalness` | Likelihood the track has no vocals (0.0 – 1.0) |
| `liveness` | Presence of live audience (0.0 – 1.0) |
| `valence` | Musical positiveness / happiness (0.0 – 1.0) |
| `tempo` | Estimated tempo in BPM |
| `duration_ms` | Track duration in milliseconds |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/amazon-music-clustering.git
cd amazon-music-clustering
```

### 2. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook Amazon_Music_Clustering.ipynb
```

> Make sure `single_genre_artists.csv` is in the same folder as the notebook before running.

---

## 🚀 Project Steps (Notebook Walkthrough)

| Step | Description |
|---|---|
| **Step 1** | Import all required libraries |
| **Step 2** | Load the dataset |
| **Step 3** | Exploratory Data Analysis — distributions, correlations |
| **Step 4** | Data Preprocessing — drop non-audio columns |
| **Step 5** | Feature Selection — 10 audio features |
| **Step 6** | Feature Scaling — StandardScaler |
| **Step 7** | Find optimal k — Elbow Method + Silhouette Score |
| **Step 8** | Apply K-Means Clustering (k=3) |
| **Step 9** | Evaluate — Silhouette Score, Davies-Bouldin, Inertia |
| **Step 10** | PCA — reduce to 2D for visualization |
| **Step 11** | Cluster Analysis — profile each cluster |
| **Step 12** | Sample songs from each cluster |
| **Step 13** | Export final CSV with cluster labels |
| **Step 14** | Print final summary |

---

## 📈 Results

### Optimal Number of Clusters: **k = 3**

Determined using the **Elbow Method** and **Silhouette Score** across k = 2 to 10.

### Cluster Breakdown

| Cluster | Name | Songs | % of Total |
|---|---|---|---|
| **Cluster 0** | 🎤 Vocal / Spoken Word Tracks | 12,513 | 13.1% |
| **Cluster 1** | 🎸 Calm / Acoustic Tracks | 30,807 | 32.1% |
| **Cluster 2** | 🎧 Energetic / Dance Tracks | 52,517 | 54.8% |

### Cluster Profiles

**Cluster 0 — Vocal / Spoken Word**
- Speechiness = **0.830** (very high — dominant feature)
- Shorter tracks (~1.6 min average)
- Typical genres: rap, hip-hop, comedy, spoken word

**Cluster 1 — Calm / Acoustic**
- Acousticness = **0.749** (very high)
- Energy = **0.311** (low), Loudness = −13.2 dB (quiet)
- Typical genres: folk, classical, ambient, vintage

**Cluster 2 — Energetic / Dance**
- Energy = **0.694** (high), Tempo = **124.9 BPM** (fast)
- Loudness = −7.6 dB (loud), Low acousticness
- Typical genres: pop, rock, dance, electronic

### Evaluation Metrics

| Metric | Value |
|---|---|
| Silhouette Score | **0.2378** |
| Davies-Bouldin Index | **1.5702** |
| Inertia | **658,335** |

---

## 📉 Visualizations

### Elbow Curve
Inertia drops sharply at k=3, confirming it as the elbow point.

### Silhouette Scores
k=3 achieves the highest Silhouette Score (0.2413) across all tested values.

### PCA Cluster Plot
2D visualization of all 95,837 songs colored by cluster, using the first two Principal Components (PC1=27.1%, PC2=18.8% variance explained).

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| scikit-learn | KMeans, StandardScaler, PCA, evaluation metrics |
| Matplotlib | Plotting |
| Seaborn | Heatmaps and styled charts |
| Jupyter Notebook | Development environment |

---

## 💼 Business Applications

- **Playlist Generation** — Automatically curate playlists using cluster-matched songs
- **Music Recommendation** — Suggest songs from the same cluster as a user's favorites
- **Artist Insights** — Identify which cluster an artist's music falls into
- **Streaming Analytics** — Segment users by their preferred cluster type

---

## 📁 Output File

The final output `amazon_music_clustered_output.csv` contains the original dataset with two new columns:

| Column | Values |
|---|---|
| `Cluster` | 0, 1, or 2 |
| `Cluster_Name` | Vocal / Spoken Word Tracks, Calm / Acoustic Tracks, Energetic / Dance Tracks |

---


---

## 📄 License

This project is for educational purposes.  
Dataset credit: Amazon Music / Spotify audio features dataset.
