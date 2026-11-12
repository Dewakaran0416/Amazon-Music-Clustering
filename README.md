# 🎧 Amazon Music Clustering

**Grouping 95,837 songs by how they sound, using unsupervised machine learning. No genre labels needed.**

K-Means clustering on 10 audio features, PCA for visualisation, and an interactive Streamlit app to explore the clusters, find similar songs and predict the cluster of a new track.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-K--Means%20%7C%20PCA-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

<!-- Demo: add a GIF or screenshot here, e.g.  ![App demo](demo.gif) -->

---

## Problem

Music platforms hold millions of tracks, and tagging every song by genre by hand is slow and inconsistent. This project groups songs automatically by their **audio characteristics**, so similar-sounding tracks land together. That supports:

- **Playlist curation:** ready-made groups of similar-sounding songs
- **Song discovery:** "more like this" recommendations
- **Artist analysis:** which songs an artist competes with
- **Listener segmentation:** understanding listening patterns

## Quick start

```bash
git clone https://github.com/Dewakaran0416/Amazon-Music-Clustering.git
cd Amazon-Music-Clustering
pip install -r requirements.txt
streamlit run app.py
```

The first load trains K-Means on all 95,837 songs, which takes about 30 seconds. After that it's cached.

To re-run the full analysis and regenerate the plots and clustered CSV:

```bash
python Amazon.py
```

## Streamlit app

| Tab | What it does |
|---|---|
| **Overview** | Key metrics, cluster sizes, and a card per cluster with its top genres |
| **Cluster profiles** | Average features per cluster, plus box plots of any feature by cluster |
| **2D view** | Interactive PCA scatter, with song and artist names on hover |
| **Similar songs** | Search any song or artist and get the 10 closest songs in its cluster |
| **Predict** | Set 10 sliders for a new song and predict its cluster instantly, with no retraining |
| **Data** | Filter by cluster and download the clustered CSV |

## Dataset

| | |
|---|---|
| File | `single_genre_artists.csv` |
| Songs | 95,837 |
| Columns | 23 |
| Missing values | None |
| Duplicates | None |

**Features used for clustering (10)**

| Feature | Meaning |
|---|---|
| `danceability` | How suitable the track is for dancing (0–1) |
| `energy` | Intensity and activity (0–1) |
| `loudness` | Overall loudness in dB |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Likelihood of no vocals (0–1) |
| `liveness` | Presence of a live audience (0–1) |
| `valence` | Musical positiveness (0–1) |
| `tempo` | Speed in BPM |
| `duration_ms` | Length in milliseconds |

**Dropped (13 columns):** IDs and names (no sound information), popularity and followers (success, not sound), genres (the thing being discovered; used only to validate afterwards), and key, mode, time_signature, explicit (categorical, so not suited to Euclidean distance).

## Approach

1. **EDA:** histograms and a correlation heatmap. Energy–loudness r = +0.73; energy–acousticness r = −0.66.
2. **Scaling:** `StandardScaler` (mean 0, std 1), because K-Means is distance-based.
3. **Choosing k:** elbow method and silhouette score for k = 2 to 10.
4. **Clustering:** `KMeans(n_clusters=3, n_init=10, random_state=42)`.
5. **Evaluation:** silhouette score and Davies-Bouldin index.
6. **Visualisation:** PCA to 2 components. This is used only for plotting; K-Means runs on all 10 features.
7. **Interpretation:** average features per cluster, then a check against genre tags the model never saw.

### Choosing k

| k | Inertia | Silhouette |
|---|---|---|
| 2 | 778,814 | 0.205 |
| **3** | **658,335** | **0.241** |
| 4 | 593,031 | 0.231 |
| 5 | 548,595 | 0.190 |
| 6 | 520,711 | 0.160 |

Silhouette peaks at **k = 3**. The elbow curve bends gradually, so silhouette was the deciding metric.

## Results

| Metric | Value |
|---|---|
| Silhouette score | 0.241 |
| Davies-Bouldin index | 1.570 |
| Inertia (SSE) | 658,335 |
| PCA variance in 2D | 45.9% (PC1 27.1% + PC2 18.8%) |

| Cluster | Songs | Defining traits | What's in it |
|---|---|---|---|
| 🔴 **0 · Speechy / Short** | 12,513 (13.1%) | Speechiness 0.83, ~97 s long | ~76% German audio dramas (hoerspiel), plus poetry and children's stories |
| 🔵 **1 · Calm / Acoustic** | 30,807 (32.1%) | Acousticness 0.75, energy 0.31, −13.2 dB | Older soft pop, chanson, soundtracks |
| 🟢 **2 · Energetic / Loud** | 52,517 (54.8%) | Energy 0.69, −7.6 dB, 125 BPM | Modern pop and rock (median release year 2000) |

**Key insight:** the genre column was never used for training, yet the clusters line up with it. The speech-heavy cluster turned out to be spoken-word content, not rap, which unsupervised learning found on its own.

A silhouette of 0.24 is modest. That's expected for music, where styles blend into each other.

## Repository structure

```
Amazon-Music-Clustering/
├── app.py                               # Streamlit app
├── Amazon.py                            # Full analysis: EDA, k selection, K-Means, PCA, profiling
├── requirements.txt                     # Python dependencies
├── single_genre_artists.csv             # Input data (95,837 songs)
├── amazon_music_clustered_output.csv    # Songs with Cluster and Cluster_Name columns
├── Figure_1.png                         # Feature distributions
├── Figure_2.png                         # Correlation heatmap
├── Figure_3.png                         # Elbow and silhouette for k = 2–10
├── Figure_4.png                         # PCA cluster plot
├── Figure_5.png                         # Feature averages per cluster (scaled 0–1 across clusters)
├── Figure_6.png                         # Cluster feature heatmap (scaled 0–1 across clusters)
├── Amazon_Music_Clustering_Report.docx  # Project report
└── Amazon Music Clustering.pptx         # Presentation
```

Figures 5 and 6 scale each feature 0–1 across the three clusters, so they show which cluster ranks highest, not absolute values. The **Cluster profiles** tab in the app shows the real averages.

## Tech stack

Python · pandas · NumPy · scikit-learn · Plotly · Matplotlib · Seaborn · Streamlit

## Limitations and next steps

- Treat outliers (durations up to ~80 minutes) and skewed features with a log transform or capping
- Try DBSCAN and Gaussian Mixture Models
- Test a higher k for finer, playlist-level groups
- Check cluster stability across random seeds

## Author

**Dewakaran:** data science and analytics, with 10+ years in US healthcare revenue cycle management.

This project is for educational purposes.
