from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "single_genre_artists.csv"
ELBOW_FIG = BASE_DIR / "Figure_3.png"

FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
ZERO_TO_ONE = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
]
K = 3
SEED = 42

COLORS = {
    "Speechy / Short": "#c0392f",
    "Calm / Acoustic": "#2c66ad",
    "Energetic / Loud": "#237445",
}
DESCRIPTIONS = {
    "Speechy / Short": "Spoken-word tracks: very high speechiness, short and live-sounding. "
                       "Mostly German audio dramas (hoerspiel), poetry and children's stories.",
    "Calm / Acoustic": "Soft, quiet, acoustic songs with the lowest energy. "
                       "Mostly older pop, chanson and soundtracks.",
    "Energetic / Loud": "Loud, fast, upbeat mainstream music. "
                        "Mostly modern pop and rock.",
}


# ---------- Helpers ----------
def show_chart(fig):
    """Draw a Plotly chart at full width on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, **kwargs):
    """Draw a dataframe at full width on old and new Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def first_genre(text):
    """Turn "['vintage chanson', 'chanson']" into "vintage chanson"."""
    text = str(text).strip("[]").replace("'", "").replace('"', "")
    return text.split(",")[0].strip() or "unknown"


def name_clusters(profile):
    """Name clusters from their average features, so names never depend on label order.

    Highest speechiness -> Speechy / Short; of the rest, highest energy ->
    Energetic / Loud; the remaining one -> Calm / Acoustic.
    """
    names = {}
    speechy = profile["speechiness"].idxmax()
    names[speechy] = "Speechy / Short"
    rest = profile.drop(index=speechy)
    energetic = rest["energy"].idxmax()
    names[energetic] = "Energetic / Loud"
    for c in rest.index:
        if c != energetic:
            names[c] = "Calm / Acoustic"
    return names


# ---------- Data and model (cached: runs once per session) ----------
@st.cache_data(show_spinner="Loading songs…")
def load_data(path):
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
    df["main_genre"] = df["genres"].apply(first_genre)
    return df


@st.cache_resource(show_spinner="Training K-Means on 95,837 songs (about 30 seconds, first run only)…")
def train_model(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    kmeans = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels, sample_size=5000, random_state=SEED)
    dbi = davies_bouldin_score(X_scaled, labels)

    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "X_scaled": X_scaled,
        "labels": labels,
        "silhouette": float(sil),
        "dbi": float(dbi),
        "inertia": float(kmeans.inertia_),
        "pca": X_pca,
        "pca_var": pca.explained_variance_ratio_,
    }


def nearest_songs(df, X_scaled, vector, cluster, n=10, exclude=None):
    """Return the n songs in the same cluster closest to a scaled feature vector."""
    idx = np.where((df["Cluster"] == cluster).to_numpy())[0]
    if exclude is not None:
        idx = idx[idx != exclude]
    dist = np.linalg.norm(X_scaled[idx] - vector, axis=1)
    best = np.argsort(dist)[:n]
    out = df.iloc[idx[best]][["name_song", "name_artists", "main_genre", "year"] + FEATURES[:3]].copy()
    out.insert(0, "distance", dist[best])
    return out.round(3)


# ---------- Page sections ----------
def section_overview(df, m):
    st.subheader("Overview")
    c = st.columns(5)
    c[0].metric("Songs", f"{len(df):,}")
    c[1].metric("Audio features", len(FEATURES))
    c[2].metric("Clusters (k)", K)
    c[3].metric("Silhouette", f"{m['silhouette']:.3f}", help="−1 to 1, higher is better. Computed on a 5,000-song sample.")
    c[4].metric("Davies-Bouldin", f"{m['dbi']:.3f}", help="Lower is better; 0 is perfect.")

    sizes = df["Cluster_Name"].value_counts().reindex(COLORS.keys())
    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(
            x=sizes.values, y=sizes.index, orientation="h", color=sizes.index,
            color_discrete_map=COLORS, text=[f"{v:,} ({v / len(df):.0%})" for v in sizes.values],
            labels={"x": "Songs", "y": ""}, title="Cluster size balance",
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=50, b=10))
        show_chart(fig)
    with right:
        st.markdown("**How to read this**")
        st.markdown(
            "Clusters are uneven because mainstream energetic music dominates the catalogue, "
            "and K-Means does not force equal sizes. A silhouette around 0.24 is modest but "
            "typical for music, where songs blend between styles."
        )

    st.markdown("#### The three clusters")
    cols = st.columns(3)
    for col, name in zip(cols, COLORS):
        part = df[df["Cluster_Name"] == name]
        top = part["main_genre"].value_counts().head(4).index.tolist()
        with col:
            st.markdown(f"<div style='border-top:5px solid {COLORS[name]};padding-top:8px'>"
                        f"<b style='font-size:1.15rem'>{name}</b></div>", unsafe_allow_html=True)
            st.caption(f"{len(part):,} songs · median year {int(part['year'].median())}")
            st.write(DESCRIPTIONS[name])
            st.markdown(
                f"- Speechiness **{part['speechiness'].mean():.2f}**\n"
                f"- Acousticness **{part['acousticness'].mean():.2f}**\n"
                f"- Energy **{part['energy'].mean():.2f}**\n"
                f"- Loudness **{part['loudness'].mean():.1f} dB**\n"
                f"- Tempo **{part['tempo'].mean():.0f} BPM**\n"
                f"- Duration **{part['duration_ms'].mean() / 1000:.0f} s**"
            )
            st.caption("Top genres: " + ", ".join(top))


def section_profiles(df):
    st.subheader("Cluster profiles")
    st.write("Average feature values per cluster, in real units. "
             "This is the table the cluster names come from.")
    profile = df.groupby("Cluster_Name")[FEATURES].mean().reindex(COLORS.keys())
    show_table(profile.style.format("{:.3f}").background_gradient(axis=0, cmap="YlGnBu"))

    long = profile[ZERO_TO_ONE].reset_index().melt(id_vars="Cluster_Name", var_name="feature", value_name="mean")
    fig = px.bar(long, x="feature", y="mean", color="Cluster_Name", barmode="group",
                 color_discrete_map=COLORS, title="Features on a 0–1 scale, real averages",
                 labels={"mean": "Average value", "feature": "", "Cluster_Name": "Cluster"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_range=[0, 1])
    show_chart(fig)

    st.markdown("#### Distribution within each cluster")
    feat = st.selectbox("Feature", FEATURES, index=FEATURES.index("energy"), key="dist_feature")
    sample = df.sample(min(15000, len(df)), random_state=SEED)
    fig = px.box(sample, x="Cluster_Name", y=feat, color="Cluster_Name", color_discrete_map=COLORS,
                 category_orders={"Cluster_Name": list(COLORS)}, points=False,
                 labels={"Cluster_Name": ""}, title=f"{feat} by cluster (15,000-song sample)")
    fig.update_layout(showlegend=False, height=400, margin=dict(l=10, r=10, t=50, b=10))
    show_chart(fig)


def section_pca(df, m):
    st.subheader("Clusters in 2D (PCA)")
    var = m["pca_var"]
    st.write(f"PCA squeezes the 10 scaled features into 2 for plotting only. "
             f"PC1 keeps {var[0]:.1%} and PC2 {var[1]:.1%} of the variance "
             f"({var.sum():.1%} in total), so some overlap at the edges is expected.")
    n = st.slider("Songs to plot", 2000, 20000, 8000, step=1000, key="pca_n")
    plot_df = df[["name_song", "name_artists", "main_genre", "Cluster_Name"]].copy()
    plot_df["PC1"], plot_df["PC2"] = m["pca"][:, 0], m["pca"][:, 1]
    plot_df = plot_df.sample(n, random_state=SEED)
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster_Name", color_discrete_map=COLORS,
                     hover_data={"name_song": True, "name_artists": True, "main_genre": True,
                                 "PC1": False, "PC2": False},
                     category_orders={"Cluster_Name": list(COLORS)},
                     opacity=0.5, labels={"Cluster_Name": "Cluster"})
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10))
    show_chart(fig)
    if ELBOW_FIG.exists():
        with st.expander("Why k = 3? Elbow and silhouette for k = 2 to 10"):
            st.image(str(ELBOW_FIG))
            st.caption("Silhouette peaks at k = 3 (0.241); k = 4 is close (0.231); k = 6 is lowest (0.160).")


def section_similar(df, m):
    st.subheader("Find similar songs")
    st.write("Search for a song. The app shows its cluster and the 10 closest songs in the same cluster.")
    query = st.text_input("Song or artist name", value="La Java", key="song_query")
    if not query.strip():
        st.info("Type a song or artist name to start.")
        return
    q = query.strip().lower()
    hits = df[df["name_song"].str.lower().str.contains(q, na=False, regex=False)
              | df["name_artists"].str.lower().str.contains(q, na=False, regex=False)].head(50)
    if hits.empty:
        st.warning(f"No songs or artists match “{query}”. Try a shorter word.")
        return
    options = hits.index.tolist()
    pick = st.selectbox("Pick a match", options, key="song_pick",
                        format_func=lambda i: f"{df.at[i, 'name_song']} — {df.at[i, 'name_artists']}")
    row = df.loc[pick]
    pos = df.index.get_loc(pick)
    name = row["Cluster_Name"]
    st.markdown(f"**{row['name_song']}** by {row['name_artists']} is in "
                f"<span style='color:{COLORS[name]};font-weight:700'>{name}</span>.",
                unsafe_allow_html=True)

    avg = df[df["Cluster_Name"] == name][ZERO_TO_ONE].mean()
    comp = pd.DataFrame({"This song": row[ZERO_TO_ONE].astype(float), "Cluster average": avg}).reset_index()
    comp = comp.melt(id_vars="index", var_name="Series", value_name="value")
    fig = px.bar(comp, x="index", y="value", color="Series", barmode="group",
                 color_discrete_map={"This song": COLORS[name], "Cluster average": "#9aa5b2"},
                 labels={"index": "", "value": "Value (0–1)", "Series": ""})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_range=[0, 1])
    show_chart(fig)

    st.markdown("**10 most similar songs** (smallest distance in scaled feature space)")
    show_table(nearest_songs(df, m["X_scaled"], m["X_scaled"][pos], row["Cluster"], exclude=pos),
               hide_index=True)


def section_predict(df, m):
    st.subheader("Predict the cluster of a new song")
    st.write("Set the audio features of a new song. The app scales them with the same fitted "
             "scaler (transform, not fit_transform) and calls kmeans.predict(). No retraining.")
    med = df[FEATURES].median()
    cols = st.columns(2)
    values = {}
    for i, f in enumerate(FEATURES):
        with cols[i % 2]:
            if f in ZERO_TO_ONE:
                values[f] = st.slider(f, 0.0, 1.0, float(round(med[f], 2)), 0.01, key=f"in_{f}")
            elif f == "loudness":
                values[f] = st.slider("loudness (dB)", -40.0, 0.0, float(round(med[f], 1)), 0.5, key="in_loud")
            elif f == "tempo":
                values[f] = st.slider("tempo (BPM)", 50.0, 220.0, float(round(med[f])), 1.0, key="in_tempo")
            else:
                secs = st.slider("duration (seconds)", 30, 600, int(med[f] / 1000), 5, key="in_dur")
                values[f] = secs * 1000.0

    new = pd.DataFrame([values])[FEATURES]
    vec = m["scaler"].transform(new)
    cluster = int(m["kmeans"].predict(vec)[0])
    name = df.loc[df["Cluster"] == cluster, "Cluster_Name"].iloc[0]
    dists = np.linalg.norm(m["kmeans"].cluster_centers_ - vec, axis=1)

    st.markdown(f"### Predicted: <span style='color:{COLORS[name]}'>{name}</span>", unsafe_allow_html=True)
    st.caption(DESCRIPTIONS[name])
    names_by_id = df.drop_duplicates("Cluster").set_index("Cluster")["Cluster_Name"]
    dist_df = pd.DataFrame({"Cluster": [names_by_id[c] for c in range(K)],
                            "Distance to centre": dists.round(3)}).sort_values("Distance to centre")
    show_table(dist_df, hide_index=True)
    st.markdown("**Closest existing songs**")
    show_table(nearest_songs(df, m["X_scaled"], vec[0], cluster), hide_index=True)


def section_data(df):
    st.subheader("Clustered data")
    choice = st.multiselect("Clusters", list(COLORS), default=list(COLORS), key="data_clusters")
    view = df[df["Cluster_Name"].isin(choice)]
    cols = ["name_song", "name_artists", "main_genre", "year", "Cluster", "Cluster_Name"] + FEATURES
    st.caption(f"{len(view):,} songs")
    show_table(view[cols].head(1000), hide_index=True)
    st.download_button("Download clustered CSV", view.drop(columns=["main_genre", "year"]).to_csv(index=False),
                       file_name="amazon_music_clustered_output.csv", mime="text/csv")


# ---------- Main ----------
def main():
    st.set_page_config(page_title="Amazon Music Clustering", layout="wide")
    st.title("Amazon Music Clustering")
    st.caption("K-Means on 10 audio features · 95,837 songs · no genre labels used for training")

    if not DATA_PATH.exists():
        st.error(f"Can't find {DATA_PATH.name}. Put it in the same folder as app.py and reload.")
        st.stop()

    df = load_data(DATA_PATH)
    m = train_model(df)
    df = df.copy()
    df["Cluster"] = m["labels"]
    names = name_clusters(df.groupby("Cluster")[FEATURES].mean())
    df["Cluster_Name"] = df["Cluster"].map(names)

    with st.sidebar:
        st.header("About")
        st.write("Groups songs by how they sound, using unsupervised K-Means clustering.")
        st.markdown(
            "**Pipeline**\n"
            "1. 10 audio features\n"
            "2. StandardScaler\n"
            f"3. KMeans(k={K}, n_init=10, random_state={SEED})\n"
            "4. Silhouette + Davies-Bouldin\n"
            "5. PCA for the 2D view"
        )
        st.metric("Inertia (SSE)", f"{m['inertia']:,.0f}")

    tabs = st.tabs(["Overview", "Cluster profiles", "2D view", "Similar songs", "Predict", "Data"])
    with tabs[0]:
        section_overview(df, m)
    with tabs[1]:
        section_profiles(df)
    with tabs[2]:
        section_pca(df, m)
    with tabs[3]:
        section_similar(df, m)
    with tabs[4]:
        section_predict(df, m)
    with tabs[5]:
        section_data(df)


if __name__ == "__main__":
    main()
