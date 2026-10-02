import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Music Analysis Platform",
    page_icon="🎵",
    layout="wide"
)

# -------------------- CARD STYLE (safe, no effect on charts) --------------------
def card(content_func):
    with st.container(border=True):
        content_func()

# -------------------- LOAD MODELS --------------------
model = joblib.load("random_forest_model.pkl")
scaler_loaded = joblib.load('scaler.joblib')
kmeans_loaded = joblib.load('kmeans_model.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('clustered_sample_data.csv')

data = load_data()

if 'kmeans_cluster' not in data.columns:
    st.error("Column 'kmeans_cluster' not found in data.")
    st.stop()

# -------------------- CONSTANTS --------------------
audio_features = [
    'danceability','energy','key','loudness','mode','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo','time_signature'
]

genre_list = [
    'acoustic','afrobeat','alt-rock','alternative','ambient','anime','black-metal','bluegrass',
    'blues','brazil','breakbeat','british','cantopop','chicago-house','children','chill',
    'classical','club','comedy','country','dance','dancehall','death-metal','deep-house',
    'detroit-techno','disco','disney','drum-and-bass','dub','dubstep','edm','electro',
    'electronic','emo','folk','forro','french','funk','garage','german','gospel','goth',
    'grindcore','groove','grunge','guitar','happy','hard-rock','hardcore','hardstyle',
    'heavy-metal','hip-hop','honky-tonk','house','idm','indian','indie-pop','indie',
    'industrial','iranian','j-dance','j-idol','j-pop','j-rock','jazz','k-pop','kids','latin',
    'latino','malay','mandopop','metal','metalcore','minimal-techno','mpb','new-age','opera',
    'pagode','party','piano','pop-film','pop','power-pop','progressive-house','psych-rock',
    'punk-rock','punk','r-n-b','reggae','reggaeton','rock-n-roll','rock','rockabilly','romance',
    'sad','salsa','samba','sertanejo','show-tunes','singer-songwriter','ska','sleep',
    'songwriter','soul','spanish','study','swedish','synth-pop','tango','techno','trance',
    'trip-hop','turkish','world-music'
]

cluster_descriptions = {
    0:"Balanced features - general pop/rock",
    1:"Balanced features - general pop/rock",
    2:"Balanced features - general pop/rock",
    3:"Balanced features - general pop/rock",
    4:"High energy & danceable - upbeat pop/EDM",
    5:"High energy & danceable - upbeat pop/EDM",
    6:"High energy & danceable - upbeat pop/EDM",
    7:"Balanced features - general pop/rock",
    8:"High speechiness - rap/spoken word",
    9:"High energy & danceable - upbeat pop/EDM",
    10:"High energy & danceable - upbeat pop/EDM",
    11:"Balanced features - general pop/rock"
}

# -------------------- LAYOUT --------------------
st.title("🎵 Music Analysis Platform")

features_list = [
    "Popularity Prediction",
    "Music Cluster Explorer",
    "Genre Evolution Visualization",
    "Artist Similarity & Trend Spotting",
    "Playlist Generator"
]

choice = st.selectbox("Choose a feature:", features_list)

# -------------------- FEATURE 1 --------------------
if choice == "Popularity Prediction":

    def popularity_content():
        st.header("⭐ Song Popularity Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            explicit = st.checkbox("Explicit")
            danceability = st.number_input("Danceability", 0.0, 1.0, 0.5)
            energy = st.number_input("Energy", 0.0, 1.0, 0.5)

        with col2:
            key = st.number_input("Key", 0, 11, 0)
            loudness = st.number_input("Loudness", -60.0, 0.0, -10.0)
            mode = st.number_input("Mode", 0, 1, 1)

        with col3:
            speechiness = st.number_input("Speechiness", 0.0, 1.0, 0.05)
            acousticness = st.number_input("Acousticness", 0.0, 1.0, 0.1)
            valence = st.number_input("Valence", 0.0, 1.0, 0.5)

        tempo = st.number_input("Tempo", 0.0, 250.0, 120.0)
        time_signature = st.number_input("Time Signature", 1, 7, 4)

        artists_label_encoded = st.number_input("Artist Label Encoded", 0, 30000, 0)
        selected_genre_name = st.selectbox("Track Genre:", genre_list)
        genre_mapping = {g: i for i, g in enumerate(genre_list)}
        track_genre_label_encoded = genre_mapping[selected_genre_name]

        if st.button("Predict Popularity"):
            input_data = np.array([[
                explicit, danceability, energy, key, loudness, mode,
                speechiness, acousticness, 0, 0, valence, tempo,
                time_signature, 1, 1, 1, 5, 5, 0,
                artists_label_encoded, track_genre_label_encoded
            ]])
            pred = model.predict(input_data)
            st.success(f"Predicted Popularity: {pred[0]}")

    card(popularity_content)

# -------------------- FEATURE 2 --------------------
elif choice == "Music Cluster Explorer":
    def cluster_content():
        st.header("🎧 Music Cluster Explorer")

        track_name = st.text_input("Track Name:")

        if track_name:
            row = data[data['track_name'].str.lower() == track_name.lower()]

            if row.empty:
                st.warning("Track not found.")
            else:
                cid = row['kmeans_cluster'].values[0]
                st.info(f"Cluster: {cluster_descriptions[cid]}")

                similar = data[
                    (data['kmeans_cluster'] == cid) &
                    (data['track_name'].str.lower() != track_name.lower())
                ]

                st.write(f"{len(similar)} similar tracks:")
                st.dataframe(similar[['track_name','artists','track_genre']].head(10))

    card(cluster_content)

# -------------------- FEATURE 3 --------------------
elif choice == "Genre Evolution Visualization":
    def evolution_content():
        st.header("📈 Genre Evolution Over Time")

        data['time_index'] = data.index
        selected = st.multiselect("Choose Genres:", genre_list, default=genre_list[:10])

        if selected:
            filtered = data[data['track_genre'].isin(selected)]
            filtered['time_bin'] = pd.cut(filtered['time_index'], bins=80, labels=False)

            grouped = (
                filtered.groupby(['time_bin','track_genre'])
                .size()
                .reset_index(name='Count')
            )

            chart = (
                alt.Chart(grouped)
                .mark_line(point=True)
                .encode(
                    x="time_bin:Q",
                    y="Count:Q",
                    color="track_genre:N",
                    tooltip=['track_genre','time_bin','Count']
                )
                .properties(width=900, height=450)
            )

            st.altair_chart(chart)

    card(evolution_content)

# -------------------- FEATURE 4 --------------------
elif choice == "Artist Similarity & Trend Spotting":
    def artist_content():
        st.header("🎤 Artist Similarity & Trends")

        artist_features = data.groupby('artists')[audio_features].mean()
        artists = artist_features.index.tolist()

        artist = st.selectbox("Choose Artist:", artists)

        from sklearn.metrics.pairwise import euclidean_distances

        vector = artist_features.loc[artist].values.reshape(1, -1)
        distances = euclidean_distances(artist_features, vector).flatten()

        sim_df = pd.DataFrame({'artist': artists, 'distance': distances})
        similar = sim_df[sim_df['artist'] != artist].nsmallest(10, 'distance')

        st.subheader("Top 10 similar artists:")
        st.dataframe(similar)

        data['time_index'] = data.index
        data['time_bin'] = pd.cut(data['time_index'], bins=50, labels=False)

        trend_data = data[data['artists'].isin([artist] + similar['artist'].tolist())]
        trend_grouped = trend_data.groupby(['time_bin','artists'])['popularity'].mean().reset_index()

        chart = (
            alt.Chart(trend_grouped)
            .mark_line(point=True)
            .encode(
                x="time_bin",
                y="popularity",
                color="artists"
            )
            .properties(width=900, height=450)
        )

        st.subheader("Popularity Trend Over Time")
        st.altair_chart(chart)

    card(artist_content)

# -------------------- FEATURE 5 --------------------
elif choice == "Playlist Generator":
    def playlist_content():
        st.header("🎶 Playlist Generator")

        d = st.slider("Danceability", 0.0, 1.0, (0.3, 0.7))
        e = st.slider("Energy", 0.0, 1.0, (0.3, 0.7))
        t = st.slider("Tempo", float(data['tempo'].min()), float(data['tempo'].max()), (80.0, 140.0))
        v = st.slider("Valence", 0.0, 1.0, (0.3, 0.7))
        a = st.slider("Acousticness", 0.0, 1.0, (0.0, 0.5))

        filtered = data[
            (data['danceability'].between(*d)) &
            (data['energy'].between(*e)) &
            (data['tempo'].between(*t)) &
            (data['valence'].between(*v)) &
            (data['acousticness'].between(*a))
        ]

        st.write(f"Found {len(filtered)} tracks:")
        st.dataframe(filtered[['track_name','artists','track_genre','danceability','energy','tempo']].head(20))

    card(playlist_content)
