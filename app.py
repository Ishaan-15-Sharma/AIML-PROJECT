import json
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
MOVIES_PATH = DATA_DIR / "ml-latest-small" / "movies.csv"
RATINGS_PATH = DATA_DIR / "ml-latest-small" / "ratings.csv"

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.stApp {background:linear-gradient(135deg,#0b1020 0%,#111827 55%,#1b1230 100%);color:#f8fafc}
.block-container {padding-top:2rem;max-width:1280px}
.hero {padding:1.6rem 1.8rem;border:1px solid #343b52;border-radius:22px;
background:linear-gradient(120deg,rgba(124,58,237,.24),rgba(14,165,233,.11));margin-bottom:1.3rem}
.hero h1 {font-size:2.7rem;margin:0;color:#fff;letter-spacing:-1px}
.hero p {color:#cbd5e1;font-size:1.05rem;margin-bottom:.4rem}
.pill {display:inline-block;background:#302653;color:#ddd6fe;padding:.2rem .55rem;border-radius:999px;font-size:.76rem;margin:.12rem}
.movie-card {background:rgba(22,30,50,.92);border:1px solid #343b52;border-radius:16px;padding:1rem;min-height:180px;margin-bottom:.7rem}
.movie-title {font-size:1.02rem;font-weight:700;color:#f8fafc;margin-bottom:.45rem}
.muted {color:#a8b3c7;font-size:.86rem}
div[data-testid="stMetric"] {background:rgba(22,30,50,.75);border:1px solid #343b52;padding:1rem;border-radius:14px}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    DATA_DIR.mkdir(exist_ok=True)
    movies_path = MOVIES_PATH
    ratings_path = RATINGS_PATH
    if not movies_path.exists() or not ratings_path.exists():
        local_movies_path = APP_DIR / "movies.csv"
        local_ratings_path = APP_DIR / "ratings.csv"
        if local_movies_path.exists() and local_ratings_path.exists():
            movies_path = local_movies_path
            ratings_path = local_ratings_path
        else:
            if not ZIP_PATH.exists():
                with st.spinner("Downloading MovieLens dataset…"):
                    urlretrieve(DATASET_URL, ZIP_PATH)
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                zf.extractall(DATA_DIR)
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    links_path = movies_path.parent / "links.csv"
    if links_path.exists():
        links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"])
        movies = movies.merge(links, on="movieId", how="left")
    else:
        movies["tmdbId"] = np.nan
    movies["genres_list"] = movies["genres"].fillna("").apply(
        lambda x: [g for g in x.split("|") if g and g != "(no genres listed)"]
    )
    stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"), rating_count=("rating", "count")
    ).reset_index()
    movies = movies.merge(stats, on="movieId", how="left")
    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")
    movies["display_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["content"] = (
        movies["display_title"].fillna("").str.lower() + " " +
        movies["genres"].fillna("").str.replace("|", " ", regex=False).str.lower()
    )
    return movies, ratings


@st.cache_resource(show_spinner="Training content and collaborative recommendation models…")
def train_models(content_tuple, rating_rows):
    content = pd.Series(content_tuple)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

    tfidf = vectorizer.fit_transform(content)
    content_sim = cosine_similarity(tfidf, dense_output=False)

    # Build sparse movie-user rating matrix, then compare movies by rating patterns.
    rating_df = pd.DataFrame(rating_rows, columns=["userId", "movieId", "rating"])
    movie_ids = sorted(rating_df["movieId"].unique())
    user_ids = sorted(rating_df["userId"].unique())
    
    movie_index = {mid: i for i, mid in enumerate(movie_ids)}
    user_index = {uid: i for i, uid in enumerate(user_ids)}
    row = rating_df["movieId"].map(movie_index).to_numpy()
    col = rating_df["userId"].map(user_index).to_numpy()
    vals = rating_df["rating"].to_numpy(dtype=float)
    rating_matrix = csr_matrix((vals, (row, col)), shape=(len(movie_ids), len(user_ids)))
    collab_sim = cosine_similarity(rating_matrix, dense_output=False)
    return content_sim, collab_sim, movie_index


def get_recommendations(selected_title, movies, content_sim, collab_sim, movie_index,
                        n_recommendations=8, genre_filter="Any genre", content_weight=0.55):
    match = movies.index[movies["title"] == selected_title]
    if len(match) == 0:
        return pd.DataFrame()
    source_idx = int(match[0])
    selected_id = int(movies.iloc[source_idx]["movieId"])
    content_scores = content_sim[source_idx].toarray().ravel()

    collab_scores = np.zeros(len(movies), dtype=float)
    if selected_id in movie_index:
        collab_row = collab_sim[movie_index[selected_id]]
        collab_scores_map = dict(zip(movie_index.keys(), collab_row.toarray().ravel()))
        collab_scores = movies["movieId"].map(collab_scores_map).fillna(0).to_numpy()

    # Normalize each score vector before blending so the scales are more comparable.
    def normalize(x):
        maximum = float(np.max(x))
        return x / maximum if maximum > 0 else x

    content_scores = normalize(content_scores)
    collab_scores = normalize(collab_scores)
    hybrid = content_weight * content_scores + (1 - content_weight) * collab_scores

    results = movies.copy()
    results["content_score"] = content_scores
    results["collab_score"] = collab_scores
    results["hybrid_score"] = hybrid
    results = results[results["title"] != selected_title]
    if genre_filter != "Any genre":
        results = results[results["genres_list"].apply(lambda gs: genre_filter in gs)]
    results = results.sort_values(
        ["hybrid_score", "avg_rating", "rating_count"], ascending=False
    ).head(n_recommendations)
    return results


class PosterLookupError(Exception):
    pass


@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_poster(tmdb_id: int, credential: str) -> bytes:
    headers = {"Accept": "application/json"}
    if credential.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {credential}"
        query = ""
    else:
        query = "?" + urlencode({"api_key": credential})
    request = Request(
        f"https://api.themoviedb.org/3/movie/{tmdb_id}{query}",
        headers=headers,
    )
    try:
        with urlopen(request, timeout=5) as response:
            movie_data = json.load(response)
    except HTTPError as exc:
        if exc.code == 401:
            raise PosterLookupError(
                "TMDb rejected the credential. Check your API key or read access token."
            ) from None
        raise PosterLookupError(
            f"TMDb returned HTTP {exc.code} while looking up this movie."
        ) from None
    except (URLError, TimeoutError, json.JSONDecodeError):
        raise PosterLookupError("Could not retrieve this poster from TMDb.") from None

    poster_path = movie_data.get("poster_path") if isinstance(movie_data, dict) else None
    if not isinstance(poster_path, str) or not poster_path.startswith("/"):
        raise PosterLookupError("TMDb has no poster for this movie.")

    try:
        with urlopen(f"https://image.tmdb.org/t/p/w342{poster_path}", timeout=10) as response:
            image_data = response.read()
    except HTTPError as exc:
        raise PosterLookupError(
            f"TMDb's image server returned HTTP {exc.code}."
        ) from None
    except (URLError, TimeoutError):
        raise PosterLookupError("Could not load this image from TMDb.") from None
    return image_data


try:
    movies, ratings = load_data()
    rating_rows = tuple(map(tuple, ratings[["userId", "movieId", "rating"]].itertuples(index=False, name=None)))
    content_sim, collab_sim, movie_index = train_models(tuple(movies["content"].tolist()), rating_rows)
except Exception as exc:
    st.error("The dataset or model could not be loaded. Check that data/ml-latest-small contains movies.csv and ratings.csv.")
    st.exception(exc)
    st.stop()

st.markdown("""
<div class="hero">
<h1>🎬 CineMatch AI</h1>
<p>Discover your next favourite movie with a hybrid machine-learning recommender.</p>
<span class="pill">TF-IDF</span><span class="pill">Cosine Similarity</span>
<span class="pill">Collaborative Filtering</span><span class="pill">Hybrid Ranking</span>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Movies", f"{len(movies):,}")
m2.metric("Ratings", f"{len(ratings):,}")
m3.metric("Users", f"{ratings['userId'].nunique():,}")


tab_recommend, tab_explore, tab_about = st.tabs(
    ["✨ Discover movies", "🔎 Explore catalogue", "🧠 How the AI works"]
)

with tab_recommend:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Choose a movie you enjoy")
        movie_titles = sorted(movies["title"].dropna().unique().tolist())
        default_title = "Toy Story (1995)" if "Toy Story (1995)" in movie_titles else movie_titles[0]
        selected_title = st.selectbox("Movie", movie_titles, index=movie_titles.index(default_title))
        chosen = movies[movies["title"] == selected_title].iloc[0]
        st.markdown(f"### {chosen['display_title']}")
        year_text = str(int(chosen["year"])) if pd.notna(chosen["year"]) else "Unknown"
        st.markdown(f"<p class='muted'>Release year: {year_text}</p>", unsafe_allow_html=True)
        st.markdown("".join(f"<span class='pill'>{g}</span>" for g in chosen["genres_list"]), unsafe_allow_html=True)
        st.write(f"**Community rating:** {chosen['avg_rating']:.1f}/5 · **Ratings:** {int(chosen['rating_count']):,}")
    with right:
        st.subheader("Tune recommendations")
        count = st.slider("Number of recommendations", 4, 12, 8)
        genres = sorted({g for gs in movies["genres_list"] for g in gs})
        genre_filter = st.selectbox("Optional genre filter", ["Any genre"] + genres)
        content_weight = st.slider(
            "Content-based influence", min_value=0, max_value=100, value=55, step=5,
            help="Higher values prioritise shared titles/genres. Lower values prioritise similar user-rating patterns."
        ) / 100
        st.caption(f"Current blend: {int(content_weight*100)}% content + {int((1-content_weight)*100)}% collaborative.")
    st.divider()
    st.subheader("Recommended for you")
    results = get_recommendations(selected_title, movies, content_sim, collab_sim, movie_index,
                                  count, genre_filter, content_weight)
    try:
        tmdb_api_key = st.secrets.get("TMDB_API_KEY", "").strip()
    except st.errors.StreamlitSecretNotFoundError:
        tmdb_api_key = ""
    if results.empty:
        st.info("No movies match that genre filter. Try another genre.")
    else:
        if not tmdb_api_key:
            st.info("Add your TMDb API key or read access token as TMDB_API_KEY in .streamlit/secrets.toml.")
        cols = st.columns(4)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 4]:
                year = str(int(row["year"])) if pd.notna(row["year"]) else "Year unknown"
                genre_html = "".join(f"<span class='pill'>{g}</span>" for g in row["genres_list"][:3])
                if tmdb_api_key and pd.notna(row["tmdbId"]):
                    try:
                        poster = get_tmdb_poster(int(row["tmdbId"]), tmdb_api_key)
                    except PosterLookupError as exc:
                        st.caption(str(exc))
                    else:
                        st.image(poster, use_container_width=True)
                else:
                    if tmdb_api_key:
                        st.caption("No TMDb ID is available for this movie.")
                st.markdown(f"""
                <div class="movie-card">
                  <div class="movie-title">{row['display_title']}</div>
                  <div class="muted">{year}</div>
                  <div style="margin:.45rem 0">{genre_html}</div>
                  <div class="muted">⭐ {row['avg_rating']:.1f}/5 · {int(row['rating_count']):,} ratings</div>
                  <div class="muted">Hybrid score: {row['hybrid_score']:.3f}</div>
                </div>""", unsafe_allow_html=True)
                with st.expander("Why this recommendation?"):
                    st.write(f"Content similarity: {row['content_score']:.3f}")
                    st.write(f"Rating-pattern similarity: {row['collab_score']:.3f}")

with tab_explore:
    st.subheader("Explore the movie catalogue")
    genres = sorted({g for gs in movies["genres_list"] for g in gs})
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        search_text = st.text_input("Search movie title", placeholder="e.g. Batman, Toy Story…")
    with c2:
        genre = st.selectbox("Genre", ["All genres"] + genres, key="explore_genre")
    with c3:
        min_rating = st.slider("Minimum average rating", 0.0, 5.0, 0.0, 0.5)
    filtered = movies.copy()
    if search_text.strip():
        filtered = filtered[filtered["display_title"].str.contains(search_text, case=False, na=False)]
    if genre != "All genres":
        filtered = filtered[filtered["genres_list"].apply(lambda gs: genre in gs)]
    filtered = filtered[filtered["avg_rating"] >= min_rating]
    filtered = filtered.sort_values(["avg_rating", "rating_count"], ascending=False).head(100)
    st.caption(f"Showing {len(filtered)} matches (maximum 100 displayed).")
    st.dataframe(filtered[["display_title", "year", "genres", "avg_rating", "rating_count"]].rename(
        columns={"display_title": "Movie", "year": "Year", "genres": "Genres",
                 "avg_rating": "Average rating", "rating_count": "Number of ratings"}),
        use_container_width=True, hide_index=True)

with tab_about:
    st.subheader("Two recommendation models, one hybrid result")
    st.markdown("""
**1. Content-based filtering:** TF-IDF converts movie titles and genres into vectors. Cosine similarity compares those vectors.

**2. Collaborative filtering:** Movies are represented by their user-rating patterns. Movies rated similarly by the same users receive higher similarity.

**3. Hybrid ranking:** The app blends the two scores. Use the slider to change the balance between content and collaborative signals.

**4. Tie-breakers:** Average rating and rating count help rank close results; they are not the core similarity algorithm.

### Limitations
This demo uses MovieLens ratings and genre/title text. It does not use plot summaries or a logged-in user's own watch history. Collaborative similarity can be weak for movies with very few ratings.
""")
    st.markdown("")
    st.markdown("")

st.caption("")
