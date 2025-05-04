import streamlit as st
import joblib
import pandas as pd
import requests
import urllib.request
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import re


st.set_page_config(page_title="Dataset Overview", layout="wide")

# -------------------------------
# Function to generate Google Drive download URL
# -------------------------------
def gdrive_url(file_id):
    return f"https://drive.google.com/uc?id={file_id}"

# -------------------------------
# Load Collaborative-Filtering Data
# -------------------------------
@st.cache_data
def load_cf_data():
    import io
    csv_ratings_id = "1SPcoSXprRZxAp0PD0v5Q83vz0lyryl6t"
    try:
        response = requests.get(gdrive_url(csv_ratings_id))
        response.raise_for_status()
        ratings = pd.read_csv(io.StringIO(response.text))
        ui = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
        sim = cosine_similarity(csr_matrix(ui.values), dense_output=False)
        return ratings, ui, sim
    except Exception as e:
        st.error(f"Failed to load ratings CSV: {e}")
        return pd.DataFrame(), pd.DataFrame(), None

sample_df, user_item_matrix, cosine_sim_user_item = load_cf_data()
if user_item_matrix is None or user_item_matrix.empty:
    st.stop()

# -------------------------------
# Load Movies Metadata
# -------------------------------
@st.cache_data
def load_movies():
    import io
    csv_movies_id = "1oRPpwyts6IDy8x7nZk5INqYnEu1GbxNV"
    try:
        response = requests.get(gdrive_url(csv_movies_id))
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df = df.rename(columns={'id': 'movieId'})
        df[['title', 'genres', 'content']] = df[['title', 'genres', 'content']].fillna('Unknown')
        return df
    except Exception as e:
        st.error(f"Failed to load movies metadata: {e}")
        return pd.DataFrame()

# -------------------------------
# Main Output
# -------------------------------
st.title("Dataset Overview")
df = load_movies()
ratings = sample_df

# -------------------------------
# Cleaned Genre Breakdown
# -------------------------------
st.subheader("Genre Breakdown")
cleaned_genres = (
    df["genres"]
    .apply(lambda x: re.sub(r"[^a-zA-Z|]", "", x))
    .str.split("|")
    .explode()
    .str.strip()
    .value_counts()
    .sort_values(ascending=False)
)
st.dataframe(cleaned_genres.rename_axis("Genre").reset_index(name="Count"))
st.bar_chart(cleaned_genres.head(20))  # Show only top 20 genres for clarity

# -------------------------------
# Rating vs. Number of Ratings
# -------------------------------
st.subheader("Rating vs. Number of Ratings")
agg = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "avg_rating", "count": "num_ratings"})
)
corr = agg["avg_rating"].corr(agg["num_ratings"])
st.write(f"Pearson corr: **{corr:.3f}**")
st.line_chart(agg.reset_index().set_index("num_ratings")["avg_rating"])
