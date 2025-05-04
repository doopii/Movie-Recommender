import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dataset Overview", layout="wide")

@st.cache_data
def load_movies(path="filtered_movies_metadata.csv"):
    df = pd.read_csv(path).rename(columns={"id":"movieId"})
    df[["genres","vote_average","vote_count"]] = df[["genres","vote_average","vote_count"]].fillna("Unknown")
    return df

@st.cache_data
def load_ratings(path="filtered_ratings.csv"):
    return pd.read_csv(path)

st.title("Dataset Overview")
df      = load_movies()
ratings = load_ratings()

st.subheader("Genre Breakdown")
counts = df["genres"].str.split("|").explode().value_counts()
st.dataframe(counts.rename_axis("Genre").reset_index(name="Count"))
st.bar_chart(counts)

st.subheader("Rating vs. Number of Ratings")
agg  = (ratings.groupby("movieId")["rating"]
           .agg(["mean","count"])
           .rename(columns={"mean":"avg_rating","count":"num_ratings"}))
corr = agg["avg_rating"].corr(agg["num_ratings"])
st.write(f"Pearson corr: **{corr:.3f}**")
st.line_chart(agg.reset_index().set_index("num_ratings")["avg_rating"])
