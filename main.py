import streamlit as st
import joblib
import pandas as pd
import requests

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Poster & Trailer Helpers
# -------------------------------
def fetch_poster(title: str) -> str:
    url = (
        "https://api.themoviedb.org/3/search/movie"
        "?api_key=8265bd1679663a7ea12ac168da84d2e8"
        f"&query={title}"
    )
    data = requests.get(url).json()
    try:
        return "https://image.tmdb.org/t/p/w500/" + data['results'][0]['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

def fetch_trailer(title: str) -> str:
    query = f"{title} official trailer"
    url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={'AIzaSyDC0zrSvxJSQ2CVU09Hxppwro9MH-H4XDo'}&maxResults=1&type=video"
    
    response = requests.get(url)
    data = response.json()
    
    try:
        video_id = data['items'][0]['id']['videoId']
        return f"https://www.youtube.com/watch?v={video_id}"
    except (KeyError, IndexError):
        return "https://www.youtube.com/results?search_query=" + query.replace(" ", "+")


# -------------------------------
# Collaborative-Filtering Data
# -------------------------------
@st.cache_data
def load_cf_data(path="filtered_ratings.csv"):
    ratings = pd.read_csv(path)
    ui      = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
    sim     = cosine_similarity(csr_matrix(ui.values), dense_output=False)
    return ratings, ui, sim

sample_df, user_item_matrix, cosine_sim_user_item = load_cf_data()




# -------------------------------
# Load Movies Metadata
# -------------------------------
def load_movies(path="filtered_movies_metadata.csv"):
    df = pd.read_csv(path)
    # <-- add this line:
    df = df.rename(columns={'id':'movieId'})
    df[['title','genres','content']] = df[['title','genres','content']].fillna('Unknown')
    return df

movies = load_movies()


# -------------------------------
# Load Models (cached per-approach)
# -------------------------------
@st.cache_resource
def load_models(opt: str):
    if opt=="Approach 1: KNN + Linear Regression":
        return {
            "clf": joblib.load("knn_classifier.pkl"),
            "reg": joblib.load("linear_regressor.pkl")
        }
    elif opt=="Approach 2: SVM":
        return {
            "clf": joblib.load("svm_classifier.pkl"),
            "reg": joblib.load("svm_regressor.pkl")
        }
    else:
        return {
            "clf":   joblib.load("random_forest_classifier.pkl"),
            "reg_rf":joblib.load("random_forest_regressor.pkl"),
            "reg_gb":joblib.load("gb_regressor.pkl")
        }


# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_with_model(
    movie_title: str,
    model_option: str,
    user_id: int = 0,
    genres: list[str] = None,
    avg_range: tuple[float,float] = None
) -> pd.DataFrame:

    m = load_models(model_option)
    df = movies.copy()
    genres = genres or []
    lo_avg, hi_avg = avg_range or (None, None)

    def cf_boost(sub_df, uid):
        if uid not in user_item_matrix.index:
            return sub_df
        row_idx = user_item_matrix.index.get_loc(uid)
        sims    = cosine_sim_user_item[row_idx].toarray().flatten()
        nbrs    = sims.argsort()[::-1][1:21]
        neigh_df = user_item_matrix.iloc[nbrs]
        liked_ids = set(neigh_df.columns[(neigh_df >= 2.5).any(axis=0)])
        filtered = sub_df[sub_df['movieId'].isin(liked_ids)]
        return filtered



    # 1) build title‐matches
    title_matches = (
        df[df['title'].str.lower().str.contains(movie_title.lower())]
          .sort_values('weighted_rating', ascending=False)
    ) if movie_title else pd.DataFrame()

    # 2) Approach 1 shortcut
    if movie_title and len(title_matches) >= 10 \
       and model_option == "Approach 1: KNN + Linear Regression":
        tm = title_matches.copy()
        if lo_avg is not None:
            tm = tm[(tm['vote_average'] >= lo_avg) & (tm['vote_average'] <= hi_avg)]
        if genres:
            pat = "|".join(genres)
            tm = tm[tm['genres'].str.contains(pat)]
        return tm.head(20)

    # 3) Approach 3 shortcut
    if movie_title and len(title_matches) >= 10 \
       and model_option == "Approach 3: Random Forest + GBM":
        tm = title_matches.copy()
        if lo_avg is not None:
            tm = tm[(tm['vote_average'] >= lo_avg) & (tm['vote_average'] <= hi_avg)]
        if genres:
            pat = "|".join(genres)
            tm = tm[tm['genres'].str.contains(pat)]
        return cf_boost(tm, user_id).sort_values('weighted_rating', ascending=False).head(20)

    # 4) exact → model → candidates
    exact = (
        df[df['title'].str.lower() == movie_title.lower()]
        if movie_title else pd.DataFrame()
    )
    if exact.empty:
        candidates = title_matches.copy() if not title_matches.empty else df.copy()
    else:
        row   = exact.iloc[[0]]
        feats = row[['vote_count','vote_average','weighted_rating']]
        cls   = m['clf'].predict(feats)[0]

        if model_option == "Approach 1: KNN + Linear Regression":
            pred = m["reg"].predict(feats)[0]
            lo, hi = max(pred-0.5,0), min(pred+0.5,10)

            if genres:
                pat = "|".join(genres)
                candidates = df[
                    (df['vote_average'] >= lo) &
                    (df['vote_average'] <= hi) &
                    df['genres'].str.contains(pat, regex=True)
                ]
            else:
                candidates = df[
                    (df['vote_average'] >= lo) &
                    (df['vote_average'] <= hi)
                ]

        elif model_option == "Approach 2: SVM":
            pred = m["reg"].predict(feats)[0]
            vc   = int(row['vote_count'])
            lo, hi = int(vc*0.8), int(vc*1.2)

            if genres:
                pat = "|".join(genres)
                candidates = df[
                    (df['vote_count']>=lo)&(df['vote_count']<=hi)&
                    df['genres'].str.contains(pat, regex=True)
                ]
            else:
                candidates = df[
                    (df['vote_count']>=lo)&(df['vote_count']<=hi)
                ]


        else:  # Approach 3 full RF+GBM path
            rf_p = m["reg_rf"].predict(feats)[0]
            gb_p = m["reg_gb"].predict(feats)[0]
            pred = (rf_p + gb_p) / 2
            lo, hi = max(pred-0.5,0), min(pred+0.5,10)
            candidates = df[
                (df['weighted_rating']>=lo)&(df['weighted_rating']<=hi)
            ]
            candidates = cf_boost(candidates, user_id)
            candidates = candidates[candidates['rating_class'] == cls]

    # enforce CF one more time on full RF+GBM path
    if model_option == "Approach 3: Random Forest + GBM":
        candidates = cf_boost(candidates, user_id)

    # 5) shared post-filters
    if lo_avg is not None:
        candidates = candidates[
            (candidates['vote_average']>=lo_avg)&
            (candidates['vote_average']<=hi_avg)
        ]
    if genres:
        pat = "|".join(genres)
        candidates = candidates[candidates['genres'].str.contains(pat)]
    min_votes = df['vote_count'].mean()
    candidates = candidates[candidates['vote_count']>=min_votes]

    # finally, always return a DataFrame (never None)
    return candidates.head(30)



# -------------------------------
# Sidebar + State + Reset
# -------------------------------
if 'recs' not in st.session_state:
    st.session_state.update({
        "movie_input":     "",
        "user_id":         int(user_item_matrix.index[0]),
        "selected_genres": [],
        "avg_range":       (
            float(movies["vote_average"].min()),
            float(movies["vote_average"].max()),
        ),
        "model_option":    "Approach 1: KNN + Linear Regression",
        "min_votes":       int(movies["vote_count"].min()),
        "recs":            pd.DataFrame()
    })

def refresh_recs():
    st.session_state.recs = recommend_with_model(
        st.session_state.movie_input,
        st.session_state.model_option,
        st.session_state.user_id,
        st.session_state.selected_genres,
        st.session_state.avg_range,
    )

def reset_filters():
    st.session_state.update({
        "movie_input":     "",
        "user_id":         0,
        "selected_genres": [],
        "avg_range":       (
            float(movies["vote_average"].min()),
            float(movies["vote_average"].max()),
        ),
        "model_option":    "Approach 1: KNN + Linear Regression"
    })
    refresh_recs()

with st.sidebar:
    st.title("Navigation")
    st.selectbox(
        "Recommendation Approach",
        [
            "Approach 1: KNN + Linear Regression",
            "Approach 2: SVM",
            "Approach 3: Random Forest + GBM"
        ],
        key="model_option", on_change=refresh_recs
    )
    st.title("Filters")
    st.text_input("Movie Title (optional)", key="movie_input", on_change=refresh_recs)
    # only show User ID for Approach 3:
    if st.session_state.model_option == "Approach 3: Random Forest + GBM":
        uid_input = st.text_input("User ID (numeric only)", value=str(st.session_state.user_id))
        if uid_input.isdigit():
            uid_val = int(uid_input)
            if uid_val in user_item_matrix.index:
                st.session_state.user_id = uid_val
                refresh_recs()
            else:
                st.warning(f"User ID {uid_val} not found.")
        else:
            st.warning("Please enter a valid numeric User ID.")
    else:
        st.markdown("_User ID only used in Approach 3_")

 
    all_genres = sorted({g for row in movies["genres"] for g in row.split()})
    st.multiselect("Genre", all_genres, key="selected_genres", on_change=refresh_recs)
    min_avg = float(movies["vote_average"].min())
    max_avg = float(movies["vote_average"].max())
    st.slider("Avg Rating Range", min_avg, max_avg,
              value=st.session_state.avg_range,
              key="avg_range", on_change=refresh_recs)
    st.number_input(
        "Min Vote Count",
        min_value=int(movies["vote_count"].min()),
        max_value=int(movies["vote_count"].max()),
        value=st.session_state.min_votes,
        step=1,
        key="min_votes",
        on_change=refresh_recs,
    )
    st.markdown("---")
    st.button("Reset All Filters", on_click=reset_filters)

    st.markdown("### Available user IDs (top 10)")
    user_counts = sample_df['userId'].value_counts().head(10)
    st.dataframe(user_counts.rename_axis("userId").reset_index(name="rating_count"))


# -------------------------------
# Main Output
# -------------------------------
st.title("Movie Recommendation System")
if st.session_state.model_option == "Approach 3: Random Forest + GBM":
    uid = st.session_state.user_id
    if uid in user_item_matrix.index:
        liked = sample_df[(sample_df['userId'] == uid) & (sample_df['rating'] >= 4.0)]
        liked_df = movies[movies['movieId'].isin(liked['movieId'])]

        if not liked_df.empty:
            st.subheader(f"Movies Liked by User {uid}")
            show_more = st.checkbox("Show More Liked Movies")
            display_df = liked_df if show_more else liked_df.head(3)

            cols = st.columns(3)
            for i, (_, row) in enumerate(display_df.iterrows()):
                poster = fetch_poster(row['title'])
                with cols[i % 3]:
                    st.image(poster, width=160)
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"⭐ {row['vote_average']:.1f} ({int(row['vote_count'])} votes)")
        else:
            st.warning(f"User ID {uid} has no liked movies (rating ≥ 4.0).")



st.write("All changes above automatically update recommendations.")

recs = st.session_state.recs
# If no user input, show top weighted movies as default recommendations
if st.session_state.movie_input.strip() == "" and recs.empty:
    recs = movies[
        movies['vote_count'] >= movies['vote_count'].mean()
    ].sort_values("weighted_rating", ascending=False).head(10)

if not recs.empty:
    recs = recs[ recs['vote_count'] >= st.session_state.min_votes ]

if recs.empty:
    st.info("No recommendations to show—adjust your filters.")
else:
    sort_option = st.selectbox(
        "Sort recommendations by:",
        [
            "Default",
            "Avg Rating (High → Low)",
            "Avg Rating (Low → High)",
            "Vote Count (High → Low)",
            "Weighted Rating (High → Low)",
            "Title (A → Z)",
            "Title (Z → A)"
        ]
    )

    df2 = recs.copy()
    if sort_option == "Avg Rating (High → Low)":
        df2 = df2.sort_values("vote_average", ascending=False)
    elif sort_option == "Avg Rating (Low → High)":
        df2 = df2.sort_values("vote_average", ascending=True)
    elif sort_option == "Vote Count (High → Low)":
        df2 = df2.sort_values("vote_count", ascending=False)
    elif sort_option == "Weighted Rating (High → Low)":
        df2 = df2.sort_values("weighted_rating", ascending=False)
    elif sort_option == "Title (A → Z)":
        df2 = df2.sort_values("title", ascending=True)
    elif sort_option == "Title (Z → A)":
        df2 = df2.sort_values("title", ascending=False)

    cols = st.columns(3)
    for i, (_, row) in enumerate(df2.iterrows()):
        with cols[i % 3]:
            poster = fetch_poster(row['title'])
            trailer = fetch_trailer(row['title'])
            card = f"""
            <div style="
                background-color:#1a1a1a;
                padding:20px;
                border-radius:14px;
                margin-bottom:26px;
                font-family:Segoe UI, sans-serif;
                color:#f0f0f0;
                height:500px;
                box-sizing:border-box;
                display:flex;
                flex-direction:column;
                justify-content:space-between;
            ">
                <div>
                    <div style="text-align:center; margin-bottom:14px;">
                        <img src="{poster}" style="
                            width:100%; height:240px;
                            object-fit:cover;
                            border-radius:10px;
                        "/>
                    </div>
                    <h4 style="
                        margin:0 0 10px 0;
                        font-size:1.2rem;
                        font-weight:600;
                        color:#ffffff;
                    ">{row['title']}</h4>
                    <p style="
                        margin:6px 0;
                        font-size:0.95rem;
                        color:#bbbbbb;
                    "><em>{row['genres']}</em></p>
                    <p style="margin:6px 0; font-size:0.95rem;">
                        ⭐ <strong>{row['vote_average']:.1f}</strong> &nbsp;💬 {int(row['vote_count']):,}
                    </p>
                    <p style="margin:6px 0 14px 0; font-size:0.95rem;">
                        🔎 Score: <strong>{row['weighted_rating']:.1f}</strong>
                    </p>
                </div>
                <a href="{trailer}" target="_blank" style="
                    font-size:0.95rem;
                    color:#4dabf7;
                    text-decoration:none;
                    font-weight:500;
                    margin-top:auto;
                ">▶ Watch Trailer</a>
            </div>
            """
            st.markdown(card, unsafe_allow_html=True)

