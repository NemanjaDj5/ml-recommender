import requests
import streamlit as st
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Movie Recommender Demo", layout="wide")
st.title("🎬 Content-Based Movie Recommender")

st.caption(
    f"Backend: {API_URL}  |  Start API with: `uvicorn api.app:app --reload`  |  Docs: "
    f"{API_URL}/docs"
)

# Inputs
user_id = st.number_input("User ID", min_value=1, value=6, step=1)
k = st.slider("Top-K Recommendations", min_value=5, max_value=30, value=10, step=1)

# Keep results between reruns
if "liked_df" not in st.session_state:
    st.session_state.liked_df = pd.DataFrame()
if "recs_df" not in st.session_state:
    st.session_state.recs_df = pd.DataFrame()

def _prettify(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns and make genres readable."""
    if df.empty:
        return df

    # Keep only columns we want to show
    cols = [c for c in ["title", "genres"] if c in df.columns]
    df = df[cols].copy()

    # Prettify genres like "Comedy|Drama" -> "Comedy, Drama"
    if "genres" in df.columns:
        df["genres"] = (
            df["genres"]
            .astype(str)
            .str.replace("|", ", ", regex=False)
            .str.replace(",", ", ")
        )

    return df

def fetch_data(user_id: int, k: int):
    try:
        rec_resp = requests.get(
            f"{API_URL}/recommend",
            params={"user_id": user_id, "k": k},
            timeout=10,
        )
        liked_resp = requests.get(
            f"{API_URL}/liked",
            params={"user_id": user_id},
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ FastAPI is not running.\n\n"
            "Start it in another terminal:\n"
            "`uvicorn api.app:app --reload --host 127.0.0.1 --port 8000`"
        )
        st.stop()

    if rec_resp.status_code != 200:
        st.error(f"❌ /recommend failed: {rec_resp.text}")
        st.stop()

    if liked_resp.status_code != 200:
        st.error(f"❌ /liked failed: {liked_resp.text}")
        st.stop()

    recs_df = pd.DataFrame(rec_resp.json())
    liked_df = pd.DataFrame(liked_resp.json())
    return liked_df, recs_df


if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        liked_df, recs_df = fetch_data(user_id=user_id, k=k)

    st.session_state.liked_df = _prettify(liked_df)
    st.session_state.recs_df = _prettify(recs_df)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ User Liked")
    if st.session_state.liked_df.empty:
        st.info("No liked movies found for this user (or cold-start user).")
    else:
        st.dataframe(st.session_state.liked_df, use_container_width=True, height=520)

with col2:
    st.subheader("⭐ Recommended")
    if st.session_state.recs_df.empty:
        st.info("Click **Recommend** to generate recommendations.")
    else:
        st.dataframe(st.session_state.recs_df, use_container_width=True, height=520)