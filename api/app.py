"""
FastAPI application for the Movie Recommender API.

Loads the trained ContentBasedRecommender model once on startup and exposes
two endpoints:
  - GET /recommend  — personalised top-K movie recommendations for a user.
  - GET /liked      — movies a user has rated at or above the rating threshold.
"""

import os
from fastapi import FastAPI, HTTPException
import polars as pl

from modeling.recommender import ContentBasedRecommender

app = FastAPI(title="Movie Recommender API")

# Defaults that work after cloning the repo
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "modeling/models/recommender_model.joblib")
MOVIES_PATH = os.getenv("MOVIES_PATH", "datasets/raw/movies.csv")
RATINGS_PATH = os.getenv("RATINGS_PATH", "datasets/raw/ratings.csv")

# Load data once on startup
movies_df = pl.read_csv(MOVIES_PATH)
ratings_df = pl.read_csv(RATINGS_PATH)

# Load trained model once on startup
load_error: str = ""
try:
    recommender = ContentBasedRecommender.load_model(
        DEFAULT_MODEL_PATH,
        movies_df=movies_df,
        ratings_df=ratings_df,
    )
except Exception as e:
    recommender = None
    load_error = str(e)


@app.get("/")
def health_check():
    """Return API health status and the active model path."""
    if recommender is None:
        return {"status": "error", "detail": f"Model failed to load: {load_error}"}
    return {"status": "ok", "model_path": DEFAULT_MODEL_PATH}


@app.get("/recommend")
def recommend(user_id: int, k: int = 10):
    """
    Return the top-K movie recommendations for a given user.

    For unknown users (cold-start), falls back to the most content-central movies.

    Args:
        user_id: The ID of the user to generate recommendations for.
        k: Number of recommendations to return (default 10).

    Returns:
        List of dicts with keys: movieId, title, genres.
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")

    recs = recommender.recommend(user_id=user_id, rec_n=k)
    return recs.to_dicts()


@app.get("/liked")
def liked(user_id: int):
    """
    Return all movies a user has rated at or above the rating threshold.

    Args:
        user_id: The ID of the user whose liked movies to retrieve.

    Returns:
        List of dicts with keys: movieId, title, genres.
    """
    if recommender is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {load_error}")

    liked_df = recommender.get_actual_liked_movies(user_id=user_id)
    return liked_df.to_dicts()