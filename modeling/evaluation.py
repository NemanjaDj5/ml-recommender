from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvalResult:
    """
    Container for leave-one-out evaluation results.

    Attributes:
        hit_rate_at_k: Fraction of users for whom the held-out liked movie
            appeared in the top-K recommendations.
        n_users: Number of users actually evaluated.
        k: Cut-off used for recommendations.
        coverage_at_k: Ratio of unique recommended movies to total recommendations
            (indicates catalogue diversity across users).
        users_evaluated: List of userId values that were included in the evaluation.
    """
    hit_rate_at_k: float
    n_users: int
    k: int
    coverage_at_k: float
    users_evaluated: List[int]


def _build_profile_from_movie_ids(
    movie_ids: List[int],
    ratings: Optional[List[float]],
    item_matrix: np.ndarray,
    movie_index: Dict[int, int],
) -> Optional[np.ndarray]:
    """
    Build a user profile vector as a weighted average of item vectors.

    Args:
        movie_ids: List of movie IDs representing the user's liked movies.
        ratings: Optional list of ratings aligned with movie_ids used as weights.
            If None, a simple unweighted mean is used.
        item_matrix: 2-D array of shape (n_movies, n_components) produced by SVD.
        movie_index: Mapping from movieId to row index in item_matrix.

    Returns:
        A 1-D numpy array representing the user profile, or None if no
        movie_ids could be mapped to a valid index.
    """
    idxs = [movie_index.get(int(mid)) for mid in movie_ids]
    pairs = [(mid, idx) for mid, idx in zip(movie_ids, idxs) if idx is not None]
    if not pairs:
        return None

    valid_movie_ids = [mid for mid, _ in pairs]
    valid_idxs = [idx for _, idx in pairs]
    vectors = item_matrix[valid_idxs]

    if ratings is None:
        return np.mean(vectors, axis=0)

    # Align ratings to valid_movie_ids
    rating_map = {int(mid): float(r) for mid, r in zip(movie_ids, ratings)}
    w = np.array([rating_map[int(mid)] for mid in valid_movie_ids], dtype=float)

    # If all weights are 0 (shouldn't happen), fall back to mean
    if np.allclose(w.sum(), 0.0):
        return np.mean(vectors, axis=0)

    return np.average(vectors, axis=0, weights=w)


def _recommend_from_profile(
    user_profile: np.ndarray,
    item_matrix: np.ndarray,
    index_to_movie_id: Dict[int, int],
    exclude_movie_ids: Set[int],
    top_k: int,
) -> List[int]:
    """
    Recommend the top-K movies most similar to a user profile vector.

    Args:
        user_profile: 1-D array representing the user's taste profile.
        item_matrix: 2-D array of shape (n_movies, n_components).
        index_to_movie_id: Mapping from row index in item_matrix to movieId.
        exclude_movie_ids: Set of movieIds to skip (already-seen movies).
        top_k: Number of recommendations to return.

    Returns:
        List of up to top_k movieIds ordered by descending cosine similarity.
    """
    sims = cosine_similarity(user_profile.reshape(1, -1), item_matrix).flatten()
    # Sort indices by similarity desc
    ranked = np.argsort(-sims)

    recs: List[int] = []
    for idx in ranked:
        mid = int(index_to_movie_id[idx])
        if mid in exclude_movie_ids:
            continue
        recs.append(mid)
        if len(recs) >= top_k:
            break
    return recs


def evaluate_leave_one_out(
    recommender,
    ratings_df: pl.DataFrame,
    like_threshold: float = 4.0,
    min_liked: int = 5,
    k: int = 10,
    max_users: int = 300,
    seed: int = 42,
) -> EvalResult:
    """
    Leave-one-out HitRate@K for your content-based recommender (fast, sensible).

    Requirements:
    - recommender.fit(...) must have been called
    - recommender has: item_matrix, movie_index, full_ratings, rating_threshold
    """
    if recommender.item_matrix is None or not recommender.movie_index:
        raise RuntimeError("Recommender must be fitted before evaluation.")

    rng = np.random.default_rng(seed)

    # Use provided threshold, but default to your model's threshold
    thr = float(like_threshold)

    liked = ratings_df.filter(pl.col("rating") >= thr)

    # Group liked movies per user (movieIds + ratings for weights)
    grouped = (
        liked.group_by("userId")
        .agg(
            pl.col("movieId").alias("movie_ids"),
            pl.col("rating").alias("ratings"),
        )
    )

    # Keep only users with enough liked items
    grouped = grouped.filter(pl.col("movie_ids").list.len() >= min_liked)

    user_rows = grouped.select(["userId", "movie_ids", "ratings"]).to_dicts()
    if not user_rows:
        return EvalResult(0.0, 0, k, 0.0, [])

    # Sample users for speed
    if len(user_rows) > max_users:
        user_rows = list(rng.choice(user_rows, size=max_users, replace=False))

    index_to_movie_id = {v: k for k, v in recommender.movie_index.items()}

    hits = 0
    all_recs: List[int] = []
    users_eval: List[int] = []

    for row in user_rows:
        user_id = int(row["userId"])
        movie_ids = [int(x) for x in row["movie_ids"]]
        ratings = [float(x) for x in row["ratings"]]

        # choose held-out liked movie
        held_out_pos = int(rng.integers(0, len(movie_ids)))
        held_out_movie = movie_ids[held_out_pos]

        train_movie_ids = [m for i, m in enumerate(movie_ids) if i != held_out_pos]
        train_ratings = [r for i, r in enumerate(ratings) if i != held_out_pos]

        # build a profile using ONLY train liked movies
        profile = _build_profile_from_movie_ids(
            train_movie_ids,
            train_ratings,
            recommender.item_matrix,
            recommender.movie_index,
        )
        if profile is None:
            continue

        # exclude everything the user has seen (not only liked)
        seen_ids = (
            ratings_df.filter(pl.col("userId") == user_id)
            .get_column("movieId")
            .to_list()
        )
        exclude = set(int(x) for x in seen_ids)

        recs = _recommend_from_profile(
            user_profile=profile,
            item_matrix=recommender.item_matrix,
            index_to_movie_id=index_to_movie_id,
            exclude_movie_ids=exclude,
            top_k=k,
        )

        hit = int(held_out_movie in set(recs))
        hits += hit
        all_recs.extend(recs)
        users_eval.append(user_id)

    n_users = len(users_eval)
    hit_rate = hits / n_users if n_users else 0.0
    coverage = len(set(all_recs)) / len(all_recs) if all_recs else 0.0

    return EvalResult(hit_rate, n_users, k, coverage, users_eval)