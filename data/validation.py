"""
This module provides functions for data validation
"""

import polars as pl

from configurations.logging_config import configure_logger

logger = configure_logger()


def validate_movies(movies: pl.DataFrame) -> None:
    """Validate movies dataframe, through check for null and duplicates value
    Args:
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns."""

    # Null check
    null_ids = movies.null_count().sum_horizontal()[0]
    if null_ids > 0:
        logger.error("Movies table contains %s null values", null_ids)
        raise ValueError("Validation failed: Null movieId found")

    # Duplicate IDs
    dup_ids = movies.height - movies["movieId"].n_unique()
    if dup_ids > 0:
        logger.error("Movies table contains %s duplicate values", dup_ids)
        raise ValueError("Validation failed: Duplicate movieId found")

    # Duplicate titles (not fatal)
    dup_titles = movies.height - movies["title"].n_unique()

    if dup_titles > 0:
        logger.warning("Movies table contains %s duplicate titles", dup_titles)

    logger.info("Movies validation completed successfully")


def validate_ratings(
    ratings: pl.DataFrame,
    movies: pl.DataFrame,
    min_ratings_per_user: int,
    min_rating: float = 0.0,
    max_rating: float = 5.0,
) -> None:
    """Validate ratings dataframe.
    Args:
        ratings: Polars DataFrame with 'movieId', 'userId', 'rating' and 'timestamp' columns.
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns.
        min_rating: Minimum rating to be considered.
        max_rating: Maximum rating to be considered.
        min_ratings_per_user: Threshold for valid number of ratings per user."""

    # Check for nulls
    null_ids = ratings.null_count().sum_horizontal()[0]

    if null_ids > 0:
        logger.error("Ratings table contains %s null values", null_ids)
        raise ValueError("Validation failed: Null ratings found")

    # Check referential integrity with movies table
    missing_from_movies_ratings = ratings.join(movies, on="movieId", how="anti").height

    if missing_from_movies_ratings > 0:
        logger.error(
            "Ratings table contains %s movieIds that are not present in movies table",
            missing_from_movies_ratings,
        )
        raise ValueError("Validation failed: Ratings dataset contains unknown movieIds")

    # Check rating range
    actual_min_rating = ratings["rating"].min()
    actual_max_rating = ratings["rating"].max()

    if actual_min_rating < min_rating or actual_max_rating > max_rating:
        logger.error(
            "Ratings table contains invalid values for rating column. "
            "Expected range: %s-%s, "
            "but found actual range: %s-%s",
            min_rating,
            max_rating,
            actual_min_rating,
            actual_max_rating,
        )
        raise ValueError("Validation failed: Rating column contains invalid values")

    # Check if all users have more then 1 rating
    user_activity = ratings.group_by("userId").len().rename({"len": "n_ratings"})
    low_user_cnt = user_activity.filter(
        pl.col("n_ratings") < min_ratings_per_user
    ).height

    if low_user_cnt > 0:
        logger.error(
            "Not every user has more than %s ratings,"
            "%s users had less than %s ratings",
            min_ratings_per_user,
            low_user_cnt,
            min_ratings_per_user,
        )
        raise ValueError(
            f"Validation failed: Not every user has more than {min_ratings_per_user} ratings."
        )

    logger.info("Ratings validation completed successfully")


def validate_tags(tags: pl.DataFrame, movies: pl.DataFrame) -> None:
    """Validate tags dataframe.
    Args:
        tags: Polars DataFrame with 'userId', 'movieId', 'tag' and 'timestamp' columns.
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns.
    """
    # Check for nulls
    null_ids = tags.null_count().sum_horizontal()[0]

    if null_ids > 0:
        logger.error("Tags table contains %s null values", null_ids)
        raise ValueError("Validation failed: Null values found in tags")

    # Check referential integrity with movies table
    missing_from_movies_tags = tags.join(movies, on="movieId", how="anti").height

    if missing_from_movies_tags > 0:
        logger.error(
            "Tags contains %s movieIds that are not present in movies table.",
            missing_from_movies_tags,
        )
        raise ValueError(
            f"Validation failed: Tags table contains {missing_from_movies_tags} unknown movieIds"
        )
    logger.info("Tags validation completed successfully")