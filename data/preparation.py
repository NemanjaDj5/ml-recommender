"""
This module provides functions for data preparation.
"""

import polars as pl
from utils.data_utils import filter_on_valid_id, convert_timestamp

from configurations.logging_config import configure_logger

logger = configure_logger()

# --------------------------------
# MOVIES CLEANING
# --------------------------------


def drop_no_genres(movies: pl.DataFrame) -> pl.DataFrame:
    """
    Drop rows where genres is equal to '(no genres listed)'.
    Args:
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns.
    Returns: (movies_kept, dropped)
    """
    movies_valid_genres = movies.filter(pl.col("genres") != "(no genres listed)")

    logger.info(
        "Movies: removed %s rows with '(no genres listed)'",
        movies.height - movies_valid_genres.height,
    )
    return movies_valid_genres


def normalize_genres(movies: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize genres to a simple space-separated, lowercase string.
    Args:
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns.
    Returns: normalized_genres
    """
    normalized_genres = movies.with_columns(
        pl.col("genres")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"\s*\|\s*", " ")
        .str.strip_chars()
    )
    logger.info("Movies: normalized genres (lowercase, spaces instead of '|').")
    return normalized_genres


def merge_duplicate_titles(
    movies: pl.DataFrame, ratings: pl.DataFrame
) -> tuple[pl.DataFrame, set]:
    """
    Merge duplicate titles.
    Args:
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns.
        ratings: Polars DataFrame with 'movieId', 'userId' and 'rating' columns.
    Returns: (merged_movies, updated_movieIds)
    """
    movies_duplicate_titles = movies.height

    rating_counts = ratings.group_by("movieId").agg(pl.len().alias("rating_count"))

    merged_movies = movies.join(rating_counts, on="movieId", how="left").with_columns(
        pl.col("rating_count").fill_null(0)
    )

    movies_no_duplicate_titles = merged_movies.group_by("title").agg(
        pl.col("movieId")
        .gather(pl.col("rating_count").arg_max())
        .first()
        .alias("movieId"),
        pl.col("genres")
        .gather(pl.col("genres").str.len_chars().arg_max())
        .first()
        .alias("genres"),
    )

    valid_ids = set(movies_no_duplicate_titles.get_column("movieId"))

    logger.info(
        "Movies: merged duplicated titles based on higher ranking count."
        "Rows before: %s , rows after: %s."
        "Number of duplicates removed %s.",
        movies_duplicate_titles,
        movies_no_duplicate_titles.height,
        movies_duplicate_titles - movies_no_duplicate_titles.height,
    )
    return movies_no_duplicate_titles, valid_ids


# ------------------------------------
# TAGS CLEANING
# ------------------------------------


def normalize_tag_column(tags: pl.DataFrame) -> pl.DataFrame:
    """
    This function is cleaning tag column from extra space, uppercase letters and
    special characters. Also adding underscore in inner space of tag column values with more words.
    After cleaning, it is grouping on movieId and removing duplicated values for specific movieId.
    Args:
        tags: Polars DataFrame with 'movieId', 'userId', 'tag' and 'timestamp' columns.
    returns: normalized_tags
    """
    normalized_tags = tags.with_columns(
        pl.col("tag")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.replace_all(r"[^a-z0-9 ]", "")
        .str.replace_all(r" ", "_")
        .alias("tag")
    )
    final_tags = (
        normalized_tags.group_by("movieId")
        .agg(pl.col("tag").unique().sort().alias("tag"))
        .with_columns(pl.col("tag").list.join(" "))
    )
    logger.info("Tags: Tag column normalized.")
    return final_tags


# -----------------------------------
# RATINGS CLEANING
# -----------------------------------


def mark_last_rating_per_user(ratings: pl.DataFrame) -> pl.DataFrame:
    """
    Add column with bool indicator for last users rating.
    Args:
        ratings: Polars DataFrame with 'movieId', 'userId', 'rating' and 'timestamp' columns.
    """
    final_ratings = ratings.with_columns(
        (pl.col("timestamp") == pl.col("timestamp").max().over("userId")).alias(
            "is_last_rating"
        )
    )
    logger.info("Ratings: added column with bool indicator for last users rating.")
    return final_ratings


# ---------------------------------
# PREPARATION MODULE WRAPPER
# --------------------------------


def run_data_preparation(
    movies: pl.DataFrame, ratings: pl.DataFrame, tags: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Orchestrates data preparation stage.
    Args:
        movies: Polars DataFrame with 'movieId', 'title' and 'genres' columns.
        ratings: Polars DataFrame with 'movieId', 'userId' and 'rating' columns.
        tags: Polars DataFrame with 'movieId', 'userId' and 'tag' columns.
    Returns: (movies_processed, ratings_processed, tags_processed)
    """
    logger.info("==== DATA PREPARATION START ====")

    logger.info("Movies dataset preparation...")
    dropped_genres = drop_no_genres(movies)

    merged_titles, valid_ids = merge_duplicate_titles(dropped_genres, ratings)

    normalized_movies = normalize_genres(merged_titles)

    logger.info("Movies cleaned, all done!")

    logger.info("Tags dataset preparation...")
    filtered_tags = filter_on_valid_id(tags, valid_ids)

    normalized_tags = normalize_tag_column(filtered_tags)

    logger.info("Tags cleaned, all done!")

    logger.info("Ratings dataset preparation...")
    filtered_ratings = filter_on_valid_id(ratings, valid_ids)

    converted_ratings = convert_timestamp(filtered_ratings)

    final_ratings = mark_last_rating_per_user(converted_ratings)

    logger.info("Ratings cleaned, all done!")

    logger.info("==== DATA PREPARATION FINISHED ====")
    return normalized_movies, normalized_tags, final_ratings