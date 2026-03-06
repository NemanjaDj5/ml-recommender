"""
This module provides functions for featurization.
"""

import polars as pl
from typing import Dict, Any

from configurations.logging_config import configure_logger

logger = configure_logger()


def assemble_movie_text(movies: pl.DataFrame, tags: pl.DataFrame) -> pl.DataFrame:
    """
    Join movies normalized genres column and tags normalized tag column. Build
    'movie_text' for TF-IDF vectorization.
    Args:
        movies: Polars DataFrame containing normalized 'genres' column.
        tags: Polars DataFrame containing normalized 'tag' column.
    Returns: final_df
    """
    merged_movies_tags = movies.join(
        tags.select("tag", "movieId"), on="movieId", how="left"
    )

    final_df = merged_movies_tags.with_columns(
        (pl.col("genres") + pl.lit(" ") + pl.col("tag").fill_null(""))
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.split(" ")
        .list.unique()
        .list.sort()
        .list.join(" ")
        .alias("movie_text")
    )
    final_df = final_df.sort("movieId")

    logger.info(
        "Created 'movie_text' column and sorted on 'movieId' column for consistency"
    )
    return final_df


def mapping_results(final_df: pl.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Build a row-index to movie metadata mapping from the featurized dataframe.

    Useful for resolving a numeric matrix row index back to a real movieId/title
    without re-joining the full dataframe (e.g. in custom inference scripts).

    Note: This function is not called by ``run_featurization``; it is provided
    as a standalone utility for downstream use.

    Args:
        final_df: Polars DataFrame containing at minimum 'movieId' and 'title' columns.
    Returns:
        movie_map: Dict mapping row index → {"movieId": int, "title": str}.
    """
    movie_ids = final_df.get_column("movieId").to_list()
    movie_titles = final_df.get_column("title").to_list()

    movie_map = {idx : {"movieId": movie_id, "title":title}
                 for idx, (movie_id, title) in enumerate(zip(movie_ids, movie_titles))}
    return movie_map



# --------------------------------------
# FEATURE MODULE WRAPPER
# --------------------------------------


def run_featurization(
    movies: pl.DataFrame, tags: pl.DataFrame
) -> tuple[pl.DataFrame, dict]:
    """
    Orchestrate featurization stage.
    Args:
        movies: Polars DataFrame containing normalized 'genres' column.
        tags: Polars DataFrame containing normalized 'tag' column.
    Returns: (featurized_df, movie_map)
    """
    logger.info("==== DATA FEATURIZATION START ====")
    logger.info("Creating feature column 'movie_text'...")

    final_df = assemble_movie_text(movies, tags)

    logger.info("Mapping all values from final_df...")
    movie_map = mapping_results(final_df)

    logger.info("==== DATA FEATURIZATION FINISHED ====")
    return final_df, movie_map