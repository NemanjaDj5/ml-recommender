"""
Module with helper functions for data steps
"""

import polars as pl

from configurations.logging_config import configure_logger

logger = configure_logger()


def filter_on_valid_id(dataset: pl.DataFrame, valid_movie_ids: set) -> pl.DataFrame:
    """
    Filter ratings based on valid movieIds.
    Args:
        dataset: Polars DataFrame with 'movieId' column.
        valid_movie_ids: set of valid movieIds.
    returns: filtered_ratings
    """
    dataset_valid_ids = dataset.filter(pl.col("movieId").is_in(valid_movie_ids))

    logger.info(
        "Ratings: removed %s rows after filtering.",
        dataset.height - dataset_valid_ids.height,
    )
    return dataset_valid_ids


def convert_timestamp(dataset: pl.DataFrame) -> pl.DataFrame:
    """
    Convert timestamps to regular datetime format.
    Args:
        dataset: Polars DataFrame with 'timestamp' column.
    returns: ratings_converted
    """
    dataset_converted = dataset.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("timestamp")
    )
    logger.info("Ratings: timestamp column converted to regular datetime format.")
    return dataset_converted
