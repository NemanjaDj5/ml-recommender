"""
General-purpose utility functions
used throughout the ml-onboarding-recommender project.
Functions include:
- Dataframe inspection function (check_schema)
"""

import polars as pl


def check_schema(data: pl.DataFrame, name: str) -> None:
    """
    Print the shape and schema of a dataframe.
    Args:
        data (pl.DataFrame): The dataframe to check.
        name (str): The name of the dataframe.
    """
    print(f"--- {name} ---")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Schema:")
    for col, dtype in data.schema.items():
        print(f"   {col:<20}{dtype}")
    print("=" * 40 + "\n")
