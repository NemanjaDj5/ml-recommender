"""
Unit tests for data utils functions.
"""

import unittest
import polars as pl
from polars.testing import assert_frame_equal

from utils.data_utils import *


class TestDataUtils(unittest.TestCase):
    """
    Unit tests for utility functions in data steps.
    """

    def setUp(self):
        self.tags = pl.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5, 6],
                "userId": [7, 8, 9, 10, 11, 12],
                "tag": [
                    "doctors",
                    "animation",
                    "netflix",
                    "pool",
                    "social",
                    "animation",
                ],
                "timestamp": [
                    964980868,
                    964982546,
                    964982951,
                    964982290,
                    964982653,
                    964982346,
                ],
            }
        )
        self.valid_ids = {1, 3, 6}

    def test_filter_on_valid_id_retains_only_valid_id(self):
        """
        Check if it removes invalid ids
        """
        result = filter_on_valid_id(self.tags, self.valid_ids)

        self.assertEqual(self.valid_ids, set(result.get_column("movieId")))

    def test_filter_on_valid_id_returns_empty_on_no_matches(self):
        """
        Check if it returns empty dataframe if here are no matches
        """
        no_match_ids = {23, 44, 66}

        result = filter_on_valid_id(self.tags, no_match_ids)

        self.assertTrue(result.is_empty())
        self.assertEqual(result.height, 0)

    def test_convert_timestamp_valid_data_type(self):
        """
        Check if changed column have valid data type
        """
        result = convert_timestamp(self.tags)

        self.assertEqual(result.get_column("timestamp").dtype, pl.Datetime)

    def test_convert_timestamp_unix_conversion(self):
        """
        Check if output timestamps have valid values
        """
        input_df = pl.DataFrame({"userId": [1], "timestamp": 964980868})
        expected_df = pl.DataFrame(
            {
                "userId": [1],
                "timestamp": pl.Series([964980868 * 1000 * 1000]).cast(
                    pl.Datetime(time_unit="us")
                ),
            }
        )

        result = convert_timestamp(input_df)

        assert_frame_equal(result, expected_df)
