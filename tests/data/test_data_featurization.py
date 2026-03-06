"""
Unit tests for featurization pipeline.
"""

import unittest
import polars as pl
from polars.testing import assert_frame_equal

from data.featurization import *


class TestDataFeaturization(unittest.TestCase):
    """
    Unit tests for data featurization.
    """

    def setUp(self):
        """
        Set up test data for use in all tests.
        """
        self.movies = pl.DataFrame(
            {
                "movieId": [1, 2, 4, 3],
                "title": ["Title_A", "Title_B", "Title_C", "Title_D"],
                "genres": ["animation", " comedy romance", "thriller action", "action"],
            }
        )
        self.tags = pl.DataFrame(
            {
                "movieId": [1, 2, 4, 3],
                "tag": [" miyazaki  japan", "romance couple", "al_pacino mafia", None],
            }
        )
        self.final_df = pl.DataFrame(
            {"movieId": [1, 2, 3], "title": ["Title_A", "Title_B", "Title_C"]}
        )

    def test_assemble_movie_text_no_duplicate_values_in_feature(self):
        """
        Check if there are no duplicated values in new feature column
        """
        expected_output = "comedy couple romance"

        result = assemble_movie_text(self.movies, self.tags)
        duplicate_value = result.filter(pl.col("movieId") == 2).select("movie_text")

        self.assertEqual(duplicate_value.item(), expected_output)

    def test_assemble_movie_text_handles_extra_space(self):
        """
        Check that extra spaces are handled correctly
        """
        expected_output = "animation japan miyazaki"

        result = assemble_movie_text(self.movies, self.tags)
        extraspace_value = result.filter(pl.col("movieId") == 1).select("movie_text")

        self.assertEqual(extraspace_value.item(), expected_output)

    def test_assemble_movie_text_is_sorted_by_movie_id(self):
        """
        Check if output is sorted by movie_id
        """
        result = assemble_movie_text(self.movies, self.tags)
        is_sorted = result.get_column("movieId").is_sorted()

        self.assertTrue(is_sorted)

    def test_assemble_movie_text_handles_null_in_tags(self):
        """
        Check if null tags are handled correctly

        """
        expected_output = "action"

        result = assemble_movie_text(self.movies, self.tags)
        null_value = result.filter(pl.col("movieId") == 3).select("movie_text")

        self.assertEqual(null_value.item(), expected_output)

    def test_mapping_results_correctly_maps_data(self):
        """
        Check if mapping_results is correctly mapped data
        """
        expected_map = {
            0: {"movieId": 1, "title": "Title_A"},
            1: {"movieId": 2, "title": "Title_B"},
            2: {"movieId": 3, "title": "Title_C"},
        }

        result = mapping_results(self.final_df)

        self.assertEqual(result, expected_map)


class TestRunFeaturization(unittest.TestCase):
    """
    Unit test for orchestrator of data featurization process.
    """

    def setUp(self):
        """
        Set up test data for use in all tests.
        :return:
        """
        self.movies = pl.DataFrame(
            {
                "movieId": [1, 2, 4, 3],
                "title": ["Title_A", "Title_B", "Title_C", "Title_D"],
                "genres": ["animation", "comedy romance", "action", "thriller action"],
            }
        )
        self.tags = pl.DataFrame(
            {
                "movieId": [1, 2, 4, 3],
                "tag": [
                    "miyazaki japan",
                    "romance couple",
                    "tarzan",
                    "al_pacino mafia",
                ],
            }
        )

    def test_run_featurization_returns_correct_output(self):
        """
        Checks if run_featurization returns correct DataFrame and
        movie map dictionary.
        """
        expected_final_df = pl.DataFrame(
            {
                "movieId": [1, 2, 3, 4],
                "title": ["Title_A", "Title_B", "Title_D", "Title_C"],
                "genres": ["animation", "comedy romance", "thriller action", "action"],
                "tag": [
                    "miyazaki japan",
                    "romance couple",
                    "al_pacino mafia",
                    "tarzan",
                ],
                "movie_text": [
                    "animation japan miyazaki",
                    "comedy couple romance",
                    "action al_pacino mafia thriller",
                    "action tarzan",
                ],
            }
        )
        expected_movie_map = {
            0: {"movieId": 1, "title": "Title_A"},
            1: {"movieId": 2, "title": "Title_B"},
            2: {"movieId": 3, "title": "Title_D"},
            3: {"movieId": 4, "title": "Title_C"},
        }

        actual_final_df, actual_movie_map = run_featurization(self.movies, self.tags)

        assert_frame_equal(actual_final_df, expected_final_df)
        self.assertEqual(actual_movie_map, expected_movie_map)
