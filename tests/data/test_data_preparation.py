"""
Unit test for data preparation pipeline
"""

import unittest
import polars as pl
from datetime import datetime
from polars.testing import assert_frame_equal

from data.preparation import *

# ----- MOVIES DATA PREPARATION TESTS -----


class TestMoviesPreparation(unittest.TestCase):
    """
    Unit tests for movies data preparation.
    """

    def setUp(self):
        """
        Set up test data for use in tests.
        """
        self.movies = pl.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "title": [
                    "Title A",
                    "Title B",
                    "Title B",
                    "Title C",
                    "Title D",
                    "Title E",
                    "Title F",
                    "Title D",
                    "Title G",
                    "Title H",
                ],
                "genres": [
                    "(no genres listed)",
                    "Drama|Romance",
                    "Drama",
                    "Thriller|Action|Mystery",
                    "Action|Adventure",
                    "Animation",
                    "Animation|Children",
                    "Action|Adventure|Children",
                    "Sci-Fi",
                    "Musical|Romance",
                ],
            }
        )
        self.ratings = pl.DataFrame(
            {
                "userId": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "movieId": [2, 3, 3, 5, 5, 8, 8, 8, 1, 10],
                "rating": [4, 5, 3, 4, 4, 2, 5, 3, 5, 4],
            }
        )

    def test_drop_no_genres_removes_rows(self):
        """
        Does it remove rows at all
        """
        result = drop_no_genres(self.movies)

        self.assertLess(result.shape[0], self.movies.shape[0])

    def test_drop_no_genres_removes_specific_rows(self):
        """
        Check if wanted value is removed
        """
        result = drop_no_genres(self.movies)

        self.assertEqual(
            result.filter(pl.col("genres") == "(no genres listed)").height, 0
        )

    def test_drop_no_genres_does_nothing_if_value_not_found(self):
        """
        Does it return unchanged dataframe if there is no value "(no genres listed)"
        """
        movies_df = pl.DataFrame(
            {
                "movieId": [1, 2, 3],
                "title": ["Title A", "Title B", "Title C"],
                "genres": ["Drama|Romance", "Drama", "Thriller|Action|Mystery"],
            }
        )
        result = drop_no_genres(movies_df)

        self.assertEqual(movies_df.shape, result.shape)

    def test_drop_no_genres_empty_dataframe_input(self):
        """
        Does it return empty dataframe if input is empty dataframe
        """
        empty_df = pl.DataFrame(
            {
                "movieId": pl.Series(dtype=pl.Int64),
                "title": pl.Series(dtype=pl.Utf8),
                "genres": pl.Series(dtype=pl.Utf8),
            }
        )

        result = drop_no_genres(empty_df)

        assert_frame_equal(result, empty_df)

    def test_normalize_genres_removes_pipes(self):
        """
        Does it remove pipes inside genres column string
        """

        result = normalize_genres(self.movies)
        result_with_pipes = result.filter(
            pl.col("genres").str.contains("|", literal=True)
        ).height

        self.assertEqual(result_with_pipes, 0)

    def test_normalize_genres_handle_extra_space_and_uppercase(self):
        """
        Does it handle extra spaces and uppercase letters in genres column
        """
        movies_df = pl.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Title A", "Title B"],
                "genres": ["  DRAMA|Romance ", "Action | AdveNTure "],
            }
        )
        expected = pl.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Title A", "Title B"],
                "genres": ["drama romance", "action adventure"],
            }
        )

        result = normalize_genres(movies_df)

        assert_frame_equal(result, expected)

    def test_normalize_genres_handle_no_pipe_string(self):
        """
        Does it handle single strings without pipes
        """
        movies_df = pl.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Title A", "Title B"],
                "genres": ["Drama ", "Action"],
            }
        )
        expected = pl.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Title A", "Title B"],
                "genres": ["drama", "action"],
            }
        )

        result = normalize_genres(movies_df)

        assert_frame_equal(result, expected)

    def test_merge_duplicate_titles_removes_duplicated_titles(self):
        """
        Does it remove duplicated titles
        """
        result, _ = merge_duplicate_titles(self.movies, self.ratings)

        self.assertEqual(result.select(pl.col("title").is_duplicated().sum()).item(), 0)

    def test_merge_duplicate_titles_picks_larger_string_genres(self):
        """
        Does it pick genres with longer string length
        """
        expected_subset_df = pl.DataFrame(
            {
                "title": ["Title B", "Title D"],
                "movieId": [3, 8],
                "genres": ["Drama|Romance", "Action|Adventure|Children"],
            }
        ).sort("title")

        result, valid_ids = merge_duplicate_titles(self.movies, self.ratings)
        actual_subset_df = result.filter(
            pl.col("title").is_in(["Title B", "Title D"])
        ).sort("title")

        assert_frame_equal(actual_subset_df, expected_subset_df)

    def test_merge_duplicate_titles_picks_more_rated_movie(self):
        """
        Does it pick movieId with more ratings
        """
        expected_ids = {1, 3, 4, 6, 7, 8, 9, 10}

        result, valid_ids = merge_duplicate_titles(self.movies, self.ratings)

        self.assertEqual(expected_ids, set(result.get_column("movieId")))

    def test_merge_duplicate_titles_valid_ids_check(self):
        """
        Are valid_ids returned correct
        """
        expected_ids = {1, 3, 4, 6, 7, 8, 9, 10}

        result, valid_ids = merge_duplicate_titles(self.movies, self.ratings)

        self.assertEqual(expected_ids, valid_ids)

    # ------ TAGS DATA PREPARATION TESTS ------

class TestTagsPreparation(unittest.TestCase):
    """
    Unit tests for tags preparation phase.
    """
    def setUp(self):
        """
        Set up test data for all unit tests.
        """
        self.tags = pl.DataFrame(
            {
                "userId": [1, 2, 3, 2, 5],
                "movieId": [1, 2, 2, 1, 3],
                "tag": ["gangster.", "doctors", "doctors", "mafia! ", "Al Pacino"],
                "timestamp":[1221442134,2343252675,5235325345,2352353234,2352552334]
            }
        )

    def test_normalize_tag_column_handle_extra_space_uppercase_special_char(self):
        """
        Check if function handles extra spaces, uppercase letters and special character removal
        """
        expected_dataframe = pl.DataFrame(
            {
                "movieId": [1, 2, 3],
                "tag": ["gangster mafia", "doctors", "al_pacino"],
            }
        )
        result = normalize_tag_column(self.tags).sort("movieId")

        assert_frame_equal(result, expected_dataframe)

    def test_normalize_tag_column_deduplication(self):
        """
        Check if deduplication works
        """
        expected_value = "doctors"

        result = normalize_tag_column(self.tags)
        dedup_check = result.filter(pl.col("movieId") == 2).select("tag")

        self.assertEqual(expected_value, dedup_check.item())

    def test_normalize_tag_column_adds_underscore(self):
        """
        Check if function adds underscore characters
        """
        expected_value = "al_pacino"

        result = normalize_tag_column(self.tags)
        dedup_check = result.filter(pl.col("movieId") == 3).select("tag")

        self.assertEqual(expected_value, dedup_check.item())

    # ----- RATINGS DATA PREPARATION TESTS -----

class TestRatingsPreparation(unittest.TestCase):
    """
    Unit tests for ratings preparation phase.
    """
    def setUp(self):
        """
        Set up test data for use in all tests.
        """
        self.ratings = pl.DataFrame(
            {
                "userId": [1, 1, 3, 4, 3],
                "movieId": [2, 4, 5, 7, 9],
                "rating": [4, 2, 1, 3, 4],
                "timestamp": [
                    datetime(2025, 1, 5, 10, 0, 0),
                    datetime(2025, 1, 6, 10, 0, 0),
                    datetime(2025, 3, 5, 15, 0, 0),
                    datetime(2025, 2, 5, 10, 0, 0),
                    datetime(2025, 3, 10, 10, 0, 0),
                ],
            }
        )

    def test_mark_last_rating_per_user_column_exist(self):
        """
        Check if it creates column with last rating per user
        """
        expected_column_name = "is_last_rating"

        result = mark_last_rating_per_user(self.ratings)

        self.assertIn(expected_column_name, result.columns)

    def test_mark_last_rating_per_movie_right_values(self):
        """
        Check if column has only 1 and 0 values
        """
        expected_values = [True, False]

        result = mark_last_rating_per_user(self.ratings)

        self.assertTrue(
            result.select(
                pl.col("is_last_rating").is_in(expected_values).all()
            ).item()
        )

    def test_mark_last_rating_per_user_only_one_last_rating(self):
        """
        Check if there is only one last rating per user
        """

        result = mark_last_rating_per_user(self.ratings)
        summed_mark_per_user = result.group_by("userId").agg(
            pl.col("is_last_rating").sum()
        )

        self.assertTrue(
            summed_mark_per_user.select(
                pl.col("is_last_rating").is_in([1]).all()
            ).item()
        )


class TestRunDataPreparation(unittest.TestCase):
    """
    Unit test for data preparation orchestration.
    """
    def setUp(self):
        """
        Set up test data for use in all tests.
        """
        self.movies = pl.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5],
                "title": ["Title_A", "Title_B", "Title_B", "Title_C", "Title_D"],
                "genres": [
                    "Animation|Comedy",
                    "(no genres listed)",
                    "Adventure|Fantasy",
                    "Comedy|Crime",
                    "Action|Thriller",
                ],
            }
        )
        self.ratings = pl.DataFrame(
            {
                "userId": [1, 2, 3, 1, 4, 5],
                "movieId": [1, 2, 3, 1, 4, 99],
                "rating": [5, 4, 3, 4, 5, 2],
                "timestamp": [
                    964982703,
                    964982704,
                    964982705,
                    964982706,
                    964982707,
                    964982708,
                ],
            }
        )
        self.tags = pl.DataFrame(
            {
                "userId": [1, 2, 3, 4],
                "movieId": [1, 3, 4, 99],
                "tag": ["funny", "action thriller", "classic", "not valid tag"],
            }
        )

    def test_run_data_preparation_produces_correct_output(self):
        """
        Check if orchestrator function produces correct output.
        """
        expected_movies = pl.DataFrame(
            {
                "title": ["Title_A", "Title_B", "Title_C", "Title_D"],
                "movieId": [1, 3, 4, 5],
                "genres": [
                    "animation comedy",
                    "adventure fantasy",
                    "comedy crime",
                    "action thriller",
                ],
            }
        )
        expected_ratings = pl.DataFrame(
            {
                "userId": [1, 1, 3, 4],
                "movieId": [1, 1, 3, 4],
                "rating": [5, 4, 3, 5],
                "timestamp": [
                    964982703000000,
                    964982706000000,
                    964982705000000,
                    964982707000000,
                ],
                "is_last_rating": [False, True, True, True],
            }
        ).with_columns(pl.col("timestamp").cast(pl.Datetime(time_unit="us")))

        expected_tags = pl.DataFrame(
            {"movieId": [1, 3, 4], "tag": ["funny", "action_thriller", "classic"]}
        )

        actual_movies, actual_tags, actual_ratings = run_data_preparation(
            self.movies, self.ratings, self.tags
        )

        assert_frame_equal(expected_movies, actual_movies.sort("movieId"))
        assert_frame_equal(expected_ratings, actual_ratings.sort("movieId"))
        assert_frame_equal(expected_tags, actual_tags.sort("movieId"))
