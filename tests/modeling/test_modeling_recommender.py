"""
Unit test for ContentBasedRecommender class
"""

import shutil
import tempfile
import unittest
import os

from unittest.mock import Mock
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal
from modeling.recommender import ContentBasedRecommender
from configurations.logging_config import configure_logger

logger = configure_logger()


class TestContentBasedRecommender(unittest.TestCase):
    """
    Unit tests for ContentBasedRecommender
    """

    RATING_THRESHOLD = 4.0
    TFIDF_NGRAM_RANGE = (1, 2)
    TFIDF_MIN_DF = 1
    TFIDF_MAX_DF = 0.95
    TFIDF_NORM = "l2"
    TFIDF_SMOOTH_IDF = True
    TFIDF_SUBLINEAR_TF = True
    TFIDF_USE_IDF = True
    SVD_COMPONENTS = 5
    SVD_RANDOM_STATE = 42

    def setUp(self):
        """
        Set up test data for use in all tests
        """
        self.final_movies = pl.DataFrame(
            {
                "movieId": [1, 2, 3, 4, 5],
                "title": ["Title_A", "Title_B", "Title_C", "Title_D", "Title_E"],
                "genres": [
                    "adventure animation",
                    "adventure",
                    "crime thriller",
                    "comedy romance",
                    "comedy",
                ],
                "tag": ["disney", "", "al_pacino", "", ""],
                "movie_text": [
                    "adventure animation disney",
                    "adventure",
                    "crime thriller al_pacino",
                    "comedy romance",
                    "comedy",
                ],
            }
        )
        self.final_ratings = pl.DataFrame(
            {
                "userId": [1, 1, 1, 2, 2, 3],
                "movieId": [1, 2, 3, 1, 2, 1],
                "rating": [5.0, 4.0, 3.0, 5.0, 4.5, 2.0],
            }
        )
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.joblib")

        self.recommender = ContentBasedRecommender(
            movies_df=self.final_movies,
            ngram_range=self.TFIDF_NGRAM_RANGE,
            rating_threshold=self.RATING_THRESHOLD,
            tfidf_min_df=self.TFIDF_MIN_DF,
            tfidf_max_df=self.TFIDF_MAX_DF,
            tfidf_norm=self.TFIDF_NORM,
            tfidf_smooth_idf=self.TFIDF_SMOOTH_IDF,
            tfidf_sublinear_tf=self.TFIDF_SUBLINEAR_TF,
            tfidf_use_idf=self.TFIDF_USE_IDF,
            svd_n_components=self.SVD_COMPONENTS,
            svd_random_state=self.SVD_RANDOM_STATE,
        )

    def tearDown(self):
        """
        Clean up the temporary directory after each test.
        """
        shutil.rmtree(self.temp_dir)

    def test_recommender_initialization(self):
        """
        Check if our recommender initializes with the correct attributes.
        """
        expected_params = {
            "ngram_range": self.TFIDF_NGRAM_RANGE,
            "rating_threshold": self.RATING_THRESHOLD,
            "tfidf_min_df": self.TFIDF_MIN_DF,
            "tfidf_max_df": self.TFIDF_MAX_DF,
            "tfidf_norm": self.TFIDF_NORM,
            "tfidf_smooth_idf": self.TFIDF_SMOOTH_IDF,
            "tfidf_sublinear_tf": self.TFIDF_SUBLINEAR_TF,
            "tfidf_use_idf": self.TFIDF_USE_IDF,
            "svd_n_components": self.SVD_COMPONENTS,
            "svd_random_state": self.SVD_RANDOM_STATE,
        }
        for attr_name, expected_value in expected_params.items():
            actual_value = getattr(self.recommender, attr_name)
            self.assertEqual(
                actual_value, expected_value, f"Mismatch for parameter {attr_name}"
            )

    def test_fit_raises_error_on_missing_feature_column(self):
        """
        Asserts that fit method raises error when a missing feature column is passed.
        """
        movies_no_movie_text = pl.DataFrame(
            {
                "movieId": [1, 2],
                "title": ["Title_A", "Title_B"],
                "genres": ["adventure animation", "adventure"],
                "tag": ["disney", ""],
            }
        )
        recommender = ContentBasedRecommender(movies_df=movies_no_movie_text)

        with self.assertRaises(ValueError) as e:
            recommender.fit(ratings_df=self.final_ratings)
        self.assertIn("movie_text", str(e.exception))

    def test_fit_handles_users_with_no_positive_ratings(self):
        """
        Asserts that fit method handles users with no positive ratings.
        """
        user_with_no_positive_ratings = 3

        self.recommender.fit(ratings_df=self.final_ratings)

        self.assertNotIn(user_with_no_positive_ratings, self.recommender.user_profiles)

    def test_fit_creates_correctly_sized_item_matrix(self):
        """
        Asserts that the item_matrix has the same number of rows as the movies_df,
        and same number of columns as the svd_n_components.
        """
        expected_rows = self.recommender.movies_df.height
        expected_cols = self.SVD_COMPONENTS

        self.recommender.fit(ratings_df=self.final_ratings)

        self.assertIsNotNone(self.recommender.item_matrix)
        actual_rows = self.recommender.item_matrix.shape[0]
        self.assertEqual(
            actual_rows,
            expected_rows,
            f"Expected {expected_rows} but got {actual_rows}",
        )
        actual_cols = self.recommender.item_matrix.shape[1]
        self.assertEqual(
            actual_cols,
            expected_cols,
            f"Expected {expected_cols} columns, " f"but got {actual_cols}.",
        )

    def test_fit_raises_error_on_empty_movies_dataframe(self):
        empty_movies_df = pl.DataFrame(
            {
                "movieId": [],
                "title": [],
                "genres": [],
                "movie_text": [],
            }
        )

        recommender = ContentBasedRecommender(movies_df=empty_movies_df)

        with self.assertRaises(ValueError) as e:
            recommender.fit(ratings_df=self.final_ratings)
        self.assertIn("The 'movies_df' must contain a non-empty 'movie_text' column.", str(e.exception))

    def test_fit_raises_error_on_empty_ratings_dataframe(self):
        """
        Asserts that ratings dataframe is not empty.
        """
        empty_ratings_df = pl.DataFrame({"userId": [], "movieId": [], "rating": []})

        with self.assertRaises(ValueError) as e:
            self.recommender.fit(ratings_df=empty_ratings_df)
        self.assertIn(
            "Ratings data must be provided via the fit() method and not be empty.",
            str(e.exception),
        )

    def test_recommend_does_not_return_seen_movies(self):
        """
        Asserts that recommended movies do not include seen movies. (even if they were not liked)
        """
        user_id = 1
        seen_movie_id = 1

        self.recommender.fit(ratings_df=self.final_ratings)
        recommendations = self.recommender.recommend(user_id=user_id)

        self.assertNotIn(seen_movie_id, recommendations.get_column("movieId").to_list())

    def test_recommend_raises_error_when_not_fitter(self):
        """
        Asserts that recommend method raises error when movie_text is not fitted.
        """
        unfitted_recommender = self.recommender

        with self.assertRaises(RuntimeError) as e:
            unfitted_recommender.recommend(user_id=1)
        self.assertIn(
            "Recommender has not been fitted. Please run fit() first.", str(e.exception)
        )

    def test_recommend_returns_popular_movies_for_cold_start_users(self):
        """
        Asserts that a cold-start user receives a list of popular movies.
        """
        cold_start_user = 5
        n_recommendations = 2

        self.recommender.fit(ratings_df=self.final_ratings)
        expected_recommendations = self.recommender._get_popular_movies(
            top_n=n_recommendations
        )
        actual_recommendations = self.recommender.recommend(
            user_id=cold_start_user, rec_n=n_recommendations
        )

        self.assertEqual(
            expected_recommendations.get_column("movieId").to_list(),
            actual_recommendations.get_column("movieId").to_list(),
        )
        self.assertEqual(actual_recommendations.height, n_recommendations)

    def test_get_popular_movies_correctly_identifies_most_popular_movies(self):
        """
        Asserts that the _get_popular_movies method returns the most popular movies.
        """
        mock_item_matrix = np.array(
            [
                [0.6, 0.3],
                [0.1, 0.1],
                [-0.5, 0.5],
            ]
        )

        mock_movie_index = {
            1: 0,
            2: 1,
            3: 2,
        }
        top_n_movies = 1
        self.recommender.item_matrix = mock_item_matrix
        self.recommender.movie_index = mock_movie_index

        popular_movies = self.recommender._get_popular_movies(top_n=top_n_movies)

        self.assertEqual(popular_movies.height, top_n_movies)
        self.assertEqual(popular_movies.get_column("movieId").to_list(), [2])

    def test_get_actual_liked_movies_returns_correct_movies(self):
        """
        Asserts if we get the actual liked movies.
        """
        user_id = 1
        expected_actual_liked_movies = [1, 2]

        self.recommender.fit(ratings_df=self.final_ratings)
        liked_movies = self.recommender.get_actual_liked_movies(user_id=user_id)

        actual_liked_movies = sorted(liked_movies.get_column("movieId").to_list())

        self.assertEqual(expected_actual_liked_movies, actual_liked_movies)

    def test_save_and_load_model(self):
        """
        Asserts that a saved model can be loaded correctly and produce same recommendations as the original model.
        """
        user_id = 1
        self.recommender.fit(ratings_df=self.final_ratings)

        original_recommendations = self.recommender.recommend(user_id=user_id)

        self.recommender.save_model(self.model_path)

        loaded_recommender = ContentBasedRecommender.load_model(
            file_path=self.model_path,
            movies_df=self.final_movies,
            ratings_df=self.final_ratings,
        )

        loaded_recommendations = loaded_recommender.recommend(user_id=user_id)

        assert_frame_equal(
            original_recommendations.sort("movieId"),
            loaded_recommendations.sort("movieId"),
        )

    def test_top_up_correctly_fills_recommendations(self):
        """
        Test that the method correctly tops up a short list of recommendations
        with the correct movies, ignoring the final order.
        """
        recommender = Mock(spec=ContentBasedRecommender)

        recommender.full_ratings = self.final_ratings
        recommender.movies_df = self.final_movies

        recommender._get_popular_movies = Mock()
        recommender._get_popular_movies.return_value = pl.DataFrame(
            {
                "movieId": [4, 5, 6, 7],
                "title": ["Pop Movie D", "Pop Movie E", "Pop Movie F", "Pop Movie G"],
                "genres": ["Popular"] * 4,
            }
        )

        user_id = 1
        n_recommendations = 5
        initial_recs = pl.DataFrame({"movieId": [1, 2], "similarity": [0.95, 0.9]})

        final_recs = ContentBasedRecommender._top_up_recommendations(
            recommender, initial_recs, n_recommendations, user_id
        )
        final_rec_ids = final_recs.get_column("movieId").to_list()

        self.assertEqual(
            len(final_rec_ids),
            n_recommendations,
            "Final recommendations list does not have the correct size.",
        )

        initial_rec_ids = initial_recs.get_column("movieId").to_list()
        top_up_ids_from_mock = (
            recommender._get_popular_movies.return_value.get_column("movieId")
            .head(3)
            .to_list()
        )
        expected_ids = set(initial_rec_ids) | set(top_up_ids_from_mock)

        self.assertSetEqual(
            set(final_rec_ids),
            expected_ids,
            "The set of final recommendations does not match the expected set.",
        )

    def test_get_movie_id_map_returns_correct_mapping(self):
        """
        Asserts that _get_movie_id_map returns the correct index-to-movieId mapping.
        """

        self.recommender.movie_index = {1: 0, 2: 1, 3: 2}

        expected_map = {0: 1, 1: 2, 2: 3}
        actual_map = self.recommender._get_movie_id_map()

        self.assertEqual(actual_map, expected_map)
