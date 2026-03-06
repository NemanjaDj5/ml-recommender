"""
Unit tests for evaluation module.
"""

import unittest
import numpy as np
import polars as pl

from modeling.evaluation import (
    EvalResult,
    _build_profile_from_movie_ids,
    _recommend_from_profile,
    evaluate_leave_one_out,
)


class TestBuildProfileFromMovieIds(unittest.TestCase):
    """
    Unit tests for _build_profile_from_movie_ids helper function.
    """

    def setUp(self):
        """
        Set up a small item matrix and movie index for use in all tests.
        """
        self.item_matrix = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]
        )
        self.movie_index = {1: 0, 2: 1, 3: 2}

    def test_returns_weighted_average_with_ratings(self):
        """
        Check that the profile is a weighted average when ratings are provided.
        """
        movie_ids = [1, 2]
        ratings = [4.0, 2.0]

        profile = _build_profile_from_movie_ids(
            movie_ids, ratings, self.item_matrix, self.movie_index
        )

        expected = np.average(
            self.item_matrix[[0, 1]], axis=0, weights=np.array([4.0, 2.0])
        )
        np.testing.assert_array_almost_equal(profile, expected)

    def test_returns_mean_when_no_ratings_provided(self):
        """
        Check that a simple mean is used when ratings is None.
        """
        movie_ids = [1, 2]

        profile = _build_profile_from_movie_ids(
            movie_ids, None, self.item_matrix, self.movie_index
        )

        expected = np.mean(self.item_matrix[[0, 1]], axis=0)
        np.testing.assert_array_almost_equal(profile, expected)

    def test_returns_none_when_no_valid_movie_ids(self):
        """
        Check that None is returned when none of the movie_ids exist in movie_index.
        """
        movie_ids = [99, 100]
        ratings = [5.0, 4.0]

        profile = _build_profile_from_movie_ids(
            movie_ids, ratings, self.item_matrix, self.movie_index
        )

        self.assertIsNone(profile)

    def test_skips_invalid_movie_ids_and_aligns_ratings(self):
        """
        Check that invalid movie_ids are skipped and ratings are aligned correctly.
        """
        movie_ids = [1, 99, 2]
        ratings = [4.0, 5.0, 2.0]

        profile = _build_profile_from_movie_ids(
            movie_ids, ratings, self.item_matrix, self.movie_index
        )

        # Only movieId 1 and 2 are valid
        expected = np.average(
            self.item_matrix[[0, 1]], axis=0, weights=np.array([4.0, 2.0])
        )
        self.assertIsNotNone(profile)
        np.testing.assert_array_almost_equal(profile, expected)


class TestRecommendFromProfile(unittest.TestCase):
    """
    Unit tests for _recommend_from_profile helper function.
    """

    def setUp(self):
        """
        Set up a small item matrix and mappings for use in all tests.
        """
        self.item_matrix = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.8, 0.2],
            ]
        )
        self.index_to_movie_id = {0: 1, 1: 2, 2: 3, 3: 4}

    def test_returns_correct_number_of_recommendations(self):
        """
        Check that the correct number of recommendations is returned.
        """
        user_profile = np.array([1.0, 0.0])
        recs = _recommend_from_profile(
            user_profile, self.item_matrix, self.index_to_movie_id, set(), top_k=2
        )

        self.assertEqual(len(recs), 2)

    def test_excludes_seen_movies(self):
        """
        Check that seen movies are excluded from recommendations.
        """
        user_profile = np.array([1.0, 0.0])
        exclude = {1, 4}

        recs = _recommend_from_profile(
            user_profile, self.item_matrix, self.index_to_movie_id, exclude, top_k=2
        )

        for movie_id in exclude:
            self.assertNotIn(movie_id, recs)

    def test_recommendations_are_ordered_by_similarity(self):
        """
        Check that returned movies are ordered by descending cosine similarity.
        """
        user_profile = np.array([1.0, 0.0])

        recs = _recommend_from_profile(
            user_profile, self.item_matrix, self.index_to_movie_id, set(), top_k=4
        )

        # movieId 1 ([1.0, 0.0]) should be most similar to the profile [1.0, 0.0]
        self.assertEqual(recs[0], 1)


class TestEvaluateLeaveOneOut(unittest.TestCase):
    """
    Unit tests for evaluate_leave_one_out function.
    """

    def setUp(self):
        """
        Set up a fitted mock recommender and ratings dataframe for use in all tests.
        """
        self.item_matrix = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )
        self.movie_index = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

        self.recommender = type("MockRecommender", (), {})()
        self.recommender.item_matrix = self.item_matrix
        self.recommender.movie_index = self.movie_index
        self.recommender.rating_threshold = 4.0

        self.ratings_df = pl.DataFrame(
            {
                "userId": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                "movieId": [1, 2, 3, 4, 5, 1, 1, 2, 3, 4, 5, 2],
                "rating": [5.0, 4.0, 4.5, 4.0, 5.0, 3.0, 5.0, 4.0, 4.5, 4.0, 5.0, 3.0],
            }
        )

    def test_returns_eval_result_instance(self):
        """
        Check that the function returns an EvalResult dataclass instance.
        """
        result = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=4.0, min_liked=3, k=2
        )

        self.assertIsInstance(result, EvalResult)

    def test_hit_rate_is_between_zero_and_one(self):
        """
        Check that hit_rate_at_k is a valid probability between 0 and 1.
        """
        result = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=4.0, min_liked=3, k=2
        )

        self.assertGreaterEqual(result.hit_rate_at_k, 0.0)
        self.assertLessEqual(result.hit_rate_at_k, 1.0)

    def test_coverage_is_between_zero_and_one(self):
        """
        Check that coverage_at_k is a valid ratio between 0 and 1.
        """
        result = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=4.0, min_liked=3, k=2
        )

        self.assertGreaterEqual(result.coverage_at_k, 0.0)
        self.assertLessEqual(result.coverage_at_k, 1.0)

    def test_k_value_is_stored_correctly(self):
        """
        Check that the k value used is correctly stored in EvalResult.
        """
        k = 3
        result = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=4.0, min_liked=3, k=k
        )

        self.assertEqual(result.k, k)

    def test_users_below_min_liked_are_excluded(self):
        """
        Check that users with fewer liked movies than min_liked are not evaluated.
        """
        result_strict = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=4.0, min_liked=10, k=2
        )

        # With min_liked=10, no user qualifies — should return empty result
        self.assertEqual(result_strict.n_users, 0)
        self.assertEqual(result_strict.hit_rate_at_k, 0.0)

    def test_returns_empty_result_when_no_qualifying_users(self):
        """
        Check that an empty EvalResult is returned when no users meet the criteria.
        """
        result = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=5.0, min_liked=10, k=2
        )

        self.assertEqual(result.n_users, 0)
        self.assertEqual(result.hit_rate_at_k, 0.0)
        self.assertEqual(result.users_evaluated, [])

    def test_n_users_matches_users_evaluated_length(self):
        """
        Check that n_users matches the length of users_evaluated list.
        """
        result = evaluate_leave_one_out(
            self.recommender, self.ratings_df, like_threshold=4.0, min_liked=3, k=2
        )

        self.assertEqual(result.n_users, len(result.users_evaluated))

    def test_raises_error_when_recommender_not_fitted(self):
        """
        Check that a RuntimeError is raised when item_matrix is None.
        """
        unfitted = type("MockRecommender", (), {})()
        unfitted.item_matrix = None
        unfitted.movie_index = {}

        with self.assertRaises(RuntimeError):
            evaluate_leave_one_out(unfitted, self.ratings_df)


if __name__ == "__main__":
    unittest.main()