"""
Module for Content-based recommender class.
"""

import os
from typing import Dict, Optional, Tuple, Any
import polars as pl
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


from configurations.logging_config import configure_logger

logger = configure_logger()


class ContentBasedRecommender:
    """
    A content-based movie recommender system using Polars for data manipulation
    and Scikit-learn for vectorization. This version is designed for simple inference
    and visual inspection, without a train/test split.
    """

    def __init__(
        self,
        movies_df: pl.DataFrame,
        ngram_range: Tuple[int, int] = (1, 2),
        rating_threshold: float = 4.0,
        tfidf_min_df: float = 1,
        tfidf_max_df: float = 0.95,
        tfidf_norm: Optional[str] = "l2",
        tfidf_smooth_idf: bool = True,
        tfidf_sublinear_tf: bool = True,
        tfidf_use_idf: bool = True,
        svd_n_components: int = 100,
        svd_random_state: int = 42,
    ):
        """
        Initializes the ContentBasedRecommender with hyperparameters and movie data.

        Args:
            movies_df: Polars dataframe containing movie_text column.
            ngram_range: Tuple of integers indicating the ngram range to use.
            rating_threshold: The minimum rating to consider a movie 'liked'.
            tfidf_min_df: Minimum TF-IDF threshold to use.
            tfidf_max_df: Maximum TF-IDF threshold to use.
            tfidf_norm: Normalization method to use.
            tfidf_smooth_idf: Smoothing method to use.
            tfidf_sublinear_tf: Sublinear TF-IDF method to use.
            tfidf_use_idf: Whether to enable inverse-document-frequency reweighting.
            svd_n_components: Number of SVD components to use.
            svd_random_state: Random state to use.
        """
        self.movies_df = movies_df
        self.rating_threshold = rating_threshold

        self.ngram_range = ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_norm = tfidf_norm
        self.tfidf_smooth_idf = tfidf_smooth_idf
        self.tfidf_sublinear_tf = tfidf_sublinear_tf
        self.tfidf_use_idf = tfidf_use_idf

        self.svd_n_components = svd_n_components
        self.svd_random_state = svd_random_state

        self.full_ratings: Optional[pl.DataFrame] = None
        self.item_matrix: Optional[np.ndarray] = None
        self.movie_index: Dict[int, int] = {}
        self.user_profiles: Dict[int, np.ndarray] = {}
        self._fitted = False

    def _fit_items(self):
        """
        Converts the 'movie_text' feature into a numerical item matrix.

        This method performs two main steps:
        1.  **TF-IDF Vectorization:** Transforms the movie text into a sparse matrix of word importance scores.
        2.  **Dimensionality Reduction:** Applies TruncatedSVD to compress the TF-IDF matrix into a dense,
        lower-dimensional matrix.
        The resulting matrix represents each movie as a vector of latent features.
        """
        if "movie_text" not in self.movies_df.columns or self.movies_df.is_empty():
            raise ValueError("The 'movies_df' must contain a non-empty 'movie_text' column.")


        self.movie_index = {
            movie_id: idx
            for idx, movie_id in enumerate(self.movies_df.get_column("movieId").to_list())}

        texts = self.movies_df.get_column("movie_text").to_list()


        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=self.ngram_range,
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            use_idf=self.tfidf_use_idf,
            smooth_idf=self.tfidf_smooth_idf,
            sublinear_tf=self.tfidf_sublinear_tf,
            norm=self.tfidf_norm,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        logger.info("TF-IDF matrix created with shape: %s", tfidf_matrix.shape)

        svd = TruncatedSVD(
            n_components=self.svd_n_components, random_state=self.svd_random_state
        )

        self.item_matrix = svd.fit_transform(tfidf_matrix)

        logger.info("Item matrix reduced with SVD to shape: %s", self.item_matrix.shape)

    def _train_user_profiles(self):
        """
        Internal method to train the user profile vectors using all data.
        Uses a robust group-by aggregation to get data, then a single
        dictionary comprehension to compute profiles.
        """
        if self.full_ratings is None or self.full_ratings.is_empty():
            raise ValueError(
                "Ratings data must be provided via the fit() method and not be empty."
            )

        item_vectors_df = pl.DataFrame(
            {"movieId": list(self.movie_index.keys()), "vector": list(self.item_matrix)}
        )

        positive_ratings_with_vectors = (
            self.full_ratings.filter(pl.col("rating") >= self.rating_threshold)
            .join(item_vectors_df, on="movieId", how="left")
            .filter(pl.col("vector").is_not_null())
        )

        user_ratings_and_vectors = positive_ratings_with_vectors.group_by("userId").agg(
            pl.col("rating").alias("ratings"), pl.col("vector").alias("vectors")
        )

        user_profiles_dict = {
            row["userId"]: np.average(
                np.stack(row["vectors"]), axis=0, weights=np.array(row["ratings"])
            )
            for row in user_ratings_and_vectors.iter_rows(named=True)
        }

        self.user_profiles = user_profiles_dict

        logger.info("Trained profiles for %s users.", len(self.user_profiles))


    def fit(self, ratings_df: pl.DataFrame):
        """
        Fits the entire recommender system on the full dataset.
        """
        logger.info("Starting recommender system training.")
        self.full_ratings = ratings_df

        self._fit_items()
        self._train_user_profiles()
        self._fitted = True
        logger.info("Recommender system training complete.")

    def recommend(self, user_id: int, rec_n: int = 10) -> pl.DataFrame:
        """
        Generates movie recommendations for a given user ID.
        For cold-start users, returns the most popular movies.
        """
        if not self._fitted:
            raise RuntimeError(
                "Recommender has not been fitted. Please run fit() first."
            )

        if user_id not in self.user_profiles:
            logger.warning(
                "User %s not found in user profiles. Returning popular movies.", user_id
            )
            return self._get_popular_movies(top_n=rec_n)

        user_profile_vector = self.user_profiles[user_id]

        similarities = cosine_similarity(
            user_profile_vector.reshape(1, -1), self.item_matrix
        ).flatten()

        index_to_movie_id = self._get_movie_id_map()
        recommendation_df = pl.DataFrame(
            {
                "movieId": [index_to_movie_id[i] for i in range(len(similarities))],
                "similarity": similarities,
            }
        ).sort(by="similarity", descending=True)

        seen_movies_df = self.full_ratings.filter(pl.col("userId") == user_id).select(
            "movieId"
        )

        unseen_recommendations_df = recommendation_df.filter(
            ~pl.col("movieId").is_in(seen_movies_df.get_column("movieId").to_list())
        )

        top_recommendations = unseen_recommendations_df.head(rec_n)

        final_recommendations = self._top_up_recommendations(
            top_recommendations, rec_n, user_id
        )

        recommendations = final_recommendations.join(
            self.movies_df, on="movieId", how="left"
        ).select(["movieId", "title", "genres"])

        return recommendations

    def _top_up_recommendations(
            self, top_recommendations: pl.DataFrame, top_n: int, user_id: int
    ) -> pl.DataFrame:
        """
        Internal method to top up a list of recommendations with popular movies
        if the list is shorter than n.
        """

        if top_recommendations.height >= top_n:
            return top_recommendations.head(top_n)


        remaining_n = top_n - top_recommendations.height

        seen_movies_ids = (
            self.full_ratings.filter(pl.col("userId") == user_id)
            .get_column("movieId")
            .to_list()
        )
        popular_movies = self._get_popular_movies(remaining_n)
        unseen_popular_movies = (
            popular_movies
            .filter(~pl.col("movieId").is_in(seen_movies_ids))
            .head(remaining_n)
            .with_columns(pl.lit(0.0).alias("similarity"))
            .select(["movieId", "similarity"])
        )

        return pl.concat([top_recommendations, unseen_popular_movies])

    def get_actual_liked_movies(self, user_id: int) -> pl.DataFrame:
        """
        Returns a table of all movies a user has liked in the dataset.
        """
        if not self._fitted:
            raise RuntimeError(
                "Recommender has not been fitted. Please run fit() first."
            )

        liked_movie_ids = (
            self.full_ratings.filter(
                (pl.col("userId") == user_id)
                & (pl.col("rating") >= self.rating_threshold)
            )
            .get_column("movieId")
            .to_list()
        )

        liked_movies = self.movies_df.filter(
            pl.col("movieId").is_in(liked_movie_ids)
        ).select(["movieId", "title", "genres"])
        return liked_movies

    def _get_popular_movies(self, top_n: int) -> pl.DataFrame:
        """
        Helper method to get the top N most 'central' movies based on
        their position in the content space.

        Note: popularity here is a content-space proxy — movies with the
        highest cosine similarity to the centroid of all item vectors.
        This is NOT based on rating counts; it favours movies that are
        most genre-representative across the catalogue.
        """
        if self.item_matrix is None:
            raise RuntimeError("Recommender has not been fitted.")

        centroid_vector = np.mean(self.item_matrix, axis=0)

        similarities = cosine_similarity(
            centroid_vector.reshape(1, -1), self.item_matrix
        ).flatten()

        top_n_indices = np.argsort(-similarities)[:top_n]

        index_to_movie_id = self._get_movie_id_map()
        top_n_movie_ids = [index_to_movie_id.get(idx) for idx in top_n_indices]

        return self.movies_df.filter(pl.col("movieId").is_in(top_n_movie_ids)).select(
            ["movieId", "title", "genres"]
        )

    def _get_movie_id_map(self) -> Dict[int, int]:
        """
        Helper method to create the index-to-movieId mapping.
        """
        if not self.movie_index:
            raise RuntimeError("movie_index must be created by fit() first.")
        return {v: k for k, v in self.movie_index.items()}

    def save_model(self, file_path: str):
        """
        Saves the trained model components to a file.
        """
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        model_components = {
            "item_matrix": self.item_matrix,
            "user_profiles": self.user_profiles,
            "movie_index": self.movie_index,
            "ngram_range": self.ngram_range,
            "rating_threshold": self.rating_threshold,
            "tfidf_min_df": self.tfidf_min_df,
            "tfidf_max_df": self.tfidf_max_df,
            "tfidf_norm": self.tfidf_norm,
            "tfidf_smooth_idf": self.tfidf_smooth_idf,
            "tfidf_sublinear_tf": self.tfidf_sublinear_tf,
            "tfidf_use_idf": self.tfidf_use_idf,
            "svd_n_components": self.svd_n_components,
            "svd_random_state": self.svd_random_state,
        }
        joblib.dump(model_components, file_path)
        logger.info("Model components saved successfully to %s", file_path)

    @classmethod
    def load_model(
        cls,
        file_path: str,
        movies_df: pl.DataFrame,
        ratings_df: Optional[pl.DataFrame] = None,
    ):
        """
        Loads a trained model from a file and returns a new class instance.
        Requires the original movies_df and full_ratings for proper functioning.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at {file_path}")

        model_components = joblib.load(file_path)

        recommender = cls(
            movies_df=movies_df,
            ngram_range=model_components["ngram_range"],
            rating_threshold=model_components["rating_threshold"],
            tfidf_min_df=model_components["tfidf_min_df"],
            tfidf_max_df=model_components["tfidf_max_df"],
            tfidf_norm=model_components["tfidf_norm"],
            tfidf_smooth_idf=model_components["tfidf_smooth_idf"],
            tfidf_sublinear_tf=model_components["tfidf_sublinear_tf"],
            tfidf_use_idf=model_components["tfidf_use_idf"],
            svd_n_components=model_components["svd_n_components"],
            svd_random_state=model_components["svd_random_state"],
        )

        recommender.item_matrix = model_components["item_matrix"]
        recommender.user_profiles = model_components["user_profiles"]
        recommender.movie_index = model_components["movie_index"]
        recommender.full_ratings = ratings_df
        recommender._fitted = True

        logger.info("Model loaded successfully from %s", file_path)
        return recommender