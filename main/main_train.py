"""
This module implements the setup, training and evaluation (through visual inspection of result
tables for the specific user and cold start user) of a Content Based recommender system using
TF-IDF and TruncatedSVD.
"""

import polars as pl

from modeling.recommender import ContentBasedRecommender
from modeling.evaluation import evaluate_leave_one_out

from data.featurization import run_featurization
from data.preparation import run_data_preparation
from data.validation import validate_movies, validate_ratings, validate_tags

from configurations.logging_config import configure_logger
from configurations import config


logger = configure_logger()


def main():
    """
    Orchestrates end-to-end recommender system pipeline.
    """
    try:
        logger.title("\n==== Data Validation ====")
        movies = pl.read_csv(config.MOVIES_CSV)
        ratings = pl.read_csv(config.RATINGS_CSV)
        tags = pl.read_csv(config.TAGS_CSV)

        validate_movies(movies)
        validate_ratings(
            ratings,
            movies,
            config.MIN_RATINGS_PER_USER,
            config.MIN_RATINGS_VALUE,
            config.MAX_RATINGS_VALUE,
        )
        validate_tags(tags, movies)

        logger.title("\n==== Data Preparation & Featurization ====")
        normalized_movies, normalized_tags, final_ratings = run_data_preparation(
            movies, ratings, tags
        )
        final_movies, _ = run_featurization(normalized_movies, normalized_tags)

        logger.title("\n==== Recommender System Training ====")
        recommender = ContentBasedRecommender(
            movies_df=final_movies,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            rating_threshold=config.RATING_THRESHOLD,
            tfidf_min_df=config.TFIDF_MIN_DF,
            tfidf_max_df=config.TFIDF_MAX_DF,
            tfidf_norm=config.TFIDF_NORM,
            tfidf_smooth_idf=config.TFIDF_SMOOTH_IDF,
            tfidf_sublinear_tf=config.TFIDF_SUBLINEAR_TF,
            tfidf_use_idf=config.TFIDF_USE_IDF,
            svd_n_components=config.SVD_COMPONENTS,
            svd_random_state=config.SVD_RANDOM_STATE,
        )

        recommender.fit(final_ratings)

        recommender.save_model(config.MODEL_PATH)

        logger.info("\n==== Recommender System Training Complete ====")

        # --- Loading & Inference ---
        logger.title("\n==== Loading model ====")
        loaded_recommender = ContentBasedRecommender.load_model(
            config.MODEL_PATH, movies_df=final_movies, ratings_df=final_ratings
        )

        logger.title("\n==== Running Evaluation ====")

        eval_result = evaluate_leave_one_out(
            recommender=loaded_recommender,
            ratings_df=final_ratings,
            like_threshold=config.RATING_THRESHOLD,
            min_liked=config.EVAL_MIN_LIKED,
            k=config.TOPK_REC,
            max_users=config.EVAL_MAX_USERS,
        )

        logger.info(
        "HitRate@%s: %.3f (evaluated on %s users)",
        eval_result.k,
        eval_result.hit_rate_at_k,
        eval_result.n_users,
        )

        logger.info(
        "Coverage@%s: %.3f",
        eval_result.k,
        eval_result.coverage_at_k
        )

        logger.title("\n==== Generating Recommendations and Visualizing Results ====")

        logger.info("Generating top 10 recommendations for user %s...", config.USER_ID)

        recommendations = loaded_recommender.recommend(
            user_id=config.USER_ID, rec_n=config.TOPK_REC
        )
        liked_movies = loaded_recommender.get_actual_liked_movies(config.USER_ID)

        logger.info("\n--- User %s - Recommendations ---", config.USER_ID)
        logger.info(recommendations)

        logger.info("\n--- User %s - Movies They Actually Liked ---", config.USER_ID)
        logger.info(liked_movies)

        logger.info(
            "\nGenerating recommendations for new user %s...", config.COLD_START_USER_ID
        )
        cold_start_recs = recommender.recommend(
            config.COLD_START_USER_ID, rec_n=config.TOPK_REC
        )
        logger.info(cold_start_recs)

        logger.info("\n==== Inference Complete ====")
    except FileNotFoundError as e:
        logger.error(
            "An error occurred: %s," "model file not found at %s.", e, config.MODEL_PATH
        )
    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)


if __name__ == "__main__":
    main()