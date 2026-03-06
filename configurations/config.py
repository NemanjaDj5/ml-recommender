"""
Configuration file with parameters for ml-recommender algorithm
"""

from pathlib import Path
from typing import Tuple, Optional

# --- Path for project and folders ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "datasets"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

# --- Path for files ---
MOVIES_CSV = RAW_DATA_DIR / "movies.csv"
RATINGS_CSV = RAW_DATA_DIR / "ratings.csv"
TAGS_CSV = RAW_DATA_DIR / "tags.csv"

# --- Path for Models ---
MODEL_PATH = PROJECT_DIR / "modeling" / "models" / "recommender_model.joblib"

# --- TF-IDF params ---
TFIDF_MIN_DF: int = 1
TFIDF_MAX_DF: float = 0.95
TFIDF_NGRAM_RANGE: Tuple[int, int] = (1, 2)
TFIDF_NORM: str = "l2"
TFIDF_USE_IDF: bool = True
TFIDF_SMOOTH_IDF: bool = True
TFIDF_SUBLINEAR_TF: bool = True

# --- SVD/LSA ---
SVD_COMPONENTS: Optional[int] = 100
SVD_RANDOM_STATE: Optional[int] = 42

# --- User profile / Eval ---
RATING_THRESHOLD: float = 4.0
TOPK_REC: int = 10
USER_ID: int = 6          # Active user used for demo inference / visual inspection
COLD_START_USER_ID: int = 611  # User with no rating history, triggers cold-start fallback

# --- Leave-one-out evaluation parameters ---
EVAL_MIN_LIKED: int = 5       # Minimum liked movies a user must have to be included in eval
EVAL_MAX_USERS: int = 300     # Max users sampled per evaluation run (for speed)


# --- Validation parameters ---
MIN_RATINGS_PER_USER: int = 2
MIN_RATINGS_VALUE: float = 0.0
MAX_RATINGS_VALUE: float = 5.0