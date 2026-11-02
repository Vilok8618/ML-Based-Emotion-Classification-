from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE


def build_pipeline(kind: str = "logreg") -> Pipeline:
    """TF-IDF (word 1-2grams + char 2-5grams) followed by a linear classifier."""
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)),
    ])
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                 random_state=RANDOM_STATE)
    elif kind == "svm":
        clf = LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model kind: {kind!r} (use 'logreg' or 'svm')")
    return Pipeline([("features", features), ("clf", clf)])
