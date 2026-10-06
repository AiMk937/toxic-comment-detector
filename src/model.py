"""Model definition plus save/load helpers."""
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.preprocess import clean_text


def build_model(max_word_features: int = 50_000, max_char_features: int = 50_000) -> Pipeline:
    # Word 1-2 grams catch phrases; character 2-5 grams catch misspellings like "id10t" or "f*ck"
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=clean_text, analyzer="word", ngram_range=(1, 2),
                                 max_features=max_word_features, sublinear_tf=True, min_df=2)),
        ("char", TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=(2, 5),
                                 max_features=max_char_features, sublinear_tf=True, min_df=2)),
    ])
    # One logistic regression per label, so a comment can be e.g. both obscene and an insult
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced"))
    return Pipeline([("tfidf", features), ("clf", clf)])


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: Path) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(f"No model at {path}. Train one first with: python -m src.train")
    return joblib.load(path)
