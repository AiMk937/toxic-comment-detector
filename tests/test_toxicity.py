"""Tests for cleaning, training, prediction and the Flask API - no dataset download needed."""
import pandas as pd
import pytest

from src.config import LABELS
from src.model import build_model, save_model
from src.preprocess import clean_text
from src.predict import ToxicityDetector

# Tiny labelled set so the pipeline can be trained in a second
TOXIC = ["you are an idiot", "shut up you stupid idiot", "i will hurt you idiot", "stupid moron go away"]
CLEAN = ["thanks for the edit", "great article, well sourced", "i added a citation", "nice work on this page"]


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    df = pd.DataFrame({"comment_text": (TOXIC + CLEAN) * 5})
    for lbl in LABELS:
        df[lbl] = ([1] * 4 + [0] * 4) * 5
    model = build_model(max_word_features=500, max_char_features=500).fit(df["comment_text"], df[LABELS])
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    save_model(model, path)
    return path


def test_clean_text_removes_urls_and_keeps_pronouns():
    assert clean_text("YOU idiot http://x.com  ") == "you idiot"
    assert clean_text(None) == ""


def test_detector_scores_every_label(model_path):
    result = ToxicityDetector(model_path).predict(["you stupid idiot", "thanks for the citation"])
    assert set(result[0]["scores"]) == set(LABELS)
    assert result[0]["scores"]["toxic"] > result[1]["scores"]["toxic"]


def test_api_rejects_bad_input_and_returns_predictions(model_path, monkeypatch):
    import app.app as web
    monkeypatch.setattr(web, "detector", ToxicityDetector(model_path))
    client = web.app.test_client()
    assert client.post("/predict", json={"comments": "not a list"}).status_code == 400
    res = client.post("/predict", json={"comments": ["you idiot", "<script>alert(1)</script>"]})
    assert res.status_code == 200
    assert len(res.get_json()["predictions"]) == 2
