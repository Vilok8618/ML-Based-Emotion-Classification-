from pathlib import Path

import pytest

from emotion_clf.data import load_csv, prepare
from emotion_clf.models import build_pipeline
from emotion_clf.predict import EmotionClassifier
from emotion_clf.preprocess import clean_text
from emotion_clf.train import main as train_main

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_emotions.csv"


def test_clean_text():
    assert clean_text("Check https://x.com @bob  I'm HAPPY!!") == "check i'm happy!!"


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        build_pipeline("nope")


def test_load_csv_requires_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_csv(str(bad))


@pytest.mark.parametrize("kind", ["logreg", "svm"])
def test_train_and_predict(tmp_path, monkeypatch, kind):
    monkeypatch.setattr("emotion_clf.train.REPORT_DIR", tmp_path / "reports")
    out = tmp_path / "m.joblib"
    train_main(["--dataset", "csv", "--path", str(SAMPLE), "--model", kind, "--out", str(out)])
    assert out.exists() and (tmp_path / "reports" / "metrics_test.json").exists()

    clf = EmotionClassifier(out)
    res = clf.predict(["i am so happy and thrilled", "i feel terrified and afraid"])
    assert res[0]["emotion"] == "joy"
    assert res[1]["emotion"] == "fear"
    assert abs(sum(res[0]["scores"].values()) - 1.0) < 1e-6


def test_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from emotion_clf import api

    out = tmp_path / "m.joblib"
    monkeypatch.setattr("emotion_clf.train.REPORT_DIR", tmp_path / "r")
    train_main(["--dataset", "csv", "--path", str(SAMPLE), "--out", str(out)])
    monkeypatch.setenv("EMOTION_MODEL_PATH", str(out))
    monkeypatch.setattr(api, "_clf", None)
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/predict", json={"texts": ["i am so happy and thrilled"]})
    assert r.status_code == 200 and r.json()[0]["emotion"] == "joy"
