import argparse
import json
from typing import Dict, List

import joblib
import numpy as np

from .config import DEFAULT_MODEL_PATH
from .preprocess import clean_text


class EmotionClassifier:
    def __init__(self, path=DEFAULT_MODEL_PATH):
        self.model = joblib.load(path)["model"]
        self.classes: List[str] = list(self.model.classes_)

    def _scores(self, texts: List[str]) -> np.ndarray:
        X = [clean_text(t) for t in texts]
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        d = self.model.decision_function(X)  # e.g. LinearSVC: softmax the margins
        e = np.exp(d - d.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts: List[str]) -> List[Dict]:
        probs = self._scores(texts)
        results = []
        for text, row in zip(texts, probs):
            order = np.argsort(row)[::-1]
            results.append({
                "text": text,
                "emotion": self.classes[order[0]],
                "confidence": float(row[order[0]]),
                "scores": {self.classes[i]: float(row[i]) for i in order},
            })
        return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Predict emotions for text.")
    p.add_argument("texts", nargs="+")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    args = p.parse_args(argv)
    clf = EmotionClassifier(args.model)
    for r in clf.predict(args.texts):
        print(json.dumps({k: r[k] for k in ("text", "emotion", "confidence")}))


if __name__ == "__main__":
    main()
