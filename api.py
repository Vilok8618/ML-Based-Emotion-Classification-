"""FastAPI service. Run: uvicorn emotion_clf.api:app --reload"""
import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_MODEL_PATH
from .predict import EmotionClassifier

app = FastAPI(title="Emotion Classification API", version="0.1.0")
_clf = None


def get_classifier() -> EmotionClassifier:
    global _clf
    if _clf is None:
        path = os.environ.get("EMOTION_MODEL_PATH", DEFAULT_MODEL_PATH)
        try:
            _clf = EmotionClassifier(path)
        except FileNotFoundError:
            raise HTTPException(503, "Model not found. Train one first (make train-hf).")
    return _clf


class PredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    text: str
    emotion: str
    confidence: float
    scores: Dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=List[Prediction])
def predict(req: PredictRequest):
    return get_classifier().predict(req.texts)
