# ML-Based Emotion Classification System

[![CI](https://github.com/<your-username>/emotion-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/emotion-classifier/actions)

Classify short text into emotions (`sadness, joy, love, anger, fear, surprise`) with a fast
classical-ML baseline, an optional fine-tuned DistilBERT, a CLI, and a REST API.

## Features
- **Baseline model:** TF-IDF (word 1–2-grams + char 2–5-grams) → Logistic Regression or Linear SVM
- **Transformer option:** fine-tune `distilbert-base-uncased` on the same data
- **Evaluation:** accuracy, macro-F1, per-class report, normalised confusion matrix
- **Serving:** CLI (`emotion-predict`) and FastAPI (`/predict`) with per-emotion scores
- **Engineering:** tests, GitHub Actions CI, `pyproject.toml`, Makefile

## Project structure
```
src/emotion_clf/   preprocess, data, models, train, evaluate, predict, api, train_transformer
tests/             pytest suite (runs offline on the bundled sample data)
scripts/           synthetic sample-data generator
data/sample/       tiny synthetic CSV for smoke tests (not a benchmark)
.github/workflows/ CI
```

## Quickstart
```bash
git clone https://github.com/<your-username>/emotion-classifier.git
cd emotion-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Smoke test on the bundled sample (offline)
make train-sample

# Real training on dair-ai/emotion (needs `pip install datasets`)
pip install datasets
make train-hf
```
Metrics and the confusion matrix are written to `reports/`; the model to `models/`.

## Predict
```bash
emotion-predict "I can't believe they cancelled it, I'm furious" "I miss her so much"
```

## REST API
```bash
make serve
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"texts": ["I am thrilled about the news!"]}'
```
Response: `[{"text": "...", "emotion": "joy", "confidence": 0.8, "scores": {...}}]`.
Set `EMOTION_MODEL_PATH` to use a model stored elsewhere. Docs at `/docs`.

## Transformer fine-tuning (optional)
```bash
pip install -r requirements-transformer.txt
python -m emotion_clf.train_transformer --epochs 3
```
GPU recommended.

## Results
Run `make train-hf` and paste your numbers here. The sample CSV is synthetic and trivially
separable, so its scores (≈1.0) are meaningless as a benchmark.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + LogReg | _fill in_ | _fill in_ |
| DistilBERT | _fill in_ | _fill in_ |

## Limitations
- Single-label, English-only, short-text focus; sarcasm and mixed emotions are hard.
- Dataset labels come from distant supervision (hashtags) and contain noise.
- Don't use for high-stakes decisions about individuals (hiring, mental-health screening, etc.).

## Ideas for extension
Multi-label emotions (GoEmotions), multilingual models, probability calibration, Dockerfile,
a Streamlit demo, speech or facial-expression modalities.

## License
MIT
