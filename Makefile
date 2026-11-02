.PHONY: install train-sample train-hf test serve
install:
	pip install -r requirements.txt && pip install -e .
train-sample:
	python -m emotion_clf.train --dataset csv --path data/sample/sample_emotions.csv
train-hf:
	python -m emotion_clf.train --dataset hf
test:
	pytest -q
serve:
	uvicorn emotion_clf.api:app --reload
