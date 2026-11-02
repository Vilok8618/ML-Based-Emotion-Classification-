"""Optional: fine-tune DistilBERT on dair-ai/emotion.

    pip install -r requirements-transformer.txt
    python -m emotion_clf.train_transformer --epochs 3
"""
import argparse

import numpy as np

from .config import HF_LABELS, MODEL_DIR


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default=str(MODEL_DIR / "distilbert-emotion"))
    args = p.parse_args(argv)

    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    ds = load_dataset("dair-ai/emotion")
    tok = AutoTokenizer.from_pretrained(args.base)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(HF_LABELS),
        id2label=dict(enumerate(HF_LABELS)),
        label2id={l: i for i, l in enumerate(HF_LABELS)})

    def metrics(eval_pred):
        logits, y = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(y, pred),
                "macro_f1": f1_score(y, pred, average="macro")}

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, learning_rate=args.lr,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        report_to="none")
    trainer = Trainer(model=model, args=targs, train_dataset=ds["train"],
                      eval_dataset=ds["validation"], tokenizer=tok,
                      compute_metrics=metrics)
    trainer.train()
    print("test:", trainer.evaluate(ds["test"]))
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
