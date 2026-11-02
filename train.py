import argparse

import joblib
from sklearn.model_selection import train_test_split

from . import __version__
from .config import DEFAULT_MODEL_PATH, RANDOM_STATE, REPORT_DIR
from .data import load_csv, load_hf, prepare
from .evaluate import evaluate
from .models import build_pipeline


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the emotion classifier.")
    p.add_argument("--dataset", choices=["hf", "csv"], default="hf")
    p.add_argument("--path", help="CSV path (text,label) when --dataset csv")
    p.add_argument("--model", choices=["logreg", "svm"], default="logreg")
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = p.parse_args(argv)

    if args.dataset == "hf":
        train, val, test = (prepare(d) for d in load_hf())
    else:
        if not args.path:
            p.error("--path is required with --dataset csv")
        df = prepare(load_csv(args.path))
        train, test = train_test_split(df, test_size=0.2, stratify=df["label"],
                                       random_state=RANDOM_STATE)
        val = None

    model = build_pipeline(args.model)
    model.fit(train["text"], train["label"])

    if val is not None:
        m = evaluate(model, val["text"], val["label"], REPORT_DIR, "validation")
        print(f"validation  acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")
    m = evaluate(model, test["text"], test["label"], REPORT_DIR, "test")
    print(f"test        acc={m['accuracy']:.3f}  macro-F1={m['macro_f1']:.3f}")

    joblib.dump({"model": model, "version": __version__}, args.out)
    print(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()
