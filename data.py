from typing import Tuple

import pandas as pd

from .config import HF_LABELS
from .preprocess import clean_text


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV with `text` and `label` columns."""
    df = pd.read_csv(path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df[["text", "label"]].dropna()


def load_hf() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dair-ai/emotion (train/validation/test) from Hugging Face."""
    from datasets import load_dataset  # lazy import: optional dependency

    ds = load_dataset("dair-ai/emotion")
    out = []
    for split in ("train", "validation", "test"):
        df = ds[split].to_pandas()
        df["label"] = df["label"].map(lambda i: HF_LABELS[i])
        out.append(df[["text", "label"]])
    return tuple(out)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    return df[df["text"].str.len() > 0].reset_index(drop=True)
