"""Generate a small synthetic CSV so the pipeline and tests run offline.

This is NOT a real benchmark -- use the Hugging Face dataset for real results.
"""
import random
import pandas as pd

random.seed(42)
STARTS = ["i feel", "i am", "today i feel", "honestly i am", "lately i have been feeling", "i just felt"]
WORDS = {
    "joy": ["so happy", "delighted", "thrilled", "cheerful", "overjoyed", "excited and glad", "full of joy"],
    "sadness": ["so sad", "heartbroken", "miserable", "down and lonely", "gloomy", "devastated", "tearful"],
    "anger": ["furious", "really angry", "irritated", "outraged", "annoyed and bitter", "enraged", "fed up"],
    "fear": ["terrified", "very scared", "anxious", "afraid", "nervous and uneasy", "panicked", "worried sick"],
    "love": ["deeply in love", "so fond of you", "affectionate", "caring and tender", "adoring", "devoted", "warm toward her"],
    "surprise": ["shocked", "astonished", "stunned", "amazed", "taken aback", "surprised", "speechless"],
}
TAILS = ["", " about everything", " right now", " after what happened", " this morning", " because of the news"]
rows = {(f"{s} {w}{t}", lab) for lab, ws in WORDS.items()
        for s in STARTS for w in ws for t in TAILS}
rows = sorted(rows)
random.shuffle(rows)
rows = rows[:900]
pd.DataFrame(rows, columns=["text", "label"]).to_csv("data/sample/sample_emotions.csv", index=False)
print(len(rows), "rows written")
