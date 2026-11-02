import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions, collapse whitespace.

    Punctuation and negations are kept on purpose: they carry emotional signal.
    """
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
