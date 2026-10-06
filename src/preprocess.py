"""Text cleaning shared by training, prediction and the web app."""
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    # Lowercases, removes URLs/IP addresses and collapses whitespace; keeps words like "you"
    # because stopword removal strips signals such as "you idiot"
    text = "" if text is None else str(text)
    text = URL_RE.sub(" ", text.lower())
    text = IP_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()
