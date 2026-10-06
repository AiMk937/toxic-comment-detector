"""Score comments for toxicity from the command line or from other code.

Usage:
    python -m src.predict "you are a genius" "nobody likes you, idiot"
"""
import argparse

from src.config import LABELS, MODEL_PATH
from src.model import load_model


class ToxicityDetector:
    """Loads the trained model once and scores lists of comments."""

    def __init__(self, model_path=MODEL_PATH, threshold: float = 0.5):
        self.model = load_model(model_path)
        self.threshold = threshold

    def predict(self, comments: list[str]) -> list[dict]:
        # Returns a probability per label plus the labels above the threshold
        comments = [str(c) for c in comments]
        proba = self.model.predict_proba(comments)
        results = []
        for text, row in zip(comments, proba):
            scores = {lbl: round(float(p), 4) for lbl, p in zip(LABELS, row)}
            flagged = [lbl for lbl, p in scores.items() if p >= self.threshold]
            results.append({"comment": text, "scores": scores, "labels": flagged, "is_toxic": bool(flagged)})
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Score comments for toxicity.")
    parser.add_argument("comments", nargs="+", help="One or more comments in quotes")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    for r in ToxicityDetector(threshold=args.threshold).predict(args.comments):
        verdict = ", ".join(r["labels"]) if r["is_toxic"] else "clean"
        print(f"{verdict:30s} | {r['comment']}")


if __name__ == "__main__":
    main()
