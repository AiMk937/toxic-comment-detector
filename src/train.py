"""Train the toxic comment classifier and evaluate it on the official Jigsaw test set.

Usage:
    python -m src.train
    python -m src.train --quick     # 20% of training data, for a fast check
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from src.config import LABEL_NAMES, LABELS, MODEL_PATH, REPORT_DIR, SEED, TEST_CSV, TEST_LABELS_CSV, TRAIN_CSV
from src.model import build_model, save_model


def load_train(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for download steps.")
    return pd.read_csv(path)


def load_test(text_path: Path, labels_path: Path) -> pd.DataFrame:
    # Joins test comments with their labels and drops rows Kaggle never scored (label -1)
    df = pd.read_csv(text_path).merge(pd.read_csv(labels_path), on="id")
    return df[df["toxic"] != -1].reset_index(drop=True)


def evaluate(model, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    proba = pd.DataFrame(model.predict_proba(df["comment_text"]), columns=LABELS)
    per_label = {}
    for lbl in LABELS:
        y, p = df[lbl], proba[lbl]
        pred = (p >= 0.5).astype(int)
        per_label[lbl] = {
            "roc_auc": roc_auc_score(y, p),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "positives": int(y.sum()),
        }
    any_true = df[LABELS].max(axis=1)
    any_pred = (proba.max(axis=1) >= 0.5).astype(int)
    summary = {
        "mean_roc_auc": sum(v["roc_auc"] for v in per_label.values()) / len(LABELS),
        "any_toxic_f1": f1_score(any_true, any_pred),
        "any_toxic_precision": precision_score(any_true, any_pred),
        "any_toxic_recall": recall_score(any_true, any_pred),
        "per_label": per_label,
    }
    return summary, proba


def save_figures(df: pd.DataFrame, proba: pd.DataFrame, summary: dict, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve for every label on one chart
    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl in LABELS:
        fpr, tpr, _ = roc_curve(df[lbl], proba[lbl])
        ax.plot(fpr, tpr, label=f"{LABEL_NAMES[lbl]} (AUC {summary['per_label'][lbl]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves - official test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Precision / recall / F1 per label at the 0.5 threshold
    rows = pd.DataFrame({LABEL_NAMES[l]: {k: summary["per_label"][l][k] for k in ("precision", "recall", "f1")}
                         for l in LABELS}).T
    fig, ax = plt.subplots(figsize=(7, 4))
    rows.plot.bar(ax=ax, color=["#93c5fd", "#2563eb", "#1e3a8a"], width=0.8)
    ax.set(ylim=(0, 1), ylabel="Score", title="Per-label scores at 0.5 threshold")
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_label_scores.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the toxic comment classifier.")
    parser.add_argument("--train", type=Path, default=TRAIN_CSV)
    parser.add_argument("--test", type=Path, default=TEST_CSV)
    parser.add_argument("--test-labels", type=Path, default=TEST_LABELS_CSV)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    parser.add_argument("--quick", action="store_true", help="Train on a 20%% sample for a fast run")
    args = parser.parse_args()

    start = time.time()
    train = load_train(args.train)
    if args.quick:
        train = train.sample(frac=0.2, random_state=SEED)
    print(f"Training on {len(train):,} comments...")
    model = build_model().fit(train["comment_text"], train[LABELS])
    save_model(model, args.model)
    print(f"Model saved to {args.model}")

    if args.test.exists() and args.test_labels.exists():
        test = load_test(args.test, args.test_labels)
        print(f"Evaluating on {len(test):,} labelled test comments...")
        summary, proba = evaluate(model, test)
        summary.update({"train_rows": len(train), "test_rows": len(test)})
        args.reports.mkdir(parents=True, exist_ok=True)
        (args.reports / "metrics.json").write_text(json.dumps(summary, indent=2))
        save_figures(test, proba, summary, args.reports / "figures")
        print(f"\nMean ROC-AUC {summary['mean_roc_auc']:.4f} | any-toxic F1 {summary['any_toxic_f1']:.4f}")
        for lbl, m in summary["per_label"].items():
            print(f"  {lbl:14s} AUC {m['roc_auc']:.4f}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")
    else:
        print("Test files not found - skipped evaluation (see data/README.md).")
    print(f"Done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
