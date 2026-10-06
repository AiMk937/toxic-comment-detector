# Data

This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset (Wikipedia talk-page comments labelled by human raters).

1. Download it from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) (you need to accept the competition rules).
2. Unzip these three files into this folder:

| File | Rows | Used for |
| --- | --- | --- |
| `train.csv` | 159,571 | Training |
| `test.csv` | 153,164 | Evaluation (comment text) |
| `test_labels.csv` | 153,164 | Evaluation (labels; rows marked -1 were never scored and are skipped) |

The CSVs are not committed because of their size. The dataset is released under CC0, with the comment text under CC-BY-SA 3.0.
