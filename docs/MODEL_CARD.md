# Model card: `lr_tfidf_position_context`

TF-IDF 1-2 grams + position features + neighbouring-sentence TF-IDF + logistic regression (served model).

## Intended use
Label each sentence of an English biomedical abstract (randomised controlled trials) with its
rhetorical section: BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS. Assistive structuring and search only:
not validated for clinical decision making.

## Training
- Data: PubMed 20k RCT @ 17ed2cb059, train split (180,040 sentences)
- Trained: 2026-10-21T09:44:19+00:00 in 89s (CPU), scikit-learn 1.9.1, Python 3.12.3
- Dev split: accuracy 0.8980, macro-F1 0.8506

## Evaluation (test split: 2,500 abstracts, 30,135 sentences)
- Accuracy **0.8921**, macro-F1 **0.8451**, weighted-F1 0.8909
- Throughput: 4,027 sentences/s (batch, single process)

| label | precision | recall | F1 | support |
|---|---|---|---|---|
| BACKGROUND | 0.775 | 0.774 | 0.775 | 3621 |
| OBJECTIVE | 0.703 | 0.625 | 0.661 | 2333 |
| METHODS | 0.922 | 0.948 | 0.935 | 9897 |
| RESULTS | 0.924 | 0.931 | 0.927 | 9713 |
| CONCLUSIONS | 0.937 | 0.918 | 0.927 | 4571 |

Confusion matrix:

| gold \ predicted | BACKGROUND | OBJECTIVE | METHODS | RESULTS | CONCLUSIONS |
|---|---|---|---|---|---|
| **BACKGROUND** | 2804 | 583 | 162 | 9 | 63 |
| **OBJECTIVE** | 742 | 1457 | 126 | 8 | 0 |
| **METHODS** | 41 | 32 | 9385 | 397 | 42 |
| **RESULTS** | 4 | 1 | 487 | 9042 | 179 |
| **CONCLUSIONS** | 25 | 1 | 16 | 334 | 4195 |

## Limitations
- Trained on RCT abstracts only; other study types (reviews, case reports) are out of domain.
- BACKGROUND vs OBJECTIVE is the main confusion: both open the abstract.
- Served on raw user text, sentence boundaries come from a rule-based splitter, while the
  test set uses gold sentence boundaries.
