# ML Challenge 2026 — Business Entity Resolution Pipeline

Unified, production-grade entity resolution system integrating **Person 1 (Data Engineering & Preprocessing)**, **Person 2 (Candidate Generation & Blocking)**, **Person 3 (Feature Engineering & Matching Model)**, and **Person 4 (Integration, Entity-Level Threshold Optimization, Persistence & Submission Validation)**.

---

## 1. System Architecture

```
dataset/ (train & test TSVs)
   │
   ▼
[Person 1: Preprocessing & Normalization] ──> src/preprocessing.py
   • Unicode NFKC normalization, case folding, and whitespace trimming
   • Ampersand replacement ('&' -> ' and ') & punctuation stripping
   • Postal code extraction (Indian 6-digit PIN & US 5/9-digit ZIP)
   • Input schema validation & source prefix verification (S1-, S2-, S3-)
   │
   ▼
[Person 2: Multi-Strategy Candidate Blocking] ──> src/blocking.py
   • 7 complementary blocking keys (exact, name tokens, addr tokens, postal, street, 3-grams, sorted-neighborhood)
   • Inverted indexing over rare tokens with relative & absolute frequency purging
   • Agreement-ranked candidate capping (heapq.nlargest, max 300 candidates/entity)
   • Zero Cartesian products (O(records x tokens))
   │
   ├──> output/candidate_pairs.tsv (Official candidate deliverable)
   │
   ▼
[Person 3: Feature Engineering & Model Training] ──> src/features.py, src/matching_model.py
   • O(1) batch lookup indexing for fast feature generation
   • 8 pairwise similarity features (token Jaccard, Levenshtein edit ratio, token-sort ratio,
     numeric address overlap, TF-IDF unigram+bigram cosine, country match)
   • Open-set country robustness (unseen test countries like France never filtered)
   • Classifier training & comparison across LogisticRegression, RandomForest, GradientBoosting, HistGradientBoosting
   │
   ▼
[Person 4: Integration, Threshold Tuning & Submission Validation] ──> main.py, src/evaluation.py, src/predict.py
   • Entity-level macro F0.5 optimization on validation ground truth (evaluating full entity sets rather than pairwise F1)
   • Correct competition metric handling: singletons correctly predicted empty score 1.0; false merges penalized with beta=0.5
   • Complete model bundle persistence (model weights + tuned threshold + TF-IDF vectorizer + feature metadata)
   • Unified CLI runner (main.py) with full end-to-end pipeline execution
   • Official submission validation (utils/validate_submission.py) ensuring 100% compliance with all 14 format rules
   │
   └──> output/matching_results.tsv (Final competition submission deliverable)
```

---

## 2. Team Responsibilities & Modules

| Role | Member Responsibilities | Source Files | Tests |
|---|---|---|---|
| **Person 1** | Data Engineering, Text Preprocessing, Postal Extraction, Schema Validation | `src/preprocessing.py`<br>`scripts/audit_data_integrity.py` | `tests/test_preprocessing.py` |
| **Person 2** | Multi-Strategy Candidate Blocking, Inverted Indexing, candidate_pairs.tsv | `src/blocking.py`<br>`src/config.py`<br>`run_blocking.py` | `tests/test_blocking.py` |
| **Person 3** | Feature Engineering, Text Similarities, TF-IDF Model, Classifier Training | `src/features.py`<br>`src/matching_model.py` | `tests/test_features.py`<br>`tests/test_matching_model.py` |
| **Person 4** | Integration, Entity-Level Macro F0.5 Tuning, Model Persistence, main.py, Validation | `main.py`<br>`src/evaluation.py`<br>`src/predict.py`<br>`utils/validate_submission.py` | `tests/test_model_persistence.py`<br>`tests/test_threshold_optimization.py`<br>`tests/test_integration.py` |

---

## 3. Key Methodological Innovations

### 3.1 Candidate Blocking (Person 2)
To avoid the $O(|S_1| \times (|S_2| + |S_3|)) \approx 1.7\text{M} \times 10\text{M} \approx 1.7 \times 10^{13}$ Cartesian product, candidate generation utilizes an inverted-index union over 7 complementary strategies:
1. `exact_name`: Identical normalized names.
2. `name_tokens`: Rare document-frequency tokens from business names.
3. `address_tokens`: Rare tokens from business addresses.
4. `postal_code`: Exact match on extracted PIN or ZIP codes.
5. `street_token`: First two address tokens when postal codes are missing.
6. `name_ngrams`: Rare character 3-grams for typo & transliteration tolerance.
7. `sorted_neighborhood`: Windowed comparisons along alphabetically sorted names.

### 3.2 Feature Engineering (Person 3)
Features operate strictly on normalized strings and precomputed TF-IDF representations:
- **Name Similarities**: Token Jaccard, Levenshtein edit distance ratio, RapidFuzz token sort ratio, and sublinear TF-IDF (unigram + bigram) cosine similarity.
- **Address Similarities**: Token Jaccard, Levenshtein edit ratio, and digit/numeric token set overlap (ensuring house numbers, street numbers, and PIN codes match).
- **Metadata**: Binary country indicator (unseen test countries like France gracefully evaluated without hardcoded filtering).

### 3.3 Official Entity-Level Macro $F_{0.5}$ Optimization (Person 4)
Standard ML classification optimizes pairwise $F_1$ or log-loss, which does not correlate with the competition metric. Our integration implements `optimize_entity_threshold()`:
- **Macro-Averaged Entity Metric**: Computes precision, recall, and $F_{0.5}$ for each Source 1 entity's set of predicted vs ground-truth matches:
  $$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
- **Singleton Rule**: Source 1 entities with no matches in $S_2/S_3$ correctly predicted with an empty set receive an entity score of **1.0**. If incorrectly paired with any candidate (false merge), they score **0.0**.
- **Threshold Search**: Scans the probability space $[0.05, 0.95]$ on a held-out validation entity split to select the decision boundary that strictly maximizes overall entity-level macro $F_{0.5}$.

### 3.4 Model Bundle Persistence (Person 4)
The trained classifier, fitted TF-IDF vectorizer, optimized decision threshold, feature column sequence, and training metadata are packaged together via `save_model_bundle()` into `reports/matching_model.joblib`. This ensures zero-leakage, perfectly reproducible standalone inference in `predict.py`.

---

## 4. Execution & Quick Start

### A. Environment Setup
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### B. Run Full End-to-End Pipeline
```bash
# Complete pipeline: train on 25k entities, tune threshold, block test set, predict, validate
python main.py --mode full
```

### C. Individual Execution Modes
```bash
# 1. Train model, tune threshold on entity macro F0.5, and save model bundle
python main.py --mode train --max-train-entities 25000

# 2. Run inference on test data using saved bundle and generate final submission
python main.py --mode predict

# 3. Evaluate existing matching_results.tsv against ground truth
python main.py --mode eval

# 4. Run official submission validator
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### D. Run Complete Test Suite
```bash
python -m pytest tests -v
```
All **121 tests** pass covering preprocessing, blocking, feature generation, model training, bundle persistence, threshold optimization, and end-to-end integration.

---

## 5. Deliverables & Output Schema

1. **`output/matching_results.tsv`** (Final Submission):
   - Tab-separated UTF-8 file.
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Every Source 1 entity from the test set is present exactly once. Singletons have an empty string. Matches are comma-separated `S2-` and `S3-` IDs.
2. **`output/candidate_pairs.tsv`** (Candidate Deliverable):
   - Tab-separated UTF-8 file.
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Contains candidate IDs evaluated per Source 1 entity.
3. **`reports/matching_model.joblib`** (Model Bundle):
   - Contains `{model, threshold, tfidf_model, feature_cols, metadata}`.
4. **`reports/predictions.csv`** (Pairwise Scoring Telemetry):
   - Detailed pairwise evaluation probabilities and binary decisions.

---

## 6. Directory Structure

```
ROOKIES_ML_challenge/
├── main.py                             # Unified end-to-end pipeline runner
├── requirements.txt                    # Project dependencies
├── README.md                           # Master pipeline documentation
├── Documentation_template.md           # Completed hackathon methodology report
├── dataset/                            # Train and test splits (S1, S2, S3, ground truth)
├── src/
│   ├── preprocessing.py                # Person 1: data cleaning, normalization, schema
│   ├── blocking.py                     # Person 2: multi-strategy inverted-index blocking
│   ├── config.py                       # Person 2: blocking configuration dataclass
│   ├── normalization_fallback.py       # Person 2: standalone normalization fallback
│   ├── features.py                     # Person 3: batch similarity feature extraction
│   ├── matching_model.py               # Person 3 & 4: training, bundle persistence
│   ├── evaluation.py                   # Person 4: entity-level macro F0.5 evaluation & tuning
│   └── predict.py                      # Person 4: model inference & submission generation
├── tests/
│   ├── test_preprocessing.py           # Person 1 unit tests
│   ├── test_blocking.py                # Person 2 unit tests
│   ├── test_features.py                # Person 3 unit tests
│   ├── test_matching_model.py          # Person 3 unit tests
│   ├── test_model_persistence.py       # Person 4 persistence tests
│   ├── test_threshold_optimization.py  # Person 4 threshold optimization tests
│   └── test_integration.py             # Person 4 end-to-end integration tests
├── utils/
│   └── validate_submission.py          # Official 14-rule competition validator
├── output/
│   ├── candidate_pairs.tsv             # Candidate pairs TSV
│   └── matching_results.tsv            # Final submission TSV
└── reports/
    ├── matching_model.joblib           # Serialized model bundle
    └── predictions.csv                 # Detailed pairwise scoring output
```
