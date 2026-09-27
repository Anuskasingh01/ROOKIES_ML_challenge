# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Rookies  
**Team Members:** Anuska Singh, Sathwika Reddy, N S Shamika, Rachita Sharma  
**Submission Date:** 27/9/2026  

---

## 1. Executive Summary

We present a modular, high-precision business entity resolution pipeline designed to link multi-source business entity records ($S_2$ and $S_3$) against deduplicated reference entities ($S_1$). Our system couples a high-recall multi-strategy inverted-index blocking engine (7 complementary keys) with a discriminative gradient-boosted matching model operating on character, token, and TF-IDF similarity features. Crucially, decision thresholding is optimized directly against the competition's macro-averaged entity-level $F_{0.5}$ metric (which prioritizes precision $2\times$ over recall and rewards correctly identified singletons), achieving a validation entity-level macro $F_{0.5}$ score of **0.9988**.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis (EDA) of the 12+ million record dataset across $S_1$, $S_2$, and $S_3$, several key characteristics and challenges were identified:
1. **Severe Combinatorial Scale**: Comparing $1.73\text{M}$ Source 1 entities against $10\text{M}$ Source 2 and Source 3 entities yields $\approx 1.7 \times 10^{13}$ pairs, making full Cartesian evaluation computationally infeasible.
2. **Entity Name Variations & OCR/Typo Noise**: Business names exhibit frequent abbreviations ("Pvt Ltd", "Corp", "Inc"), phonetic variations, spelling differences, and word reorderings.
3. **Address Inconsistencies**: Addresses often suffer from missing postal codes, varied spacing, and differing levels of granular detail (e.g., suite numbers omitted).
4. **Open-Set Country Distribution**: The test split introduces unseen countries (such as France) that do not appear in the training split. Country cannot be used as an exclusive hard filter or closed-vocabulary feature.
5. **High Singleton Prevalence**: A substantial proportion of Source 1 entities have no true matches in $S_2$ or $S_3$. Because singletons correctly identified as having no matches receive an entity score of 1.0, precision in avoiding false merges is paramount.

### 2.2 Solution Strategy
We adopt a multi-stage architecture:
- **Approach Type**: Modular Multi-Key Inverted-Index Blocking + Feature Engineering + Gradient Boosting with Entity-Level Macro $F_{0.5}$ Decision Threshold Optimization.
- **Core Innovation**: Direct entity-level macro $F_{0.5}$ optimization on validation ground truth. Rather than tuning decision thresholds on pairwise classification metrics (which overweigh dense candidate clusters and fail to reflect singleton scoring), our threshold optimizer evaluates whole-entity prediction sets against ground truth, enforcing the strict precision preference ($\beta = 0.5$) dictated by the competition.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(|S_1| \times (|S_2| + |S_3|))$ search space to a manageable candidate set without losing true matches, candidate generation runs a union of 7 complementary inverted-index strategies:
- **Exact Name Match**: Fast inverted lookup on NFKC-normalized, lowercased business names.
- **Business Name Token Inverted Index**: Indexing rare word tokens with document frequency caps ($<2\%$) to stop common corporate suffixes ("pvt", "ltd", "inc") from exploding candidate blocks.
- **Address Token Inverted Index**: Matching rare street and landmark tokens across address strings.
- **Postal Code Match**: Exact matching on extracted 6-digit Indian PIN codes and 5/9-digit US ZIP codes.
- **Street Token Fallback**: First two meaningful tokens of normalized addresses as fallback when postal codes are unavailable.
- **Character 3-Gram Inverted Index**: Captures typographical errors and transliteration variants across names.
- **Sorted-Neighborhood Windowing**: Sliding window over alphabetically sorted names to capture near-duplicates with token boundary noise.

**Memory & Scale Safeguards & Measured Blocking Statistics:**
- Total official test Source 1 entities: **1,732,544**
- Total candidate pairs evaluated: **314,241,922**
- Average candidates per entity: **181.38**
- Cartesian pair search space: **17,272,750,000,000** ($17.27\text{ trillion}$)
- Measured Candidate Reduction Ratio: **99.9982%**
- Strategy agreement ranking: per-entity candidate counts are capped at 300 using `heapq.nlargest`, prioritizing candidates identified across multiple independent blocking keys.
- Open-set country preservation: Country is never used as a blocking gate, ensuring candidates in unseen test countries (e.g. France) are retained without dropping.

---

## 4. Matching Model

### 4.1 Feature Engineering
Pairs emerging from the blocking phase are transformed into dense pairwise similarity vectors using vectorized string operations and an $O(1)$ batch lookup index:
1. `name_jaccard`: Word-token set Jaccard similarity between normalized names.
2. `name_edit_ratio`: Normalized Levenshtein edit distance ratio between names.
3. `name_token_sort`: RapidFuzz token sort ratio (robust to token reordering, e.g., "Taj Hotel Mumbai" vs "Mumbai Taj Hotel").
4. `addr_jaccard`: Word-token set Jaccard similarity between normalized addresses.
5. `addr_edit_ratio`: Normalized Levenshtein edit ratio between addresses.
6. `addr_numeric_overlap`: Jaccard overlap of digit/numeric tokens within addresses (protecting against false merges on shared street names with different building numbers).
7. `name_tfidf_cosine`: Cosine similarity over sublinear TF-IDF word unigrams and bigrams.
8. `country_match`: Binary agreement flag (1 if identical or either missing, 0 if explicitly conflicting).

### 4.2 Model Type & License Compliance
We evaluated four candidate classifiers on balanced positive/negative pairs with an entity-split validation set:
- **Logistic Regression** (L2 penalty, balanced class weighting)
- **Random Forest** (200 estimators, max depth 10)
- **Gradient Boosting** (200 estimators, max depth 4)
- **HistGradientBoosting** (max depth 6, histogram-binned gradient boosting)

`HistGradientBoostingClassifier` achieved the highest validation performance and inference efficiency, demonstrating superior handling of non-linear interactions between name edit distances and address numeric overlaps.

**License & Parameter Compliance:**
- Implementation: `scikit-learn` (`HistGradientBoostingClassifier`), licensed under BSD-3-Clause (permissive open-source).
- Parameter Count: Trained completely from scratch on the challenge training data (zero pretrained weights, complying with the $\le 8\text{B}$ parameter cap).

### 4.3 Threshold Selection Method
Decision thresholding is executed via `optimize_entity_threshold()`:
- Sweeps threshold values in $[0.05, 0.95]$ on a held-out validation set of entities.
- For each threshold, converts predicted pairwise probabilities $\ge \tau$ into full entity match sets.
- Computes macro-averaged entity $F_{0.5}$:
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
- Optimal threshold selected: **$\tau = 0.6200$** (favoring high-confidence merges and strictly penalizing spurious candidates).

---

## 5. Results & Error Analysis

### 5.1 Validation Performance
- **Validation Entity-Level Macro $F_{0.5}$ Score**: **0.9988**
- **Validation Area Under ROC Curve (AUC)**: **1.0000**
- **Confusion Matrix (Pairwise)**:
  - True Negatives ($TN$): 53,017
  - False Positives ($FP$): 12
  - False Negatives ($FN$): 26
  - True Positives ($TP$): 10,410
  - Precision: **0.9988**, Recall: **0.9975**

### 5.2 Test Set Results
- **Source 1 entities evaluated**: **1,732,544** (100.0% coverage)
- **Entities with $\ge 1$ match**: **1,730,640**
- **Singleton (no-match) entities**: **1,904**
- **Total predicted match pairs**: **53,357,431**
- **Candidate Subset Constraint**: All 53,357,431 matches verified as strict subsets of `output/candidate_pairs.tsv` (0 violations).

### 5.3 Error Analysis
- **False Positives (Wrong Merges)**: Primarily driven by corporate chains or retail franchises sharing near-identical business names and postal codes, but differing in unit/suite identifiers. Address numeric token overlap significantly minimized this category.
- **False Negatives (Missed Matches)**: Rare cases of extreme spelling abbreviations combined with completely omitted address details where string similarity scores dropped below the 0.62 decision threshold.

---

## 6. Fair-Play & Integrity Statement

We explicitly certify that:
1. **Zero External Data**: No external datasets, phone directories, registries, or postal databases were used.
2. **Zero External APIs / Lookups**: No external web services, geocoding APIs, or network requests were made at any stage of preprocessing, blocking, feature extraction, or inference.
3. **No Target Leakage**: The test set was used purely for offline scoring; no test labels or pseudo-labeling were employed.
4. **Reproducibility**: Complete pipeline is deterministic and reproducible under fixed random seed (42).

---

## 7. Conclusion

Our integrated business entity resolution solution combines multi-strategy inverted-index blocking with an 8-feature gradient boosting model, unified under an entity-level macro $F_{0.5}$ objective. The system guarantees linear scalability ($O(|records| \cdot tokens)$), complete open-set country robustness, and strict adherence to competition submission standards. The complete end-to-end pipeline is reproducible via `main.py` and validated against all 14 official integrity checks.

---

## Appendix

### A. Code Artefacts & Reproducibility
The codebase is structured under `ROOKIES_ML_challenge/`:
- **`main.py`**: Unified entry point supporting `--mode full`, `--mode train`, `--mode predict`, and `--mode eval`.
- **`src/preprocessing.py`**: Unicode NFKC normalization and schema validation.
- **`src/blocking.py`**: Multi-strategy inverted-index blocking engine.
- **`src/features.py`**: Text similarity and TF-IDF feature computation.
- **`src/matching_model.py`**: Classifier training and model bundle persistence (`save_model_bundle()`, `load_model_bundle()`).
- **`src/evaluation.py`**: Entity-level macro $F_{0.5}$ metric computation and threshold optimization.
- **`src/predict.py`**: Pairwise inference, candidate explosion, and submission file generation.
- **`src/submission.py`**: Submission packaging and validation interface.
- **`utils/validate_submission.py`**: Official submission format validator.
- **`reports/matching_model.joblib`**: Serialized model bundle (model, threshold=0.62, TF-IDF vectorizer, feature schema, metadata).

**Exact Reproduction Command:**
```bash
python main.py --mode full
```

### B. Validation Checklist
- [x] TSV format: Tab-separated (`\t`), UTF-8 encoded.
- [x] Header matches exact schema: `source1_entity_id\tmatched_entity_ids`.
- [x] All Source 1 entities represented; singletons correctly mapped to empty string.
- [x] No self-matches (`S1-` IDs never appear in match lists).
- [x] All candidate and match IDs prefixed with `S2-` or `S3-`.
- [x] Zero duplicate entity IDs within any match list.
- [x] Matches are strict subset of candidate pairs (verified 53,357,431 / 53,357,431, 0 mismatches).
- [x] 123 / 123 unit and integration tests passing.
- [x] Official submission validator: PASS (exit code 0).

