# ML Challenge 2026 — Business Entity Resolution Pipeline

Integrated pipeline combining **Person 1 (Data Engineering & Preprocessing)** and **Person 2 (Candidate Generation & Blocking)**.

---

## 1. System Architecture

```
dataset/ (train & test TSVs)
   │
   ▼
[Person 1: Preprocessing & Normalization] ──> src/preprocessing.py
   • Unicode NFKC normalization
   • Case folding & whitespace trimming
   • Ampersand replacement ('&' -> ' and ')
   • Punctuation removal & postal code extraction
   • Input schema & source prefix validation
   │
   ▼
[Person 2: Multi-Strategy Candidate Blocking] ──> src/blocking.py
   • Inverted indices over rare tokens & character 3-grams
   • Multi-strategy union (7 complementary keys)
   • Block purging: relative (max_df) & absolute caps
   • Agreement-ranked candidate capping (heapq.nlargest)
   │
   ├──> output/candidate_pairs.tsv (Official submission format)
   └──> output/candidate_pairs_scored.tsv (Sidecar agreement scores for Person 3)
```

---

## 2. Person 1 — Data Engineering & Preprocessing

- **`src/preprocessing.py`**:
  - `load_tsv(path, usecols)`: Safely loads TSV with strict string types and no NaN coercion.
  - `normalize_name(series)`: Vectorized normalization for business names.
  - `normalize_address(series)`: Vectorized address normalization.
  - `extract_postal_code(series)`: Extracts Indian 6-digit PIN and US 5(+4)-digit ZIP codes.
  - `normalize_dataframe(df)`: Appends normalized name, address, country, and postal code columns.
  - `validate_schema(df, expected_source)`: Asserts required columns and source ID prefixes (`S1-`, `S2-`, `S3-`).
- **`scripts/audit_data_integrity.py`**:
  - Scans all train and test TSVs to verify schema validity, row counts, duplicate entity IDs, and country distributions.
- **`utils/validate_submission.py`**:
  - Automated validator verifying compliance with competition rules (UTF-8, tab separation, required S1 coverage, no invalid candidate IDs). Supports `--candidate-only` mode during blocking development.

---

## 3. Person 2 — Candidate Generation / Blocking

Generates high-recall candidate pairs without computing quadratic Cartesian products (`O(records x tokens)` rather than `O(S1 x (S2+S3))`).

### Blocking Strategies (Union of 7 Complementary Keys)

| Strategy | Catches | Mechanism |
|---|---|---|
| `exact_name` | Clean duplicates | Exact match on normalized name |
| `name_tokens` | Word reorderings, partial name matches | Inverted index, doc-frequency stoplisted |
| `address_tokens` | Address variants | Same, on address tokens |
| `postal_code` | Reliable geo match | Exact match on extracted PIN/ZIP |
| `street_token` | Missing postal code | First 2 address tokens as fallback key |
| `name_ngrams` | Typos, transliteration | Character 3-grams, min-shared-count threshold |
| `sorted_neighborhood` | Word-boundary typos & heavy variants | Sort by name, slide window, pair within window |

*Country is never used to gate candidates (open set, France appears only in test set).*

### Memory & Scale Safeguards
1. **Index-level purge** (`max_postings_per_key` + `max_*_document_frequency`): Tokens/n-grams appearing in >2% of records or >500 postings are purged.
2. **Agreement-ranked per-entity cap** (`max_candidates_per_entity`, default 300): Candidates supported by multiple independent strategies are prioritized using `heapq.nlargest`.

---

## 4. Execution & Quick Start

### A. Run Entire Integrated Pipeline
```bash
python run_all.py
```
This automatically runs:
1. Data integrity & schema audit
2. Full test suite (Person 1 + Person 2)
3. Candidate blocking on test set
4. Candidate output format & ID consistency validation
5. Recall evaluation & strategy ablation report on training split

### B. Individual Steps

```bash
# 1. Run full test suite
python -m pytest tests -v

# 2. Run data integrity audit
python scripts/audit_data_integrity.py

# 3. Generate candidate pairs for test split
python run_blocking.py --split test --out output/candidate_pairs.tsv

# 4. Validate output format against competition rules
python utils/validate_submission.py --candidate output/candidate_pairs.tsv --candidate-only --test-dir dataset/test --check-ids

# 5. Evaluate blocking recall and strategy ablation on training set
python evaluate_blocking.py --ablation
```

---

## 5. Output Format

1. **`output/candidate_pairs.tsv`** (Submission deliverable):
   Tab-separated TSV with columns `source1_entity_id` and `candidate_entity_ids` (comma-separated S2/S3 IDs).
2. **`output/candidate_pairs_scored.tsv`** (Model feature sidecar):
   Contains `source1_entity_id`, `candidate_entity_id`, and `n_strategies_agreeing` — high-signal prior feature for Person 3's matching model.

---

## 6. Directory Structure

```
├── README.md                      # Comprehensive project documentation
├── Documentation_template.md      # Methodology write-up template
├── requirements.txt               # Dependencies (pandas, pytest)
├── run_all.py                     # Master pipeline execution script
├── run_blocking.py                # Root runner CLI for blocking
├── evaluate_blocking.py           # Root runner for recall evaluation
├── dataset/
│   ├── train/                     # train_source1/2/3.tsv, train_ground_truth.tsv
│   └── test/                      # test_source1/2/3.tsv
├── src/
│   ├── preprocessing.py           # Person 1: data engineering & normalization
│   ├── blocking.py                # Person 2: multi-strategy inverted index blocking
│   ├── config.py                  # Blocking configuration & thresholds
│   ├── normalization_fallback.py  # Fallback normalization routines
│   ├── run_blocking.py            # CLI entry point for candidate generation
│   └── evaluate_blocking.py       # Blocking recall & ablation evaluator
├── tests/
│   ├── test_preprocessing.py      # Person 1 test suite (normalization, schema, helpers)
│   └── test_blocking.py           # Person 2 test suite (recovery, singletons, deduplication)
├── scripts/
│   └── audit_data_integrity.py    # Data integrity audit script
├── utils/
│   └── validate_submission.py     # Competition submission format validator
└── output/
    ├── candidate_pairs.tsv        # Generated candidates for test set
    └── candidate_pairs_scored.tsv # Agreement score sidecar
```
