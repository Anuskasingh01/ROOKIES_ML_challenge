import sys
import os
import time
import random
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    normalize_sources,
    build_lookup_index,
    fit_tfidf,
    build_pair_features_batch,
)

def main():
    print("=== Step 1: Loading TSV files ===")
    t0 = time.time()
    
    # We can load the required columns to save significant memory and speed up IO
    cols = ["entity_id", "business_name", "business_address", "country"]
    
    print("Loading train_source1.tsv...")
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", usecols=cols, dtype=str, keep_default_na=False)
    print(f"  Source 1 loaded: {len(s1_df):,} rows ({time.time() - t0:.2f}s)")
    
    t_step = time.time()
    print("Loading train_source2.tsv...")
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", usecols=cols, dtype=str, keep_default_na=False)
    print(f"  Source 2 loaded: {len(s2_df):,} rows ({time.time() - t_step:.2f}s)")
    
    t_step = time.time()
    print("Loading train_source3.tsv...")
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", usecols=cols, dtype=str, keep_default_na=False)
    print(f"  Source 3 loaded: {len(s3_df):,} rows ({time.time() - t_step:.2f}s)")
    print(f"All sources loaded in {time.time() - t0:.2f}s\n")

    print("=== Step 2: Running normalize_sources() ===")
    t_norm = time.time()
    s1_df, s2_df, s3_df = normalize_sources(s1_df, s2_df, s3_df)
    print(f"normalize_sources() completed in {time.time() - t_norm:.2f}s\n")

    print("=== Step 3: Building lookup index for each source ===")
    t_idx = time.time()
    s1_idx = build_lookup_index(s1_df)
    s2_idx = build_lookup_index(s2_df)
    s3_idx = build_lookup_index(s3_df)
    s23_idx = pd.concat([s2_idx, s3_idx])
    print(f"Lookup indexes built in {time.time() - t_idx:.2f}s")
    print(f"  s1_idx: {len(s1_idx):,} records")
    print(f"  s23_idx: {len(s23_idx):,} records\n")

    print("=== Step 4: Fitting TF-IDF on combined names ===")
    t_tfidf = time.time()
    tfidf_model = fit_tfidf(s1_df, s2_df, s3_df)
    print(f"fit_tfidf() completed in {time.time() - t_tfidf:.2f}s")
    print(f"  Vocabulary size: {len(tfidf_model.vocabulary_):,}\n")

    print("=== Step 5: Selecting 10 true matches and 10 non-matches ===")
    # Load ground truth
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000, keep_default_na=False)
    gt_valid = gt_df[gt_df["matched_entity_ids"] != ""].copy()
    
    # Pick 10 diverse true matches
    true_pairs = []
    all_matched_cands = set()
    for _, row in gt_valid.iterrows():
        s1_id = row["source1_entity_id"]
        cands = [c.strip() for c in row["matched_entity_ids"].split(",") if c.strip()]
        for c in cands:
            all_matched_cands.add((s1_id, c))
            if len(true_pairs) < 10 and (s1_id, c) not in true_pairs:
                # verify both exist in our indexes
                if s1_id in s1_idx.index and c in s23_idx.index:
                    true_pairs.append({
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": c,
                        "pair_type": "MATCH (Ground Truth)",
                    })
        if len(true_pairs) >= 10:
            break

    # Pick 10 random non-matching pairs
    random.seed(42)
    s1_sample = list(s1_idx.index[:5000])
    s23_sample = list(s23_idx.index[:10000])
    
    non_matching_pairs = []
    while len(non_matching_pairs) < 10:
        s1_cand = random.choice(s1_sample)
        other_cand = random.choice(s23_sample)
        if (s1_cand, other_cand) not in all_matched_cands:
            non_matching_pairs.append({
                "source1_entity_id": s1_cand,
                "candidate_entity_id": other_cand,
                "pair_type": "NON-MATCH (Random)",
            })

    test_pairs_df = pd.DataFrame(true_pairs + non_matching_pairs)
    print(f"Selected {len(true_pairs)} true matches and {len(non_matching_pairs)} non-matches.\n")

    print("=== Step 6: Running build_pair_features_batch() on all 20 pairs ===")
    t_feat = time.time()
    # Pass pairs without extra columns
    feature_input = test_pairs_df[["source1_entity_id", "candidate_entity_id"]]
    features_df = build_pair_features_batch(feature_input, s1_idx, s23_idx, tfidf_model)
    features_df["pair_type"] = test_pairs_df["pair_type"]
    print(f"build_pair_features_batch() completed in {time.time() - t_feat:.2f}s\n")

    # Add entity names & addresses for easy visual inspection
    s1_info = s1_idx.loc[features_df["source1_entity_id"]].reset_index()
    s23_info = s23_idx.loc[features_df["candidate_entity_id"]].reset_index()

    features_df["s1_name"] = s1_info["name_norm"].values
    features_df["cand_name"] = s23_info["name_norm"].values
    features_df["s1_addr"] = s1_info["address_norm"].values
    features_df["cand_addr"] = s23_info["address_norm"].values

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", 40)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("=" * 100)
    print("DETAILED FEATURE COMPARISON TABLE (20 PAIRS)")
    print("=" * 100)
    
    summary_cols = [
        "pair_type",
        "source1_entity_id",
        "candidate_entity_id",
        "name_jaccard",
        "name_edit_ratio",
        "name_token_sort",
        "name_tfidf_cosine",
        "addr_jaccard",
        "addr_edit_ratio",
        "addr_numeric_overlap",
        "country_match",
    ]
    print(features_df[summary_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("RECORD DETAILS (NAMES & ADDRESSES)")
    print("=" * 100)
    detail_cols = [
        "pair_type",
        "source1_entity_id",
        "s1_name",
        "candidate_entity_id",
        "cand_name",
        "s1_addr",
        "cand_addr",
    ]
    print(features_df[detail_cols].to_string(index=False))

if __name__ == "__main__":
    main()
