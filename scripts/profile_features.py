import sys
import os
import time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    normalize_sources,
    build_lookup_index,
    fit_tfidf,
    _token_jaccard,
    _edit_ratio,
    _token_sort_ratio,
    _numeric_overlap,
)

def profile():
    # 1. Load small subset to get realistic data for profiling
    print("Loading 100k sample from sources for profiling...")
    s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=100000, dtype=str, keep_default_na=False)
    s2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=200000, dtype=str, keep_default_na=False)
    s3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=200000, dtype=str, keep_default_na=False)

    s1, s2, s3 = normalize_sources(s1, s2, s3)
    s1_idx = build_lookup_index(s1)
    s2_idx = build_lookup_index(s2)
    s3_idx = build_lookup_index(s3)
    s23_idx = pd.concat([s2_idx, s3_idx])

    print(f"s1_idx size: {len(s1_idx):,}, s23_idx size: {len(s23_idx):,}")

    # Generate test pairs of various sizes: 20 pairs, 1,000 pairs, 10,000 pairs
    for N in [20, 1000, 10000]:
        print(f"\n=================== PROFILING N = {N:,} PAIRS ===================")
        s1_ids = np.random.choice(s1_idx.index, size=N, replace=True)
        cand_ids = np.random.choice(s23_idx.index, size=N, replace=True)
        pairs_df = pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_id": cand_ids})

        # --- Sub-test A: Merge / Lookup strategies ---
        # 1. Current approach in features.py: s.rename() + pd.merge()
        t0 = time.time()
        s1_renamed = s1_idx.rename(columns={"name_norm": "name_norm_s1", "address_norm": "address_norm_s1", "country_norm": "country_norm_s1"})
        t_rename1 = time.time() - t0
        t0 = time.time()
        s23_renamed = s23_idx.rename(columns={"name_norm": "name_norm_s2", "address_norm": "address_norm_s2", "country_norm": "country_norm_s2"})
        t_rename2 = time.time() - t0

        t0 = time.time()
        m1 = pairs_df.merge(s1_renamed, left_on="source1_entity_id", right_index=True, how="left")
        m2 = m1.merge(s23_renamed, left_on="candidate_entity_id", right_index=True, how="left")
        t_merge = time.time() - t0

        print(f"[Merge Strategy: Current s.rename + pd.merge]")
        print(f"  s1_idx.rename: {t_rename1*1000:.2f} ms")
        print(f"  s23_idx.rename: {t_rename2*1000:.2f} ms  <-- Notice this on 10.3M rows!")
        print(f"  pd.merge total: {t_merge*1000:.2f} ms")

        # 2. Optimized lookup: direct reindexing / loc on pairs only
        t0 = time.time()
        # Look up only the needed rows directly from index
        s1_lookup = s1_idx.reindex(pairs_df["source1_entity_id"]).reset_index(drop=True)
        s1_lookup.columns = ["name_norm_s1", "address_norm_s1", "country_norm_s1"]
        s23_lookup = s23_idx.reindex(pairs_df["candidate_entity_id"]).reset_index(drop=True)
        s23_lookup.columns = ["name_norm_s2", "address_norm_s2", "country_norm_s2"]
        merged_fast = pd.concat([pairs_df.reset_index(drop=True), s1_lookup, s23_lookup], axis=1)
        t_reindex = time.time() - t0
        print(f"[Lookup Strategy: reindex / slice by pairs]: {t_reindex*1000:.2f} ms (Speedup: {t_merge/max(t_reindex,1e-6):.1f}x)")

        # --- Sub-test B: String similarity helpers (.apply vs list comprehension / zip) ---
        df_test = merged_fast.fillna("")
        
        # 1. Current row-wise .apply(lambda r: ..., axis=1)
        t0 = time.time()
        res_apply = df_test.apply(lambda r: _edit_ratio(r["name_norm_s1"], r["name_norm_s2"]), axis=1)
        t_apply = time.time() - t0

        # 2. List comprehension with zip
        t0 = time.time()
        res_zip = [_edit_ratio(a, b) for a, b in zip(df_test["name_norm_s1"], df_test["name_norm_s2"])]
        t_zip = time.time() - t0
        print(f"[String Sim: _edit_ratio on {N} rows]")
        print(f"  df.apply(axis=1): {t_apply*1000:.2f} ms")
        print(f"  list comp with zip: {t_zip*1000:.2f} ms (Speedup: {t_apply/max(t_zip,1e-6):.1f}x)")

        # 3. All 6 string features via zip
        t0 = time.time()
        n1 = df_test["name_norm_s1"].tolist()
        n2 = df_test["name_norm_s2"].tolist()
        a1 = df_test["address_norm_s1"].tolist()
        a2 = df_test["address_norm_s2"].tolist()
        j_n = [_token_jaccard(x, y) for x, y in zip(n1, n2)]
        e_n = [_edit_ratio(x, y) for x, y in zip(n1, n2)]
        s_n = [_token_sort_ratio(x, y) for x, y in zip(n1, n2)]
        j_a = [_token_jaccard(x, y) for x, y in zip(a1, a2)]
        e_a = [_edit_ratio(x, y) for x, y in zip(a1, a2)]
        num_a = [_numeric_overlap(x, y) for x, y in zip(a1, a2)]
        t_all_str = time.time() - t0
        print(f"  All 6 string features via list comp: {t_all_str*1000:.2f} ms")

    # --- Sub-test C: TF-IDF Vocabulary & Transform Profiling ---
    print("\n=================== PROFILING TF-IDF (VOCAB SIZE & TRANSFORM) ===================")
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_names = pd.concat([s1["name_norm"], s2["name_norm"], s3["name_norm"]])
    print(f"Profiling on {len(all_names):,} names...")

    configs = [
        ("Current (min_df=1, no limit)", dict(analyzer="word", ngram_range=(1,2), min_df=1, sublinear_tf=True)),
        ("min_df=3 (filters singleton noise)", dict(analyzer="word", ngram_range=(1,2), min_df=3, sublinear_tf=True)),
        ("min_df=5 (filters rare typos)", dict(analyzer="word", ngram_range=(1,2), min_df=5, sublinear_tf=True)),
        ("max_features=100k, min_df=3", dict(analyzer="word", ngram_range=(1,2), min_df=3, max_features=100000, sublinear_tf=True)),
        ("max_features=50k, min_df=2", dict(analyzer="word", ngram_range=(1,2), min_df=2, max_features=50000, sublinear_tf=True)),
    ]

    for label, kwargs in configs:
        t0 = time.time()
        vec = TfidfVectorizer(**kwargs)
        vec.fit(all_names)
        t_fit = time.time() - t0
        vocab_size = len(vec.vocabulary_)

        # Benchmark transform on 10,000 names
        sample_names = np.random.choice(all_names, size=10000)
        t0 = time.time()
        mat = vec.transform(sample_names)
        t_trans = time.time() - t0

        print(f"{label:35s} | Vocab: {vocab_size:8,d} | Fit: {t_fit:5.2f}s | Transform 10k: {t_trans*1000:6.1f} ms")

if __name__ == "__main__":
    profile()
