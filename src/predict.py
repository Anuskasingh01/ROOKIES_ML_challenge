"""
src/predict.py
----------------
Inference pipeline: takes Person 2's candidate_pairs.tsv, scores every
candidate pair with the trained matching model, and outputs:

    1. Per-pair predictions (internal / debugging use):
       source1_entity_id, candidate_entity_id, match_probability, predicted_match

    2. matching_results.tsv (the required submission format):
       one row per Source 1 entity, with all its predicted matches
       comma-joined into a single string (empty string for singletons).
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Generator, Iterable

import joblib
import numpy as np
import pandas as pd

from src.matching_model import load_model_bundle
from src.features import (
    normalize_sources,
    build_lookup_index,
    fit_tfidf,
    explode_candidate_pairs,
    build_pair_features_batch,
)


def iter_candidate_batches(
    candidate_source: pd.DataFrame | str | Path,
    batch_size: int = 50000,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield successive DataFrames of candidate pairs with columns
    ['source1_entity_id', 'candidate_entity_id'] in chunks of at most batch_size.
    Accepts either an in-memory DataFrame or a file path to candidate_pairs.tsv.
    """
    batch: list[tuple[str, str]] = []

    if isinstance(candidate_source, (str, Path)):
        with open(candidate_source, "r", encoding="utf-8") as f:
            header = f.readline()  # skip header
            for line in f:
                s1, sep, rest = line.partition("\t")
                if not sep:
                    continue
                rest = rest.rstrip("\r\n")
                if not rest:
                    continue
                for cid in rest.split(","):
                    cid = cid.strip()
                    if cid:
                        batch.append((s1, cid))
                        if len(batch) >= batch_size:
                            yield pd.DataFrame(batch, columns=["source1_entity_id", "candidate_entity_id"])
                            batch = []
    else:
        df = candidate_source
        cand_col = "candidate_ids" if "candidate_ids" in df.columns else "candidate_entity_ids"
        for s1, cands in zip(df["source1_entity_id"], df[cand_col]):
            if isinstance(cands, str):
                cands = [c.strip() for c in cands.split(",") if c.strip()]
            elif not isinstance(cands, (list, tuple, set)):
                continue
            for cid in cands:
                batch.append((s1, cid))
                if len(batch) >= batch_size:
                    yield pd.DataFrame(batch, columns=["source1_entity_id", "candidate_entity_id"])
                    batch = []

    if batch:
        yield pd.DataFrame(batch, columns=["source1_entity_id", "candidate_entity_id"])


def generate_predictions(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame | str | Path,
    model_path: str = "reports/matching_model.joblib",
    output_path: str = "reports/predictions.csv",
    feature_cols: list[str] | None = None,
    batch_size: int = 50000,
) -> pd.DataFrame:
    """
    Score candidate pairs in memory-safe sequential batches using the
    trained matching model bundle and write predictions to disk.

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Raw (unnormalized) source DataFrames, as loaded from the .tsv files.
    candidate_pairs_df : pd.DataFrame | str | Path
        Person 2's output — either DataFrame with columns [source1_entity_id,
        candidate_entity_ids], or path to candidate_pairs.tsv on disk.
    model_path : str
        Path to the persisted model bundle joblib file.
    output_path : str
        Where to write the per-pair predictions CSV.
    feature_cols : list[str] | None
        Explicit feature column list. If None, loaded from bundle.
    batch_size : int
        Number of candidate pairs to process per streaming batch (default: 50,000).

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id, match_probability, predicted_match
    """
    print("Loading saved model bundle...")
    saved = load_model_bundle(model_path)
    model = saved["model"]
    threshold = float(saved["threshold"])
    tfidf = saved.get("tfidf_model")
    if feature_cols is None and saved.get("feature_cols"):
        feature_cols = saved["feature_cols"]
    print(f"Loaded model={type(model).__name__}, threshold={threshold:.4f}")

    print("Normalizing source data...")
    s1_norm, s2_norm, s3_norm = normalize_sources(s1_df, s2_df, s3_df)

    print("Building lookup indexes...")
    s1_idx = build_lookup_index(s1_norm)
    s23_idx = pd.concat([build_lookup_index(s2_norm), build_lookup_index(s3_norm)])
    del s1_norm, s2_norm, s3_norm
    gc.collect()

    if tfidf is None:
        print("Fitting TF-IDF (fallback)...")
        tfidf = fit_tfidf(s1_df, s2_df, s3_df)
    else:
        print("Using persisted TF-IDF vectorizer from model bundle...")

    out_dir = os.path.dirname(str(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass

    matched_chunks: list[pd.DataFrame] = []
    all_batch_preds: list[pd.DataFrame] = []
    total_pairs = 0
    total_matches = 0
    first_batch = True
    batch_idx = 0
    t_start = time.time()

    print(f"Scoring candidate pairs in streaming batches of {batch_size:,}...")
    for batch_df in iter_candidate_batches(candidate_pairs_df, batch_size=batch_size):
        batch_idx += 1
        n_pairs = len(batch_df)
        total_pairs += n_pairs

        # Compute features for this batch (memory-safe O(batch_size))
        features = build_pair_features_batch(batch_df, s1_idx, s23_idx, tfidf)

        if feature_cols is None:
            feature_cols = [
                c for c in features.columns
                if c not in ("source1_entity_id", "candidate_entity_id", "label")
            ]

        X = features[feature_cols].fillna(0.0)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)

        batch_results = pd.DataFrame({
            "source1_entity_id": batch_df["source1_entity_id"],
            "candidate_entity_id": batch_df["candidate_entity_id"],
            "match_probability": probs,
            "predicted_match": preds,
        })

        # Track positive matches
        matched_in_batch = batch_results[batch_results["predicted_match"] == 1]
        if not matched_in_batch.empty:
            total_matches += len(matched_in_batch)
            matched_chunks.append(
                matched_in_batch[["source1_entity_id", "candidate_entity_id", "match_probability", "predicted_match"]].copy()
            )

        # Incrementally write batch to output_path
        batch_results.to_csv(output_path, mode="a", header=first_batch, index=False)
        first_batch = False

        # In small runs (e.g. unit tests), retain full predictions in memory
        if total_pairs <= 50000:
            all_batch_preds.append(batch_results)

        del features, X, probs, preds, batch_results
        if batch_idx % 200 == 0:
            elapsed = time.time() - t_start
            rate = total_pairs / elapsed if elapsed > 0 else 0
            print(f"  [Batch {batch_idx:,}] Scored {total_pairs:,} pairs | {total_matches:,} matches ({rate:,.0f} pairs/s)")

    if total_pairs == 0:
        print("No candidate pairs to score — writing empty predictions file.")
        empty = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "match_probability", "predicted_match",
        ])
        empty.to_csv(output_path, index=False)
        return empty

    elapsed = time.time() - t_start
    rate = total_pairs / elapsed if elapsed > 0 else 0
    print(f"Scoring completed in {elapsed:.1f}s: {total_pairs:,} total pairs scored ({rate:,.0f} pairs/s), {total_matches:,} matches.")

    if total_pairs <= 50000:
        return pd.concat(all_batch_preds, ignore_index=True)
    elif matched_chunks:
        return pd.concat(matched_chunks, ignore_index=True)
    else:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "match_probability", "predicted_match",
        ])


def generate_matching_results(
    predictions_df: pd.DataFrame,
    all_source1_ids: pd.Series,
    output_path: str = "output/matching_results.tsv",
) -> pd.DataFrame:
    """
    Aggregate per-pair predictions into the final submission format required
    by matching_results.tsv: one row per Source 1 entity, with all its
    predicted matches comma-joined into a single string.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of generate_predictions() — must contain source1_entity_id,
        candidate_entity_id, predicted_match.
    all_source1_ids : pd.Series
        Every entity_id from the Source 1 test file (e.g. test_source1_df["entity_id"]).
        Ensures every S1 entity gets exactly one row, including singletons
        that had no candidates at all (and so never appear in predictions_df).
    output_path : str
        Where to write the tab-separated output file. Must be output/matching_results.tsv
        per the challenge submission spec.

    Returns
    -------
    pd.DataFrame with columns [source1_entity_id, matched_entity_ids].
    """
    matched = predictions_df[predictions_df["predicted_match"] == 1].copy()

    # Defensive de-dup, in case a candidate appears twice for the same S1 entity
    matched = matched.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

    grouped = (
        matched.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(ids)))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )

    # Ensure every S1 test entity appears exactly once, singletons get ""
    all_ids_df = pd.DataFrame({"source1_entity_id": all_source1_ids.unique()})
    result = all_ids_df.merge(grouped, on="source1_entity_id", how="left")
    result["matched_entity_ids"] = result["matched_entity_ids"].fillna("")

    out_dir = os.path.dirname(str(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(output_path, sep="\t", index=False)
    print(f"Saved matching_results.tsv with {len(result)} rows to {output_path}")
    print(f"Entities with >=1 match: {(result['matched_entity_ids'] != '').sum()}")
    print(f"Singleton (no-match) entities: {(result['matched_entity_ids'] == '').sum()}")

    return result


def predict_matches(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame | str | Path,
    model_path: str = "reports/matching_model.joblib",
    predictions_output_path: str = "reports/predictions.csv",
    matching_results_output_path: str = "output/matching_results.tsv",
    batch_size: int = 50000,
) -> pd.DataFrame:
    """
    Shared-interface entry point for Person 4's main.py, matching the
    predict_matches() name specified in the team's interface contract.
    """
    predictions = generate_predictions(
        s1_df, s2_df, s3_df, candidate_pairs_df,
        model_path=model_path,
        output_path=predictions_output_path,
        batch_size=batch_size,
    )
    results = generate_matching_results(
        predictions,
        all_source1_ids=s1_df["entity_id"],
        output_path=matching_results_output_path,
    )
    return results