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

import os
import joblib
import pandas as pd

from src.matching_model import load_model_bundle
from src.features import (
    normalize_sources,
    build_lookup_index,
    fit_tfidf,
    explode_candidate_pairs,
    build_pair_features_batch,
)


def generate_predictions(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    model_path: str = "reports/matching_model.joblib",
    output_path: str = "reports/predictions.csv",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Score every candidate pair from Person 2's blocking step with the
    trained matching model and write per-pair predictions to disk.

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Raw (unnormalized) source DataFrames, as loaded from the .tsv files.
    candidate_pairs_df : pd.DataFrame
        Person 2's output — columns [source1_entity_id, candidate_entity_ids],
        where candidate_entity_ids is a comma-separated string (possibly "").
    model_path : str
        Path to the joblib file saved by train_model(), expected to contain
        a dict {"model": fitted_classifier, "threshold": float}.
    output_path : str
        Where to write the per-pair predictions CSV (not the final submission
        file — see generate_matching_results() for that).
    feature_cols : list[str] | None
        Explicit feature column list. If None, inferred the same way
        train_model() infers it (all columns except ID/label columns).

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id, candidate_entity_id, match_probability, predicted_match

    Notes
    -----
    Source1 entities with no candidates at all (singletons from Person 2's
    blocking step) produce no rows here — there is nothing to compare them
    against. generate_matching_results() re-adds them as empty-match rows
    when building the final submission file.
    """
    print("Loading saved model bundle...")
    saved = load_model_bundle(model_path)
    model = saved["model"]
    threshold = saved["threshold"]
    tfidf = saved.get("tfidf_model")
    if feature_cols is None and saved.get("feature_cols"):
        feature_cols = saved["feature_cols"]
    print(f"Loaded model={type(model).__name__}, threshold={threshold:.4f}")

    print("Normalizing source data...")
    s1_df, s2_df, s3_df = normalize_sources(s1_df, s2_df, s3_df)

    print("Building lookup indexes...")
    s1_idx = build_lookup_index(s1_df)
    s23_idx = pd.concat([build_lookup_index(s2_df), build_lookup_index(s3_df)])

    if tfidf is None:
        print("Fitting TF-IDF (fallback)...")
        tfidf = fit_tfidf(s1_df, s2_df, s3_df)
    else:
        print("Using persisted TF-IDF vectorizer from model bundle...")

    print("Exploding candidate pairs...")
    pairs = explode_candidate_pairs(candidate_pairs_df)
    print(f"Total pairs to score: {len(pairs)}")

    out_dir = os.path.dirname(str(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if len(pairs) == 0:
        print("No candidate pairs to score — writing empty predictions file.")
        empty = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "match_probability", "predicted_match",
        ])
        empty.to_csv(output_path, index=False)
        return empty

    print("Building features for all pairs (this may take a while)...")
    features = build_pair_features_batch(pairs, s1_idx, s23_idx, tfidf)

    if feature_cols is None:
        feature_cols = [
            c for c in features.columns
            if c not in ("source1_entity_id", "candidate_entity_id", "label")
        ]

    X = features[feature_cols].fillna(0.0)

    print("Scoring pairs...")
    probs = model.predict_proba(X)[:, 1]

    predictions = features[["source1_entity_id", "candidate_entity_id"]].copy()
    predictions["match_probability"] = probs
    predictions["predicted_match"] = (probs >= threshold).astype(int)

    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions)} predictions to {output_path}")
    print(f"Predicted matches: {predictions['predicted_match'].sum()} / {len(predictions)}")

    return predictions


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

    Notes
    -----
    Follows every rule in the challenge README:
      - Exactly one row per Source 1 entity (including singletons, via the
        left-merge against all_source1_ids).
      - matched_entity_ids is "" (not NaN) for entities with no matches.
      - No duplicate entity IDs within a single ID list (defensive drop_duplicates).
      - Written tab-separated, matching the required output format.
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
    candidate_pairs_df: pd.DataFrame,
    model_path: str = "reports/matching_model.joblib",
    predictions_output_path: str = "reports/predictions.csv",
    matching_results_output_path: str = "output/matching_results.tsv",
) -> pd.DataFrame:
    """
    Shared-interface entry point for Person 4's main.py, matching the
    predict_matches() name specified in the team's interface contract.

    Thin wrapper around generate_predictions() + generate_matching_results():
    scores every candidate pair, then aggregates into the final
    matching_results.tsv submission format in one call.

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Raw (unnormalized) source DataFrames.
    candidate_pairs_df : pd.DataFrame
        Person 2's candidate_pairs.tsv, loaded as a DataFrame — columns
        [source1_entity_id, candidate_entity_ids].
    model_path : str
        Path to the trained model saved by train_model().
    predictions_output_path : str
        Where to write the intermediate per-pair predictions (debugging/audit trail).
    matching_results_output_path : str
        Where to write the final output/matching_results.tsv submission file.

    Returns
    -------
    pd.DataFrame with columns [source1_entity_id, matched_entity_ids] —
    the final matching_results.tsv content, one row per Source 1 entity.
    """
    predictions = generate_predictions(
        s1_df, s2_df, s3_df, candidate_pairs_df,
        model_path=model_path,
        output_path=predictions_output_path,
    )
    results = generate_matching_results(
        predictions,
        all_source1_ids=s1_df["entity_id"],
        output_path=matching_results_output_path,
    )
    return results