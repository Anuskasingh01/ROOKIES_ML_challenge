"""
src/evaluation.py
------------------
Entity-level macro-averaged F0.5 evaluation — the actual challenge scoring
metric. This is distinct from pairwise classification metrics
(precision/recall/F1 on individual pairs), which train_model() already
reports for model comparison but which do NOT match how the leaderboard
scores submissions.

The challenge computes F0.5 PER Source 1 entity, then averages across all
entities (macro-average), including singletons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _entity_f05(true_ids: set[str], pred_ids: set[str]) -> float:
    """
    Compute F0.5 for a single Source 1 entity given its true and predicted
    match sets.

    Special cases (per challenge README):
    - True singleton (no true matches), correctly predicted empty -> 1.0
    - True singleton, but model predicted >=1 match (false merge) -> 0.0
    - Has true matches, but model predicted none -> 0.0 (recall = 0)
    """
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    return (1.25 * precision * recall) / (0.25 * precision + recall)


def evaluate_predictions(
    matching_results_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict:
    """
    Compute the official challenge metric: macro-averaged F0.5 per Source 1
    entity, exactly as described in the README's Evaluation Criteria section.

    Parameters
    ----------
    matching_results_df : pd.DataFrame
        Your pipeline's output — columns [source1_entity_id, matched_entity_ids],
        where matched_entity_ids is a comma-separated string (or "" for none).
        This is the same format as output/matching_results.tsv.
    ground_truth_df : pd.DataFrame
        Columns [source1_entity_id, matched_entity_ids] from a held-out
        validation split of train_ground_truth.tsv. NEVER pass test data
        here — no test ground truth exists, and this must only be used
        during your own validation, not real submission.

    Returns
    -------
    dict with:
        macro_f05           : float  the official metric — mean F0.5 across all entities
        num_entities         : int
        num_singletons_true  : int   entities with no true matches
        num_singletons_pred  : int   entities predicted as no-match
        singleton_accuracy   : float fraction of true singletons correctly predicted empty
        per_entity_f05       : pd.DataFrame  entity-level breakdown for error analysis
    """
    def to_id_set(s: str) -> set[str]:
        if pd.isna(s) or s == "":
            return set()
        return set(s.split(","))

    gt = ground_truth_df.set_index("source1_entity_id")["matched_entity_ids"]
    pred = matching_results_df.set_index("source1_entity_id")["matched_entity_ids"]

    # Align on ground truth's entities — every S1 entity in the validation
    # split must have a prediction; missing predictions are treated as
    # "no match predicted" (empty set), consistent with a real submission
    # gap being penalized rather than silently ignored.
    common_ids = gt.index
    rows = []
    for s1_id in common_ids:
        true_ids = to_id_set(gt.loc[s1_id])
        pred_val = pred.loc[s1_id] if s1_id in pred.index else ""
        pred_ids = to_id_set(pred_val)
        f05 = _entity_f05(true_ids, pred_ids)
        rows.append({
            "source1_entity_id": s1_id,
            "num_true_matches": len(true_ids),
            "num_pred_matches": len(pred_ids),
            "f0.5": f05,
        })

    per_entity = pd.DataFrame(rows)

    num_singletons_true = (per_entity["num_true_matches"] == 0).sum()
    singleton_rows = per_entity[per_entity["num_true_matches"] == 0]
    singleton_accuracy = (
        (singleton_rows["num_pred_matches"] == 0).mean()
        if len(singleton_rows) > 0 else float("nan")
    )

    results = {
        "macro_f05": per_entity["f0.5"].mean(),
        "num_entities": len(per_entity),
        "num_singletons_true": int(num_singletons_true),
        "num_singletons_pred": int((per_entity["num_pred_matches"] == 0).sum()),
        "singleton_accuracy": singleton_accuracy,
        "per_entity_f05": per_entity,
    }

    print(f"Macro F0.5: {results['macro_f05']:.4f}")
    print(f"Entities evaluated: {results['num_entities']}")
    print(f"True singletons: {results['num_singletons_true']}, "
          f"singleton accuracy: {results['singleton_accuracy']:.4f}")

    return results


def optimize_entity_threshold(
    val_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    probability_col: str = "match_probability",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """
    Search for the decision threshold that maximizes the official challenge
    metric: entity-level macro-averaged F0.5.

    Parameters
    ----------
    val_pairs_df : pd.DataFrame
        Validation candidate pairs with columns:
        [source1_entity_id, candidate_entity_id, probability_col]
    ground_truth_df : pd.DataFrame
        Ground truth matching labels for the validation entities with columns:
        [source1_entity_id, matched_entity_ids] (comma-separated string or empty)
    probability_col : str
        Name of the probability column (default 'match_probability')
    thresholds : np.ndarray | None
        1D array of candidate thresholds to evaluate. If None, defaults to
        np.linspace(0.05, 0.95, 91) (step of 0.01).

    Returns
    -------
    best_threshold : float
        Threshold maximizing validation entity-level macro F0.5.
    best_score : float
        The maximum macro F0.5 achieved.
    sweep_df : pd.DataFrame
        DataFrame with columns ['threshold', 'macro_f05', 'singleton_accuracy']
        recording performance across the entire sweep.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    def parse_ids(val) -> set[str]:
        if pd.isna(val) or str(val).strip() == "":
            return set()
        return {x.strip() for x in str(val).split(",") if x.strip()}

    gt_dict = {
        row["source1_entity_id"]: parse_ids(row["matched_entity_ids"])
        for _, row in ground_truth_df.iterrows()
    }
    all_s1_ids = list(gt_dict.keys())

    # Pre-index candidates and probabilities by source1_entity_id
    cands_by_s1: dict[str, list[tuple[str, float]]] = {s1: [] for s1 in all_s1_ids}
    if not val_pairs_df.empty:
        for s1_id, cand_id, prob in zip(
            val_pairs_df["source1_entity_id"],
            val_pairs_df["candidate_entity_id"],
            val_pairs_df[probability_col],
        ):
            if s1_id in cands_by_s1:
                cands_by_s1[s1_id].append((str(cand_id), float(prob)))

    records = []
    best_threshold = 0.5
    best_f05 = -1.0

    for thr in thresholds:
        thr_float = round(float(thr), 4)
        scores = []
        singleton_correct = 0
        total_singletons = 0

        for s1_id in all_s1_ids:
            true_set = gt_dict[s1_id]
            pred_set = {cid for cid, p in cands_by_s1[s1_id] if p >= thr_float}
            score = _entity_f05(true_set, pred_set)
            scores.append(score)

            if len(true_set) == 0:
                total_singletons += 1
                if len(pred_set) == 0:
                    singleton_correct += 1

        macro_f05 = float(np.mean(scores)) if scores else 0.0
        singleton_acc = (singleton_correct / total_singletons) if total_singletons > 0 else 1.0

        records.append({
            "threshold": thr_float,
            "macro_f05": macro_f05,
            "singleton_accuracy": singleton_acc,
        })

        # Tie-break: prefer higher threshold to favor precision under F0.5
        if macro_f05 > best_f05 or (abs(macro_f05 - best_f05) < 1e-9 and thr_float > best_threshold):
            best_f05 = macro_f05
            best_threshold = thr_float

    sweep_df = pd.DataFrame(records)
    print(f"Optimal Entity Macro F0.5: {best_f05:.4f} at threshold: {best_threshold:.4f}")
    return best_threshold, best_f05, sweep_df