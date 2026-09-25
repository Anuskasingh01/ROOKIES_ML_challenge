"""
Blocking-quality evaluation, run only against TRAINING data (ground truth
required). This is what tells you, before ever touching the matching model,
whether the candidate generation stage is good enough - a match lost here
can never be recovered downstream, so recall is the headline number.
"""
from __future__ import annotations

import argparse
import logging
from typing import Dict, Set

import pandas as pd

from config import BlockingConfig, DEFAULT_CONFIG, Paths
from blocking import generate_candidates

logger = logging.getLogger("evaluate_blocking")


def _parse_id_list(cell) -> Set[str]:
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return {x.strip() for x in str(cell).split(",") if x.strip()}


def load_ground_truth(path: str) -> Dict[str, Set[str]]:
    gt = pd.read_csv(path, sep="\t", dtype=str)
    return {
        row["source1_entity_id"]: _parse_id_list(row["matched_entity_ids"])
        for _, row in gt.iterrows()
    }


def blocking_recall_report(candidates_df: pd.DataFrame, ground_truth: Dict[str, Set[str]]) -> dict:
    """
    Returns:
      overall_recall: fraction of ALL true (s1, matched_id) pairs that appear
        somewhere in the candidate set (this is the true recall ceiling for
        the matching model downstream).
      entity_level_recall: fraction of Source 1 entities for which EVERY
        true match was captured (stricter; a single missed match on an
        entity with 3 true matches still counts as a "miss" here).
      avg_candidates_per_entity, reduction_ratio, entities_with_zero_candidates.
    """
    cand_map = dict(zip(candidates_df["source1_entity_id"], candidates_df["candidate_ids"]))

    total_true_pairs = 0
    recovered_true_pairs = 0
    entities_fully_recovered = 0
    entities_with_any_true_match = 0
    zero_candidate_entities = 0
    total_candidates = 0

    for s1_id, true_matches in ground_truth.items():
        cands = set(cand_map.get(s1_id, []))
        total_candidates += len(cands)
        if len(cands) == 0:
            zero_candidate_entities += 1

        if not true_matches:
            continue  # singleton - nothing to recover, doesn't affect recall

        entities_with_any_true_match += 1
        total_true_pairs += len(true_matches)
        found = true_matches & cands
        recovered_true_pairs += len(found)
        if found == true_matches:
            entities_fully_recovered += 1

    n_entities = len(ground_truth)
    overall_recall = recovered_true_pairs / total_true_pairs if total_true_pairs else 1.0
    entity_level_recall = (
        entities_fully_recovered / entities_with_any_true_match
        if entities_with_any_true_match else 1.0
    )

    return {
        "n_source1_entities": n_entities,
        "avg_candidates_per_entity": total_candidates / n_entities if n_entities else 0.0,
        "entities_with_zero_candidates": zero_candidate_entities,
        "overall_pair_recall": overall_recall,
        "entity_level_full_recall": entity_level_recall,
        "true_pairs_total": total_true_pairs,
        "true_pairs_recovered": recovered_true_pairs,
    }


def strategy_ablation_report(
    s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]], cfg: BlockingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Runs each strategy alone (and the full union) to show which strategies
    are actually earning their keep - useful evidence for the methodology doc."""
    from blocking import STRATEGY_REGISTRY

    rows = []
    for name in STRATEGY_REGISTRY:
        cands = generate_candidates(s1_df, s2_df, s3_df, cfg, strategies=[name])
        report = blocking_recall_report(cands, ground_truth)
        report["strategy"] = name
        rows.append(report)

    cands_all = generate_candidates(s1_df, s2_df, s3_df, cfg, strategies=list(STRATEGY_REGISTRY.keys()))
    report_all = blocking_recall_report(cands_all, ground_truth)
    report_all["strategy"] = "ALL_UNIONED"
    rows.append(report_all)

    return pd.DataFrame(rows).set_index("strategy")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate blocking recall on training data")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--ablation", action="store_true", help="Also run each strategy in isolation")
    args = parser.parse_args()

    paths = Paths()
    if args.train_dir:
        paths = Paths(train_dir=args.train_dir)

    s1 = pd.read_csv(paths.train_source1, sep="\t", dtype=str)
    s2 = pd.read_csv(paths.train_source2, sep="\t", dtype=str)
    s3 = pd.read_csv(paths.train_source3, sep="\t", dtype=str)
    ground_truth = load_ground_truth(str(paths.train_ground_truth))

    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    report = blocking_recall_report(candidates, ground_truth)

    print("\n=== Blocking quality (full union of strategies) ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    n_s1, n_s2, n_s3 = len(s1), len(s2), len(s3)
    naive_pairs = n_s1 * (n_s2 + n_s3)
    actual_pairs = sum(len(c) for c in candidates["candidate_ids"])
    reduction_ratio = 1 - (actual_pairs / naive_pairs) if naive_pairs else 0.0
    print(f"  reduction_ratio_vs_cartesian: {reduction_ratio:.6f}")
    print(f"  candidate_pairs_generated: {actual_pairs}  (naive cartesian would be {naive_pairs})")

    if args.ablation:
        print("\n=== Per-strategy ablation ===")
        ablation = strategy_ablation_report(s1, s2, s3, ground_truth, DEFAULT_CONFIG)
        pd.set_option("display.width", 160)
        print(ablation[[
            "overall_pair_recall", "entity_level_full_recall",
            "avg_candidates_per_entity", "entities_with_zero_candidates",
        ]])


if __name__ == "__main__":
    main()
