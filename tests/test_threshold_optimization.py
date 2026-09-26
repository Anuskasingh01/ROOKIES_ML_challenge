"""
tests/test_threshold_optimization.py
------------------------------------
Focused unit tests for entity-level macro F0.5 threshold search:
- optimize_entity_threshold()
- train_model(val_ground_truth_df=...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import optimize_entity_threshold, _entity_f05
from src.matching_model import train_model


def test_optimize_entity_threshold_perfect_recovery():
    # S1-1 has true match S2-1, candidate pair prob=0.9
    # S1-1 has false candidate S2-2, prob=0.3
    # S1-2 is a singleton, has false candidate S2-3, prob=0.4
    val_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "match_probability": 0.90},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-2", "match_probability": 0.30},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-3", "match_probability": 0.40},
    ])
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    ])

    best_thr, best_score, sweep_df = optimize_entity_threshold(
        val_pairs, gt, thresholds=np.array([0.2, 0.5, 0.85])
    )

    # At threshold 0.85:
    # S1-1 predicts {S2-1} (exact match -> F0.5 = 1.0)
    # S1-2 predicts set() (singleton correct -> F0.5 = 1.0)
    # Macro F0.5 = 1.0
    assert best_score == 1.0
    assert best_thr == 0.85
    assert len(sweep_df) == 3


def test_optimize_entity_threshold_precision_favored():
    # S1-1 true matches: S2-1, S3-1
    # Candidate probs: S2-1 (0.85), S3-1 (0.60), S2-decoy (0.55)
    # At thr=0.50: predicts {S2-1, S3-1, S2-decoy} -> TP=2, FP=1, FN=0 -> P=2/3, R=1 -> F0.5 = 0.714
    # At thr=0.70: predicts {S2-1} -> TP=1, FP=0, FN=1 -> P=1, R=1/2 -> F0.5 = 0.833
    # Precision is weighted 2x over recall, so thr=0.70 must win over thr=0.50!
    val_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "match_probability": 0.85},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "match_probability": 0.60},
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-decoy", "match_probability": 0.55},
    ])
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
    ])

    best_thr, best_score, _ = optimize_entity_threshold(
        val_pairs, gt, thresholds=np.array([0.50, 0.70])
    )

    assert best_thr == 0.70
    assert abs(best_score - 0.833) < 0.01


def test_optimize_entity_threshold_all_singletons():
    # Only singletons with varying false candidate scores
    val_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "match_probability": 0.3},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2", "match_probability": 0.6},
    ])
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    ])

    best_thr, best_score, sweep_df = optimize_entity_threshold(
        val_pairs, gt, thresholds=np.array([0.2, 0.5, 0.7])
    )

    # At thr=0.7: neither passes, both predicted empty -> 100% singleton accuracy -> score 1.0
    assert best_thr == 0.7
    assert best_score == 1.0


def test_train_model_with_val_ground_truth():
    # Synthetic feature dataset with entity IDs
    data = []
    for i in range(20):
        s1 = f"S1-{i}"
        # True match
        data.append({
            "source1_entity_id": s1,
            "candidate_entity_id": f"S2-{i}",
            "name_jaccard": 0.9,
            "addr_jaccard": 0.8,
            "country_match": 1.0,
            "label": 1,
        })
        # Negative near-miss
        data.append({
            "source1_entity_id": s1,
            "candidate_entity_id": f"S2-fake-{i}",
            "name_jaccard": 0.3,
            "addr_jaccard": 0.1,
            "country_match": 1.0,
            "label": 0,
        })

    features_df = pd.DataFrame(data)
    val_gt = pd.DataFrame([
        {"source1_entity_id": f"S1-{i}", "matched_entity_ids": f"S2-{i}"}
        for i in range(15, 20)
    ])

    model, threshold, importance = train_model(
        features_df=features_df,
        feature_cols=["name_jaccard", "addr_jaccard", "country_match"],
        val_ground_truth_df=val_gt,
        random_state=42,
    )

    assert model is not None
    assert 0.05 <= threshold <= 0.95
    assert importance is not None
    assert "name_jaccard" in importance.index
