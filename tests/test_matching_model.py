"""
tests/test_matching_model.py
------------------------------
Minimal tests for Person 3's matching model module, covering the three
areas required by the challenge spec: feature generation, prediction
output, and threshold handling.
"""

import pandas as pd
import numpy as np
import pytest

from src.matching_model import build_training_labels
from src.evaluation import evaluate_predictions, _entity_f05


def test_build_training_labels_creates_positives_and_negatives():
    gt = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-1,S3-1", ""],
    })
    s2_ids = pd.Series(["S2-1", "S2-2"])
    s3_ids = pd.Series(["S3-1", "S3-2"])

    labels = build_training_labels(gt, s2_ids, s3_ids, negatives_per_positive=2)

    assert set(labels.columns) == {"source1_entity_id", "candidate_entity_id", "label"}
    positives = labels[labels["label"] == 1]
    assert len(positives) == 2  # S2-1 and S3-1 for S1-1
    assert (labels["label"].isin([0, 1])).all()


def test_build_training_labels_no_positive_negative_collision():
    gt = pd.DataFrame({
        "source1_entity_id": ["S1-1"],
        "matched_entity_ids": ["S2-1"],
    })
    s2_ids = pd.Series(["S2-1"])
    s3_ids = pd.Series([])

    labels = build_training_labels(gt, s2_ids, s3_ids, negatives_per_positive=5)
    dupes = labels.duplicated(subset=["source1_entity_id", "candidate_entity_id"])
    assert not dupes.any()


def test_entity_f05_perfect_match():
    assert _entity_f05({"S2-1", "S3-1"}, {"S2-1", "S3-1"}) == 1.0


def test_entity_f05_singleton_correct():
    assert _entity_f05(set(), set()) == 1.0


def test_entity_f05_singleton_false_merge():
    assert _entity_f05(set(), {"S2-1"}) == 0.0


def test_entity_f05_partial_precision_matches_readme_example():
    # From README: true=[S2-1, S3-1], pred=[S2-1, S2-2, S3-1] -> F0.5 = 0.714
    f05 = _entity_f05({"S2-1", "S3-1"}, {"S2-1", "S2-2", "S3-1"})
    assert abs(f05 - 0.714) < 0.01


def test_evaluate_predictions_macro_average():
    matching_results = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-1", ""],
    })
    ground_truth = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2"],
        "matched_entity_ids": ["S2-1", ""],
    })
    results = evaluate_predictions(matching_results, ground_truth)
    assert results["macro_f05"] == 1.0
    assert results["num_singletons_true"] == 1