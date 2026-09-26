"""
tests/test_integration.py
-------------------------
End-to-end integration tests for the full pipeline:
- Training -> Threshold Tuning -> Model Bundle Persistence -> Blocking -> Inference -> Submission Validation
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.test_blocking import make_synthetic
from main import (
    run_training_pipeline,
    run_inference_pipeline,
    run_validation,
)


@pytest.fixture
def synthetic_dataset_dir(tmp_path) -> Path:
    """Create a complete synthetic train and test directory."""
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    s1, s2, s3 = make_synthetic()

    # Save test sources
    s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    # Save train sources
    s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    s2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    s3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)

    # Ground truth
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-732914", "matched_entity_ids": "S2-118820,S3-905477"},
        {"source1_entity_id": "S1-889301", "matched_entity_ids": "S2-397155,S3-651230"},
        {"source1_entity_id": "S1-999999", "matched_entity_ids": ""},
        {"source1_entity_id": "S1-FR0001", "matched_entity_ids": "S2-FR0002"},
    ])
    gt.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    return tmp_path


def test_end_to_end_python_pipeline(synthetic_dataset_dir, tmp_path):
    train_dir = synthetic_dataset_dir / "train"
    test_dir = synthetic_dataset_dir / "test"
    output_dir = tmp_path / "output"
    reports_dir = tmp_path / "reports"
    model_path = reports_dir / "matching_model.joblib"

    # Step 1: Train & persist bundle
    trained_bundle = run_training_pipeline(
        train_dir=train_dir,
        model_path=model_path,
        val_size=0.25,
        negatives_per_positive=2,
        seed=42,
    )
    assert model_path.exists()
    assert trained_bundle["model"] is not None
    assert 0.05 <= trained_bundle["threshold"] <= 0.95

    # Step 2: Infer & generate outputs
    cand_path, match_path = run_inference_pipeline(
        test_dir=test_dir,
        model_path=model_path,
        output_dir=output_dir,
        reports_dir=reports_dir,
    )

    assert cand_path.exists()
    assert match_path.exists()

    cand_df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
    match_df = pd.read_csv(match_path, sep="\t", dtype=str, keep_default_na=False)

    # Required columns
    assert list(cand_df.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert list(match_df.columns) == ["source1_entity_id", "matched_entity_ids"]

    # All 4 Source 1 entities must be present
    assert len(cand_df) == 4
    assert len(match_df) == 4
    assert sorted(match_df["source1_entity_id"]) == ["S1-732914", "S1-889301", "S1-999999", "S1-FR0001"]

    # Singletons must be empty string, not NaN
    singleton_row = match_df[match_df["source1_entity_id"] == "S1-999999"]
    assert singleton_row["matched_entity_ids"].iloc[0] == ""

    # Step 3: Run official submission validator
    is_valid = run_validation(
        matching_path=match_path,
        candidate_path=cand_path,
        test_dir=test_dir,
        check_ids=True,
    )
    assert is_valid is True


def test_main_cli_execution(synthetic_dataset_dir, tmp_path):
    output_dir = tmp_path / "cli_output"
    reports_dir = tmp_path / "cli_reports"
    model_path = reports_dir / "cli_model.joblib"

    cmd = [
        sys.executable,
        "main.py",
        "--mode", "full",
        "--data-dir", str(synthetic_dataset_dir),
        "--output-dir", str(output_dir),
        "--reports-dir", str(reports_dir),
        "--model-path", str(model_path),
        "--check-ids",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI failed with error:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    assert (output_dir / "candidate_pairs.tsv").exists()
    assert (output_dir / "matching_results.tsv").exists()
    assert "ALL VALIDATION CHECKS PASSED SUCCESSFULLY" in result.stdout
