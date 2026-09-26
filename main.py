"""
main.py
-------
Unified end-to-end execution runner for the Business Entity Resolution Challenge.

Integrates:
- Person 1: Preprocessing, Data Normalization, and Schema Validation
- Person 2: Candidate Generation / Blocking (Multi-strategy Inverted Indexing)
- Person 3: Feature Engineering, Model Training, and Pairwise Inference
- Person 4: Integration, Entity-Level Macro F0.5 Threshold Tuning,
            Model Persistence, Submission Generation, and Official Validation.

Modes:
    --mode full    : Run complete pipeline (train -> tune -> block -> predict -> validate)
    --mode train   : Train matching model, tune entity-level threshold, save bundle
    --mode predict : Load saved bundle, generate candidates and predictions for test set
    --mode eval    : Evaluate an existing matching_results.tsv against ground truth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Person 1: Preprocessing
from src.preprocessing import (
    load_tsv,
    validate_schema,
    normalize_dataframe,
)

# Person 2: Candidate Blocking
from src.config import BlockingConfig, DEFAULT_CONFIG
from src.blocking import (
    generate_candidates,
    write_candidate_pairs_tsv,
)

# Person 3: Feature Engineering
from src.features import (
    normalize_sources,
    build_lookup_index,
    fit_tfidf,
    build_pair_features_batch,
    explode_candidate_pairs,
)

# Person 3 & 4: Model Training, Evaluation, and Persistence
from src.matching_model import (
    build_training_labels,
    train_model,
    save_model_bundle,
    load_model_bundle,
)
from src.evaluation import (
    evaluate_predictions,
    optimize_entity_threshold,
)
from src.predict import (
    generate_predictions,
    generate_matching_results,
)

# Organizer submission validator
from utils.validate_submission import validate


def setup_logger(name: str = "main") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def print_banner(text: str) -> None:
    width = 75
    border = "=" * width
    logger.info("\n" + border + f"\n  {text}\n" + border)


def run_training_pipeline(
    train_dir: Path,
    model_path: Path,
    val_size: float = 0.2,
    max_train_entities: int | None = None,
    negatives_per_positive: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train classifier, tune decision threshold on validation entity-level macro F0.5,
    and persist model bundle to disk.
    """
    print_banner("STEP 1: MODEL TRAINING & THRESHOLD OPTIMIZATION")

    s1_path = train_dir / "train_source1.tsv"
    s2_path = train_dir / "train_source2.tsv"
    s3_path = train_dir / "train_source3.tsv"
    gt_path = train_dir / "train_ground_truth.tsv"

    logger.info("Loading training sources: %s, %s, %s, %s", s1_path, s2_path, s3_path, gt_path)
    s1_df = load_tsv(str(s1_path))
    s2_df = load_tsv(str(s2_path))
    s3_df = load_tsv(str(s3_path))
    gt_df = load_tsv(str(gt_path))

    validate_schema(s1_df, "S1")
    validate_schema(s2_df, "S2")
    validate_schema(s3_df, "S3")
    logger.info("Schema validation passed for all training sources.")

    # Optional subsampling for training efficiency
    if max_train_entities is not None and len(s1_df) > max_train_entities:
        logger.info("Sampling %d entities from training set (seed=%d)...", max_train_entities, seed)
        sampled_s1_ids = s1_df["entity_id"].sample(n=max_train_entities, random_state=seed)
        s1_df = s1_df[s1_df["entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)
        gt_df = gt_df[gt_df["source1_entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)

    # Train / Validation Split by Source 1 entity ID (prevents entity data leakage)
    all_s1_ids = s1_df["entity_id"].unique()
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids, test_size=val_size, random_state=seed
    )
    val_s1_set = set(val_s1_ids)

    train_gt = gt_df[~gt_df["source1_entity_id"].isin(val_s1_set)].reset_index(drop=True)
    val_gt = gt_df[gt_df["source1_entity_id"].isin(val_s1_set)].reset_index(drop=True)
    logger.info("Entity Split: Train entities=%d, Val entities=%d", len(train_s1_ids), len(val_s1_ids))

    # Preprocessing & Normalization
    logger.info("Normalizing source data...")
    s1_norm, s2_norm, s3_norm = normalize_sources(s1_df, s2_df, s3_df)

    # Fit TF-IDF model on training entity names
    logger.info("Fitting TF-IDF model on entity names...")
    tfidf_model = fit_tfidf(s1_norm, s2_norm, s3_norm)

    # Lookup indexes for O(1) feature merging
    s1_idx = build_lookup_index(s1_norm)
    s23_idx = pd.concat([build_lookup_index(s2_norm), build_lookup_index(s3_norm)])

    # Generate candidate pairs and labels for training
    logger.info("Generating training positive and negative pairs...")
    train_labels = build_training_labels(
        train_gt,
        s2_norm["entity_id"],
        s3_norm["entity_id"],
        negatives_per_positive=negatives_per_positive,
        random_state=seed,
    )

    logger.info("Generating validation pairs for entity threshold tuning...")
    val_labels = build_training_labels(
        val_gt,
        s2_norm["entity_id"],
        s3_norm["entity_id"],
        negatives_per_positive=negatives_per_positive,
        random_state=seed,
    )

    # Combine pairs for unified feature computation
    all_pairs = pd.concat([train_labels, val_labels], ignore_index=True)
    logger.info("Computing features for %d total training/validation pairs...", len(all_pairs))
    features_df = build_pair_features_batch(all_pairs, s1_idx, s23_idx, tfidf_model)
    features_df["label"] = all_pairs["label"]

    feature_cols = [
        c for c in features_df.columns
        if c not in ("label", "source1_entity_id", "candidate_entity_id")
    ]
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)

    # Train model and tune threshold using entity-level macro F0.5
    logger.info("Training classifier and tuning entity-level macro F0.5 threshold...")
    model, threshold, importance = train_model(
        features_df=features_df,
        label_col="label",
        feature_cols=feature_cols,
        val_ground_truth_df=val_gt,
        random_state=seed,
    )

    # Persist model bundle (model, threshold, tfidf_model, feature_cols)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model_bundle(
        model=model,
        threshold=threshold,
        tfidf_model=tfidf_model,
        path=str(model_path),
        feature_cols=feature_cols,
        extra_metadata={
            "n_train_entities": len(train_s1_ids),
            "n_val_entities": len(val_s1_ids),
            "random_seed": seed,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    logger.info("Model bundle successfully persisted to %s", model_path)

    return {
        "model": model,
        "threshold": threshold,
        "tfidf_model": tfidf_model,
        "feature_cols": feature_cols,
    }


def run_inference_pipeline(
    test_dir: Path,
    model_path: Path,
    output_dir: Path,
    reports_dir: Path,
    blocking_config: BlockingConfig = DEFAULT_CONFIG,
    sample_size: int | None = None,
) -> tuple[Path, Path]:
    """
    Run candidate blocking on test set, compute features using the persisted
    model bundle, generate predictions, and write final outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cand_out = output_dir / "candidate_pairs.tsv"
    match_out = output_dir / "matching_results.tsv"
    preds_out = reports_dir / "predictions.csv"

    # 1. Blocking / Candidate Generation
    print_banner("STEP 2: CANDIDATE GENERATION / BLOCKING (TEST SPLIT)")
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    logger.info("Loading test sources from %s", test_dir)
    s1_test = load_tsv(str(s1_path))
    s2_test = load_tsv(str(s2_path))
    s3_test = load_tsv(str(s3_path))

    validate_schema(s1_test, "S1")
    validate_schema(s2_test, "S2")
    validate_schema(s3_test, "S3")
    logger.info("Test schema validation passed. Total S1 entities: %d", len(s1_test))

    if sample_size is not None and len(s1_test) > sample_size:
        logger.info("Sampling %d test entities for rapid execution...", sample_size)
        s1_test = s1_test.head(sample_size)

    t0 = time.time()
    candidates_df = generate_candidates(s1_test, s2_test, s3_test, blocking_config)
    logger.info("Blocking completed in %.2fs", time.time() - t0)

    write_candidate_pairs_tsv(candidates_df, str(cand_out))
    logger.info("Wrote candidate pairs to %s (%d rows)", cand_out, len(candidates_df))

    # 2. Pairwise Feature Computation & Model Prediction
    print_banner("STEP 3: MATCHING MODEL INFERENCE & SCORING")
    preds = generate_predictions(
        s1_df=s1_test,
        s2_df=s2_test,
        s3_df=s3_test,
        candidate_pairs_df=candidates_df,
        model_path=str(model_path),
        output_path=str(preds_out),
    )
    logger.info("Pairwise scoring completed. Predictions written to %s", preds_out)

    # 3. Final Submission Matching Results Generation
    print_banner("STEP 4: SUBMISSION FILE GENERATION")
    results_df = generate_matching_results(
        predictions_df=preds,
        all_source1_ids=s1_test["entity_id"],
        output_path=str(match_out),
    )
    logger.info("Final submission written to %s (%d rows)", match_out, len(results_df))

    return cand_out, match_out


def run_validation(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> bool:
    """Run organizer validator on the generated submission."""
    print_banner("STEP 5: SUBMISSION INTEGRITY VALIDATION")
    logger.info("Running utils/validate_submission.py...")

    errors, warnings = validate(
        matching_path=str(matching_path),
        candidate_path=str(candidate_path),
        test_dir=str(test_dir),
        check_ids=check_ids,
    )

    if warnings:
        logger.warning("Validation warnings (%d):", len(warnings))
        for w in warnings:
            logger.warning("  - %s", w)

    if errors:
        logger.error("VALIDATION FAILED with %d errors:", len(errors))
        for err in errors:
            logger.error("  [ERROR] %s", err)
        return False

    logger.info("ALL VALIDATION CHECKS PASSED SUCCESSFULLY (PASS - exit code 0)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution End-to-End Execution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "predict", "eval"],
        default="full",
        help="Pipeline execution mode (default: full)",
    )
    parser.add_argument(
        "--data-dir",
        default="dataset",
        help="Root directory containing train/ and test/ folders",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Directory with training TSVs (default: <data-dir>/train)",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Directory with test TSVs (default: <data-dir>/test)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where output TSVs are saved (default: output)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for reports and logs (default: reports)",
    )
    parser.add_argument(
        "--model-path",
        default="reports/matching_model.joblib",
        help="Path to persisted model bundle (default: reports/matching_model.joblib)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional entity sample size for quick dry-runs",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation split ratio for threshold optimization (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Run strict ID existence check in validator",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip final submission validation check",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = Path(args.train_dir) if args.train_dir else data_dir / "train"
    test_dir = Path(args.test_dir) if args.test_dir else data_dir / "test"
    output_dir = Path(args.output_dir)
    reports_dir = Path(args.reports_dir)
    model_path = Path(args.model_path)

    start_time = time.time()
    print_banner(f"STARTING PIPELINE EXECUTION (MODE: {args.mode.upper()})")

    if args.mode in ("full", "train"):
        run_training_pipeline(
            train_dir=train_dir,
            model_path=model_path,
            val_size=args.val_size,
            max_train_entities=args.sample_size,
            seed=args.seed,
        )

    if args.mode in ("full", "predict"):
        cand_out, match_out = run_inference_pipeline(
            test_dir=test_dir,
            model_path=model_path,
            output_dir=output_dir,
            reports_dir=reports_dir,
            sample_size=args.sample_size,
        )

        if not args.skip_validation:
            success = run_validation(
                matching_path=match_out,
                candidate_path=cand_out,
                test_dir=test_dir,
                check_ids=args.check_ids,
            )
            if not success:
                logger.error("Submission failed validation!")
                sys.exit(1)

    if args.mode == "eval":
        match_out = output_dir / "matching_results.tsv"
        gt_out = train_dir / "train_ground_truth.tsv"
        logger.info("Evaluating %s against %s", match_out, gt_out)
        matching_df = pd.read_csv(match_out, sep="\t", dtype=str)
        gt_df = pd.read_csv(gt_out, sep="\t", dtype=str)
        evaluate_predictions(matching_df, gt_df)

    elapsed = time.time() - start_time
    print_banner(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")


if __name__ == "__main__":
    main()
