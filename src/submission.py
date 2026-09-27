"""
src/submission.py
-----------------
Submission packaging and validation utilities for the Business Entity Resolution Challenge.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from utils.validate_submission import validate

logger = logging.getLogger("submission")


def validate_submission(
    matching_path: str | Path = "output/matching_results.tsv",
    candidate_path: str | Path = "output/candidate_pairs.tsv",
    test_dir: str | Path = "dataset/test",
    check_ids: bool = False,
) -> bool:
    """
    Validate submission files against competition format and integrity rules.

    Parameters
    ----------
    matching_path : str | Path
        Path to output/matching_results.tsv
    candidate_path : str | Path
        Path to output/candidate_pairs.tsv
    test_dir : str | Path
        Directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv
    check_ids : bool
        Whether to run strict existence checks on candidate IDs (default: False)

    Returns
    -------
    bool
        True if all validation checks passed (0 errors), False otherwise.
    """
    errors, warnings = validate(
        matching_path=str(matching_path),
        candidate_path=str(candidate_path),
        test_dir=str(test_dir),
        check_ids=check_ids,
    )

    if warnings:
        for w in warnings:
            logger.warning("[VALIDATOR WARNING] %s", w)

    if errors:
        for err in errors:
            logger.error("[VALIDATOR ERROR] %s", err)
        return False

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    valid = validate_submission()
    if valid:
        print("PASS - Submission validation successful.")
        sys.exit(0)
    else:
        print("FAIL - Submission validation failed.")
        sys.exit(1)
