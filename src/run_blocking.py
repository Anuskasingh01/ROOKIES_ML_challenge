"""
CLI: generate candidate_pairs.tsv for the test set (or train set, for
debugging) using blocking.py. This is what Person 4's main.py will call
into, and what you (Person 2) run standalone to produce your deliverable.

Usage:
    python run_blocking.py --split test \
        --source1 dataset/test/test_source1.tsv \
        --source2 dataset/test/test_source2.tsv \
        --source3 dataset/test/test_source3.tsv \
        --out output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from config import DEFAULT_CONFIG, Paths
from blocking import generate_candidates, write_candidate_pairs_tsv

try:
    from preprocessing import validate_schema
except ImportError:
    try:
        from src.preprocessing import validate_schema
    except ImportError:
        validate_schema = None

logger = logging.getLogger("run_blocking")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the blocking / candidate-generation stage")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--source1", default=None)
    parser.add_argument("--source2", default=None)
    parser.add_argument("--source3", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    paths = Paths()
    if args.split == "train":
        s1_path = args.source1 or paths.train_source1
        s2_path = args.source2 or paths.train_source2
        s3_path = args.source3 or paths.train_source3
    else:
        s1_path = args.source1 or paths.test_source1
        s2_path = args.source2 or paths.test_source2
        s3_path = args.source3 or paths.test_source3
    out_path = args.out or paths.candidate_pairs_out

    logger.info("Loading %s / %s / %s", s1_path, s2_path, s3_path)
    # dtype=str everywhere: entity_id, postal codes, etc. must never be
    # silently coerced to numeric (e.g. a PIN code losing a leading zero).
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    if validate_schema is not None:
        try:
            validate_schema(s1, "S1")
            validate_schema(s2, "S2")
            validate_schema(s3, "S3")
            logger.info("Schema validation passed for S1, S2, and S3 (Person 1 Data Engineering)")
        except Exception as err:
            logger.warning("Schema validation warning: %s", err)

    t0 = time.time()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    elapsed = time.time() - t0
    logger.info("Blocking finished in %.1fs for %d Source 1 entities", elapsed, len(s1))

    import os
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    write_candidate_pairs_tsv(candidates, str(out_path))

    total_candidates = sum(len(c) for c in candidates["candidate_ids"])
    zero = sum(1 for c in candidates["candidate_ids"] if len(c) == 0)
    logger.info(
        "avg candidates/entity=%.2f, entities with zero candidates=%d/%d",
        total_candidates / len(s1) if len(s1) else 0.0, zero, len(s1),
    )


if __name__ == "__main__":
    main()
