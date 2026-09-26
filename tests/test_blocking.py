"""
Smoke tests using small synthetic frames (mirroring the video's own "Acme"
example), so blocking.py can be verified before the real dataset lands.
Run with:  python -m pytest tests/test_blocking.py -v
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
from config import DEFAULT_CONFIG
from blocking import generate_candidates, candidates_to_tsv_rows


def make_synthetic():
    s1 = pd.DataFrame([
        {"entity_id": "S1-732914", "business_name": "Acme Robotics Inc",
         "business_address": "500 Market St, San Jose", "country": "US"},
        {"entity_id": "S1-889301", "business_name": "Delta Foods",
         "business_address": "8 Oak Ave, Austin", "country": "US"},
        {"entity_id": "S1-999999", "business_name": "Totally Unique Widgets Co",
         "business_address": "1 Nowhere Ln, Fargo", "country": "US"},  # singleton
        {"entity_id": "S1-FR0001", "business_name": "Boulangerie Lumiere",
         "business_address": "12 Rue de Paris, Paris", "country": "France"},  # unseen country
    ])
    s2 = pd.DataFrame([
        {"entity_id": "S2-118820", "business_name": "Acme Robotics Incorporated",
         "business_address": "500 Market Street, San Jose CA", "country": "US"},
        {"entity_id": "S2-397155", "business_name": "Delta Foods Co",
         "business_address": "8 Oak Avenue, Austin", "country": "US"},
        {"entity_id": "S2-663049", "business_name": "Bright Cafe",
         "business_address": "22 Pine Street, Reno", "country": "US"},
        {"entity_id": "S2-FR0002", "business_name": "Boulangerie Lumiere SARL",
         "business_address": "12 Rue de Paris, 75001 Paris", "country": "France"},
    ])
    s3 = pd.DataFrame([
        {"entity_id": "S3-905477", "business_name": "Acme Robotics",
         "business_address": "Nr City Hall, San Jose", "country": "US"},
        {"entity_id": "S3-063118", "business_name": "Acme Bakery",  # decoy: same block, different biz
         "business_address": "500 Market St, San Jose", "country": "US"},
        {"entity_id": "S3-651230", "business_name": "Delta Foods Ltd",
         "business_address": "Oak Ave, Austin", "country": "US"},
        {"entity_id": "S3-472088", "business_name": "Kappa Motors",
         "business_address": "90 Lake Dr, Fargo", "country": "US"},
    ])
    return s1, s2, s3


def test_true_matches_are_recovered():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    cand_map = dict(zip(candidates["source1_entity_id"], candidates["candidate_ids"]))

    assert "S2-118820" in cand_map["S1-732914"]
    assert "S3-905477" in cand_map["S1-732914"]
    assert "S2-397155" in cand_map["S1-889301"]
    assert "S3-651230" in cand_map["S1-889301"]


def test_unrelated_records_not_pulled_in_by_unrelated_blocks():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    cand_map = dict(zip(candidates["source1_entity_id"], candidates["candidate_ids"]))
    # Kappa Motors (Fargo) should not show up as a candidate for Acme Robotics (San Jose)
    assert "S3-472088" not in cand_map["S1-732914"]


def test_singleton_can_have_empty_or_small_candidate_set():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    cand_map = dict(zip(candidates["source1_entity_id"], candidates["candidate_ids"]))
    assert "S1-999999" in cand_map  # every S1 entity must appear, even with no/few candidates


def test_unseen_country_is_not_dropped():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    cand_map = dict(zip(candidates["source1_entity_id"], candidates["candidate_ids"]))
    assert "S1-FR0001" in cand_map
    assert "S2-FR0002" in cand_map["S1-FR0001"]


def test_no_duplicate_candidate_ids():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    for ids in candidates["candidate_ids"]:
        assert len(ids) == len(set(ids))


def test_every_s1_entity_present_exactly_once():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    assert sorted(candidates["source1_entity_id"]) == sorted(s1["entity_id"])
    assert candidates["source1_entity_id"].is_unique


def test_tsv_row_format():
    s1, s2, s3 = make_synthetic()
    candidates = generate_candidates(s1, s2, s3, DEFAULT_CONFIG)
    tsv_rows = candidates_to_tsv_rows(candidates)
    assert list(tsv_rows.columns) == ["source1_entity_id", "candidate_entity_ids"]
    for val in tsv_rows["candidate_entity_ids"]:
        assert isinstance(val, str)  # never NaN, empty string is fine


if __name__ == "__main__":
    # allow running without pytest installed
    import inspect
    mod = sys.modules[__name__]
    tests = [f for name, f in inspect.getmembers(mod) if name.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")
