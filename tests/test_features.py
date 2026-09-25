import sys
import os

import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import (
    normalize_sources,
    build_lookup_index,
    explode_candidate_pairs,
    fit_tfidf,
    build_pair_features_batch,
    _token_jaccard,
    _edit_ratio,
    _token_sort_ratio,
    _numeric_overlap,
)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_source_df(records):
    """Build a minimal source DataFrame from a list of (id, name, addr, country) tuples."""
    return pd.DataFrame(
        records,
        columns=["entity_id", "business_name", "business_address", "country"],
    )


# ---------------------------------------------------------------------------
# Block 1 — explode_candidate_pairs
# ---------------------------------------------------------------------------

class TestExplodeCandidatePairs:
    """Tests for explode_candidate_pairs()."""

    def _input_df(self, rows):
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])

    def test_normal_comma_separated(self):
        df = self._input_df([("S1-001", "S2-001,S3-002")])
        result = explode_candidate_pairs(df)

        assert list(result.columns) == ["source1_entity_id", "candidate_entity_id"]
        assert len(result) == 2
        assert set(result["candidate_entity_id"]) == {"S2-001", "S3-002"}
        assert (result["source1_entity_id"] == "S1-001").all()

    def test_empty_string_produces_no_rows(self):
        """A singleton with no candidates (empty string) must not appear in output."""
        df = self._input_df([("S1-002", "")])
        result = explode_candidate_pairs(df)
        assert len(result) == 0

    def test_nan_produces_no_rows(self):
        """A singleton with NaN candidates must not appear in output."""
        df = self._input_df([("S1-003", None)])
        result = explode_candidate_pairs(df)
        assert len(result) == 0

    def test_whitespace_around_ids_is_stripped(self):
        """IDs padded with spaces must be stripped before appearing in output."""
        df = self._input_df([("S1-004", " S2-003 , S3-004 ")])
        result = explode_candidate_pairs(df)

        assert len(result) == 2
        assert set(result["candidate_entity_id"]) == {"S2-003", "S3-004"}

    def test_trailing_comma_produces_no_empty_id(self):
        """A trailing comma must not produce an empty-string candidate ID."""
        df = self._input_df([("S1-005", "S2-005,")])
        result = explode_candidate_pairs(df)

        assert len(result) == 1
        assert result.iloc[0]["candidate_entity_id"] == "S2-005"

    def test_mixed_cases_combined(self):
        """All cases together: row count and IDs match expectations for each source1 entity."""
        df = self._input_df([
            ("S1-001", "S2-001,S3-002"),   # normal → 2 rows
            ("S1-002", ""),                # empty  → 0 rows
            ("S1-003", None),              # NaN    → 0 rows
            ("S1-004", " S2-003 , S3-004 "), # whitespace → 2 rows, stripped
            ("S1-005", "S2-005,"),         # trailing comma → 1 row
        ])
        result = explode_candidate_pairs(df)

        assert len(result) == 5  # 2 + 0 + 0 + 2 + 1
        assert len(result[result["source1_entity_id"] == "S1-001"]) == 2
        assert len(result[result["source1_entity_id"] == "S1-002"]) == 0
        assert len(result[result["source1_entity_id"] == "S1-003"]) == 0
        assert len(result[result["source1_entity_id"] == "S1-004"]) == 2
        assert len(result[result["source1_entity_id"] == "S1-005"]) == 1

    def test_output_columns(self):
        df = self._input_df([("S1-001", "S2-001")])
        result = explode_candidate_pairs(df)
        assert list(result.columns) == ["source1_entity_id", "candidate_entity_id"]


# ---------------------------------------------------------------------------
# Block 2 — _token_jaccard
# ---------------------------------------------------------------------------

class TestTokenJaccard:
    """Tests for _token_jaccard(a, b)."""

    # --- empty / missing inputs ---

    def test_empty_left_returns_zero(self):
        assert _token_jaccard("", "foo bar") == 0.0

    def test_empty_right_returns_zero(self):
        assert _token_jaccard("foo bar", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert _token_jaccard("", "") == 0.0

    # --- normal inputs with known expected outputs ---

    def test_identical_strings_return_one(self):
        assert _token_jaccard("abc def", "abc def") == 1.0

    def test_single_token_exact_match(self):
        assert _token_jaccard("foo", "foo") == 1.0

    def test_no_overlap_returns_zero(self):
        # {"a","b","c"} ∩ {"d","e","f"} = ∅
        assert _token_jaccard("a b c", "d e f") == 0.0

    def test_partial_overlap(self):
        # {"abc","def"} ∩ {"abc","xyz"} = {"abc"}  → 1/3
        assert _token_jaccard("abc def", "abc xyz") == pytest.approx(1 / 3, abs=1e-9)

    def test_subset_overlap(self):
        # {"a","b"} vs {"a","b","c"}  →  2 / 3
        assert _token_jaccard("a b", "a b c") == pytest.approx(2 / 3, abs=1e-9)

    def test_no_abbreviation_collapsing(self):
        # Confirms normalizer does NOT collapse inc/incorporated:
        # {"inc"} ∩ {"incorporated"} = ∅
        assert _token_jaccard("inc", "incorporated") == 0.0

    def test_st_vs_street_no_collapse(self):
        # Same principle for address abbreviations
        assert _token_jaccard("st", "street") == 0.0

    def test_return_value_range(self):
        result = _token_jaccard("one two three", "two three four")
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Block 3 — _edit_ratio
# ---------------------------------------------------------------------------

class TestEditRatio:
    """Tests for _edit_ratio(a, b) — uses rapidfuzz, returns float in [0, 1]."""

    # --- empty / missing inputs ---

    def test_empty_left_returns_zero(self):
        assert _edit_ratio("", "foo") == 0.0

    def test_empty_right_returns_zero(self):
        assert _edit_ratio("foo", "") == 0.0

    def test_both_empty_returns_zero(self):
        # Convention: both-empty → 0.0 (unknown, not similar).
        # Deliberately NOT 1.0 — avoids spuriously boosting pairs with two missing names.
        assert _edit_ratio("", "") == 0.0

    # --- normal inputs ---

    def test_identical_strings_return_one(self):
        assert _edit_ratio("abc", "abc") == 1.0

    def test_very_similar_strings(self):
        # "starbucks" vs "starbux" — close typo, expect high similarity
        result = _edit_ratio("starbucks", "starbux")
        assert result > 0.7

    def test_moderately_similar_strings(self):
        # "kitten" vs "sitting" — classical Levenshtein example, neither identical nor zero
        result = _edit_ratio("kitten", "sitting")
        assert 0.5 < result < 0.9

    def test_very_different_strings(self):
        assert _edit_ratio("aaa", "zzz") < 0.5

    def test_return_value_in_unit_interval(self):
        # rapidfuzz returns 0–100; we divide by 100 — must always be in [0, 1]
        assert 0.0 <= _edit_ratio("hello", "hello world") <= 1.0

    def test_symmetry(self):
        # edit ratio should be symmetric
        assert _edit_ratio("abc", "abcd") == pytest.approx(_edit_ratio("abcd", "abc"), abs=1e-9)


# ---------------------------------------------------------------------------
# Block 4 — _token_sort_ratio
# ---------------------------------------------------------------------------

class TestTokenSortRatio:
    """Tests for _token_sort_ratio(a, b) — word-order invariant, uses rapidfuzz."""

    # --- empty / missing inputs ---

    def test_empty_left_returns_zero(self):
        assert _token_sort_ratio("", "foo bar") == 0.0

    def test_empty_right_returns_zero(self):
        assert _token_sort_ratio("foo bar", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert _token_sort_ratio("", "") == 0.0

    # --- normal inputs ---

    def test_identical_strings_return_one(self):
        assert _token_sort_ratio("foo bar", "foo bar") == 1.0

    def test_word_order_swap_three_tokens(self):
        # Core feature value: "bank of india" sorted → "bank india of"
        #                     "india bank of" sorted → "bank india of"
        # After sort both are identical → ratio should be 1.0
        assert _token_sort_ratio("bank of india", "india bank of") == pytest.approx(1.0, abs=1e-9)

    def test_word_order_swap_two_tokens(self):
        assert _token_sort_ratio("alpha beta", "beta alpha") == pytest.approx(1.0, abs=1e-9)

    def test_completely_different_tokens(self):
        assert _token_sort_ratio("alpha beta", "gamma delta") < 0.5

    def test_return_value_in_unit_interval(self):
        result = _token_sort_ratio("one two three", "four five six")
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Block 5 — _numeric_overlap
# ---------------------------------------------------------------------------

class TestNumericOverlap:
    """Tests for _numeric_overlap(a, b) — Jaccard over digit-only tokens."""

    # --- empty / missing inputs ---

    def test_empty_left_returns_zero(self):
        assert _numeric_overlap("", "123 main st") == 0.0

    def test_empty_right_returns_zero(self):
        assert _numeric_overlap("123 main st", "") == 0.0

    def test_both_empty_returns_zero(self):
        assert _numeric_overlap("", "") == 0.0

    # --- no digit tokens ---

    def test_no_digits_on_either_side_returns_zero(self):
        assert _numeric_overlap("no digits here", "also none") == 0.0

    def test_only_non_digit_tokens_returns_zero(self):
        # Confirms feature is strictly digit-token based: "st" ≠ "street" as numbers
        assert _numeric_overlap("st", "street") == 0.0

    # --- normal inputs with known expected outputs ---

    def test_full_overlap_single_number(self):
        # {"123"} ∩ {"123"} / {"123"} ∪ {"123"} = 1.0
        assert _numeric_overlap("123 main st", "123 broadway") == 1.0

    def test_full_overlap_multiple_shared_numbers(self):
        # {"12","34"} ∩ {"12","34"} = 1.0
        assert _numeric_overlap("12 34 main", "12 34 other") == 1.0

    def test_partial_overlap(self):
        # {"123","456"} vs {"123","789"}  →  1 / 3
        assert _numeric_overlap("123 456 main", "123 789 road") == pytest.approx(1 / 3, abs=1e-9)

    def test_no_overlap_different_numbers(self):
        assert _numeric_overlap("123 main st", "456 elm ave") == 0.0

    def test_return_value_in_unit_interval(self):
        result = _numeric_overlap("10 20 street", "10 30 avenue")
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Block 6 — build_pair_features_batch: happy path
# ---------------------------------------------------------------------------

class TestBuildPairFeaturesBatchHappyPath:
    """
    End-to-end test on a small synthetic dataset with hand-crafted records
    where expected feature values can be computed analytically.

    Synthetic data:
        S1-001 "starbucks corp"    / "123 main st seattle"   / us
        S1-002 "taj hotels"        / "apollo bunder mumbai"  / india

        S2-001 "starbucks corporation" / "123 main street seattle" / us  ← near-match to S1-001
        S2-002 "taj hotel resorts"     / "apollo bunder mumbai"    / india ← near-match to S1-002
        S3-001 "completely different"  / "999 other ave"           / us   ← non-match to S1-001

    Pairs scored:
        (S1-001, S2-001) — near match
        (S1-001, S3-001) — non-match
        (S1-002, S2-002) — near match
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        s1_raw = _make_source_df([
            ("S1-001", "starbucks corp",  "123 main st seattle",  "us"),
            ("S1-002", "taj hotels",      "apollo bunder mumbai", "india"),
        ])
        s2_raw = _make_source_df([
            ("S2-001", "starbucks corporation", "123 main street seattle", "us"),
            ("S2-002", "taj hotel resorts",     "apollo bunder mumbai",    "india"),
        ])
        s3_raw = _make_source_df([
            ("S3-001", "completely different",  "999 other ave",           "us"),
        ])

        s1_df, s2_df, s3_df = normalize_sources(s1_raw, s2_raw, s3_raw)
        s1_idx  = build_lookup_index(s1_df)
        s23_idx = pd.concat([build_lookup_index(s2_df), build_lookup_index(s3_df)])
        tfidf   = fit_tfidf(s1_df, s2_df, s3_df)

        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001"},
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-001"},
            {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-002"},
        ])

        self.result  = build_pair_features_batch(pairs, s1_idx, s23_idx, tfidf)
        self.r_near  = self._row("S1-001", "S2-001")  # near-match
        self.r_far   = self._row("S1-001", "S3-001")  # non-match
        self.r_taj   = self._row("S1-002", "S2-002")  # near-match (address identical)

    def _row(self, s1_id, cand_id):
        mask = (
            (self.result["source1_entity_id"] == s1_id) &
            (self.result["candidate_entity_id"] == cand_id)
        )
        return self.result[mask].iloc[0]

    # --- structural checks ---

    def test_output_row_count(self):
        assert len(self.result) == 3

    def test_all_feature_columns_present(self):
        expected = {
            "source1_entity_id", "candidate_entity_id",
            "name_jaccard", "name_edit_ratio", "name_token_sort",
            "addr_jaccard", "addr_edit_ratio", "addr_numeric_overlap",
            "name_tfidf_cosine", "country_match",
        }
        assert expected.issubset(set(self.result.columns))

    def test_no_nan_in_output(self):
        assert not self.result.isnull().any().any()

    def test_norm_columns_dropped(self):
        """Raw norm string columns must not appear in the returned DataFrame."""
        for col in ["name_norm_s1", "address_norm_s1", "country_norm_s1",
                    "name_norm_s2", "address_norm_s2", "country_norm_s2"]:
            assert col not in self.result.columns

    # --- (S1-001, S2-001): "starbucks corp" vs "starbucks corporation" ---
    #
    # name tokens: {"starbucks","corp"} ∩ {"starbucks","corporation"}
    #              = {"starbucks"}  →  1/3  ≈ 0.333
    #
    # addr tokens: {"123","main","st","seattle"} ∩ {"123","main","street","seattle"}
    #              = {"123","main","seattle"}  →  3/5 = 0.6
    #
    # addr numeric: {"123"} ∩ {"123"} / union = 1.0

    def test_near_match_name_jaccard(self):
        assert self.r_near["name_jaccard"] == pytest.approx(1 / 3, abs=1e-9)

    def test_near_match_name_edit_ratio_high(self):
        assert self.r_near["name_edit_ratio"] > 0.6

    def test_near_match_name_token_sort_high(self):
        assert self.r_near["name_token_sort"] > 0.6

    def test_near_match_addr_jaccard(self):
        # {"123","main","seattle"} / {"123","main","st","street","seattle"} = 3/5
        assert self.r_near["addr_jaccard"] == pytest.approx(3 / 5, abs=1e-9)

    def test_near_match_addr_numeric_overlap_full(self):
        # Both addresses contain "123" — full numeric overlap
        assert self.r_near["addr_numeric_overlap"] == 1.0

    def test_near_match_tfidf_cosine_positive(self):
        # Both names contain "starbucks" → cosine > 0
        assert self.r_near["name_tfidf_cosine"] > 0.0

    def test_near_match_country_match(self):
        assert self.r_near["country_match"] == 1

    # --- (S1-001, S3-001): "starbucks corp" vs "completely different" ---

    def test_non_match_name_jaccard_zero(self):
        # No shared name tokens
        assert self.r_far["name_jaccard"] == 0.0

    def test_non_match_name_edit_ratio_low(self):
        assert self.r_far["name_edit_ratio"] < 0.5

    def test_non_match_addr_jaccard_zero(self):
        # No shared address tokens
        assert self.r_far["addr_jaccard"] == 0.0

    def test_non_match_addr_numeric_overlap_zero(self):
        # "123" ∩ "999" = ∅
        assert self.r_far["addr_numeric_overlap"] == 0.0

    def test_non_match_tfidf_cosine_zero(self):
        # No shared vocabulary → cosine = 0.0
        assert self.r_far["name_tfidf_cosine"] == pytest.approx(0.0, abs=1e-9)

    def test_non_match_country_match(self):
        # Both "us" — country still matches even though names don't
        assert self.r_far["country_match"] == 1

    # --- (S1-002, S2-002): "taj hotels" vs "taj hotel resorts" ---
    #
    # name tokens: {"taj","hotels"} ∩ {"taj","hotel","resorts"}
    #              = {"taj"}  →  1/4 = 0.25
    #   ("hotels" ≠ "hotel" — no abbreviation collapsing)
    #
    # addr tokens: "apollo bunder mumbai" identical on both sides → 1.0
    # addr numeric: no digits in either address → 0.0

    def test_taj_name_jaccard(self):
        # Only "taj" is shared; "hotels" ≠ "hotel"
        assert self.r_taj["name_jaccard"] == pytest.approx(1 / 4, abs=1e-9)

    def test_taj_tfidf_cosine_positive(self):
        # Both names share "taj" → cosine > 0
        assert self.r_taj["name_tfidf_cosine"] > 0.0

    def test_taj_addr_jaccard_full(self):
        # Identical addresses after normalization
        assert self.r_taj["addr_jaccard"] == 1.0

    def test_taj_addr_numeric_overlap_zero(self):
        # Neither address contains digit tokens
        assert self.r_taj["addr_numeric_overlap"] == 0.0

    def test_taj_country_match(self):
        assert self.r_taj["country_match"] == 1

    # --- all feature values are in expected numeric ranges ---

    def test_all_jaccard_features_in_unit_interval(self):
        for col in ["name_jaccard", "addr_jaccard"]:
            assert (self.result[col] >= 0.0).all()
            assert (self.result[col] <= 1.0).all()

    def test_all_ratio_features_in_unit_interval(self):
        for col in ["name_edit_ratio", "name_token_sort", "addr_edit_ratio",
                    "addr_numeric_overlap", "name_tfidf_cosine"]:
            assert (self.result[col] >= 0.0).all()
            assert (self.result[col] <= 1.0).all()

    def test_country_match_is_binary(self):
        assert set(self.result["country_match"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Block 7 — build_pair_features_batch: missing / NaN data robustness
# ---------------------------------------------------------------------------

class TestBuildPairFeaturesBatchMissingData:
    """
    Verifies that build_pair_features_batch() does not crash when
    business_name or business_address is None/NaN in the source files,
    and returns 0.0 for all affected similarity features.

    S1-003: business_name is None  (name_norm → "" after normalize_sources)
    S2-003: business_address is None (address_norm → "" after normalize_sources)
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        s1_raw = _make_source_df([
            ("S1-003", None,          "456 elm ave",   "us"),  # missing name
        ])
        s2_raw = _make_source_df([
            ("S2-003", "omega group", None,             "us"),  # missing address
        ])
        # s3 needs at least one row for fit_tfidf to have a valid corpus
        s3_raw = _make_source_df([
            ("S3-999", "dummy filler", "dummy address", "us"),
        ])

        s1_df, s2_df, s3_df = normalize_sources(s1_raw, s2_raw, s3_raw)
        s1_idx  = build_lookup_index(s1_df)
        s23_idx = pd.concat([build_lookup_index(s2_df), build_lookup_index(s3_df)])
        tfidf   = fit_tfidf(s1_df, s2_df, s3_df)

        pairs = pd.DataFrame([
            {"source1_entity_id": "S1-003", "candidate_entity_id": "S2-003"},
        ])

        self.result = build_pair_features_batch(pairs, s1_idx, s23_idx, tfidf)
        self.row    = self.result.iloc[0]

    def test_does_not_raise(self):
        """The fixture runs build_pair_features_batch without error — this test just confirms it."""
        assert self.result is not None

    def test_output_has_one_row(self):
        assert len(self.result) == 1

    def test_no_nan_in_output(self):
        assert not self.result.isnull().any().any()

    # Missing name on S1 side → _token_jaccard("", ...) and friends all return 0.0
    def test_missing_name_jaccard_is_zero(self):
        assert self.row["name_jaccard"] == 0.0

    def test_missing_name_edit_ratio_is_zero(self):
        assert self.row["name_edit_ratio"] == 0.0

    def test_missing_name_token_sort_is_zero(self):
        assert self.row["name_token_sort"] == 0.0

    def test_missing_name_tfidf_cosine_is_zero(self):
        # Empty string → all-zero TF-IDF vector → cosine = 0.0
        assert self.row["name_tfidf_cosine"] == pytest.approx(0.0, abs=1e-9)

    # Missing address on S2 side → _token_jaccard(..., "") and friends all return 0.0
    def test_missing_addr_jaccard_is_zero(self):
        assert self.row["addr_jaccard"] == 0.0

    def test_missing_addr_edit_ratio_is_zero(self):
        assert self.row["addr_edit_ratio"] == 0.0

    def test_missing_addr_numeric_overlap_is_zero(self):
        assert self.row["addr_numeric_overlap"] == 0.0

    def test_country_match_unaffected_by_missing_fields(self):
        # country is present and identical ("us" == "us") → 1
        assert self.row["country_match"] == 1


if __name__ == "__main__":
    pytest.main(["-v", __file__])
