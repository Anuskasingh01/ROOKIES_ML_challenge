"""
src/features.py
---------------
Feature engineering for business entity resolution.

Pipeline startup order (call once before scoring any pairs):
    1. normalize_sources()       — add *_norm columns to each source DataFrame
    2. build_lookup_index()      — build entity_id-indexed DataFrames (×3)
    3. fit_tfidf()               — fit TF-IDF on the union of all name_norm values
    4. explode_candidate_pairs() — expand candidate_pairs.tsv to one row per pair
    5. build_pair_features_batch() — compute all features in a single vectorized pass

NOTE on .apply() calls (token_jaccard, edit_ratio, token_sort_ratio,
numeric_overlap): these are row-wise Python-level operations and are the
current performance bottleneck for large candidate sets. They are correct
and readable at present, but should be revisited if blocking produces
candidate counts in the millions — at that scale, parallelisation via
joblib, Cython extensions, or Rapids cuDF string ops would be appropriate.
Person 2 to confirm actual candidate volumes once blocking is finalized.
"""

from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz as _rfuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import (
    normalize_name,
    normalize_address,
    normalize_country,
    extract_address_numbers,
)

# ---------------------------------------------------------------------------
# 1. Upfront normalization
# ---------------------------------------------------------------------------

def normalize_sources(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add *_norm columns to each source DataFrame in-place (on copies).

    Called ONCE at pipeline startup — never inside a per-pair loop.

    Returns
    -------
    Tuple of (s1_df, s2_df, s3_df) each with three new columns:
        name_norm, address_norm, country_norm
    All columns are guaranteed to be str with "" for missing values (never NaN).

    Notes
    -----
    normalize_name() propagates NaN when business_name is missing because its
    first operation (str.normalize('NFKC')) does not call fillna internally.
    We apply .fillna("") explicitly here so all three norm columns are
    consistently empty-string for unknown values across every source.
    normalize_address() and normalize_country() already call fillna internally,
    but we apply it again for safety and symmetry.
    """
    dfs = []
    for df in (s1_df, s2_df, s3_df):
        df = df.copy()
        df["name_norm"]    = normalize_name(df["business_name"]).fillna("")
        df["address_norm"] = normalize_address(df["business_address"]).fillna("")
        df["country_norm"] = normalize_country(df["country"]).fillna("")
        dfs.append(df)
    return tuple(dfs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 2. Lookup index
# ---------------------------------------------------------------------------

def build_lookup_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame indexed by entity_id, containing only the three
    norm columns.  Used for O(1) batch-merge access in build_pair_features_batch().

    Parameters
    ----------
    df : pd.DataFrame
        A source DataFrame that has already been through normalize_sources().

    Returns
    -------
    pd.DataFrame with index=entity_id and columns [name_norm, address_norm, country_norm].
    """
    return df.set_index("entity_id")[["name_norm", "address_norm", "country_norm"]]


# ---------------------------------------------------------------------------
# 3. Candidate pair explosion
# ---------------------------------------------------------------------------

def explode_candidate_pairs(candidate_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a candidate_pairs.tsv DataFrame from one-row-per-source1-entity to
    one-row-per-candidate-pair.

    Parameters
    ----------
    candidate_pairs_df : pd.DataFrame
        Loaded candidate_pairs.tsv with columns:
            source1_entity_id   : str
            candidate_entity_ids: str  (comma-separated, may be "" for singletons)

    Returns
    -------
    pd.DataFrame with columns [source1_entity_id, candidate_entity_id].
    Rows where candidate_entity_ids was empty are excluded (singletons
    have no candidates to score).

    Examples
    --------
    Input row:  S1-00001 | "S2-00047,S3-00812"
    Output rows:
        S1-00001 | S2-00047
        S1-00001 | S3-00812
    """
    df = candidate_pairs_df.copy()

    # Support Person 2's in-memory column name 'candidate_ids'
    if "candidate_ids" in df.columns and "candidate_entity_ids" not in df.columns:
        df = df.rename(columns={"candidate_ids": "candidate_entity_ids"})

    if len(df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])

    # Check if elements are already lists
    first_val = df["candidate_entity_ids"].dropna().iloc[0] if len(df["candidate_entity_ids"].dropna()) > 0 else ""
    if isinstance(first_val, (list, tuple, set)):
        df = df.explode("candidate_entity_ids")
        df["candidate_entity_id"] = df["candidate_entity_ids"].fillna("").astype(str).str.strip()
        df = df[df["candidate_entity_id"] != ""]
        return df[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)

    # Filter out rows with no candidates before exploding to avoid empty strings
    # in the result. Keep NaN and "" both as "no candidates".
    df["candidate_entity_ids"] = df["candidate_entity_ids"].fillna("")
    df = df[df["candidate_entity_ids"] != ""]

    # Split the comma-separated string into a list, strip whitespace around IDs
    df["candidate_entity_id"] = df["candidate_entity_ids"].str.split(",")
    df = df.explode("candidate_entity_id")
    df["candidate_entity_id"] = df["candidate_entity_id"].str.strip()

    # Drop any empty strings that could appear after stripping (e.g. trailing comma)
    df = df[df["candidate_entity_id"] != ""]

    return df[["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. TF-IDF model
# ---------------------------------------------------------------------------

def fit_tfidf(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the union of all name_norm values across all
    three sources.  Must be called after normalize_sources().

    Parameters
    ----------
    s1_df, s2_df, s3_df : pd.DataFrame
        Source DataFrames that already contain a name_norm column.

    Returns
    -------
    TfidfVectorizer
        A fitted model ready for .transform() calls.

    Notes
    -----
    - analyzer="word", ngram_range=(1, 2): unigrams + bigrams.
      Bigrams capture partial phrase matches (e.g. "bank of" in noisy names).
    - Empty strings ("") produce all-zero vectors → cosine similarity = 0.0
      automatically, so no special-casing is needed for missing names.
    - sublinear_tf=True dampens the effect of very frequent tokens (e.g. "ltd",
      "pvt") without removing them entirely, complementing the idf weighting.
    """
    all_names = pd.concat(
        [s1_df["name_norm"], s2_df["name_norm"], s3_df["name_norm"]],
        ignore_index=True,
    )
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    vectorizer.fit(all_names)
    return vectorizer


# ---------------------------------------------------------------------------
# 5. Per-feature helper functions
# ---------------------------------------------------------------------------

def _token_jaccard(a: str, b: str) -> float:
    """
    Jaccard similarity over whitespace-tokenised word sets.

    Empty/missing string on either side → 0.0 (no shared tokens).

    NOTE: this is called via .apply() row-wise — see module-level note
    about future vectorisation if candidate volume is very large.
    """
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _edit_ratio(a: str, b: str) -> float:
    """
    Normalised edit-distance similarity using rapidfuzz.fuzz.ratio().
    Returns a float in [0, 1] (rapidfuzz returns 0–100; we divide by 100).

    Empty/missing string on either side → 0.0 (unknown, treat as dissimilar).
    Both empty → 0.0 (no information; avoid spuriously boosting a pair).

    Uses rapidfuzz instead of difflib for 10–100× speed improvement on large
    candidate sets with no API change.

    NOTE: called via .apply() — see module-level note on future vectorisation.
    """
    if not a or not b:
        return 0.0
    return _rfuzz.ratio(a, b) / 100.0


def _token_sort_ratio(a: str, b: str) -> float:
    """
    rapidfuzz token_sort_ratio: sorts tokens alphabetically before comparing,
    making it robust to word-order transpositions (e.g. "Bank of India" vs
    "India Bank of").  Returns float in [0, 1].

    Empty/missing string on either side → 0.0.

    NOTE: called via .apply() — see module-level note on future vectorisation.
    """
    if not a or not b:
        return 0.0
    return _rfuzz.token_sort_ratio(a, b) / 100.0


def _numeric_overlap(a: str, b: str) -> float:
    """
    Jaccard similarity over the sets of digit-only tokens in each string.
    Captures street numbers and PIN codes even when abbreviated tokens differ
    (e.g. "St" vs "Street" — the numbers still match).

    Empty/missing string or no digits on either side → 0.0.

    NOTE: called via .apply() — see module-level note on future vectorisation.
    """
    if not a or not b:
        return 0.0
    # re.findall is used directly (consistent with extract_address_numbers logic)
    sa = set(re.findall(r"\d+", a))
    sb = set(re.findall(r"\d+", b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# 6. Batch feature computation  (primary hot path)
# ---------------------------------------------------------------------------

def build_pair_features_batch(
    pairs_df: pd.DataFrame,
    s1_idx: pd.DataFrame,
    s23_idx: pd.DataFrame,
    tfidf_model: TfidfVectorizer,
) -> pd.DataFrame:
    """
    Compute all similarity features for every candidate pair in one batched pass.

    This is the primary hot path — designed to avoid per-pair Python-level
    lookups.  All norm strings are attached via pd.merge (one merge per source),
    and TF-IDF cosine similarity is computed fully vectorised using sparse
    matrix operations.  Row-wise .apply() is used for the remaining string
    similarity features (see module-level note on future optimisation).

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Exploded pairs with columns [source1_entity_id, candidate_entity_id].
        Typically the output of explode_candidate_pairs().
    s1_idx : pd.DataFrame
        Lookup index for source 1.  Output of build_lookup_index(s1_df).
    s23_idx : pd.DataFrame
        Combined lookup index for source 2 and source 3.
        Build with: pd.concat([build_lookup_index(s2_df), build_lookup_index(s3_df)]).
    tfidf_model : TfidfVectorizer
        Fitted model from fit_tfidf().

    Returns
    -------
    pd.DataFrame
        One row per candidate pair.  Contains the original ID columns plus:
            name_jaccard        : float  Jaccard over name tokens
            name_edit_ratio     : float  Normalised edit distance (rapidfuzz)
            name_token_sort     : float  Token-sort ratio (rapidfuzz)
            addr_jaccard        : float  Jaccard over address tokens
            addr_edit_ratio     : float  Normalised edit distance on addresses
            addr_numeric_overlap: float  Jaccard over digit tokens in address
            name_tfidf_cosine   : float  TF-IDF cosine similarity on names
            country_match       : int    1 if country_norm values are equal else 0

    Notes
    -----
    Merge strategy: we use left merges so that pairs referencing an entity_id
    not present in the index (which should not happen in a well-formed pipeline,
    but can occur during unit testing with synthetic data) produce NaN rather
    than silently dropping the row.  Those NaN norm values are filled with ""
    immediately after the merges so all downstream helpers receive clean strings.
    """
    # ------------------------------------------------------------------
    # Step 1: Attach source-1 norm fields
    # ------------------------------------------------------------------
    df = pairs_df.copy()
    df = df.merge(
        s1_idx.rename(columns={
            "name_norm":    "name_norm_s1",
            "address_norm": "address_norm_s1",
            "country_norm": "country_norm_s1",
        }),
        left_on="source1_entity_id",
        right_index=True,
        how="left",
    )

    # ------------------------------------------------------------------
    # Step 2: Attach source-2/3 norm fields
    # ------------------------------------------------------------------
    df = df.merge(
        s23_idx.rename(columns={
            "name_norm":    "name_norm_s2",
            "address_norm": "address_norm_s2",
            "country_norm": "country_norm_s2",
        }),
        left_on="candidate_entity_id",
        right_index=True,
        how="left",
    )

    # ------------------------------------------------------------------
    # Step 3: Safety fill — ensure no NaN reaches the helpers
    # (can occur if a candidate ID is absent from the index)
    # ------------------------------------------------------------------
    norm_cols = [
        "name_norm_s1", "address_norm_s1", "country_norm_s1",
        "name_norm_s2", "address_norm_s2", "country_norm_s2",
    ]
    df[norm_cols] = df[norm_cols].fillna("")

    # ------------------------------------------------------------------
    # Step 4: Row-wise string similarity features
    #
    # NOTE (future optimisation): the four .apply() calls below are
    # Python-level row iterations.  They are the dominant cost for large
    # candidate sets.  If Person 2's blocking produces O(millions) of
    # pairs, consider replacing these with parallelised approaches
    # (joblib Parallel + rapidfuzz.process, or pandas-on-Spark / cuDF).
    # Revisit once actual candidate counts are known.
    # ------------------------------------------------------------------

    # Name features
    df["name_jaccard"] = df.apply(
        lambda r: _token_jaccard(r["name_norm_s1"], r["name_norm_s2"]), axis=1
    )
    df["name_edit_ratio"] = df.apply(
        lambda r: _edit_ratio(r["name_norm_s1"], r["name_norm_s2"]), axis=1
    )
    df["name_token_sort"] = df.apply(
        lambda r: _token_sort_ratio(r["name_norm_s1"], r["name_norm_s2"]), axis=1
    )

    # Address features
    df["addr_jaccard"] = df.apply(
        lambda r: _token_jaccard(r["address_norm_s1"], r["address_norm_s2"]), axis=1
    )
    df["addr_edit_ratio"] = df.apply(
        lambda r: _edit_ratio(r["address_norm_s1"], r["address_norm_s2"]), axis=1
    )
    df["addr_numeric_overlap"] = df.apply(
        lambda r: _numeric_overlap(r["address_norm_s1"], r["address_norm_s2"]), axis=1
    )

    # ------------------------------------------------------------------
    # Step 5: TF-IDF cosine similarity on names — fully vectorised
    #
    # transform() accepts a list/Series of strings and returns a sparse
    # matrix.  element_wise_cosine avoids the full N×N matrix product by
    # computing only the diagonal (each row paired with its counterpart).
    # ------------------------------------------------------------------
    tfidf_s1 = tfidf_model.transform(df["name_norm_s1"])
    tfidf_s2 = tfidf_model.transform(df["name_norm_s2"])

    # Row-paired dot product: equivalent to cosine_similarity(A, B).diagonal()
    # but without materialising the full N×N matrix — O(N × vocab) not O(N²).
    norms_s1 = np.asarray(tfidf_s1.power(2).sum(axis=1)).flatten() ** 0.5
    norms_s2 = np.asarray(tfidf_s2.power(2).sum(axis=1)).flatten() ** 0.5
    dot_products = np.asarray(tfidf_s1.multiply(tfidf_s2).sum(axis=1)).flatten()

    # Safe division: pre-allocate zeros then only divide where denom > 0.
    # Using np.where(denom > 0, dot/denom, 0) triggers a RuntimeWarning because
    # NumPy evaluates both branches before selecting — the 0/0 case fires even
    # though its result is discarded.  The explicit assignment below avoids that.
    denom = norms_s1 * norms_s2
    cosine = np.zeros(len(denom), dtype=np.float64)
    valid = denom > 0
    cosine[valid] = dot_products[valid] / denom[valid]
    df["name_tfidf_cosine"] = cosine

    # ------------------------------------------------------------------
    # Step 6: Country exact match (boolean as int)
    # ------------------------------------------------------------------
    df["country_match"] = (
        df["country_norm_s1"] == df["country_norm_s2"]
    ).astype(int)

    # ------------------------------------------------------------------
    # Step 7: Drop raw norm string columns — callers only need features
    # ------------------------------------------------------------------
    df = df.drop(columns=norm_cols)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Single-pair wrapper (debugging / interactive inspection only)
# ---------------------------------------------------------------------------

def build_pair_features(
    s1_id: str,
    other_id: str,
    s1_idx: pd.DataFrame,
    s23_idx: pd.DataFrame,
    tfidf_model: TfidfVectorizer,
) -> dict:
    """
    Compute features for a single candidate pair.

    This is a thin wrapper around build_pair_features_batch() intended for
    debugging and interactive inspection only — not the production hot path.

    Returns
    -------
    dict mapping feature name → float value.
    """
    single_pair = pd.DataFrame([{
        "source1_entity_id":  s1_id,
        "candidate_entity_id": other_id,
    }])
    result = build_pair_features_batch(single_pair, s1_idx, s23_idx, tfidf_model)
    return result.iloc[0].to_dict()
