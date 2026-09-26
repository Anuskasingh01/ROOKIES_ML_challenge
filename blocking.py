"""
Person 2 - Candidate Generation / Blocking
============================================

Produces, for every Source 1 entity, the set of Source 2 / Source 3 record
IDs worth scoring with the matching model. This module owns
`output/candidate_pairs.tsv` end to end.

Design goals (in priority order, matching the challenge's own guidance):
  1. Recall first - a true match lost here can never be recovered downstream.
  2. Multiple *complementary*, cheap blocking keys, unioned - never a single
     key, so one noisy field doesn't sink recall for a whole entity.
  3. No Cartesian product, ever. Everything below is built on inverted
     indices, so the cost scales with (records x avg tokens), not with
     (records_S1 x records_S2S3).
  4. No external data / APIs / geocoding of any kind - every signal comes
     only from the fields already in the provided TSVs.
  5. Country is NEVER used to filter or gate candidates - it's an open set
     (France appears only in test) - it is only ever used as a passthrough
     feature later, not here.

Strategies implemented, each returning {s1_entity_id: set(candidate_ids)},
then unioned:
  A. Exact normalized-name block
  B. Token blocking on business name (inverted index, document-frequency
     stoplisted so legal suffixes like "inc"/"ltd"/"pvt" can't blow up a block)
  C. Token blocking on address (same mechanism)
  D. Postal-code block (when extractable)
  E. Street-token block (first meaningful address token, as a fallback when
     there's no postal code)
  F. Character n-gram blocking on the business name (typo / transliteration
     tolerant; only rare n-grams are indexed to keep blocks small)
  G. Sorted-neighborhood pass on normalized name (catches near-duplicates
     that share no exact token, e.g. heavy typos at word boundaries)

Expected input columns (see config.py): entity_id, business_name,
business_address, country, and ideally normalized_name / normalized_address /
postal_code from Person 1's preprocessing.py. Missing normalized columns are
derived on the fly via normalization_fallback.py so this module runs
standalone.
"""
from __future__ import annotations

import heapq
import logging
from collections import defaultdict, Counter
from typing import Dict, Iterable, List, Set

import pandas as pd

try:
    from preprocessing import normalize_dataframe, validate_schema
except ImportError:
    try:
        from src.preprocessing import normalize_dataframe, validate_schema
    except ImportError:
        normalize_dataframe = None
        validate_schema = None

from config import BlockingConfig, DEFAULT_CONFIG
from normalization_fallback import (
    normalize_name as fallback_normalize_name,
    normalize_address as fallback_normalize_address,
    extract_postal_code as fallback_extract_postal_code,
)

logger = logging.getLogger("blocking")


# --------------------------------------------------------------------------
# Preparation
# --------------------------------------------------------------------------

def ensure_normalized_columns(df: pd.DataFrame, cfg: BlockingConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Guarantee normalized_name / normalized_address / postal_code exist.

    If Person 1's preprocessing already produced these columns, they are used
    as-is (this function never overwrites real upstream normalization).
    Otherwise, Person 1's authoritative preprocessing module (preprocessing.py)
    is invoked to generate normalized columns. Falls back to local fallback
    only if preprocessing.py is unavailable.
    """
    df = df.copy()

    # If Person 1's preprocessed columns already exist under Person 1 naming, map them
    if "business_name_normalized" in df.columns and cfg.norm_name_col not in df.columns:
        df[cfg.norm_name_col] = df["business_name_normalized"]
    if "business_address_normalized" in df.columns and cfg.norm_address_col not in df.columns:
        df[cfg.norm_address_col] = df["business_address_normalized"]

    # If normalized columns are missing and Person 1's normalize_dataframe is available, run it
    if (cfg.norm_name_col not in df.columns or cfg.norm_address_col not in df.columns) and normalize_dataframe is not None:
        try:
            normed = normalize_dataframe(df)
            if cfg.norm_name_col not in df.columns:
                df[cfg.norm_name_col] = normed.get(cfg.norm_name_col, normed.get("business_name_normalized", ""))
            if cfg.norm_address_col not in df.columns:
                df[cfg.norm_address_col] = normed.get(cfg.norm_address_col, normed.get("business_address_normalized", ""))
            if cfg.postal_col not in df.columns and "postal_code" in normed.columns:
                df[cfg.postal_col] = normed["postal_code"]
        except Exception as e:
            logger.warning("Error running Person 1 normalize_dataframe, using fallback: %s", e)

    # Local fallback for any still-missing columns
    if cfg.norm_name_col not in df.columns:
        df[cfg.norm_name_col] = df[cfg.name_col].map(fallback_normalize_name)

    if cfg.norm_address_col not in df.columns:
        df[cfg.norm_address_col] = df[cfg.address_col].map(fallback_normalize_address)

    if cfg.postal_col not in df.columns:
        df[cfg.postal_col] = df[cfg.address_col].map(fallback_extract_postal_code)
    else:
        df[cfg.postal_col] = df[cfg.postal_col].fillna("").astype(str)

    df[cfg.norm_name_col] = df[cfg.norm_name_col].fillna("").astype(str)
    df[cfg.norm_address_col] = df[cfg.norm_address_col].fillna("").astype(str)
    if cfg.country_col in df.columns:
        df[cfg.country_col] = df[cfg.country_col].fillna("").astype(str)
    else:
        df[cfg.country_col] = ""

    return df


def _tokens(text: str, min_len: int) -> List[str]:
    if not text:
        return []
    return [t for t in text.split(" ") if len(t) >= min_len]


def _char_ngrams(text: str, n: int) -> Set[str]:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


# --------------------------------------------------------------------------
# Inverted-index construction (built once per candidate pool = S2 union S3)
# --------------------------------------------------------------------------

class InvertedIndex:
    """token/ngram -> list of entity_ids, with document-frequency stoplisting.

    Built once from the S2+S3 pool and reused for every S1 lookup, which is
    what keeps this whole pipeline sub-quadratic: index construction is
    O(records x tokens_per_record), and each S1 lookup is
    O(tokens_in_s1_record x avg_postings_per_token).
    """

    def __init__(self, max_document_frequency: float, max_absolute_postings: int = 2000):
        self._index: Dict[str, List[str]] = defaultdict(list)
        self._max_df = max_document_frequency
        self._max_absolute = max_absolute_postings
        self._n_docs = 0
        self._stoplisted: Set[str] = set()

    def add(self, entity_id: str, keys: Iterable[str]) -> None:
        for k in set(keys):  # dedupe within a single record's own keys
            self._index[k].append(entity_id)

    def finalize_with_pool_size(self, n_docs: int) -> None:
        """Call once after all `add()` calls, passing the number of records
        the index was built over. Drops keys that are too common to be
        useful ("block purging") instead of ever hand-listing legal suffixes
        or common street words - this generalizes across countries,
        including ones unseen in training (e.g. France).

        Two caps are enforced, whichever is stricter:
          - relative: max_df * n_docs (e.g. "drop if in >2% of records")
          - absolute: max_absolute_postings (protects against low-cardinality
            pools where even a small percentage is still thousands of rows)
        """
        self._n_docs = n_docs
        relative_cap = max(1, int(self._max_df * max(1, n_docs)))
        cap = min(relative_cap, self._max_absolute)
        for k, ids in list(self._index.items()):
            if len(ids) > cap:
                self._stoplisted.add(k)
                del self._index[k]

    def lookup(self, keys: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for k in keys:
            out.update(self._index.get(k, ()))
        return out

    def lookup_min_shared(self, keys: Iterable[str], min_shared: int) -> Set[str]:
        """Return ids appearing in >= min_shared of the given keys' postings.
        Used for n-gram blocking, where a single shared trigram is noise but
        several shared rare trigrams is a strong typo-tolerant signal."""
        counts: Counter = Counter()
        for k in keys:
            for eid in self._index.get(k, ()):
                counts[eid] += 1
        return {eid for eid, c in counts.items() if c >= min_shared}

    @property
    def stoplisted_keys(self) -> Set[str]:
        return self._stoplisted


def _build_pool(s2_df: pd.DataFrame, s3_df: pd.DataFrame, cfg: BlockingConfig) -> pd.DataFrame:
    pool = pd.concat([s2_df, s3_df], ignore_index=True, sort=False)
    return pool


# --------------------------------------------------------------------------
# Individual blocking strategies -> {s1_id: set(candidate_ids)}
# --------------------------------------------------------------------------

def block_exact_name(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    exact_index: Dict[str, List[str]] = defaultdict(list)
    for eid, name in zip(pool[cfg.id_col], pool[cfg.norm_name_col]):
        if name:
            exact_index[name].append(eid)

    result: Dict[str, Set[str]] = {}
    for eid, name in zip(s1[cfg.id_col], s1[cfg.norm_name_col]):
        result[eid] = set(exact_index.get(name, ())) if name else set()
    return result


def block_name_tokens(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    idx = InvertedIndex(cfg.max_token_document_frequency, cfg.max_postings_per_key)
    for eid, name in zip(pool[cfg.id_col], pool[cfg.norm_name_col]):
        idx.add(eid, _tokens(name, cfg.min_token_length))
    idx.finalize_with_pool_size(len(pool))

    result: Dict[str, Set[str]] = {}
    for eid, name in zip(s1[cfg.id_col], s1[cfg.norm_name_col]):
        result[eid] = idx.lookup(_tokens(name, cfg.min_token_length))
    return result


def block_address_tokens(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    idx = InvertedIndex(cfg.max_token_document_frequency, cfg.max_postings_per_key)
    for eid, addr in zip(pool[cfg.id_col], pool[cfg.norm_address_col]):
        idx.add(eid, _tokens(addr, cfg.min_token_length))
    idx.finalize_with_pool_size(len(pool))

    result: Dict[str, Set[str]] = {}
    for eid, addr in zip(s1[cfg.id_col], s1[cfg.norm_address_col]):
        result[eid] = idx.lookup(_tokens(addr, cfg.min_token_length))
    return result


def block_postal_code(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    if not cfg.use_postal_code_block:
        return {eid: set() for eid in s1[cfg.id_col]}

    index: Dict[str, List[str]] = defaultdict(list)
    for eid, pin in zip(pool[cfg.id_col], pool[cfg.postal_col]):
        if pin:
            index[pin].append(eid)

    result: Dict[str, Set[str]] = {}
    for eid, pin in zip(s1[cfg.id_col], s1[cfg.postal_col]):
        result[eid] = set(index.get(pin, ())) if pin else set()
    return result


def block_street_token(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    """Fallback address block for records with no postal code: block on the
    first two address tokens (typically house-number-ish + street name),
    document-frequency stoplisted the same way as name tokens."""
    if not cfg.use_street_token_block:
        return {eid: set() for eid in s1[cfg.id_col]}

    def street_key(addr: str) -> List[str]:
        toks = _tokens(addr, cfg.min_token_length)
        return toks[:2]

    idx = InvertedIndex(cfg.max_token_document_frequency, cfg.max_postings_per_key)
    for eid, addr in zip(pool[cfg.id_col], pool[cfg.norm_address_col]):
        idx.add(eid, street_key(addr))
    idx.finalize_with_pool_size(len(pool))

    result: Dict[str, Set[str]] = {}
    for eid, addr in zip(s1[cfg.id_col], s1[cfg.norm_address_col]):
        result[eid] = idx.lookup(street_key(addr))
    return result


def block_name_ngrams(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    field_col = cfg.norm_name_col if cfg.ngram_field == "name" else None

    def field_text(row_name: str, row_addr: str) -> str:
        if cfg.ngram_field == "name":
            return row_name
        return f"{row_name} {row_addr}"

    idx = InvertedIndex(cfg.max_ngram_document_frequency, cfg.max_ngram_postings_per_key)
    for eid, name, addr in zip(pool[cfg.id_col], pool[cfg.norm_name_col], pool[cfg.norm_address_col]):
        text = field_text(name, addr)
        idx.add(eid, _char_ngrams(text, cfg.ngram_size))
    idx.finalize_with_pool_size(len(pool))

    result: Dict[str, Set[str]] = {}
    for eid, name, addr in zip(s1[cfg.id_col], s1[cfg.norm_name_col], s1[cfg.norm_address_col]):
        text = field_text(name, addr)
        grams = _char_ngrams(text, cfg.ngram_size)
        result[eid] = idx.lookup_min_shared(grams, cfg.min_shared_ngrams)
    return result


def block_sorted_neighborhood(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockingConfig) -> Dict[str, Set[str]]:
    """Classic sorted-neighborhood: sort S1 and the candidate pool together
    by normalized name, slide a fixed window, and pair everything inside a
    window. O(n log n), catches near-duplicates that share no exact token
    (e.g. "acme robotic" vs "acme robotics") without any per-token index."""
    w = cfg.sorted_neighborhood_window

    combined = pd.concat(
        [
            s1[[cfg.id_col, cfg.norm_name_col]].assign(_src="s1"),
            pool[[cfg.id_col, cfg.norm_name_col]].assign(_src="pool"),
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(cfg.norm_name_col, kind="mergesort").reset_index(drop=True)

    result: Dict[str, Set[str]] = {eid: set() for eid in s1[cfg.id_col]}
    n = len(combined)
    ids = combined[cfg.id_col].to_numpy()
    srcs = combined["_src"].to_numpy()

    for i in range(n):
        if srcs[i] != "s1":
            continue
        s1_id = ids[i]
        lo, hi = max(0, i - w), min(n, i + w + 1)
        for j in range(lo, hi):
            if j == i or srcs[j] != "pool":
                continue
            result[s1_id].add(ids[j])
    return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "exact_name": block_exact_name,
    "name_tokens": block_name_tokens,
    "address_tokens": block_address_tokens,
    "postal_code": block_postal_code,
    "street_token": block_street_token,
    "name_ngrams": block_name_ngrams,
    "sorted_neighborhood": block_sorted_neighborhood,
}


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    cfg: BlockingConfig = DEFAULT_CONFIG,
    strategies: Iterable[str] = tuple(STRATEGY_REGISTRY.keys()),
    return_strategy_breakdown: bool = False,
):
    """Main entry point. Returns a long-form DataFrame with one row per
    (source1_entity_id, candidate_entity_id) pair - the exact set later fed
    to the matching model. Also returns per-strategy stats when requested,
    useful for the recall/reduction-ratio report.
    """
    s1 = ensure_normalized_columns(s1_df, cfg)
    s2 = ensure_normalized_columns(s2_df, cfg)
    s3 = ensure_normalized_columns(s3_df, cfg)
    pool = _build_pool(s2, s3, cfg)

    logger.info("S1=%d  S2=%d  S3=%d  pool=%d", len(s1), len(s2), len(s3), len(pool))

    # IMPORTANT: we track a *vote count* per (s1_id, candidate_id) - how many
    # independent strategies flagged it - rather than a plain set. A cap
    # truncated by arbitrary ID sort order can silently discard a true match
    # that happened to sort late; truncating by "fewest strategies agree on
    # it first" is both safer for recall and a meaningful confidence signal
    # Person 3 can use as a feature (see candidate_pairs_scored.tsv below).
    per_entity_votes: Dict[str, Counter] = defaultdict(Counter)
    per_strategy_counts: Dict[str, int] = {}

    for name in strategies:
        fn = STRATEGY_REGISTRY[name]
        logger.info("Running blocking strategy: %s", name)
        strategy_result = fn(s1, pool, cfg)

        # Interim per-strategy cap: bounds memory/CPU from any single
        # pathological block BEFORE it ever reaches the union step. Looser
        # than the final max_candidates_per_entity so it practically never
        # binds on healthy data. Deterministic tie-break (sorted) since,
        # within one strategy, every candidate is equally "voted" and there
        # is no agreement signal yet to prioritize by.
        interim_cap = cfg.max_candidates_per_entity_per_strategy
        n_strategy_capped = 0
        total_pairs = 0
        for eid, cands in strategy_result.items():
            if len(cands) > interim_cap:
                cands = sorted(cands)[:interim_cap]
                n_strategy_capped += 1
            total_pairs += len(cands)
            per_entity_votes[eid].update(cands)

        per_strategy_counts[name] = total_pairs
        logger.info("  -> %d pairs contributed%s", total_pairs,
                     f" ({n_strategy_capped} entities hit interim cap)" if n_strategy_capped else "")

    # Final safety cap: applied post-union, and now rank-preserving - keep
    # the candidates with the MOST strategy agreement first, only truncating
    # (deterministically, ties broken by id) once genuinely oversized.
    rows = []
    scored_rows = []
    n_capped = 0
    for eid in s1[cfg.id_col]:
        votes = per_entity_votes.get(eid, None)
        if not votes:
            rows.append((eid, []))
            continue
        cap = cfg.max_candidates_per_entity
        if len(votes) > cap:
            n_capped += 1
            # heapq.nlargest is O(n log cap) instead of a full O(n log n)
            # sort - matters a lot when a pathological block leaves an
            # entity with tens of thousands of vote-counter entries.
            top = heapq.nlargest(cap, votes.items(), key=lambda kv: kv[1])
            # top has only `cap` items now - cheap to fully sort for a
            # deterministic final tie-break on id.
            ranked = sorted(top, key=lambda kv: (-kv[1], kv[0]))
        else:
            ranked = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
        cand_ids = sorted(cid for cid, _ in ranked)
        rows.append((eid, cand_ids))
        for cid, vote_count in ranked:
            scored_rows.append((eid, cid, vote_count))

    if n_capped:
        logger.warning(
            "%d / %d Source 1 entities hit the max_candidates_per_entity cap (%d) - "
            "consider tightening max_token_document_frequency if this is a large fraction.",
            n_capped, len(s1), cfg.max_candidates_per_entity,
        )

    out = pd.DataFrame(rows, columns=[cfg.id_col, "candidate_ids"])
    out = out.rename(columns={cfg.id_col: "source1_entity_id"})
    out.attrs["scored_rows"] = scored_rows  # stashed for write_candidate_pairs_tsv's sidecar file

    if return_strategy_breakdown:
        return out, per_strategy_counts
    return out


def candidates_to_tsv_rows(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Convert the internal list-column representation into the submission
    format: one row per S1 entity, candidate_entity_ids as a comma-joined
    string, empty string (not NaN) when there are no candidates."""
    out = candidates_df.copy()
    out["candidate_entity_ids"] = out["candidate_ids"].apply(lambda ids: ",".join(ids))
    return out[["source1_entity_id", "candidate_entity_ids"]]


def write_candidate_pairs_tsv(
    candidates_df: pd.DataFrame, out_path: str, write_scored_sidecar: bool = True
) -> None:
    """Writes the required output/candidate_pairs.tsv (submission format).

    Also writes an OPTIONAL, non-submission sidecar
    `<out_path stem>_scored.tsv` with one row per (s1_id, candidate_id,
    n_strategies_agreeing) - not part of the challenge's required output,
    but a free, already-computed feature for Person 3's matching model
    ("how many independent blocking strategies flagged this pair" is a
    surprisingly strong prior signal). Safe to ignore if unused.
    """
    tsv_df = candidates_to_tsv_rows(candidates_df)
    tsv_df.to_csv(out_path, sep="\t", index=False)
    logger.info("Wrote %s (%d rows)", out_path, len(tsv_df))

    scored_rows = candidates_df.attrs.get("scored_rows")
    if write_scored_sidecar and scored_rows:
        import os
        stem, ext = os.path.splitext(out_path)
        sidecar_path = f"{stem}_scored{ext}"
        scored_df = pd.DataFrame(
            scored_rows, columns=["source1_entity_id", "candidate_entity_id", "n_strategies_agreeing"]
        )
        scored_df.to_csv(sidecar_path, sep="\t", index=False)
        logger.info("Wrote optional sidecar %s (%d rows, for Person 3's features - not scored)", sidecar_path, len(scored_df))
