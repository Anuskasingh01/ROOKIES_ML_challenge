"""
Shared configuration for the blocking / candidate-generation stage (Person 2).

Nothing here reaches out to the network or any external service - all tunables
control purely local, in-memory index-building over the TSVs Person 1 hands off.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BlockingConfig:
    # ---- I/O -------------------------------------------------------------
    id_col: str = "entity_id"
    name_col: str = "business_name"
    address_col: str = "business_address"
    country_col: str = "country"

    # Columns Person 1's preprocessing module is expected to add. The loader
    # falls back to deriving these on the fly if they're missing, so this
    # module can run standalone before Person 1's output is finalized.
    norm_name_col: str = "normalized_name"
    norm_address_col: str = "normalized_address"
    postal_col: str = "postal_code"

    # ---- Token blocking ----------------------------------------------------
    # A name/address token is dropped as a blocking key if it appears in more
    # than this fraction of ALL records in a given source pool - this auto-
    # stoplists legal-suffix noise ("inc", "ltd", "pvt", "co") without a
    # hand-maintained list, and without ever touching country-specific words.
    max_token_document_frequency: float = 0.02
    min_token_length: int = 2
    # Absolute "block purging" cap, applied in ADDITION to the relative
    # document-frequency cutoff above. Protects against pathological cases
    # a relative threshold can miss - e.g. a pool with very low name
    # cardinality (many records sharing near-identical names/addresses) -
    # where even a "rare" (by percentage) token still has a huge absolute
    # posting list. Whichever cap (relative or absolute) is stricter wins.
    max_postings_per_key: int = 500

    # ---- Character n-gram blocking (typo / transliteration tolerance) ----
    ngram_size: int = 3
    # Only n-grams rarer than this (as a fraction of records) are used as
    # index keys - common n-grams ("ing", "com") would blow up block size
    # for near-zero recall benefit.
    max_ngram_document_frequency: float = 0.05
    min_shared_ngrams: int = 4          # candidate must share >= this many rare n-grams
    ngram_field: str = "name"           # "name" or "name_address" (concat)
    max_ngram_postings_per_key: int = 800  # absolute purge cap, see max_postings_per_key

    # ---- Sorted neighborhood (catches leading-character typos / reorderings) --
    sorted_neighborhood_window: int = 5

    # ---- Address key blocking ---------------------------------------------
    # Used only when a postal code is present; falls back gracefully when not.
    use_postal_code_block: bool = True
    use_street_token_block: bool = True

    # ---- Safety valve -------------------------------------------------------
    # Hard ceiling on candidates kept per Source-1 entity after all strategies
    # are unioned. This only trims pathological outliers (a S1 record that,
    # despite document-frequency stoplisting, still matches thousands of
    # records) - it is NOT meant to bind in the common case, and it is applied
    # AFTER union, never per-strategy, so no single strategy is silently
    # capped before contributing its recall.
    max_candidates_per_entity: int = 300
    # Interim per-strategy, per-entity cap applied BEFORE strategies are
    # unioned (deliberately looser than max_candidates_per_entity so a
    # single strategy can't itself balloon memory/CPU on a pathological
    # block before the final union+cap ever runs).
    max_candidates_per_entity_per_strategy: int = 250

    # Reproducibility - candidate ordering is deterministic (sorted), so no
    # actual randomness is used in this stage, but kept here for parity with
    # the rest of the pipeline's seed handling.
    random_seed: int = 42


DEFAULT_CONFIG = BlockingConfig()


@dataclass(frozen=True)
class Paths:
    train_dir: Path = Path("dataset/train")
    test_dir: Path = Path("dataset/test")
    output_dir: Path = Path("output")

    @property
    def train_source1(self) -> Path: return self.train_dir / "train_source1.tsv"
    @property
    def train_source2(self) -> Path: return self.train_dir / "train_source2.tsv"
    @property
    def train_source3(self) -> Path: return self.train_dir / "train_source3.tsv"
    @property
    def train_ground_truth(self) -> Path: return self.train_dir / "train_ground_truth.tsv"

    @property
    def test_source1(self) -> Path: return self.test_dir / "test_source1.tsv"
    @property
    def test_source2(self) -> Path: return self.test_dir / "test_source2.tsv"
    @property
    def test_source3(self) -> Path: return self.test_dir / "test_source3.tsv"

    @property
    def candidate_pairs_out(self) -> Path: return self.output_dir / "candidate_pairs.tsv"
