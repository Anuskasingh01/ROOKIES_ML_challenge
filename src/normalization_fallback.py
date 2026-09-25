"""
Minimal, dependency-free normalization used ONLY as a fallback so blocking.py
can run standalone (e.g. for a teammate smoke-testing blocking before Person 1's
preprocessing.py is finalized). The real, authoritative normalization is
Person 1's job - if `normalized_name` / `normalized_address` columns are
already present in the frame, this module is never called.
"""
from __future__ import annotations
import re
import unicodedata

_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "lp", "plc",
    "pvt", "private", "pte",
    "gmbh", "sa", "sas", "srl", "bv", "nv",
}

_ADDRESS_ABBREV = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "highway": "hwy", "apartment": "apt", "suite": "ste", "floor": "fl",
    "near": "nr",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _base_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(raw: str) -> str:
    text = _base_clean(raw)
    tokens = [t for t in text.split(" ") if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_address(raw: str) -> str:
    text = _base_clean(raw)
    tokens = [_ADDRESS_ABBREV.get(t, t) for t in text.split(" ") if t]
    return " ".join(tokens)


def extract_postal_code(raw_address: str) -> str:
    """Best-effort postal/PIN code extraction - pure regex, no geocoding."""
    if not raw_address:
        return ""
    text = str(raw_address)
    # 6-digit Indian PIN, 5(+4) digit US ZIP, generic 4-6 digit trailing code
    m = re.search(r"\b(\d{6})\b", text) or re.search(r"\b(\d{5}(?:-\d{4})?)\b", text)
    return m.group(1) if m else ""
