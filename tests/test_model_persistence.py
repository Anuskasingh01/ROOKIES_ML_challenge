"""
tests/test_model_persistence.py
---------------------------------
Unit tests for model bundle persistence:
- save_model_bundle()
- load_model_bundle()
- integration with predict.py
"""

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.matching_model import save_model_bundle, load_model_bundle
from src.predict import generate_predictions


@pytest.fixture
def dummy_bundle_components():
    # Train a minimal dummy classifier
    X = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)

    # Fit a minimal TF-IDF model
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    tfidf.fit(["acme corp", "beta llc", "gamma inc"])

    threshold = 0.625
    feature_cols = ["feat_name_ratio", "feat_addr_ratio"]
    metadata = {"model_name": "RandomForest", "val_macro_f05": 0.85}

    return clf, threshold, tfidf, feature_cols, metadata


def test_save_and_load_bundle_roundtrip(tmp_path, dummy_bundle_components):
    clf, threshold, tfidf, feature_cols, metadata = dummy_bundle_components
    bundle_path = tmp_path / "model_bundle.joblib"

    saved_path = save_model_bundle(
        model=clf,
        threshold=threshold,
        tfidf_model=tfidf,
        path=bundle_path,
        feature_cols=feature_cols,
        extra_metadata=metadata,
    )

    assert os.path.exists(saved_path)
    assert os.path.getsize(saved_path) > 0

    bundle = load_model_bundle(saved_path, require_tfidf=True)
    assert isinstance(bundle, dict)
    assert bundle["threshold"] == threshold
    assert bundle["feature_cols"] == feature_cols
    assert bundle["metadata"] == metadata

    # Check model functionality
    test_X = np.array([[0.85, 0.85]])
    orig_prob = clf.predict_proba(test_X)[:, 1]
    loaded_prob = bundle["model"].predict_proba(test_X)[:, 1]
    np.testing.assert_allclose(orig_prob, loaded_prob)

    # Check TF-IDF functionality
    test_text = ["acme corp"]
    orig_vec = tfidf.transform(test_text).toarray()
    loaded_vec = bundle["tfidf_model"].transform(test_text).toarray()
    np.testing.assert_allclose(orig_vec, loaded_vec)


def test_load_model_bundle_nonexistent_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(FileNotFoundError, match="Model bundle not found"):
        load_model_bundle(missing_path)


def test_load_model_bundle_invalid_format(tmp_path):
    bad_path = tmp_path / "not_a_dict.joblib"
    joblib.dump(["not", "a", "dict"], bad_path)

    with pytest.raises(ValueError, match="Expected dict bundle"):
        load_model_bundle(bad_path)


def test_load_model_bundle_missing_required_keys(tmp_path):
    missing_model_path = tmp_path / "no_model.joblib"
    joblib.dump({"threshold": 0.5}, missing_model_path)
    with pytest.raises(KeyError, match="missing 'model' key"):
        load_model_bundle(missing_model_path)

    missing_threshold_path = tmp_path / "no_threshold.joblib"
    joblib.dump({"model": "dummy_clf"}, missing_threshold_path)
    with pytest.raises(KeyError, match="missing 'threshold' key"):
        load_model_bundle(missing_threshold_path)


def test_load_model_bundle_legacy_backward_compatibility(tmp_path):
    legacy_path = tmp_path / "legacy.joblib"
    joblib.dump({"model": "legacy_model", "threshold": 0.45}, legacy_path)

    bundle = load_model_bundle(legacy_path, require_tfidf=False)
    assert bundle["model"] == "legacy_model"
    assert bundle["threshold"] == 0.45
    assert bundle["tfidf_model"] is None
    assert bundle["feature_cols"] is None
    assert bundle["metadata"] == {}


def test_load_model_bundle_require_tfidf_flag(tmp_path):
    no_tfidf_path = tmp_path / "no_tfidf.joblib"
    joblib.dump({"model": "clf", "threshold": 0.5}, no_tfidf_path)

    with pytest.raises(KeyError, match="does not contain required 'tfidf_model'"):
        load_model_bundle(no_tfidf_path, require_tfidf=True)


def test_save_creates_parent_directory(tmp_path, dummy_bundle_components):
    clf, threshold, tfidf, _, _ = dummy_bundle_components
    deep_path = tmp_path / "sub" / "dir" / "bundle.joblib"

    assert not (tmp_path / "sub" / "dir").exists()
    saved = save_model_bundle(clf, threshold, tfidf, deep_path)
    assert os.path.exists(saved)


def test_predict_with_saved_bundle(tmp_path):
    # Fit and save a bundle with 8 features
    from src.features import fit_tfidf

    s1_df = pd.DataFrame([{
        "entity_id": "S1-1",
        "business_name": "Acme Widgets Inc",
        "business_address": "123 Main St",
        "country": "US",
    }])
    s2_df = pd.DataFrame([{
        "entity_id": "S2-1",
        "business_name": "Acme Widgets Co",
        "business_address": "123 Main Street",
        "country": "US",
    }])
    s3_df = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    from src.features import normalize_sources
    s1_n, s2_n, s3_n = normalize_sources(s1_df, s2_df, s3_df)
    tfidf = fit_tfidf(s1_n, s2_n, s3_n)

    # Minimal classifier trained on 8 features
    feat_cols = [
        "name_jaccard", "name_edit_ratio", "name_token_sort",
        "addr_jaccard", "addr_edit_ratio", "addr_numeric_overlap",
        "name_tfidf_cosine", "country_match",
    ]
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    dummy_X = np.ones((4, len(feat_cols)))
    dummy_y = np.array([0, 1, 0, 1])
    clf.fit(dummy_X, dummy_y)

    bundle_path = tmp_path / "test_model.joblib"
    save_model_bundle(
        model=clf,
        threshold=0.5,
        tfidf_model=tfidf,
        path=bundle_path,
        feature_cols=feat_cols,
    )

    cands_df = pd.DataFrame([{
        "source1_entity_id": "S1-1",
        "candidate_entity_ids": "S2-1",
    }])

    pred_out = tmp_path / "preds.csv"
    preds = generate_predictions(
        s1_df=s1_df,
        s2_df=s2_df,
        s3_df=s3_df,
        candidate_pairs_df=cands_df,
        model_path=str(bundle_path),
        output_path=str(pred_out),
    )

    assert len(preds) == 1
    assert preds.iloc[0]["source1_entity_id"] == "S1-1"
    assert preds.iloc[0]["candidate_entity_id"] == "S2-1"
    assert 0.0 <= preds.iloc[0]["match_probability"] <= 1.0
    assert preds.iloc[0]["predicted_match"] in (0, 1)
    assert os.path.exists(pred_out)
