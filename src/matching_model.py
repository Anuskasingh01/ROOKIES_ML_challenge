"""
src/matching_model.py
----------------------
Ground truth conversion, model training, threshold tuning, and prediction
generation for the pairwise entity-matching classifier.
"""

import os
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from src.evaluation import _entity_f05, optimize_entity_threshold


def build_training_labels(
    ground_truth_df: pd.DataFrame,
    source2_ids: pd.Series,
    source3_ids: pd.Series,
    negatives_per_positive: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convert train_ground_truth.tsv into row-level (source1_entity_id,
    candidate_entity_id, label) pairs for training a pairwise classifier.

    Positives: every ID listed in matched_entity_ids (label=1).
    Negatives: randomly sampled S2/S3 entity_ids that are NOT a true match
    for that source1_entity_id (label=0).

    NOTE (known limitation): negatives here are randomly sampled from the
    full candidate pool, not drawn from Person 2's actual blocked candidate
    set. Random negatives are much easier to distinguish than the
    near-miss non-matches a trained blocker will surface in production
    (e.g. same city/country, overlapping name tokens, different business).
    Ideally negatives would be resampled from Person 2's candidate_pairs.tsv
    once available so the model learns on realistic hard negatives instead
    of trivially-different random pairs.

    Parameters
    ----------
    ground_truth_df : pd.DataFrame
        Loaded train_ground_truth.tsv, columns [source1_entity_id, matched_entity_ids].
    source2_ids, source3_ids : pd.Series
        All entity_id values from source2_df / source3_df — the negative sampling pool.
    negatives_per_positive : int
        Negatives drawn per true positive for that S1 entity (min 1, so
        singletons also get negative examples).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns [source1_entity_id, candidate_entity_id, label],
    shuffled.
    """
    rng = np.random.default_rng(random_state)

    gt = ground_truth_df.copy()
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    # ---- Positives (vectorized explode) ----
    pos = gt[gt["matched_entity_ids"] != ""].copy()
    pos["candidate_entity_id"] = pos["matched_entity_ids"].str.split(",")
    pos = pos.explode("candidate_entity_id")
    pos["candidate_entity_id"] = pos["candidate_entity_id"].str.strip()
    pos = pos[pos["candidate_entity_id"] != ""]
    pos["label"] = 1
    pos = pos[["source1_entity_id", "candidate_entity_id", "label"]].reset_index(drop=True)

    # ---- How many negatives each S1 entity needs ----
    n_pos_per_s1 = pos.groupby("source1_entity_id").size()
    all_s1_ids = gt["source1_entity_id"]
    n_neg_per_s1 = (
        n_pos_per_s1.reindex(all_s1_ids, fill_value=0)
        .clip(lower=1)
        * negatives_per_positive
    ).astype(int)

    # ---- Vectorized negative sampling ----
    s1_repeated = np.repeat(all_s1_ids.to_numpy(), n_neg_per_s1.to_numpy())
    candidate_pool = np.concatenate([source2_ids.to_numpy(), source3_ids.to_numpy()])
    sampled_candidates = rng.choice(candidate_pool, size=len(s1_repeated), replace=True)

    neg = pd.DataFrame({
        "source1_entity_id": s1_repeated,
        "candidate_entity_id": sampled_candidates,
        "label": 0,
    })

    # ---- Remove accidental collisions with true positives ----
    pos_pairs = pos[["source1_entity_id", "candidate_entity_id"]].assign(_is_pos=1)
    neg = neg.merge(pos_pairs, on=["source1_entity_id", "candidate_entity_id"], how="left")
    neg = neg[neg["_is_pos"].isna()].drop(columns="_is_pos").reset_index(drop=True)

    labeled = pd.concat([pos, neg], ignore_index=True)
    return labeled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_model(
    features_df: pd.DataFrame,
    label_col: str = "label",
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    val_ground_truth_df: pd.DataFrame | None = None,
):
    """
    Compare several classifiers (Logistic Regression, Random Forest,
    Gradient Boosting, HistGradientBoosting) on the pairwise feature matrix,
    evaluate each on a held-out split, and for each model pick the decision
    threshold that maximizes validation F0.5.
    
    If val_ground_truth_df is provided, tunes the threshold on the official
    entity-level macro F0.5 metric. Otherwise falls back to pairwise F0.5.
    Returns the best-performing model by F0.5.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in features_df.columns
            if c not in (label_col, "source1_entity_id", "candidate_entity_id")
        ]

    X = features_df[feature_cols].fillna(0.0)
    y = features_df[label_col]

    val_pairs_meta = None
    if val_ground_truth_df is not None and "source1_entity_id" in features_df.columns:
        val_s1_set = set(val_ground_truth_df["source1_entity_id"])
        val_mask = features_df["source1_entity_id"].isin(val_s1_set)
        train_mask = ~val_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        val_pairs_meta = features_df[val_mask][["source1_entity_id", "candidate_entity_id"]].copy()
    else:
        indices = np.arange(len(features_df))
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, indices, test_size=test_size, stratify=y, random_state=random_state
        )
        if "source1_entity_id" in features_df.columns and "candidate_entity_id" in features_df.columns:
            val_pairs_meta = features_df.iloc[idx_val][["source1_entity_id", "candidate_entity_id"]].copy()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=random_state
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=6, random_state=random_state
        ),
    }

    best_name, best_model, best_f05, best_threshold = None, None, -1.0, 0.5
    results = {}

    for name, clf in candidates.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)

        if val_ground_truth_df is not None and val_pairs_meta is not None and not val_pairs_meta.empty:
            val_pairs = val_pairs_meta.copy()
            val_pairs["match_probability"] = val_probs
            thr, f05, _ = optimize_entity_threshold(val_pairs, val_ground_truth_df)
            print(f"{name}: AUC={auc:.4f}, best Entity Macro F0.5={f05:.4f} at threshold={thr:.4f}")
        else:
            precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
            f05_scores = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-12)
            best_idx = int(np.nanargmax(f05_scores[:-1]))
            thr = float(thresholds[best_idx])
            f05 = float(f05_scores[best_idx])
            print(f"{name}: AUC={auc:.4f}, best Pairwise F0.5={f05:.4f} at threshold={thr:.4f}")

        results[name] = {"auc": auc, "f0.5": f05, "threshold": thr}

        if f05 > best_f05:
            best_name, best_model, best_f05, best_threshold = name, clf, f05, thr

    print(f"\n=== Best model: {best_name} (F0.5={best_f05:.4f}, threshold={best_threshold:.4f}) ===")

    val_preds = (best_model.predict_proba(X_val)[:, 1] >= best_threshold).astype(int)
    print(classification_report(y_val, val_preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, val_preds))

    if hasattr(best_model, "feature_importances_"):
        feature_importance = pd.Series(
            best_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        print("\nTop features:")
        print(feature_importance.head(10))
    elif hasattr(best_model, "coef_"):
        feature_importance = pd.Series(
            np.abs(best_model.coef_[0]), index=feature_cols
        ).sort_values(ascending=False)
        print("\nTop features (|coefficient|):")
        print(feature_importance.head(10))
    return best_model, best_threshold, feature_importance


def save_model_bundle(
    model: Any,
    threshold: float,
    tfidf_model: Any = None,
    path: str | os.PathLike = "reports/matching_model.joblib",
    feature_cols: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> str:
    """
    Save the trained matching model, prediction threshold, TF-IDF vectorizer,
    and associated metadata into a single joblib bundle file.

    Parameters
    ----------
    model : classifier object
        Trained model (e.g. RandomForestClassifier, LogisticRegression).
    threshold : float
        Decision threshold for predicting a positive match.
    tfidf_model : TfidfVectorizer or None
        Fitted TF-IDF model on entity names, used to transform candidates at test time.
    path : str or Path
        Destination path for the .joblib file.
    feature_cols : list[str] | None
        List of feature column names the model was trained on.
    extra_metadata : dict | None
        Optional dictionary of additional metadata (e.g. training scores, metrics).

    Returns
    -------
    str : Path to the saved bundle.
    """
    dir_name = os.path.dirname(str(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    bundle = {
        "model": model,
        "threshold": float(threshold),
        "tfidf_model": tfidf_model,
        "feature_cols": list(feature_cols) if feature_cols is not None else None,
        "metadata": extra_metadata or {},
    }
    joblib.dump(bundle, str(path))
    return str(path)


def load_model_bundle(
    path: str | os.PathLike = "reports/matching_model.joblib",
    require_tfidf: bool = False,
) -> dict:
    """
    Load a model bundle from disk and validate required keys.

    Parameters
    ----------
    path : str or Path
        Path to the saved .joblib file.
    require_tfidf : bool
        If True, raises ValueError if 'tfidf_model' is not in the bundle.

    Returns
    -------
    dict with keys:
        - 'model': trained classifier
        - 'threshold': float
        - 'tfidf_model': TfidfVectorizer (or None if older bundle)
        - 'feature_cols': list[str] | None
        - 'metadata': dict
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Model bundle not found at {path_str}")

    bundle = joblib.load(path_str)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dict bundle in {path_str}, got {type(bundle).__name__}")

    if "model" not in bundle:
        raise KeyError(f"Corrupted bundle at {path_str}: missing 'model' key")
    if "threshold" not in bundle:
        raise KeyError(f"Corrupted bundle at {path_str}: missing 'threshold' key")

    if require_tfidf and ("tfidf_model" not in bundle or bundle["tfidf_model"] is None):
        raise KeyError(f"Bundle at {path_str} does not contain required 'tfidf_model'")

    # Guarantee standard keys exist for backwards compatibility with raw dicts
    if "tfidf_model" not in bundle:
        bundle["tfidf_model"] = None
    if "feature_cols" not in bundle:
        bundle["feature_cols"] = None
    if "metadata" not in bundle:
        bundle["metadata"] = {}

    return bundle