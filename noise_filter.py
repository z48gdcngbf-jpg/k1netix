"""
Layer 2 — Noise Filter

K1netix brief alignment:
  - This module is the "AI triage" cost gate.
  - In production it should use XGBoost trained on labelled historical clashes.
  - In the competition artifact it must still work without XGBoost, so it
    falls back to scikit-learn RandomForest.

Modes:
  - weak    : trains on heuristic labels, no ground truth needed
  - labeled : trains on a CSV with guid + label columns
  - predict : loads a saved model and scores new data
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    MODEL_BACKEND = "xgboost"
except Exception:
    XGBClassifier = None
    MODEL_BACKEND = "random_forest_fallback"

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from feature_engineering import extract_features, generate_weak_labels


warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "has_title", "has_description", "title_length", "desc_length",
    "desc_word_count", "desc_density",
    "has_assigned_to", "has_due_date", "has_stage", "has_bim_snippet",
    "num_ref_links", "num_comments", "num_viewpoints", "num_labels",
    "num_related_topics", "avg_comment_length", "num_unique_commenters",
    "status_rank", "is_open", "is_closed", "is_in_progress",
    "priority_rank", "days_since_created", "days_open", "has_been_modified",
    "noise_keyword_hit", "clash_keyword_hit",
    "type_encoded", "status_encoded",
]

MODEL_PATH = Path("models/xgb_noise_filter.pkl")


def build_model():
    """Build the best available classifier."""
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def train(
    df: pd.DataFrame,
    labels: pd.Series | None = None,
    save: bool = True,
    verbose: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Train the noise filter.

    If labels are missing, this uses weak labels generated from the clash text
    and metadata. That is fine for the artifact, but real K1netix should later
    replace this with labelled historical project data.
    """
    X = df[FEATURE_COLS].fillna(0).values

    if labels is None:
        y = generate_weak_labels(df).values
        label_source = "heuristic weak supervision"
    else:
        y = labels.values
        label_source = "ground truth"

    if len(set(y)) < 2:
        y = _force_two_classes(y)
        label_source += " with fallback class balancing"

    if verbose:
        print(f"Training noise filter on {len(X)} samples")
        print(f"Backend: {MODEL_BACKEND}")
        print(f"Labels: {label_source}")
        print(f"Class distribution -> noise: {(y == 0).sum()} | real: {(y == 1).sum()}")

    model = build_model()
    metrics = {
        "backend": MODEL_BACKEND,
        "label_source": label_source,
        "n_samples": len(X),
        "n_noise": int((y == 0).sum()),
        "n_real": int((y == 1).sum()),
    }

    if len(X) >= 6 and min((y == 0).sum(), (y == 1).sum()) >= 2:
        n_splits = min(5, int(min((y == 0).sum(), (y == 1).sum())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
        metrics["cv_roc_auc_std"] = float(cv_scores.std())
        if verbose:
            print(f"CV ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    else:
        metrics["cv_note"] = "Skipped CV because dataset is too small or one class is rare."

    model.fit(X, y)

    if hasattr(model, "feature_importances_"):
        importances = dict(zip(FEATURE_COLS, model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        metrics["top_features"] = {k: round(float(v), 4) for k, v in top_features}

        if verbose:
            print("\nTop features:")
            for feat, imp in top_features:
                print(f"  {feat:<30} {imp:.4f}")

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        if verbose:
            print(f"\nModel saved -> {MODEL_PATH}")

    return model, metrics


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_PATH}. Run train() first.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(
    df: pd.DataFrame,
    model=None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Score issues.

    Returns df with:
      - noise_prob
      - real_prob
      - is_noise
      - noise_reason
    """
    if model is None:
        model = load_model()

    X = df[FEATURE_COLS].fillna(0).values
    probs = model.predict_proba(X)

    if probs.shape[1] == 1:
        real_prob = np.full(len(df), 0.5)
        noise_prob = np.full(len(df), 0.5)
    else:
        noise_prob = probs[:, 0]
        real_prob = probs[:, 1]

    result = df.copy()
    result["real_prob"] = real_prob
    result["noise_prob"] = noise_prob
    result["is_noise"] = result["noise_prob"] >= threshold
    result["noise_reason"] = result.apply(_explain_noise, axis=1)

    return result


def _explain_noise(row: pd.Series) -> str:
    if not row.get("is_noise", False):
        return ""

    reasons = []
    if row.get("noise_keyword_hit"):
        reasons.append("noise keyword in title/description")
    if not row.get("has_description"):
        reasons.append("no description")
    if row.get("num_comments", 0) == 0 and row.get("num_viewpoints", 0) == 0:
        reasons.append("no comments or viewpoints")
    if row.get("desc_word_count", 0) < 3:
        reasons.append("very short description")
    if row.get("noise_prob", 0) > 0.85:
        reasons.append(f"high noise probability ({row['noise_prob']:.0%})")

    return " | ".join(reasons) if reasons else "low quality signal"


def run(
    bcf_json: dict[str, Any],
    labeled_csv: str | None = None,
    threshold: float = 0.5,
    retrain: bool = True,
) -> dict[str, Any]:
    """
    Full Layer 2 noise filter.
    """
    issues = bcf_json.get("issues", [])
    if not issues:
        return {"real_issues": [], "noise_issues": [], "metrics": {}, "df_scored": None}

    df = extract_features(issues)

    labels = None
    if labeled_csv:
        label_df = pd.read_csv(labeled_csv)
        label_map = dict(zip(label_df["guid"], label_df["label"]))
        labels = df["guid"].map(label_map)
        if labels.isna().any():
            print(f"{labels.isna().sum()} issues missing labels; filling with weak labels")
            weak = generate_weak_labels(df)
            labels = labels.fillna(weak)
        labels = labels.astype(int)

    if retrain or not MODEL_PATH.exists():
        model, metrics = train(df, labels=labels)
    else:
        model = load_model()
        metrics = {"note": "loaded saved model", "backend": MODEL_BACKEND}

    df_scored = predict(df, model=model, threshold=threshold)

    score_map = df_scored.set_index("guid")[
        ["real_prob", "noise_prob", "is_noise", "noise_reason"]
    ].to_dict("index")

    real_issues = []
    noise_issues = []

    for issue in issues:
        guid = issue.get("guid") or issue.get("clash_id") or issue.get("id") or ""
        score = score_map.get(guid, {})
        issue["_noise_filter"] = {
            "real_prob": round(float(score.get("real_prob", 0.5)), 4),
            "noise_prob": round(float(score.get("noise_prob", 0.5)), 4),
            "is_noise": bool(score.get("is_noise", False)),
            "noise_reason": score.get("noise_reason", ""),
        }
        if issue["_noise_filter"]["is_noise"]:
            noise_issues.append(issue)
        else:
            real_issues.append(issue)

    print("\nNoise filter complete:")
    print(f"  Real clashes : {len(real_issues)}")
    print(f"  Noise        : {len(noise_issues)}")

    return {
        "real_issues": real_issues,
        "noise_issues": noise_issues,
        "metrics": metrics,
        "df_scored": df_scored,
    }


def _force_two_classes(y: np.ndarray) -> np.ndarray:
    """Keep tiny demo datasets trainable even when weak labels create one class."""
    y = np.array(y, dtype=int)
    if len(y) >= 2:
        y[0] = 0
        y[-1] = 1
    return y


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "bcf_output.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = run(data)
    print(json.dumps(result["metrics"], indent=2))
