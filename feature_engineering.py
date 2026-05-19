"""
Shared feature engineering for the K1netix prototype.

This file is required by:
  - noise_filter.py
  - clusterer.py

It converts BCF-like issue dictionaries into numeric/text features that can be
used by the noise filter and the clash grouper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


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


STATUS_RANK = {
    "open": 3,
    "active": 3,
    "in progress": 2,
    "in_progress": 2,
    "review": 2,
    "closed": 0,
    "resolved": 0,
}


PRIORITY_RANK = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}


NOISE_KEYWORDS = [
    "minor",
    "duplicate",
    "tolerance",
    "false positive",
    "soft clash",
    "ignore",
    "acceptable",
]


CLASH_KEYWORDS = [
    "fire",
    "escape",
    "headroom",
    "damper",
    "beam",
    "structural",
    "clearance",
    "riser",
    "access",
    "duct",
    "pipe",
    "cable",
]


def extract_features(issues: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw BCF-like issues into a feature DataFrame."""
    rows = []

    for issue in issues:
        title = str(issue.get("title", "") or "")
        desc = str(issue.get("description", "") or "")
        status = str(issue.get("status", "") or "").lower()
        priority = str(issue.get("priority", "") or "").lower()
        issue_type = str(issue.get("type", "") or "").lower()

        comments = issue.get("comments", []) or []
        viewpoints = issue.get("viewpoints", []) or []
        labels = issue.get("labels", []) or []
        related_topics = issue.get("related_topics", []) or []
        ref_links = issue.get("reference_links", []) or issue.get("ref_links", []) or []

        comment_lengths = [
            len(str(c.get("text", c))) if isinstance(c, dict) else len(str(c))
            for c in comments
        ]
        commenters = [
            str(c.get("author", ""))
            for c in comments
            if isinstance(c, dict) and c.get("author")
        ]

        created_at = _parse_date(issue.get("created_at") or issue.get("creation_date"))
        modified_at = _parse_date(issue.get("modified_at") or issue.get("modified_date"))

        days_since_created = _days_since(created_at)
        days_open = days_since_created if status not in ("closed", "resolved") else 0
        text = f"{title} {desc}".lower()

        rows.append({
            "guid": issue.get("guid") or issue.get("clash_id") or issue.get("id") or "",
            "has_title": int(bool(title.strip())),
            "has_description": int(bool(desc.strip())),
            "title_length": len(title),
            "desc_length": len(desc),
            "desc_word_count": len(desc.split()),
            "desc_density": len(desc.split()) / max(len(desc), 1),
            "has_assigned_to": int(bool(issue.get("assigned_to"))),
            "has_due_date": int(bool(issue.get("due_date"))),
            "has_stage": int(bool(issue.get("stage"))),
            "has_bim_snippet": int(bool(issue.get("bim_snippet") or issue.get("element_guids"))),
            "num_ref_links": len(ref_links),
            "num_comments": len(comments),
            "num_viewpoints": len(viewpoints),
            "num_labels": len(labels),
            "num_related_topics": len(related_topics),
            "avg_comment_length": float(np.mean(comment_lengths)) if comment_lengths else 0.0,
            "num_unique_commenters": len(set(commenters)),
            "status_rank": STATUS_RANK.get(status, 1),
            "is_open": int(status in ("open", "active")),
            "is_closed": int(status in ("closed", "resolved")),
            "is_in_progress": int(status in ("in progress", "in_progress", "review")),
            "priority_rank": PRIORITY_RANK.get(priority, 1),
            "days_since_created": days_since_created,
            "days_open": days_open,
            "has_been_modified": int(bool(modified_at and created_at and modified_at != created_at)),
            "noise_keyword_hit": int(any(keyword in text for keyword in NOISE_KEYWORDS)),
            "clash_keyword_hit": int(any(keyword in text for keyword in CLASH_KEYWORDS)),
            "type_encoded": abs(hash(issue_type)) % 20,
            "status_encoded": abs(hash(status)) % 20,
            "title": title,
            "description": desc,
            "status": status,
            "priority": priority,
            "type": issue_type,
        })

    df = pd.DataFrame(rows)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    if "guid" not in df.columns:
        df["guid"] = ""

    return df


def generate_weak_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create weak labels for a demo when there is no human-labelled training set.

    1 = likely real/actionable clash.
    0 = likely noise/minor issue.
    """
    score = (
        df["clash_keyword_hit"] * 2.0
        + df["priority_rank"]
        + df["num_comments"] * 0.3
        + df["num_viewpoints"] * 0.5
        + df["has_bim_snippet"]
        - df["noise_keyword_hit"] * 2.0
    )
    return (score >= 2.5).astype(int)


def build_cluster_matrix(df: pd.DataFrame, tfidf_weight: float = 0.7):
    """Build a mixed text + numeric feature matrix for clustering."""
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler

    text = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()

    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words="english",
    ).fit_transform(text)

    numeric = df[FEATURE_COLS].fillna(0)
    numeric_scaled = StandardScaler().fit_transform(numeric)

    return hstack([
        tfidf * tfidf_weight,
        numeric_scaled * (1 - tfidf_weight),
    ])


def _parse_date(value: Any):
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _days_since(date_value) -> int:
    if not date_value:
        return 0

    now = datetime.now(timezone.utc)
    if date_value.tzinfo is None:
        date_value = date_value.replace(tzinfo=timezone.utc)

    return max((now - date_value).days, 0)

