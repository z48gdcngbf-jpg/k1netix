"""
Layer 2 — Clash Grouper

K1netix brief alignment:
  - This groups many related real clashes into one Pre-RFI candidate.
  - HDBSCAN is preferred because it can detect natural clusters and noise.
  - If HDBSCAN is not installed, the artifact falls back to deterministic
    grouping so the iPaaS demo still runs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

try:
    import hdbscan
except Exception:
    hdbscan = None

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from feature_engineering import FEATURE_COLS, build_cluster_matrix, extract_features


def reduce_dimensions(matrix, n_components: int = 30):
    """Reduce high-dimensional TF-IDF + numeric matrix."""
    n_samples = matrix.shape[0]
    if n_samples < 5:
        return matrix

    n_comp = min(n_components, n_samples - 1, matrix.shape[1] - 1)
    if n_comp < 2:
        return matrix

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    return svd.fit_transform(matrix)


def cluster_clashes(
    issues: list[dict[str, Any]],
    min_cluster_size: int = 2,
    min_samples: int = 1,
    cluster_selection_epsilon: float = 0.0,
    tfidf_weight: float = 0.7,
) -> tuple[pd.DataFrame, Any]:
    """
    Cluster real non-noise issues.

    Returns:
      DataFrame with cluster labels, fitted clusterer or None.
    """
    if not issues:
        return pd.DataFrame(), None

    df = extract_features(issues)

    # HDBSCAN needs enough points to build a neighbour graph. Layer 2 runs
    # clustering per discipline, so some groups may only contain 1-2 clashes.
    # For those tiny groups, deterministic grouping is the correct artifact
    # behaviour: keep the pipeline moving and still create Pre-RFI candidates.
    if hdbscan is None or len(issues) <= max(2, min_cluster_size):
        return _fallback_grouping(df), None

    matrix = build_cluster_matrix(df, tfidf_weight=tfidf_weight)
    reduced = reduce_dimensions(matrix, n_components=min(30, len(issues) - 1))

    # HDBSCAN's prediction-data path expects dense numeric arrays.
    # For very small discipline groups, reduce_dimensions may return the original
    # sparse TF-IDF matrix, so convert it before fitting.
    if hasattr(reduced, "toarray"):
        reduced = reduced.toarray()
    reduced = np.asarray(reduced, dtype=float)

    safe_min_cluster_size = max(2, min(min_cluster_size, len(issues)))
    safe_min_samples = max(1, min(min_samples, len(issues) - 1))

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=safe_min_cluster_size,
        min_samples=safe_min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )

    try:
        labels = clusterer.fit_predict(reduced)
    except Exception:
        return _fallback_grouping(df), None

    df["cluster_id"] = labels
    df["cluster_prob"] = getattr(clusterer, "probabilities_", np.ones(len(df)))
    df["cluster_outlier"] = getattr(clusterer, "outlier_scores_", np.zeros(len(df)))
    df["is_unclustered"] = labels == -1

    return df, clusterer


def summarise_clusters(
    issues: list[dict[str, Any]],
    df_clustered: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build cluster summary objects. Each cluster can become one Pre-RFI."""
    guid_to_issue = {
        issue.get("guid") or issue.get("clash_id") or issue.get("id") or "": issue
        for issue in issues
    }
    clusters = defaultdict(list)

    for _, row in df_clustered.iterrows():
        cid = int(row["cluster_id"])
        guid = row["guid"]
        clusters[cid].append({
            "guid": guid,
            "cluster_prob": float(row["cluster_prob"]),
            "cluster_outlier": float(row["cluster_outlier"]),
            "issue": guid_to_issue.get(guid, {}),
        })

    summaries = []
    for cid, members in clusters.items():
        is_unclustered = cid == -1
        issue_list = [member["issue"] for member in members]

        statuses = [i.get("status", "") for i in issue_list if i.get("status")]
        priorities = [i.get("priority", "") for i in issue_list if i.get("priority")]
        types = [i.get("type", "") for i in issue_list if i.get("type")]
        assignees = [i.get("assigned_to", "") for i in issue_list if i.get("assigned_to")]

        titles = [i.get("title", "") for i in issue_list if i.get("title")]
        cluster_label = _derive_cluster_label(titles)
        priority_score = _compute_cluster_priority(issue_list)

        summaries.append({
            "cluster_id": cid,
            "is_unclustered": is_unclustered,
            "cluster_label": cluster_label,
            "issue_count": len(members),
            "guids": [member["guid"] for member in members],
            "avg_membership": round(float(np.mean([m["cluster_prob"] for m in members])), 3),
            "dominant_status": _mode(statuses),
            "dominant_priority": _mode(priorities),
            "dominant_type": _mode(types),
            "assignees": sorted(set(assignees)),
            "priority_score": priority_score,
            "issues": issue_list,
        })

    summaries.sort(key=lambda x: (x["is_unclustered"], -x["priority_score"]))
    return summaries


def evaluate_clustering(df: pd.DataFrame) -> dict[str, Any]:
    """Return clustering quality metrics."""
    if df.empty:
        return {"n_clusters": 0, "n_noise_pts": 0, "n_clustered": 0, "noise_ratio": 0}

    labels = df["cluster_id"].values
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    n_clustered = int((labels != -1).sum())

    metrics = {
        "n_clusters": n_clusters,
        "n_noise_pts": n_noise,
        "n_clustered": n_clustered,
        "noise_ratio": round(n_noise / max(len(labels), 1), 3),
        "backend": "hdbscan" if hdbscan is not None else "deterministic_fallback",
    }

    if n_clusters >= 2 and n_clustered >= 4:
        mask = labels != -1
        try:
            feat_cols = [c for c in FEATURE_COLS if c in df.columns]
            X = StandardScaler().fit_transform(df.loc[mask, feat_cols].fillna(0))
            metrics["silhouette_score"] = round(float(silhouette_score(X, labels[mask])), 3)
        except Exception:
            pass

    return metrics


def run(
    real_issues: list[dict[str, Any]],
    min_cluster_size: int = 2,
    tfidf_weight: float = 0.7,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full Layer 2 clustering step."""
    if not real_issues:
        return {"clusters": [], "unclustered": [], "metrics": {}, "df_clustered": None}

    df_clustered, clusterer = cluster_clashes(
        real_issues,
        min_cluster_size=min_cluster_size,
        tfidf_weight=tfidf_weight,
    )

    cluster_summaries = summarise_clusters(real_issues, df_clustered)
    metrics = evaluate_clustering(df_clustered)

    proper_clusters = [c for c in cluster_summaries if not c["is_unclustered"]]
    unclustered = next((c for c in cluster_summaries if c["is_unclustered"]), None)
    unclustered_issues = unclustered["issues"] if unclustered else []

    if verbose:
        print("\nClustering complete:")
        print(f"  Backend         : {metrics['backend']}")
        print(f"  Clusters formed : {metrics['n_clusters']}")
        print(f"  Unclustered pts : {metrics['n_noise_pts']}")
        print(f"  Silhouette      : {metrics.get('silhouette_score', 'N/A')}")
        for cluster in proper_clusters:
            print(
                f"  [{cluster['cluster_id']}] {cluster['cluster_label'][:50]:<50} "
                f"({cluster['issue_count']} issues, priority={cluster['priority_score']})"
            )

    return {
        "clusters": proper_clusters,
        "unclustered": unclustered_issues,
        "metrics": metrics,
        "df_clustered": df_clustered,
    }


def _fallback_grouping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic grouping for artifact demos.

    It groups issues by repeated BIM/coordination signals in the title:
    fire, structural, MEP, access, or general coordination. This is not meant
    to replace HDBSCAN; it keeps the iPaaS demo working when dependencies are
    missing.
    """
    labels = []
    mapping: dict[str, int] = {}

    for _, row in df.iterrows():
        text = f"{row.get('title', '')} {row.get('description', '')}".lower()
        if any(term in text for term in ("fire", "escape", "damper")):
            key = "fire_safety"
        elif any(term in text for term in ("beam", "structural", "slab")):
            key = "structural_mep"
        elif any(term in text for term in ("duct", "pipe", "cable", "tray")):
            key = "mep_routing"
        elif any(term in text for term in ("access", "clearance", "riser")):
            key = "access_clearance"
        else:
            key = "general_coordination"

        if key not in mapping:
            mapping[key] = len(mapping)
        labels.append(mapping[key])

    df = df.copy()
    df["cluster_id"] = labels
    df["cluster_prob"] = 1.0
    df["cluster_outlier"] = 0.0
    df["is_unclustered"] = False
    return df


def _derive_cluster_label(titles: list[str]) -> str:
    if not titles:
        return "Unnamed Cluster"
    if len(titles) == 1:
        return titles[0][:80]

    try:
        vec = TfidfVectorizer(max_features=5, stop_words="english")
        vec.fit(titles)
        keywords = vec.get_feature_names_out().tolist()
        return " | ".join(keyword.title() for keyword in keywords[:4])
    except Exception:
        return titles[0][:80]


PRIORITY_WEIGHTS = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _compute_cluster_priority(issues: list[dict[str, Any]]) -> float:
    score = 0.0
    for issue in issues:
        priority = (issue.get("priority") or "").lower()
        score += PRIORITY_WEIGHTS.get(priority, 1)
        score += len(issue.get("comments", [])) * 0.5
        score += len(issue.get("viewpoints", [])) * 0.3
        if (issue.get("status") or "").lower() in ("open", "active"):
            score += 1.0
    return round(score / max(len(issues), 1), 3)


def _mode(values: list[str]) -> str:
    return max(set(values), key=values.count) if values else ""


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "bcf_output.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = run(data.get("issues", []))
    print(json.dumps(result["metrics"], indent=2))
