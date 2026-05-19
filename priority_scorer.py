"""
Layer 2 — Priority Scorer
Implements the K1netix Prioritisation Score formula:

  Score = w1*LLM_certainty + w2*data_completeness + w3*classifier_prob + w4*regulation_match

Layer 2 computes:
  - data_completeness   (from BCF field fill-rate)
  - classifier_prob     (from XGBoost output, with ambiguity penalty)

Layer 3 will fill:
  - LLM_certainty       (placeholder = 0.5 until RAG/LLM runs)
  - regulation_match    (placeholder = 0.5 until RAG/LLM runs)

Scores are computed per-issue AND per-cluster (aggregate).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict

# ── Default weights (equal) ───────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "w1_llm_certainty":      0.25,
    "w2_data_completeness":  0.25,
    "w3_classifier_prob":    0.25,
    "w4_regulation_match":   0.25,
}

# BCF fields that indicate a well-described, actionable clash
# Each field has a weight reflecting its importance for completeness
COMPLETENESS_FIELDS: dict[str, float] = {
    "title":         0.15,
    "description":   0.20,
    "assigned_to":   0.15,
    "status":        0.10,
    "priority":      0.10,
    "due_date":      0.08,
    "stage":         0.05,
    "labels":        0.07,   # non-empty list
    "viewpoints":    0.05,   # non-empty list
    "comments":      0.05,   # non-empty list
}

# Ambiguity zone for classifier probability — penalise uncertain predictions
AMBIGUITY_LOW  = 0.4
AMBIGUITY_HIGH = 0.6
AMBIGUITY_PENALTY = 0.15   # deduct this when prob is in the grey zone


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class PriorityScore:
    # Component scores (0–1 each)
    llm_certainty:      float   # w1 — placeholder until Layer 3
    data_completeness:  float   # w2 — computed here
    classifier_prob:    float   # w3 — from XGBoost
    regulation_match:   float   # w4 — placeholder until Layer 3

    # Weights used
    weights: dict

    # Final composite score
    composite: float

    # Severity band
    band:  str       # CRITICAL · HIGH · MEDIUM · LOW
    color: str       # for UI display

    # Explanation
    breakdown: dict

    def to_dict(self) -> dict:
        return asdict(self)


BAND_THRESHOLDS = [
    (0.75, "CRITICAL", "#ff4444"),
    (0.55, "HIGH",     "#ff9900"),
    (0.35, "MEDIUM",   "#f0c040"),
    (0.00, "LOW",      "#6bff9e"),
]

def _band(score: float) -> tuple[str, str]:
    for threshold, label, color in BAND_THRESHOLDS:
        if score >= threshold:
            return label, color
    return "LOW", "#6bff9e"


# ── Data completeness calculator ──────────────────────────────────────────────

def compute_data_completeness(issue: dict) -> tuple[float, dict]:
    """
    Fill-rate of critical BCF fields, weighted by importance.
    Returns (completeness_score, per_field_detail).
    """
    detail = {}
    total_weight  = sum(COMPLETENESS_FIELDS.values())
    earned_weight = 0.0

    for field, weight in COMPLETENESS_FIELDS.items():
        val = issue.get(field)
        if isinstance(val, list):
            present = len(val) > 0
        elif isinstance(val, str):
            present = bool(val.strip())
        else:
            present = val is not None

        detail[field] = {"present": present, "weight": weight}
        if present:
            earned_weight += weight

    score = round(earned_weight / total_weight, 4)
    return score, detail


# ── Classifier probability adjuster ──────────────────────────────────────────

def adjust_classifier_prob(raw_prob: float) -> tuple[float, str]:
    """
    Penalise predictions in the ambiguous zone (0.4–0.6).
    Returns (adjusted_prob, explanation).
    """
    if AMBIGUITY_LOW <= raw_prob <= AMBIGUITY_HIGH:
        adjusted = max(0.0, raw_prob - AMBIGUITY_PENALTY)
        note = f"ambiguity penalty applied ({raw_prob:.2f} → {adjusted:.2f})"
    else:
        adjusted = raw_prob
        note = "no penalty"

    return round(adjusted, 4), note


# ── Per-issue scorer ──────────────────────────────────────────────────────────

def score_issue(
    issue: dict,
    weights: dict | None = None,
    llm_certainty: float = 0.5,        # Layer 3 will override
    regulation_match: float = 0.5,     # Layer 3 will override
) -> PriorityScore:
    """
    Compute the K1netix Prioritisation Score for a single issue.

    Args:
        issue            : issue dict (must have _noise_filter from XGBoost step)
        weights          : override DEFAULT_WEIGHTS if needed
        llm_certainty    : from Layer 3 RAG (placeholder = 0.5)
        regulation_match : from Layer 3 RAG (placeholder = 0.5)
    """
    w = weights or DEFAULT_WEIGHTS

    # w2 — data completeness
    completeness, completeness_detail = compute_data_completeness(issue)

    # w3 — classifier probability (with ambiguity penalty)
    nf = issue.get("_noise_filter") or {}
    raw_prob = float(nf.get("real_prob", 0.5))
    adj_prob, prob_note = adjust_classifier_prob(raw_prob)

    # Composite score
    composite = (
        w["w1_llm_certainty"]     * llm_certainty    +
        w["w2_data_completeness"] * completeness      +
        w["w3_classifier_prob"]   * adj_prob          +
        w["w4_regulation_match"]  * regulation_match
    )
    composite = round(min(max(composite, 0.0), 1.0), 4)

    band, color = _band(composite)

    breakdown = {
        "llm_certainty":     {"value": llm_certainty,   "weight": w["w1_llm_certainty"],
                               "note": "placeholder — Layer 3 will update"},
        "data_completeness": {"value": completeness,     "weight": w["w2_data_completeness"],
                               "fields": completeness_detail},
        "classifier_prob":   {"value": adj_prob,         "weight": w["w3_classifier_prob"],
                               "raw": raw_prob,           "note": prob_note},
        "regulation_match":  {"value": regulation_match, "weight": w["w4_regulation_match"],
                               "note": "placeholder — Layer 3 will update"},
    }

    return PriorityScore(
        llm_certainty=llm_certainty,
        data_completeness=completeness,
        classifier_prob=adj_prob,
        regulation_match=regulation_match,
        weights=w,
        composite=composite,
        band=band,
        color=color,
        breakdown=breakdown,
    )


# ── Per-cluster scorer ────────────────────────────────────────────────────────

def score_cluster(
    cluster: dict,
    weights: dict | None = None,
    llm_certainty: float = 0.5,
    regulation_match: float = 0.5,
) -> dict:
    """
    Aggregate priority score for a whole cluster.
    Uses max classifier_prob and mean data_completeness across member issues.

    Returns the cluster dict enriched with _priority_score.
    """
    issues = cluster.get("issues", [])
    if not issues:
        return cluster

    # Score each member
    member_scores = [
        score_issue(iss, weights=weights,
                    llm_certainty=llm_certainty,
                    regulation_match=regulation_match)
        for iss in issues
    ]

    # Aggregate
    import statistics

    agg_completeness = round(statistics.mean(s.data_completeness for s in member_scores), 4)
    agg_classifier   = round(max(s.classifier_prob for s in member_scores), 4)   # worst-case = max risk
    agg_composite    = round(statistics.mean(s.composite for s in member_scores), 4)

    w = weights or DEFAULT_WEIGHTS
    cluster_composite = (
        w["w1_llm_certainty"]     * llm_certainty      +
        w["w2_data_completeness"] * agg_completeness    +
        w["w3_classifier_prob"]   * agg_classifier      +
        w["w4_regulation_match"]  * regulation_match
    )
    cluster_composite = round(min(max(cluster_composite, 0.0), 1.0), 4)

    band, color = _band(cluster_composite)

    cluster["_priority_score"] = {
        "composite":         cluster_composite,
        "band":              band,
        "color":             color,
        "components": {
            "llm_certainty":     llm_certainty,
            "data_completeness": agg_completeness,
            "classifier_prob":   agg_classifier,
            "regulation_match":  regulation_match,
        },
        "weights":           w,
        "member_scores":     [s.composite for s in member_scores],
        "note": "LLM certainty and regulation match are Layer 3 placeholders (0.5)"
    }

    return cluster


# ── Batch scoring ─────────────────────────────────────────────────────────────

def score_all_issues(issues: list[dict], weights: dict | None = None) -> list[dict]:
    """Score all issues and inject _priority_score into each."""
    for issue in issues:
        ps = score_issue(issue, weights=weights)
        issue["_priority_score"] = ps.to_dict()
    return issues


def score_all_clusters(clusters: list[dict], weights: dict | None = None) -> list[dict]:
    """Score all clusters and re-sort by composite score descending."""
    scored = [score_cluster(c, weights=weights) for c in clusters]
    scored.sort(key=lambda c: c.get("_priority_score", {}).get("composite", 0), reverse=True)
    return scored


# ── Layer 3 update hook ───────────────────────────────────────────────────────

def update_with_layer3(
    item: dict,
    llm_certainty: float,
    regulation_match: float,
    weights: dict | None = None,
) -> dict:
    """
    Called by Layer 3 to replace placeholder values with real LLM outputs.
    Works on both individual issues and cluster dicts.
    """
    existing = item.get("_priority_score") or {}
    w = weights or DEFAULT_WEIGHTS

    data_completeness = existing.get("data_completeness") or existing.get(
        "components", {}
    ).get("data_completeness", 0.5)

    classifier_prob = existing.get("classifier_prob") or existing.get(
        "components", {}
    ).get("classifier_prob", 0.5)

    composite = (
        w["w1_llm_certainty"]     * llm_certainty    +
        w["w2_data_completeness"] * data_completeness +
        w["w3_classifier_prob"]   * classifier_prob   +
        w["w4_regulation_match"]  * regulation_match
    )
    composite = round(min(max(composite, 0.0), 1.0), 4)
    band, color = _band(composite)

    if "_priority_score" in item:
        item["_priority_score"]["composite"]           = composite
        item["_priority_score"]["band"]                = band
        item["_priority_score"]["color"]               = color
        item["_priority_score"]["llm_certainty"]       = llm_certainty
        item["_priority_score"]["regulation_match"]    = regulation_match
        if "components" in item["_priority_score"]:
            item["_priority_score"]["components"]["llm_certainty"]    = llm_certainty
            item["_priority_score"]["components"]["regulation_match"] = regulation_match
        item["_priority_score"]["note"] = "fully scored — Layer 3 complete"

    return item


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else "layer2_output.json"
    with open(path) as f:
        data = json.load(f)

    issues   = data.get("real_issues", [])
    clusters = data.get("clusters", [])

    scored_issues   = score_all_issues(issues)
    scored_clusters = score_all_clusters(clusters)

    bands = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for iss in scored_issues:
        b = iss.get("_priority_score", {}).get("band", "LOW")
        bands[b] = bands.get(b, 0) + 1

    print("\n📊 Priority Score Distribution (issues)")
    for band, count in bands.items():
        print(f"  {band:<10} {count}")

    print("\n🔢 Top 5 clusters by priority:")
    for c in scored_clusters[:5]:
        ps = c.get("_priority_score", {})
        print(f"  [{ps.get('band','?')}] {c.get('cluster_label','?')[:50]:<50} "
              f"score={ps.get('composite','?')}")