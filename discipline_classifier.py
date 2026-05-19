"""
Layer 2 — Discipline Classifier
Tags each BCF clash with its involved discipline(s) and clash pair type.

Disciplines: MECH · ELEC · PLMB · FP · STR · ARCH · CIVIL
Clash pairs:  MEP-STR · MEP-ARCH · STR-ARCH · MEP-MEP · etc.

No IFC required — works purely from BCF title, description, type, labels.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Keyword dictionaries ──────────────────────────────────────────────────────
# Each entry: discipline → list of keyword patterns (lowercase, partial match)

DISCIPLINE_KEYWORDS: dict[str, list[str]] = {
    "MECH": [
        "duct", "hvac", "air handling", "ahu", "fcu", "fan coil", "diffuser",
        "grille", "damper", "exhaust", "supply air", "return air", "ventilation",
        "chiller", "cooling", "heating", "vav", "crac", "mechanical",
        "air conditioning", "ac unit", "cassette unit", "plenum",
    ],
    "ELEC": [
        "cable", "conduit", "cable tray", "tray", "wireway", "bus duct",
        "busbar", "panel", "switchboard", "switchgear", "electrical",
        "mdb", "db board", "transformer", "raceway", "containment",
        "low voltage", "lv", "mv", "medium voltage", "electrical room",
        "riser", "electrical shaft",
    ],
    "PLMB": [
        "pipe", "piping", "water", "drain", "drainage", "sanitary",
        "soil", "waste", "hot water", "cold water", "hwsd", "cwsd",
        "plumbing", "sewer", "stormwater", "rainwater", "vent pipe",
        "rcp", "chilled water", "condenser water", "chws", "chwr",
        "cws", "cwr", "domestic water",
    ],
    "FP": [
        "sprinkler", "fire", "fire main", "fire protection", "fire suppression",
        "fm200", "halon", "deluge", "standpipe", "fire hose", "hydrant",
        "fire riser", "afss", "fire alarm", "smoke detector",
    ],
    "STR": [
        "beam", "column", "slab", "foundation", "footing", "pile",
        "structural", "steel", "rebar", "reinforcement", "concrete",
        "truss", "girder", "joist", "shear wall", "core wall",
        "transfer plate", "transfer beam", "pt slab", "post-tension",
        "precast", "rc", "reinforced concrete",
    ],
    "ARCH": [
        "wall", "partition", "door", "window", "ceiling", "false ceiling",
        "architectural", "finish", "facade", "cladding", "curtain wall",
        "floor finish", "screed", "render", "gypsum", "drylining",
        "raised floor", "access floor", "soffit", "bulkhead", "coving",
    ],
    "CIVIL": [
        "civil", "site", "road", "pavement", "earthwork", "excavation",
        "retaining wall", "ground beam", "utility", "manhole",
        "culvert", "storm drain", "site services",
    ],
}

# Clash type keywords that appear in BCF "type" field
TYPE_DISCIPLINE_MAP: dict[str, str] = {
    "mechanical":  "MECH",
    "electrical":  "ELEC",
    "plumbing":    "PLMB",
    "fire":        "FP",
    "structural":  "STR",
    "architectural": "ARCH",
    "civil":       "CIVIL",
    "mep":         "MEP_GROUP",  # generic MEP — will be resolved from text
    "coordination": None,        # too vague
}

# MEP super-group
MEP_DISCIPLINES = {"MECH", "ELEC", "PLMB", "FP"}

# Human-readable discipline names
DISCIPLINE_LABELS = {
    "MECH":  "Mechanical",
    "ELEC":  "Electrical",
    "PLMB":  "Plumbing",
    "FP":    "Fire Protection",
    "STR":   "Structural",
    "ARCH":  "Architecture",
    "CIVIL": "Civil",
}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DisciplineTag:
    primary:     str             # e.g. "MECH"
    secondary:   Optional[str]   # e.g. "STR"
    clash_pair:  str             # e.g. "MECH-STR"
    group:       str             # e.g. "MEP-STR"
    confidence:  float           # 0-1
    scores:      dict = field(default_factory=dict)  # per-discipline raw scores
    is_mep:      bool = False
    is_cross_discipline: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── Scorer ─────────────────────────────────────────────────────────────────────

def _score_text(text: str) -> dict[str, float]:
    """
    Score a text string against all discipline keyword lists.
    Returns raw hit count per discipline.
    """
    if not text:
        return {d: 0.0 for d in DISCIPLINE_KEYWORDS}

    t = text.lower()
    scores: dict[str, float] = {}

    for disc, keywords in DISCIPLINE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in t)
        # Weight longer keyword matches more (more specific = more confident)
        weighted = sum(len(kw.split()) for kw in keywords if kw in t)
        scores[disc] = hits + weighted * 0.5

    return scores


def _type_field_boost(type_str: str) -> dict[str, float]:
    """Extra score boost from the BCF 'type' field."""
    boosts = {d: 0.0 for d in DISCIPLINE_KEYWORDS}
    if not type_str:
        return boosts
    t = type_str.lower().strip()
    for key, disc in TYPE_DISCIPLINE_MAP.items():
        if key in t and disc and disc in boosts:
            boosts[disc] += 3.0  # strong signal
    return boosts


def _label_boost(labels: list[str]) -> dict[str, float]:
    """Extra score boost from BCF 'labels' list."""
    boosts = {d: 0.0 for d in DISCIPLINE_KEYWORDS}
    for label in labels:
        label_scores = _score_text(label)
        for d, s in label_scores.items():
            boosts[d] += s * 1.5  # labels are explicit — weight more
    return boosts


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_discipline(issue: dict) -> DisciplineTag:
    """
    Classify a single BCF issue into discipline(s).

    Scoring pipeline:
      1. Score title + description text
      2. Boost from BCF 'type' field
      3. Boost from BCF 'labels'
      4. Pick top-2 disciplines → primary + secondary
      5. Derive clash pair and group
    """
    title   = issue.get("title") or ""
    desc    = issue.get("description") or ""
    itype   = issue.get("type") or ""
    labels  = issue.get("labels") or []

    combined = f"{title} {desc}"

    # Aggregate scores
    text_scores  = _score_text(combined)
    type_boosts  = _type_field_boost(itype)
    label_boosts = _label_boost(labels)

    total: dict[str, float] = {}
    for d in DISCIPLINE_KEYWORDS:
        total[d] = text_scores[d] + type_boosts[d] + label_boosts[d]

    total_sum = sum(total.values())

    # Normalised scores (0–1)
    norm: dict[str, float] = {
        d: round(v / total_sum, 4) if total_sum > 0 else 0.0
        for d, v in total.items()
    }

    # Sort by score descending
    ranked = sorted(norm.items(), key=lambda x: x[1], reverse=True)

    # Primary discipline
    primary, primary_score = ranked[0]

    # If top score is zero → unclassified
    if primary_score == 0.0:
        primary       = "ARCH"        # safe fallback
        primary_score = 0.1
        secondary     = None
        confidence    = 0.1
    else:
        # Secondary: only accept if score > 15% of primary
        secondary_candidates = [
            (d, s) for d, s in ranked[1:]
            if s > 0 and s >= primary_score * 0.15
        ]
        secondary = secondary_candidates[0][0] if secondary_candidates else None
        confidence = round(primary_score, 3)

    # Clash pair (canonical order: higher hierarchy first)
    hierarchy = ["STR", "ARCH", "CIVIL", "MECH", "PLMB", "FP", "ELEC"]

    def hierarchy_rank(d: str) -> int:
        try:
            return hierarchy.index(d)
        except ValueError:
            return 99

    if secondary:
        pair_discs = sorted([primary, secondary], key=hierarchy_rank)
        clash_pair = f"{pair_discs[0]}-{pair_discs[1]}"
    else:
        clash_pair = primary

    # Group (MEP-STR, MEP-ARCH, STR-ARCH, MEP-MEP, ARCH-ARCH, etc.)
    def to_group(d: str) -> str:
        return "MEP" if d in MEP_DISCIPLINES else d

    if secondary:
        g1, g2 = to_group(primary), to_group(secondary)
        group_parts = sorted([g1, g2], key=lambda x: ["STR","ARCH","CIVIL","MEP"].index(x)
                             if x in ["STR","ARCH","CIVIL","MEP"] else 9)
        group = f"{group_parts[0]}-{group_parts[1]}" if g1 != g2 else g1
    else:
        group = to_group(primary)

    is_mep = primary in MEP_DISCIPLINES or (secondary or "") in MEP_DISCIPLINES
    is_cross = secondary is not None and secondary != primary

    return DisciplineTag(
        primary=primary,
        secondary=secondary,
        clash_pair=clash_pair,
        group=group,
        confidence=confidence,
        scores=norm,
        is_mep=is_mep,
        is_cross_discipline=is_cross,
    )


# ── Batch classifier ──────────────────────────────────────────────────────────

def classify_all(issues: list[dict]) -> list[dict]:
    """
    Classify all issues and inject _discipline key into each issue dict.
    Returns the same list with _discipline added in-place.
    """
    for issue in issues:
        tag = classify_discipline(issue)
        issue["_discipline"] = tag.to_dict()
    return issues


def group_by_discipline(issues: list[dict]) -> dict[str, list[dict]]:
    """
    Group classified issues by their primary discipline.
    Requires classify_all() to have been run first.
    """
    groups: dict[str, list[dict]] = {}
    for issue in issues:
        disc = (issue.get("_discipline") or {}).get("primary", "UNKNOWN")
        groups.setdefault(disc, []).append(issue)
    return groups


def group_by_clash_pair(issues: list[dict]) -> dict[str, list[dict]]:
    """
    Group classified issues by their clash pair (e.g. MECH-STR).
    Requires classify_all() to have been run first.
    """
    groups: dict[str, list[dict]] = {}
    for issue in issues:
        pair = (issue.get("_discipline") or {}).get("clash_pair", "UNKNOWN")
        groups.setdefault(pair, []).append(issue)
    return groups


def discipline_summary(issues: list[dict]) -> dict:
    """
    Return a summary dict of discipline distribution.
    """
    from collections import Counter

    primary_counts  = Counter()
    pair_counts     = Counter()
    group_counts    = Counter()
    low_confidence  = 0

    for iss in issues:
        d = iss.get("_discipline") or {}
        primary_counts[d.get("primary", "UNKNOWN")] += 1
        pair_counts[d.get("clash_pair", "UNKNOWN")]  += 1
        group_counts[d.get("group", "UNKNOWN")]       += 1
        if d.get("confidence", 1.0) < 0.3:
            low_confidence += 1

    return {
        "by_primary":    dict(primary_counts.most_common()),
        "by_clash_pair": dict(pair_counts.most_common()),
        "by_group":      dict(group_counts.most_common()),
        "low_confidence_count": low_confidence,
        "total": len(issues),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    path = sys.argv[1] if len(sys.argv) > 1 else "bcf_output.json"
    with open(path) as f:
        data = json.load(f)

    issues = classify_all(data.get("issues", []))
    summary = discipline_summary(issues)

    print("\n📐 Discipline Classification Summary")
    print("=" * 40)
    for disc, count in summary["by_primary"].items():
        label = DISCIPLINE_LABELS.get(disc, disc)
        bar   = "█" * count
        print(f"  {label:<20} {bar} {count}")
    print(f"\n  Clash pairs:")
    for pair, count in summary["by_clash_pair"].items():
        print(f"    {pair:<20} {count}")
    print(f"\n  Low confidence: {summary['low_confidence_count']} issues")