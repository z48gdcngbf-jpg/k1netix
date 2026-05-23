from __future__ import annotations
import streamlit as st
import zipfile
import json
import xml.etree.ElementTree as ET
import io
import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from layer4 import render_layer4_ui

from doc_ingester    import ingest_document, ingest_multiple
from vector_store    import (upsert_chunks, query, collection_stats,
                              list_documents, delete_document)
from deepseek_client import DeepSeekClient, build_compliance_prompt
from priority_scorer import update_with_layer3

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K1netix",
    page_icon="🏗️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #f0c040; }

    .metric-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-left: 3px solid #f0c040;
        border-radius: 4px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-family: 'IBM Plex Mono', monospace; color: #f0c040; font-weight: 600; }

    .issue-card {
        background: #1a1d27;
        border: 1px solid #2a2d3a;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
    }
    .issue-card:hover { border-color: #f0c040; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-right: 4px;
    }
    .badge-open    { background: #2a1a1a; color: #ff6b6b; border: 1px solid #ff6b6b44; }
    .badge-closed  { background: #1a2a1a; color: #6bff9e; border: 1px solid #6bff9e44; }
    .badge-inprog  { background: #1a1a2a; color: #6bb5ff; border: 1px solid #6bb5ff44; }
    .badge-default { background: #2a2a1a; color: #f0c040; border: 1px solid #f0c04044; }

    .stDataFrame { background: #1a1d27; }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #3a3d4a;
        border-radius: 8px;
        padding: 10px;
        background: #1a1d27;
    }
    .stButton > button {
        background: #f0c040;
        color: #0f1117;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border: none;
        border-radius: 4px;
    }
    .stButton > button:hover { background: #ffd060; color: #0f1117; }
    .stTabs [data-baseweb="tab"] { color: #888; }
    .stTabs [aria-selected="true"] { color: #f0c040; border-bottom-color: #f0c040; }
</style>
""", unsafe_allow_html=True)


# ── BCF Parser ────────────────────────────────────────────────────────────────

def safe_find_text(element, path, ns=""):
    """Safely extract text from an XML element."""
    if element is None:
        return None
    tag = f"{ns}{path}" if ns else path
    found = element.find(tag)
    return found.text.strip() if found is not None and found.text else None


def parse_markup(markup_xml: bytes, ns_map: dict) -> dict:
    """Parse a markup.bcf XML file into a dict."""
    root = ET.fromstring(markup_xml)
    ns = ns_map.get("markup", "")

    # Topic
    topic = root.find(f"{ns}Topic") or root.find("Topic")
    if topic is None:
        return {}

    def tf(path):
        return safe_find_text(topic, path, "")

    issue = {
        "guid":         topic.get("Guid", topic.get("guid", "")),
        "type":         topic.get("TopicType", ""),
        "status":       topic.get("TopicStatus", ""),
        "title":        tf("Title"),
        "description":  tf("Description"),
        "priority":     tf("Priority"),
        "assigned_to":  tf("AssignedTo"),
        "creation_date": tf("CreationDate"),
        "creation_author": tf("CreationAuthor"),
        "modified_date": tf("ModifiedDate"),
        "modified_author": tf("ModifiedAuthor"),
        "due_date":     tf("DueDate"),
        "stage":        tf("Stage"),
        "index":        tf("Index"),
        "labels":       [],
        "comments":     [],
        "viewpoints":   [],
        "related_topics": [],
        "bim_snippet":  None,
        "reference_links": [],
    }

    # Labels
    for label in topic.findall("Label"):
        if label.text:
            issue["labels"].append(label.text.strip())

    # Reference links
    for link in topic.findall("ReferenceLink"):
        if link.text:
            issue["reference_links"].append(link.text.strip())

    # BIM Snippet
    snippet = topic.find("BimSnippet")
    if snippet is not None:
        issue["bim_snippet"] = {
            "type": snippet.get("SnippetType", ""),
            "is_external": snippet.get("isExternal", "false"),
            "reference": safe_find_text(snippet, "Reference"),
            "reference_schema": safe_find_text(snippet, "ReferenceSchema"),
        }

    # Comments
    for comment_el in root.findall("Comment"):
        issue["comments"].append({
            "guid":    comment_el.get("Guid", ""),
            "date":    safe_find_text(comment_el, "Date"),
            "author":  safe_find_text(comment_el, "Author"),
            "comment": safe_find_text(comment_el, "Comment"),
            "viewpoint_ref": safe_find_text(comment_el, "Viewpoint"),
        })

    # Viewpoints references
    for vp_el in root.findall("Viewpoints"):
        issue["viewpoints"].append({
            "guid":      vp_el.get("Guid", ""),
            "viewpoint": safe_find_text(vp_el, "Viewpoint"),
            "snapshot":  safe_find_text(vp_el, "Snapshot"),
            "index":     safe_find_text(vp_el, "Index"),
        })

    # Related topics
    for rt in topic.findall("RelatedTopic"):
        issue["related_topics"].append(rt.get("Guid", ""))

    return issue


def parse_viewpoint(vp_xml: bytes) -> dict:
    """Parse a .bcfv viewpoint file."""
    root = ET.fromstring(vp_xml)
    vp = {"camera": None, "components": [], "lines": [], "clipping_planes": []}

    # Perspective camera
    pc = root.find(".//PerspectiveCamera")
    if pc is not None:
        def v3(el, tag):
            e = el.find(tag)
            if e is None:
                return None
            return {
                "x": safe_find_text(e, "X"),
                "y": safe_find_text(e, "Y"),
                "z": safe_find_text(e, "Z"),
            }
        vp["camera"] = {
            "type": "perspective",
            "point":     v3(pc, "CameraViewPoint"),
            "direction": v3(pc, "CameraDirection"),
            "up_vector": v3(pc, "CameraUpVector"),
            "field_of_view": safe_find_text(pc, "FieldOfView"),
        }

    # Orthogonal camera
    oc = root.find(".//OrthogonalCamera")
    if oc is not None and vp["camera"] is None:
        vp["camera"] = {
            "type": "orthogonal",
            "point":     {"x": safe_find_text(oc, "CameraViewPoint/X"),
                          "y": safe_find_text(oc, "CameraViewPoint/Y"),
                          "z": safe_find_text(oc, "CameraViewPoint/Z")},
            "view_to_world_scale": safe_find_text(oc, "ViewToWorldScale"),
        }

    # Components
    for comp in root.findall(".//Component"):
        vp["components"].append({
            "ifc_guid":      comp.get("IfcGuid", ""),
            "originating_system": comp.get("OriginatingSystem", ""),
            "authoring_tool_id":  comp.get("AuthoringToolId", ""),
        })

    # Clipping planes
    for cp in root.findall(".//ClippingPlane"):
        vp["clipping_planes"].append({
            "location":  {"x": safe_find_text(cp, "Location/X"),
                          "y": safe_find_text(cp, "Location/Y"),
                          "z": safe_find_text(cp, "Location/Z")},
            "direction": {"x": safe_find_text(cp, "Direction/X"),
                          "y": safe_find_text(cp, "Direction/Y"),
                          "z": safe_find_text(cp, "Direction/Z")},
        })

    return vp


def parse_project(project_xml: bytes) -> dict:
    """Parse project.bcfp"""
    root = ET.fromstring(project_xml)
    proj = root.find(".//Project") or root
    return {
        "project_id":   proj.get("ProjectId", ""),
        "name":         safe_find_text(proj, "Name"),
        "extension_schema": safe_find_text(proj, "ExtensionSchema"),
    }


def parse_version(version_xml: bytes) -> dict:
    """Parse bcf.version"""
    root = ET.fromstring(version_xml)
    return {
        "version_id":     root.get("VersionId", ""),
        "detailed_version": safe_find_text(root, "DetailedVersion"),
    }


def bcfzip_to_json(file_bytes: bytes) -> dict:
    """Convert a bcfzip file bytes to a structured JSON-serialisable dict."""
    result = {
        "version":  {},
        "project":  {},
        "issues":   [],
        "snapshots": {},   # guid -> base64 png
        "parse_errors": [],
    }

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        names = z.namelist()

        # Version
        version_files = [n for n in names if n.endswith("bcf.version") or n == "bcf.version"]
        if version_files:
            try:
                result["version"] = parse_version(z.read(version_files[0]))
            except Exception as e:
                result["parse_errors"].append(f"version parse error: {e}")

        # Project
        project_files = [n for n in names if n.endswith("project.bcfp") or n == "project.bcfp"]
        if project_files:
            try:
                result["project"] = parse_project(z.read(project_files[0]))
            except Exception as e:
                result["parse_errors"].append(f"project parse error: {e}")

        # Issues (each GUID folder contains markup.bcf)
        markup_files = [n for n in names if n.endswith("markup.bcf")]
        for mf in markup_files:
            folder = mf.rsplit("/", 1)[0] if "/" in mf else ""
            try:
                issue = parse_markup(z.read(mf), {})
                if not issue:
                    continue

                # Attach viewpoint data
                vp_files = [n for n in names
                            if n.startswith(folder + "/") and n.endswith(".bcfv")]
                for vpf in vp_files:
                    try:
                        vp_data = parse_viewpoint(z.read(vpf))
                        vp_filename = vpf.rsplit("/", 1)[-1]
                        for vp_ref in issue["viewpoints"]:
                            if vp_ref.get("viewpoint") == vp_filename:
                                vp_ref["data"] = vp_data
                    except Exception as e:
                        result["parse_errors"].append(f"viewpoint parse error {vpf}: {e}")

                # Attach snapshots as base64
                snap_files = [n for n in names
                              if n.startswith(folder + "/") and
                              (n.endswith(".png") or n.endswith(".jpg") or n.endswith(".jpeg"))]
                for sf in snap_files:
                    snap_filename = sf.rsplit("/", 1)[-1]
                    img_b64 = base64.b64encode(z.read(sf)).decode()
                    ext = sf.rsplit(".", 1)[-1].lower()
                    mime = "image/png" if ext == "png" else "image/jpeg"
                    data_uri = f"data:{mime};base64,{img_b64}"
                    for vp_ref in issue["viewpoints"]:
                        if vp_ref.get("snapshot") == snap_filename:
                            vp_ref["snapshot_data"] = data_uri
                    # Also store in top-level by guid
                    result["snapshots"][issue.get("guid", folder)] = data_uri

                result["issues"].append(issue)

            except Exception as e:
                result["parse_errors"].append(f"markup parse error {mf}: {e}")

    return result


# ── UI helpers ────────────────────────────────────────────────────────────────

STATUS_COLORS = {
    "open":        "badge-open",
    "closed":      "badge-closed",
    "in progress": "badge-inprog",
    "resolved":    "badge-closed",
    "active":      "badge-inprog",
}

def status_badge(status: str) -> str:
    if not status:
        return ""
    cls = STATUS_COLORS.get(status.lower(), "badge-default")
    return f'<span class="badge {cls}">{status}</span>'


def type_badge(t: str) -> str:
    if not t:
        return ""
    return f'<span class="badge badge-default">{t}</span>'


def issues_to_dataframe(issues: list) -> pd.DataFrame:
    rows = []
    for iss in issues:
        rows.append({
            "GUID":          iss.get("guid", ""),
            "Title":         iss.get("title", ""),
            "Status":        iss.get("status", ""),
            "Type":          iss.get("type", ""),
            "Priority":      iss.get("priority", ""),
            "Assigned To":   iss.get("assigned_to", ""),
            "Created":       iss.get("creation_date", ""),
            "Author":        iss.get("creation_author", ""),
            "Comments":      len(iss.get("comments", [])),
            "Viewpoints":    len(iss.get("viewpoints", [])),
        })
    return pd.DataFrame(rows)


# ── Main App ──────────────────────────────────────────────────────────────────

st.markdown("## 🏗️ K1netix")
st.markdown("Upload a `.bcfzip` file to inspect issues, comments, and viewpoints.")

uploaded = st.file_uploader(
    "Drop your BCFzip file here",
    type=["bcfzip", "zip"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding: 60px 0; color: #555;">
        <div style="font-size: 48px;">📂</div>
        <div style="font-family: 'IBM Plex Mono', monospace; margin-top: 8px;">
            Awaiting .bcfzip file…
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Parse ─────────────────────────────────────────────────────────────────────
with st.spinner("Parsing BCF file…"):
    file_bytes = uploaded.read()
    data = bcfzip_to_json(file_bytes)

# Make Layer 1 output available to later pipeline stages without requiring
# a manually-created bcf_output.json file. This is the in-memory handoff that
# makes the prototype behave like an iPaaS pipeline.
st.session_state["bcf_json"] = data

# Also write a convenience copy beside main.py for debugging or CLI use.
bcf_output_path = Path(__file__).with_name("bcf_output.json")
with open(bcf_output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

if data["parse_errors"]:
    with st.expander("⚠️ Parse warnings", expanded=False):
        for err in data["parse_errors"]:
            st.warning(err)

issues = data["issues"]

# ── Summary metrics ───────────────────────────────────────────────────────────
project_name = data["project"].get("name") or uploaded.name
bcf_version  = data["version"].get("version_id", "?")

st.markdown(f"### {project_name} &nbsp; `BCF {bcf_version}`", unsafe_allow_html=True)

total     = len(issues)
open_cnt  = sum(1 for i in issues if (i.get("status") or "").lower() in ("open", "active"))
closed_cnt = sum(1 for i in issues if (i.get("status") or "").lower() in ("closed", "resolved"))
comments  = sum(len(i.get("comments", [])) for i in issues)

col1, col2, col3, col4 = st.columns(4)
for col, label, value in [
    (col1, "Total Issues",    total),
    (col2, "Open",            open_cnt),
    (col3, "Closed/Resolved", closed_cnt),
    (col4, "Total Comments",  comments),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_table, tab_cards, tab_json = st.tabs(["📊 Table View", "🗂 Issue Cards", "📄 Raw JSON"])

# ── Table View ────────────────────────────────────────────────────────────────
with tab_table:
    df = issues_to_dataframe(issues)
    if df.empty:
        st.info("No issues found in this BCF file.")
    else:
        # Filters
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            statuses = ["All"] + sorted(df["Status"].dropna().unique().tolist())
            sel_status = st.selectbox("Filter by Status", statuses)
        with fcol2:
            types = ["All"] + sorted(df["Type"].dropna().unique().tolist())
            sel_type = st.selectbox("Filter by Type", types)
        with fcol3:
            search = st.text_input("Search title…", placeholder="keyword")

        filtered = df.copy()
        if sel_status != "All":
            filtered = filtered[filtered["Status"] == sel_status]
        if sel_type != "All":
            filtered = filtered[filtered["Type"] == sel_type]
        if search:
            filtered = filtered[filtered["Title"].str.contains(search, case=False, na=False)]

        st.dataframe(
            filtered.drop(columns=["GUID"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing {len(filtered)} of {len(df)} issues")

        # Download JSON
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Download as JSON",
            data=json_str,
            file_name=uploaded.name.replace(".bcfzip", ".json").replace(".zip", ".json"),
            mime="application/json",
        )

# ── Issue Cards ───────────────────────────────────────────────────────────────
with tab_cards:
    if not issues:
        st.info("No issues found.")
    else:
        for iss in issues:
            title       = iss.get("title") or "(Untitled)"
            status      = iss.get("status", "")
            itype       = iss.get("type", "")
            assigned    = iss.get("assigned_to", "—")
            priority    = iss.get("priority", "—")
            desc        = iss.get("description", "")
            comments_l  = iss.get("comments", [])
            viewpoints  = iss.get("viewpoints", [])
            created     = iss.get("creation_date", "")
            author      = iss.get("creation_author", "")

            with st.expander(
                f"{status_badge(status)} {type_badge(itype)} **{title}**",
                expanded=False
            ):
                icol1, icol2 = st.columns([2, 1])
                with icol1:
                    if desc:
                        st.markdown(f"**Description:** {desc}")
                    st.markdown(
                        f"**Assigned to:** `{assigned}` &nbsp;|&nbsp; "
                        f"**Priority:** `{priority}` &nbsp;|&nbsp; "
                        f"**Author:** `{author}`",
                        unsafe_allow_html=True
                    )
                    if created:
                        st.caption(f"Created: {created}")

                    # Comments
                    if comments_l:
                        st.markdown("**💬 Comments**")
                        for c in comments_l:
                            st.markdown(
                                f"> {c.get('comment', '')}  \n"
                                f"<small>— {c.get('author', '?')} &nbsp; {c.get('date', '')}</small>",
                                unsafe_allow_html=True
                            )

                with icol2:
                    # Show snapshot if available
                    for vp in viewpoints:
                        snap = vp.get("snapshot_data")
                        if snap:
                            st.image(snap, caption="Viewpoint snapshot", use_column_width=True)
                            break

                    # Viewpoint camera info
                    for vp in viewpoints:
                        cam = (vp.get("data") or {}).get("camera")
                        if cam:
                            st.markdown(f"**📷 Camera:** `{cam.get('type', '?')}`")
                            comps = (vp.get("data") or {}).get("components", [])
                            if comps:
                                st.markdown(f"**🔩 Components:** {len(comps)} selected")
                            break

# ── Raw JSON ──────────────────────────────────────────────────────────────────
with tab_json:
    # Remove snapshot binary data for readability
    display_data = json.loads(json.dumps(data))  # deep copy
    display_data.pop("snapshots", None)
    for iss in display_data.get("issues", []):
        for vp in iss.get("viewpoints", []):
            vp.pop("snapshot_data", None)

    st.json(display_data, expanded=2)




"""
Layer 2 — Full Pipeline (updated)
BCF JSON → Discipline Classify → XGBoost filter → HDBSCAN cluster → Priority Score

Entry points:
  run_layer2(bcf_json)           → dict with all results
  render_layer2_ui(bcf_json)     → Streamlit panel
"""

import json
from pathlib import Path
from collections import defaultdict

from discipline_classifier import (
    classify_all, discipline_summary, group_by_discipline,
    DISCIPLINE_LABELS, MEP_DISCIPLINES,
)
from noise_filter  import run as run_noise_filter
from clusterer     import run as run_clusterer
from priority_scorer import (
    score_all_issues, score_all_clusters, DEFAULT_WEIGHTS,
)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_layer2(
    bcf_json: dict,
    noise_threshold: float  = 0.5,
    min_cluster_size: int   = 2,
    tfidf_weight: float     = 0.7,
    labeled_csv: str | None = None,
    weights: dict | None    = None,
    verbose: bool           = True,
) -> dict:
    total = len(bcf_json.get("issues", []))
    if verbose:
        _header(f"LAYER 2 — K1netix AI TRIAGE  ({total} issues)")

    # ── Step 1: Discipline classification ─────────────────────────────
    _log(verbose, "1/4  Discipline classification")
    all_issues   = classify_all(bcf_json.get("issues", []))
    disc_summary = discipline_summary(all_issues)

    if verbose:
        for disc, count in disc_summary["by_primary"].items():
            label = DISCIPLINE_LABELS.get(disc, disc)
            print(f"       {label:<22} {count}")
        print(f"       Low-confidence tags: {disc_summary['low_confidence_count']}")

    # ── Step 2: XGBoost noise filter ──────────────────────────────────
    _log(verbose, "2/4  XGBoost noise filter")
    bcf_json_classified = {**bcf_json, "issues": all_issues}

    filter_result = run_noise_filter(
        bcf_json_classified,
        labeled_csv=labeled_csv,
        threshold=noise_threshold,
        retrain=True,
    )

    real_issues   = filter_result["real_issues"]
    noise_issues  = filter_result["noise_issues"]
    noise_metrics = filter_result["metrics"]
    df_scored     = filter_result["df_scored"]

    _log(verbose, f"       Real: {len(real_issues)}  |  Noise: {len(noise_issues)}")

    # ── Step 3: HDBSCAN clustering per discipline ──────────────────────
    _log(verbose, "3/4  HDBSCAN clash grouping")

    all_clusters            = []
    all_unclustered         = []
    cluster_metrics_agg     = defaultdict(int)
    disc_groups             = group_by_discipline(real_issues)

    for disc, disc_issues in disc_groups.items():
        if not disc_issues:
            continue

        cluster_result = run_clusterer(
            disc_issues,
            min_cluster_size=min_cluster_size,
            tfidf_weight=tfidf_weight,
            verbose=False,
        )

        for c in cluster_result["clusters"]:
            c["discipline"]       = disc
            c["discipline_label"] = DISCIPLINE_LABELS.get(disc, disc)
            all_clusters.append(c)

        all_unclustered.extend(cluster_result["unclustered"])
        m = cluster_result["metrics"]
        cluster_metrics_agg["n_clusters"]  += m.get("n_clusters", 0)
        cluster_metrics_agg["n_noise_pts"] += m.get("n_noise_pts", 0)

    _log(verbose, f"       {cluster_metrics_agg['n_clusters']} groups across "
                  f"{len(disc_groups)} disciplines")

    # ── Step 4: Priority scoring ───────────────────────────────────────
    _log(verbose, "4/4  Prioritisation scoring")

    real_issues  = score_all_issues(real_issues,  weights=weights)
    all_clusters = score_all_clusters(all_clusters, weights=weights)
    noise_issues = score_all_issues(noise_issues,  weights=weights)

    if verbose and all_clusters:
        print()
        for c in all_clusters[:5]:
            ps    = c.get("_priority_score", {})
            label = c.get("cluster_label", "?")[:45]
            disc  = c.get("discipline_label", "?")
            score = ps.get("composite", "?")
            band  = ps.get("band", "?")
            print(f"       [{band:<8}] {label:<45} {disc:<15} score={score}")

    if verbose:
        _header("Layer 2 complete → ready for Layer 3 (RAG + DeepSeek)")

    return {
        "real_issues":        real_issues,
        "noise_issues":       noise_issues,
        "clusters":           all_clusters,
        "unclustered":        all_unclustered,
        "by_discipline":      group_by_discipline(real_issues),
        "by_clash_pair":      _group_by_clash_pair(real_issues),
        "discipline_summary": disc_summary,
        "noise_metrics":      noise_metrics,
        "cluster_metrics":    dict(cluster_metrics_agg),
        "df_scored":          df_scored,
    }


def _group_by_clash_pair(issues: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for iss in issues:
        pair = (iss.get("_discipline") or {}).get("clash_pair", "UNKNOWN")
        groups.setdefault(pair, []).append(iss)
    return groups

def _log(verbose: bool, msg: str):
    if verbose:
        print(f"  {msg}")

def _header(msg: str):
    print(f"\n{'─'*62}\n  {msg}\n{'─'*62}")


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def render_layer2_ui(bcf_json: dict):
    import streamlit as st
    import pandas as pd

    st.markdown("## ⚙️ Layer 2 — AI Triage")
    st.caption("Discipline classification → noise filter → clash grouping → priority scoring")

    with st.expander("⚙️ Parameters", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            noise_threshold  = st.slider("Noise threshold",    0.0, 1.0, 0.5, 0.01)
        with pc2:
            min_cluster_size = st.slider("Min cluster size",   2, 10, 2)
        with pc3:
            tfidf_weight     = st.slider("Text/numeric ratio", 0.0, 1.0, 0.7, 0.1)

        st.markdown("**Priority weights** (should sum to 1.0)")
        wc1, wc2, wc3, wc4 = st.columns(4)
        with wc1:
            w1 = st.number_input("LLM certainty",     0.0, 1.0, 0.25, 0.05,
                                  help="Placeholder — Layer 3 will populate")
        with wc2:
            w2 = st.number_input("Data completeness", 0.0, 1.0, 0.25, 0.05)
        with wc3:
            w3 = st.number_input("Classifier prob",   0.0, 1.0, 0.25, 0.05)
        with wc4:
            w4 = st.number_input("Regulation match",  0.0, 1.0, 0.25, 0.05,
                                  help="Placeholder — Layer 3 will populate")

        if abs(w1 + w2 + w3 + w4 - 1.0) > 0.01:
            st.warning(f"Weights sum to {w1+w2+w3+w4:.2f} — should equal 1.0")

    custom_weights = {
        "w1_llm_certainty":     w1,
        "w2_data_completeness": w2,
        "w3_classifier_prob":   w3,
        "w4_regulation_match":  w4,
    }

    if st.button("▶️  Run Layer 2", type="primary"):
        with st.spinner("Classifying · filtering · clustering · scoring…"):
            result = run_layer2(
                bcf_json,
                noise_threshold=noise_threshold,
                min_cluster_size=min_cluster_size,
                tfidf_weight=tfidf_weight,
                weights=custom_weights,
                verbose=False,
            )
        st.session_state["layer2_result"] = result
        st.success("Layer 2 complete ✓")

    result = st.session_state.get("layer2_result")
    if not result:
        return

    # ── Summary metrics ────────────────────────────────────────────────
    st.divider()
    cm = result["cluster_metrics"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Issues",   len(bcf_json.get("issues", [])))
    m2.metric("Real Clashes",   len(result["real_issues"]))
    m3.metric("Noise Removed",  len(result["noise_issues"]))
    m4.metric("Clash Groups",   cm.get("n_clusters", 0))
    m5.metric("Disciplines",    len(result["by_discipline"]))

    tabs = st.tabs(["📐 Disciplines", "🗂 Clash Groups", "🚫 Noise", "📊 Priority", "📄 JSON"])

    # ── Tab 1: Disciplines ─────────────────────────────────────────────
    with tabs[0]:
        ds = result["discipline_summary"]
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**By Primary Discipline**")
            rows = [{"Discipline": DISCIPLINE_LABELS.get(d, d), "Count": c}
                    for d, c in ds["by_primary"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with d2:
            st.markdown("**By Clash Pair**")
            rows = [{"Clash Pair": p, "Count": c}
                    for p, c in ds["by_clash_pair"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        if ds["low_confidence_count"] > 0:
            st.warning(f"{ds['low_confidence_count']} issues had low classification confidence — "
                       "consider enriching BCF titles/descriptions.")

    # ── Tab 2: Clash groups ────────────────────────────────────────────
    with tabs[1]:
        clusters = result["clusters"]
        if not clusters:
            st.info("No clusters formed. Try lowering min cluster size.")
        else:
            band_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            for c in clusters:
                ps    = c.get("_priority_score", {})
                band  = ps.get("band", "?")
                score = ps.get("composite", 0)
                label = c.get("cluster_label", "Unnamed")
                disc  = c.get("discipline_label", "")
                count = c.get("issue_count", 0)
                icon  = band_icon.get(band, "⚪")

                with st.expander(
                    f"{icon} **[{band}]** {label[:50]}  —  {disc}  "
                    f"`{count} issues`  score={score}",
                    expanded=False
                ):
                    cc1, cc2, cc3, cc4 = st.columns(4)
                    cc1.metric("Score",   score)
                    cc2.metric("Completeness",
                               f"{ps.get('components',{}).get('data_completeness',0):.0%}")
                    cc3.metric("Classifier",
                               f"{ps.get('components',{}).get('classifier_prob',0):.0%}")
                    cc4.metric("Issues",  count)

                    issue_rows = [{
                        "Title":    i.get("title","")[:60],
                        "Status":   i.get("status",""),
                        "Priority": i.get("priority",""),
                        "Score":    (i.get("_priority_score") or {}).get("composite",""),
                    } for i in c.get("issues", [])]
                    st.dataframe(pd.DataFrame(issue_rows), hide_index=True,
                                 use_container_width=True)

    # ── Tab 3: Noise ───────────────────────────────────────────────────
    with tabs[2]:
        noise = result["noise_issues"]
        if not noise:
            st.success("No noise detected!")
        else:
            rows = [{
                "Title":      i.get("title",""),
                "Discipline": DISCIPLINE_LABELS.get(
                                  (i.get("_discipline") or {}).get("primary",""), "?"),
                "Noise Prob": f"{(i.get('_noise_filter') or {}).get('noise_prob',0):.0%}",
                "Reason":     (i.get("_noise_filter") or {}).get("noise_reason",""),
            } for i in noise]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Tab 4: Priority table ──────────────────────────────────────────
    with tabs[3]:
        rows = []
        for i in result["real_issues"]:
            ps   = i.get("_priority_score") or {}
            disc = i.get("_discipline") or {}
            rows.append({
                "Title":       i.get("title","")[:55],
                "Discipline":  DISCIPLINE_LABELS.get(disc.get("primary",""),"?"),
                "Clash Pair":  disc.get("clash_pair",""),
                "Band":        ps.get("band",""),
                "Score":       ps.get("composite", 0),
                "Completeness": f"{ps.get('data_completeness',0):.0%}",
                "Classifier":  f"{ps.get('classifier_prob',0):.0%}",
            })

        df_p = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df_p.empty and "Score" in df_p.columns:
            df_p = df_p.sort_values("Score", ascending=False)

        st.dataframe(df_p, hide_index=True, use_container_width=True)
    
        st.download_button(
            "⬇️  Download Layer 2 JSON",
            data=_export_json(result),
            file_name="k1netix_layer2.json",
            mime="application/json",
        )

    # ── Tab 5: JSON ────────────────────────────────────────────────────
    with tabs[4]:
        st.json(json.loads(_export_json(result)), expanded=2)


def _export_json(result: dict) -> str:
    export = {
        "discipline_summary": result["discipline_summary"],
        "cluster_metrics":    result["cluster_metrics"],
        "noise_metrics":      result.get("noise_metrics", {}),
        "clusters": result["clusters"],
        "noise_issues": [
            {"guid": i.get("guid"), "title": i.get("title"),
             "discipline": i.get("_discipline"),
             "noise_filter": i.get("_noise_filter")}
            for i in result["noise_issues"]
        ],
        # Ready-to-consume input for Layer 3
        "layer3_inputs": [
            {
                "cluster_id":     c.get("cluster_id"),
                "cluster_label":  c.get("cluster_label"),
                "discipline":     c.get("discipline"),
                "discipline_label": c.get("discipline_label"),
                "clash_pair":     (c.get("issues") or [{}])[0]
                                  .get("_discipline", {}).get("clash_pair"),
                "priority_score": (c.get("_priority_score") or {}).get("composite"),
                "band":           (c.get("_priority_score") or {}).get("band"),
                "issue_count":    c.get("issue_count"),
                "issues":         c.get("issues", []),
            }
            for c in result["clusters"]
        ],
    }
    return json.dumps(export, indent=2, default=str)






# ── Layer 3 — Compliance Pipeline ────────────────────────────────────────────
# Chains: Layer 2 clusters → RAG retrieval → DeepSeek compliance check
 
 
# ── RAG retrieval ─────────────────────────────────────────────────────────────
 
def retrieve_clauses(
    cluster: dict,
    n_results: int = 8,
    jurisdiction: Optional[str] = None,
) -> list[dict]:
    """
    Build a rich query from the cluster and retrieve relevant regulation clauses.
    Uses discipline filter for more precise retrieval.
    """
    issues = cluster.get("issues", [])
    label  = cluster.get("cluster_label", "")
    disc   = cluster.get("discipline", "")
 
    # Build a semantically rich query string
    titles = " ".join(i.get("title","") for i in issues[:5] if i.get("title"))
    descs  = " ".join(i.get("description","")[:100] for i in issues[:3] if i.get("description"))
    pair   = (issues[0].get("_discipline") or {}).get("clash_pair","") if issues else ""
 
    query_text = f"{label} {pair} {disc} {titles} {descs}".strip()[:500]
 
    # Map internal discipline codes to vector store filter values
    disc_filter = {
        "MECH": "MEP", "ELEC": "MEP", "PLMB": "MEP", "FP": "FP",
        "STR": "STR", "ARCH": "ARCH", "CIVIL": "CIVIL",
    }.get(disc, None)
 
    return query(
        query_text          = query_text,
        n_results           = n_results,
        discipline_filter   = disc_filter,
        jurisdiction_filter = jurisdiction,
    )
 
 
# ── Per-cluster compliance check ──────────────────────────────────────────────
 
def check_cluster(
    cluster: dict,
    client: DeepSeekClient,
    n_clauses: int = 8,
    jurisdiction: Optional[str] = None,
) -> dict:
    """
    Run RAG retrieval + DeepSeek compliance check for one cluster.
    Returns the cluster dict enriched with _compliance key.
    """
    retrieved = retrieve_clauses(cluster, n_results=n_clauses, jurisdiction=jurisdiction)
 
    # If no regulations loaded, return a graceful degraded result
    if not retrieved:
        cluster["_compliance"] = {
            "compliance_status":          "INSUFFICIENT_DATA",
            "llm_certainty":              0.1,
            "regulation_match_quality":   0.0,
            "clause_checks":              [],
            "constructability_assessment": "No regulation documents have been loaded into the knowledge base.",
            "primary_concern":            "Regulation knowledge base is empty.",
            "recommended_action":         "Upload relevant building regulations in the Layer 3 panel.",
            "suggested_solutions":        [],
            "requires_specialist":        True,
            "specialist_discipline":      cluster.get("discipline_label",""),
            "summary":                    "Compliance check could not be performed: no regulations loaded.",
            "data_gaps":                  ["No regulation documents in vector store"],
            "retrieved_clauses":          [],
            "error":                      None,
        }
        return cluster
 
    prompt = build_compliance_prompt(cluster, retrieved)
 
    try:
        result = client.chat_json(user_prompt=prompt)
 
        # Attach retrieved clauses for transparency
        result["retrieved_clauses"] = [
            {
                "text":          c["text"][:300],
                "source":        c["source"],
                "section":       c["section"],
                "section_title": c["section_title"],
                "score":         c["score"],
            }
            for c in retrieved
        ]
        result["error"] = None
        cluster["_compliance"] = result
 
        # Update priority score with Layer 3 values
        llm_cert  = float(result.get("llm_certainty", 0.5))
        reg_match = float(result.get("regulation_match_quality", 0.5))
        cluster   = update_with_layer3(cluster, llm_cert, reg_match)
 
    except Exception as e:
        cluster["_compliance"] = {
            "compliance_status":        "INSUFFICIENT_DATA",
            "llm_certainty":            0.1,
            "regulation_match_quality": 0.0,
            "clause_checks":            [],
            "summary":                  f"Compliance check failed: {str(e)[:200]}",
            "error":                    str(e),
            "retrieved_clauses":        retrieved,
        }
 
    return cluster
 
 
# ── Full Layer 3 pipeline ─────────────────────────────────────────────────────
 
def run_layer3(
    layer2_result: dict,
    api_key: Optional[str] = None,
    n_clauses: int = 8,
    jurisdiction: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Run Layer 3 on all clusters from Layer 2.
 
    Args:
        layer2_result : output dict from layer2/pipeline.py run_layer2()
        api_key       : DeepSeek API key (or set DEEPSEEK_API_KEY env var)
        n_clauses     : number of regulation clauses to retrieve per cluster
        jurisdiction  : filter regulations by jurisdiction (e.g. "UK", "Singapore")
 
    Returns:
        layer2_result enriched with _compliance on every cluster,
        plus a top-level compliance_summary.
    """
    clusters = layer2_result.get("clusters", [])
    if not clusters:
        return {**layer2_result, "compliance_summary": {"error": "No clusters to check"}}
 
    client = DeepSeekClient(api_key=api_key)
 
    if verbose:
        print(f"\n{'─'*62}")
        print(f"  LAYER 3 — COMPLIANCE CHECK ({len(clusters)} clusters)")
        store = collection_stats()
        print(f"  Knowledge base: {store['total_chunks']} regulation chunks")
        print(f"{'─'*62}")
 
    enriched_clusters = []
    statuses = {"COMPLIANT": 0, "NON_COMPLIANT": 0, "NEEDS_REVIEW": 0, "INSUFFICIENT_DATA": 0}
 
    for i, cluster in enumerate(clusters):
        label = cluster.get("cluster_label","?")[:45]
        disc  = cluster.get("discipline_label","")
        if verbose:
            print(f"  [{i+1}/{len(clusters)}] {label:<45} ({disc})")
 
        enriched = check_cluster(cluster, client,
                                 n_clauses=n_clauses,
                                 jurisdiction=jurisdiction)
        enriched_clusters.append(enriched)
 
        status = enriched.get("_compliance",{}).get("compliance_status","INSUFFICIENT_DATA")
        statuses[status] = statuses.get(status, 0) + 1
 
        if verbose:
            cert  = enriched.get("_compliance",{}).get("llm_certainty",0)
            score = (enriched.get("_priority_score") or {}).get("composite","?")
            print(f"         → {status:<20} certainty={cert:.2f}  final_score={score}")
 
        # Small delay to avoid rate limiting
        time.sleep(0.3)
 
    compliance_summary = {
        "total_clusters_checked": len(enriched_clusters),
        "status_breakdown":       statuses,
        "tokens_used":            client.token_usage(),
        "knowledge_base_chunks":  collection_stats()["total_chunks"],
    }
 
    if verbose:
        print(f"\n  ✅ Layer 3 complete")
        print(f"     {statuses}")
        print(f"     Tokens used: {client.token_usage():,}")
 
    return {
        **layer2_result,
        "clusters":           enriched_clusters,
        "compliance_summary": compliance_summary,
    }
 
 
# ── Streamlit UI ──────────────────────────────────────────────────────────────
 
def render_layer3_ui(layer2_result: Optional[dict] = None):
    import streamlit as st
    import pandas as pd
 
    st.markdown("## 🔍 Layer 3 — Regulation Compliance")
    st.caption("Upload building regulations → RAG retrieval → DeepSeek clause-by-clause check")
 
    # ── Section A: Knowledge base management ──────────────────────────
    st.markdown("### 📚 Knowledge Base")
 
    kb_col1, kb_col2 = st.columns([2, 1])
 
    with kb_col1:
        reg_files = st.file_uploader(
            "Upload regulation documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            key="reg_upload",
        )
 
        jc1, jc2 = st.columns(2)
        with jc1:
            jurisdiction = st.selectbox(
                "Jurisdiction",
                ["general", "UK", "Singapore", "UAE", "Australia", "USA", "EU"],
                key="jurisdiction",
            )
        with jc2:
            doc_type = st.selectbox(
                "Document type",
                ["auto", "regulation", "standard", "spec", "guidance"],
                key="doc_type",
            )
 
        if reg_files:
            if st.button("📥  Ingest documents", key="ingest_btn"):
                progress = st.progress(0)
                total_chunks = 0
                for idx, f in enumerate(reg_files):
                    with st.spinner(f"Ingesting {f.name}…"):
                        try:
                            chunks = ingest_document(
                                f.read(), f.name,
                                jurisdiction = jurisdiction,
                                document_type = doc_type,
                            )
                            result = upsert_chunks(chunks)
                            total_chunks += result["upserted"]
                            st.success(f"✓ {f.name} — {result['upserted']} chunks")
                        except Exception as e:
                            st.error(f"✗ {f.name}: {e}")
                    progress.progress((idx + 1) / len(reg_files))
 
                st.success(f"✅ {total_chunks} total chunks added to knowledge base")
                st.rerun()
 
    with kb_col2:
        stats = collection_stats()
        st.metric("Chunks in KB", stats["total_chunks"])
 
        docs = list_documents()
        if docs:
            st.markdown("**Loaded documents:**")
            for d in docs:
                dc1, dc2 = st.columns([3, 1])
                dc1.caption(f"📄 {d['source']} ({d['jurisdiction']})")
                if dc2.button("🗑", key=f"del_{d['source']}",
                               help=f"Remove {d['source']}"):
                    delete_document(d["source"])
                    st.rerun()
        else:
            st.info("No documents loaded yet.")
 
    st.divider()
 
    # ── Section B: Run compliance check ───────────────────────────────
    st.markdown("### 🤖 DeepSeek Compliance Check")
 
    if layer2_result is None:
        layer2_result = st.session_state.get("layer2_result")
 
    if not layer2_result or not layer2_result.get("clusters"):
        st.info("Run Layer 2 first to generate clash groups.")
        return
 
    clusters = layer2_result.get("clusters", [])
    st.caption(f"{len(clusters)} clash groups ready for compliance check")
 
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        api_key_input = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.environ.get("DEEPSEEK_API_KEY",""),
            help="Get your key at platform.deepseek.com",
        )
    with rc2:
        n_clauses = st.slider("Clauses per check", 4, 15, 8,
                               help="More clauses = better coverage but higher cost")
    with rc3:
        check_jurisdiction = st.selectbox(
            "Filter regulations by",
            ["all"] + ["UK","Singapore","UAE","Australia","USA","EU","general"],
            key="check_juris",
        )
 
    check_juris = None if check_jurisdiction == "all" else check_jurisdiction
 
    if stats["total_chunks"] == 0:
        st.warning("⚠️ Knowledge base is empty — upload regulation documents above first.")
 
    if st.button("▶️  Run Compliance Check", type="primary", key="run_l3"):
        if not api_key_input:
            st.error("Enter your DeepSeek API key.")
        else:
            os.environ["DEEPSEEK_API_KEY"] = api_key_input
            prog_bar = st.progress(0)
            status_box = st.empty()
 
            enriched_clusters = []
            client = DeepSeekClient(api_key=api_key_input)
            stat_counts = {"COMPLIANT": 0, "NON_COMPLIANT": 0,
                           "NEEDS_REVIEW": 0, "INSUFFICIENT_DATA": 0}
 
            for i, cluster in enumerate(clusters):
                label = cluster.get("cluster_label","?")[:40]
                status_box.info(f"Checking [{i+1}/{len(clusters)}]: {label}…")
 
                enriched = check_cluster(
                    cluster, client,
                    n_clauses=n_clauses,
                    jurisdiction=check_juris,
                )
                enriched_clusters.append(enriched)
                s = enriched.get("_compliance",{}).get("compliance_status","INSUFFICIENT_DATA")
                stat_counts[s] = stat_counts.get(s, 0) + 1
                prog_bar.progress((i + 1) / len(clusters))
                time.sleep(0.2)
 
            status_box.empty()
 
            full_result = {
                **layer2_result,
                "clusters": enriched_clusters,
                "compliance_summary": {
                    "total_checked":          len(enriched_clusters),
                    "status_breakdown":       stat_counts,
                    "tokens_used":            client.token_usage(),
                    "knowledge_base_chunks":  stats["total_chunks"],
                },
            }
            st.session_state["layer3_result"] = full_result
            st.success(f"✅ Compliance check complete · {client.token_usage():,} tokens used")
            st.rerun()
 
    # ── Section C: Results ─────────────────────────────────────────────
    result = st.session_state.get("layer3_result")
    if not result:
        return
 
    comp_sum = result.get("compliance_summary", {})
    breakdown = comp_sum.get("status_breakdown", {})
 
    st.divider()
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Non-Compliant",    breakdown.get("NON_COMPLIANT", 0))
    sm2.metric("Needs Review",     breakdown.get("NEEDS_REVIEW", 0))
    sm3.metric("Compliant",        breakdown.get("COMPLIANT", 0))
    sm4.metric("Insufficient Data",breakdown.get("INSUFFICIENT_DATA", 0))
    sm5.metric("Tokens Used",      f"{comp_sum.get('tokens_used',0):,}")
 
    st.divider()
 
    # Results tabs
    rtab1, rtab2, rtab3 = st.tabs(["🗂 Cluster Results", "📊 Summary Table", "📄 JSON"])
 
    # ── Cluster results ────────────────────────────────────────────────
    with rtab1:
        STATUS_ICON = {
            "NON_COMPLIANT":    "🔴",
            "NEEDS_REVIEW":     "🟠",
            "COMPLIANT":        "🟢",
            "INSUFFICIENT_DATA":"⚪",
        }
 
        for c in result["clusters"]:
            comp   = c.get("_compliance") or {}
            status = comp.get("compliance_status","INSUFFICIENT_DATA")
            ps     = c.get("_priority_score") or {}
            band   = ps.get("band","?")
            score  = ps.get("composite",0)
            label  = c.get("cluster_label","?")
            disc   = c.get("discipline_label","")
            icon   = STATUS_ICON.get(status,"⚪")
            cert   = comp.get("llm_certainty",0)
            rm     = comp.get("regulation_match_quality",0)
 
            with st.expander(
                f"{icon} **[{band}]** {label[:50]}  —  {disc}  "
                f"`score={score}`  `{status}`",
                expanded=(status == "NON_COMPLIANT"),
            ):
                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("Final Score",    score)
                ec2.metric("LLM Certainty",  f"{cert:.0%}")
                ec3.metric("Reg. Match",     f"{rm:.0%}")
                ec4.metric("Status",         status)
 
                # Primary concern + recommendation
                if comp.get("primary_concern"):
                    st.error(f"**⚠️ Primary concern:** {comp['primary_concern']}")
                if comp.get("recommended_action"):
                    st.warning(f"**💡 Recommended action:** {comp['recommended_action']}")
 
                # Summary
                if comp.get("summary"):
                    st.markdown(f"**Summary:** {comp['summary']}")
 
                # Constructability
                if comp.get("constructability_assessment"):
                    with st.expander("🏗️ Constructability assessment"):
                        st.markdown(comp["constructability_assessment"])
 
                # Clause checks
                clause_checks = comp.get("clause_checks",[])
                if clause_checks:
                    st.markdown("**📋 Clause-by-Clause Checks**")
                    clause_rows = [{
                        "Clause":     ch.get("clause_ref",""),
                        "Summary":    ch.get("clause_summary",""),
                        "Applies":    "✓" if ch.get("applies") else "✗",
                        "Status":     ch.get("status",""),
                        "Certainty":  f"{ch.get('certainty',0):.0%}",
                        "Reasoning":  ch.get("reasoning",""),
                    } for ch in clause_checks]
                    st.dataframe(pd.DataFrame(clause_rows),
                                 hide_index=True, use_container_width=True)
 
                # Suggested solutions
                solutions = comp.get("suggested_solutions",[])
                if solutions:
                    st.markdown("**🔧 Suggested Solutions**")
                    for j, sol in enumerate(solutions, 1):
                        st.markdown(f"{j}. {sol}")
 
                # Specialist flag
                if comp.get("requires_specialist"):
                    st.info(f"🧑‍🔬 Specialist review recommended: "
                            f"{comp.get('specialist_discipline','')}")
 
                # Retrieved clauses
                retrieved = comp.get("retrieved_clauses",[])
                if retrieved:
                    with st.expander(f"📑 Retrieved regulation clauses ({len(retrieved)})"):
                        for rc in retrieved:
                            st.markdown(
                                f"**{rc.get('source','')} § {rc.get('section','')}**  "
                                f"— {rc.get('section_title','')}  "
                                f"`score={rc.get('score',0):.2f}`"
                            )
                            st.caption(rc.get("text","")[:300] + "…")
 
                # Data gaps
                gaps = comp.get("data_gaps",[])
                if gaps:
                    st.caption(f"⚠️ Data gaps: {' · '.join(gaps)}")
 
                # Error
                if comp.get("error"):
                    st.error(f"Error: {comp['error']}")
 
    # ── Summary table ──────────────────────────────────────────────────
    with rtab2:
        rows = []
        for c in result["clusters"]:
            comp = c.get("_compliance") or {}
            ps   = c.get("_priority_score") or {}
            rows.append({
                "Cluster":        c.get("cluster_label","")[:50],
                "Discipline":     c.get("discipline_label",""),
                "Band":           ps.get("band",""),
                "Final Score":    ps.get("composite",0),
                "Status":         comp.get("compliance_status",""),
                "LLM Certainty":  f"{comp.get('llm_certainty',0):.0%}",
                "Reg. Match":     f"{comp.get('regulation_match_quality',0):.0%}",
                "Issues":         c.get("issue_count",0),
                "Specialist":     "Yes" if comp.get("requires_specialist") else "No",
            })
 
        df_sum = pd.DataFrame(rows).sort_values("Final Score", ascending=False)
        st.dataframe(df_sum, hide_index=True, use_container_width=True)
 
        st.download_button(
            "⬇️  Download Layer 3 JSON",
            data=_export_layer3_json(result),
            file_name="k1netix_layer3.json",
            mime="application/json",
        )
 
    # ── JSON ───────────────────────────────────────────────────────────
    with rtab3:
        st.json(json.loads(_export_layer3_json(result)), expanded=2)
 
 
def _export_layer3_json(result: dict) -> str:
    export = {
        "compliance_summary": result.get("compliance_summary", {}),
        "clusters": [
            {
                "cluster_id":       c.get("cluster_id"),
                "cluster_label":    c.get("cluster_label"),
                "discipline":       c.get("discipline"),
                "discipline_label": c.get("discipline_label"),
                "issue_count":      c.get("issue_count"),
                "priority_score":   c.get("_priority_score"),
                "compliance":       c.get("_compliance"),
            }
            for c in result.get("clusters", [])
        ],
    }
    return json.dumps(export, indent=2, default=str)
 
 
# ── Main app entry point ──────────────────────────────────────────────────────

render_layer2_ui(st.session_state.get("bcf_json"))
render_layer3_ui(st.session_state.get("layer2_result"))
render_layer4_ui(st.session_state.get("layer3_result"))
