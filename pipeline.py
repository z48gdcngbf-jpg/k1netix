"""
Layer 3 — Compliance Pipeline
Chains: Layer 2 clusters → RAG retrieval → DeepSeek compliance check → updated priority scores

Entry points:
  run_layer3(clusters, api_key)   → enriched clusters with compliance results
  render_layer3_ui(layer2_result) → Streamlit panel
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from doc_ingester   import ingest_document, ingest_multiple
from vector_store   import (upsert_chunks, query, collection_stats,
                             list_documents, delete_document)
from deepseek_client import DeepSeekClient, build_compliance_prompt

# Layer 2 priority scorer update hook
sys.path.insert(0, str(Path(__file__).parent.parent / "layer2"))
from priority_scorer import update_with_layer3


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


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "k1netix_layer2.json"
    with open(path) as f:
        layer2_result = json.load(f)

    result   = run_layer3(layer2_result)
    out_path = Path(path).with_name("k1netix_layer3.json")
    with open(out_path, "w") as f:
        json.dump(json.loads(_export_layer3_json(result)), f, indent=2, default=str)
    print(f"\n📄 Output → {out_path}")
