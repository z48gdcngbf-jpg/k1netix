"""
Layer 3 — Vector Store
"""
from __future__ import annotations

# CRITICAL: SQLite fix for Streamlit Cloud — must come before chromadb import
__import__("pysqlite3")
import sys as _sys
_sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")

import json
from pathlib import Path
from typing import Optional

import json
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "k1netix_regulations"
DEFAULT_DB_PATH    = "./chroma_db"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"   # ~80MB download on first run


# ── Lazy singletons ───────────────────────────────────────────────────────────
_client     = None
_embedder   = None


def _get_client(db_path: str = DEFAULT_DB_PATH):
    global _client
    if _client is None:
        import chromadb
        _client = chromadb.PersistentClient(path=db_path)
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _embed(texts: list[str]) -> list[list[float]]:
    model = _get_embedder()
    return model.encode(texts, show_progress_bar=False).tolist()


# ── Collection access ─────────────────────────────────────────────────────────

def get_collection(
    name: str = DEFAULT_COLLECTION,
    db_path: str = DEFAULT_DB_PATH,
):
    client = _get_client(db_path)
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_stats(
    name: str = DEFAULT_COLLECTION,
    db_path: str = DEFAULT_DB_PATH,
) -> dict:
    col = get_collection(name, db_path)
    count = col.count()
    return {"collection": name, "total_chunks": count, "db_path": db_path}


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_chunks(
    chunks: list,           # list of RegulationChunk
    collection_name: str = DEFAULT_COLLECTION,
    db_path: str = DEFAULT_DB_PATH,
    batch_size: int = 64,
) -> dict:
    """
    Embed and upsert regulation chunks into ChromaDB.
    Uses upsert so re-ingesting the same document is safe (idempotent).
    """
    col = get_collection(collection_name, db_path)

    total = len(chunks)
    added = 0

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        ids        = [c.chunk_id for c in batch]
        texts      = [c.text for c in batch]
        embeddings = _embed(texts)
        metadatas  = [{
            "source":        c.source,
            "document_type": c.document_type,
            "jurisdiction":  c.jurisdiction,
            "discipline":    c.discipline,
            "section":       c.section,
            "section_title": c.section_title,
            "page":          str(c.page or ""),
            "word_count":    c.word_count,
            "keywords":      json.dumps(c.keywords),
        } for c in batch]

        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        added += len(batch)

    return {"upserted": added, "total": total}


# ── Query ─────────────────────────────────────────────────────────────────────

def query(
    query_text: str,
    n_results: int = 8,
    discipline_filter: Optional[str] = None,
    jurisdiction_filter: Optional[str] = None,
    document_type_filter: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """
    Semantic search over stored regulation chunks.

    Returns list of dicts, each with: text, score, metadata.
    Score is cosine similarity (higher = more relevant).
    """
    col = get_collection(collection_name, db_path)

    if col.count() == 0:
        return []

    # Build ChromaDB where filter
    where_clauses = []
    if discipline_filter and discipline_filter not in ("general", "UNKNOWN"):
        where_clauses.append({"discipline": {"$in": [discipline_filter, "general"]}})
    if jurisdiction_filter:
        where_clauses.append({"jurisdiction": {"$in": [jurisdiction_filter, "general"]}})
    if document_type_filter:
        where_clauses.append({"document_type": document_type_filter})

    where = {"$and": where_clauses} if len(where_clauses) > 1 else \
            where_clauses[0] if where_clauses else None

    query_embedding = _embed([query_text])[0]

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=min(n_results, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    try:
        results = col.query(**kwargs)
    except Exception:
        # Filter may have no matches — retry without filter
        kwargs.pop("where", None)
        results = col.query(**kwargs)

    hits = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        similarity = round(1 - dist, 4)   # cosine distance → similarity
        hits.append({
            "text":          doc,
            "score":         similarity,
            "source":        meta.get("source", ""),
            "section":       meta.get("section", ""),
            "section_title": meta.get("section_title", ""),
            "discipline":    meta.get("discipline", ""),
            "document_type": meta.get("document_type", ""),
            "jurisdiction":  meta.get("jurisdiction", ""),
            "page":          meta.get("page", ""),
            "keywords":      json.loads(meta.get("keywords", "[]")),
        })

    # Sort by score descending
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_document(
    source_filename: str,
    collection_name: str = DEFAULT_COLLECTION,
    db_path: str = DEFAULT_DB_PATH,
):
    """Remove all chunks belonging to a specific source document."""
    col = get_collection(collection_name, db_path)
    col.delete(where={"source": source_filename})


def list_documents(
    collection_name: str = DEFAULT_COLLECTION,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """List all unique source documents in the store."""
    col = get_collection(collection_name, db_path)
    if col.count() == 0:
        return []

    # Sample all metadatas (ChromaDB doesn't support GROUP BY)
    results = col.get(include=["metadatas"])
    seen = {}
    for meta in results["metadatas"]:
        src = meta.get("source", "unknown")
        if src not in seen:
            seen[src] = {
                "source":        src,
                "document_type": meta.get("document_type", ""),
                "jurisdiction":  meta.get("jurisdiction", ""),
                "chunk_count":   0,
            }
        seen[src]["chunk_count"] += 1

    return list(seen.values())


if __name__ == "__main__":
    print(json.dumps(collection_stats(), indent=2))
    docs = list_documents()
    print(f"Documents in store: {len(docs)}")
    for d in docs:
        print(f"  {d['source']} — {d['chunk_count']} chunks ({d['jurisdiction']})")
