"""
Layer 3 — Document Ingester
Loads building regulations, standards, and project specs.
Chunks them into retrievable passages with rich metadata.

Supported formats: PDF · TXT · DOCX · MD
"""

from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


# ── Chunk data class ──────────────────────────────────────────────────────────

@dataclass
class RegulationChunk:
    chunk_id:      str          # deterministic hash of content
    text:          str          # the actual clause / passage text
    source:        str          # filename
    document_type: str          # "regulation" | "standard" | "spec" | "guidance"
    jurisdiction:  str          # e.g. "UK" | "Singapore" | "UAE" | "general"
    discipline:    str          # "MEP" | "STR" | "ARCH" | "FP" | "general"
    section:       str          # e.g. "3.2.1"
    section_title: str          # e.g. "Fire Resistance of Structural Elements"
    page:          Optional[int]
    word_count:    int
    keywords:      list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Discipline keyword map for auto-tagging ───────────────────────────────────

DISCIPLINE_SIGNALS = {
    "MEP":  ["duct","pipe","hvac","mechanical","electrical","plumbing",
             "sprinkler","ventilation","drainage","cable","conduit"],
    "STR":  ["structural","beam","column","slab","load","foundation",
             "reinforcement","concrete","steel","moment","shear"],
    "ARCH": ["egress","door","window","wall","corridor","room","space",
             "occupancy","accessibility","means of escape","finish"],
    "FP":   ["fire","smoke","flame","combustion","suppression","alarm",
             "detection","rating","resistance","compartment","evacuation"],
    "CIVIL":["site","road","drainage","utility","excavation","earthwork"],
}

DOCUMENT_TYPE_SIGNALS = {
    "regulation": ["building act","building regulation","fire code","bye-law",
                   "nfpa","bs ","en ","din ","approved document"],
    "standard":   ["iso ","bs iso","astm","standard specification","british standard"],
    "spec":       ["project specification","employer requirement","particular spec",
                   "schedule of works"],
    "guidance":   ["guidance","advisory","recommendation","best practice","guide"],
}


def _detect_discipline(text: str) -> str:
    t = text.lower()
    scores = {d: sum(1 for kw in kws if kw in t)
              for d, kws in DISCIPLINE_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def _detect_doc_type(text: str, filename: str) -> str:
    combined = (text + " " + filename).lower()
    scores = {t: sum(1 for kw in kws if kw in combined)
              for t, kws in DOCUMENT_TYPE_SIGNALS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "regulation"


def _extract_keywords(text: str, n: int = 10) -> list[str]:
    """Simple TF-based keyword extraction — no external lib needed."""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stopwords = {"that","this","with","from","have","been","shall","should",
                 "must","will","where","when","such","each","into","than",
                 "they","their","there","which","also","more","other"}
    freq = {}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]


def _make_chunk_id(source: str, section: str, text: str) -> str:
    raw = f"{source}|{section}|{text[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Splitters ─────────────────────────────────────────────────────────────────

def _split_by_sections(text: str) -> list[tuple[str, str, str]]:
    """
    Split text into (section_number, section_title, body) tuples.
    Handles patterns like:  3.2  Fire Safety   or   Clause 4.1 — Means of Escape
    """
    pattern = re.compile(
        r'(?:^|\n)'
        r'((?:Clause\s+)?(\d+(?:\.\d+){0,3})\s*[—–-]?\s*([A-Z][^\n]{0,80}))',
        re.MULTILINE
    )

    matches = list(pattern.finditer(text))
    if not matches:
        # No structured sections — split by paragraphs
        return [("", "", para.strip())
                for para in text.split("\n\n") if para.strip()]

    chunks = []
    for i, m in enumerate(matches):
        sec_num   = m.group(2) or ""
        sec_title = m.group(3).strip() if m.group(3) else ""
        start     = m.end()
        end       = matches[i+1].start() if i+1 < len(matches) else len(text)
        body      = text[start:end].strip()
        if body:
            chunks.append((sec_num, sec_title, body))

    return chunks


def _sliding_window(text: str, chunk_size: int = 400,
                    overlap: int = 80) -> list[str]:
    """
    Sliding window splitter for long unstructured text.
    Splits on sentence boundaries within the window.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current, count = [], [], 0

    for sent in sentences:
        words = len(sent.split())
        if count + words > chunk_size and current:
            chunks.append(" ".join(current))
            # Keep last few sentences for overlap
            overlap_sents = []
            overlap_count = 0
            for s in reversed(current):
                overlap_count += len(s.split())
                if overlap_count >= overlap:
                    break
                overlap_sents.insert(0, s)
            current = overlap_sents
            count   = sum(len(s.split()) for s in current)
        current.append(sent)
        count += words

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Format-specific readers ───────────────────────────────────────────────────

def _read_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return [(i+1, page.get_text()) for i, page in enumerate(doc)]
    except ImportError:
        pass
    try:
        import pdfplumber
        import io
        pages = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                pages.append((i+1, page.extract_text() or ""))
        return pages
    except ImportError:
        raise ImportError(
            "PDF parsing requires PyMuPDF or pdfplumber.\n"
            "Run: pip install pymupdf  or  pip install pdfplumber"
        )


def _read_docx(file_bytes: bytes) -> str:
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("DOCX parsing requires python-docx. Run: pip install python-docx")


# ── Main ingester ─────────────────────────────────────────────────────────────

def ingest_document(
    file_bytes: bytes,
    filename: str,
    jurisdiction: str = "general",
    document_type: str = "auto",
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[RegulationChunk]:
    """
    Parse a regulation document and return a list of RegulationChunk objects.

    Args:
        file_bytes    : raw file bytes
        filename      : original filename (used for type detection)
        jurisdiction  : e.g. "UK", "Singapore", "UAE", "general"
        document_type : "auto" | "regulation" | "standard" | "spec" | "guidance"
        chunk_size    : target words per chunk
        overlap       : overlap words between chunks
    """
    ext = Path(filename).suffix.lower()
    chunks: list[RegulationChunk] = []

    # ── Read raw text ──────────────────────────────────────────────────
    if ext == ".pdf":
        pages = _read_pdf(file_bytes)
        # Process per page for page number metadata
        for page_num, page_text in pages:
            if not page_text.strip():
                continue
            doc_type = document_type if document_type != "auto" else _detect_doc_type(page_text, filename)
            sections = _split_by_sections(page_text)
            for sec_num, sec_title, body in sections:
                if len(body.split()) < 15:
                    continue
                sub_chunks = _sliding_window(body, chunk_size, overlap)
                for sub in sub_chunks:
                    if not sub.strip():
                        continue
                    disc = _detect_discipline(sub)
                    chunks.append(RegulationChunk(
                        chunk_id      = _make_chunk_id(filename, sec_num, sub),
                        text          = sub.strip(),
                        source        = filename,
                        document_type = doc_type,
                        jurisdiction  = jurisdiction,
                        discipline    = disc,
                        section       = sec_num,
                        section_title = sec_title,
                        page          = page_num,
                        word_count    = len(sub.split()),
                        keywords      = _extract_keywords(sub),
                    ))

    elif ext in (".txt", ".md"):
        raw = file_bytes.decode("utf-8", errors="ignore")
        doc_type = document_type if document_type != "auto" else _detect_doc_type(raw, filename)
        sections = _split_by_sections(raw)
        for sec_num, sec_title, body in sections:
            if len(body.split()) < 15:
                continue
            sub_chunks = _sliding_window(body, chunk_size, overlap)
            for sub in sub_chunks:
                if not sub.strip():
                    continue
                disc = _detect_discipline(sub)
                chunks.append(RegulationChunk(
                    chunk_id      = _make_chunk_id(filename, sec_num, sub),
                    text          = sub.strip(),
                    source        = filename,
                    document_type = doc_type,
                    jurisdiction  = jurisdiction,
                    discipline    = disc,
                    section       = sec_num,
                    section_title = sec_title,
                    page          = None,
                    word_count    = len(sub.split()),
                    keywords      = _extract_keywords(sub),
                ))

    elif ext == ".docx":
        raw = _read_docx(file_bytes)
        doc_type = document_type if document_type != "auto" else _detect_doc_type(raw, filename)
        sections = _split_by_sections(raw)
        for sec_num, sec_title, body in sections:
            if len(body.split()) < 15:
                continue
            sub_chunks = _sliding_window(body, chunk_size, overlap)
            for sub in sub_chunks:
                disc = _detect_discipline(sub)
                chunks.append(RegulationChunk(
                    chunk_id      = _make_chunk_id(filename, sec_num, sub),
                    text          = sub.strip(),
                    source        = filename,
                    document_type = doc_type,
                    jurisdiction  = jurisdiction,
                    discipline    = disc,
                    section       = sec_num,
                    section_title = sec_title,
                    page          = None,
                    word_count    = len(sub.split()),
                    keywords      = _extract_keywords(sub),
                ))
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF, TXT, DOCX, or MD.")

    return chunks


def ingest_multiple(
    files: list[tuple[bytes, str]],
    jurisdiction: str = "general",
) -> list[RegulationChunk]:
    """
    Ingest multiple documents at once.
    files: list of (bytes, filename) tuples.
    """
    all_chunks = []
    for file_bytes, filename in files:
        try:
            chunks = ingest_document(file_bytes, filename, jurisdiction=jurisdiction)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"⚠️ Failed to ingest {filename}: {e}")
    return all_chunks


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    with open(path, "rb") as f:
        chunks = ingest_document(f.read(), Path(path).name)
    print(f"✅ Ingested {len(chunks)} chunks from {path}")
    for c in chunks[:3]:
        print(f"\n  [{c.section}] {c.section_title[:50]}")
        print(f"  Discipline: {c.discipline} | Words: {c.word_count}")
        print(f"  {c.text[:120]}…")
