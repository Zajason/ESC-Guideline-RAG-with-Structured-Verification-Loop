from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

import io

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(".").resolve()
DATA_ROOT = PROJECT_ROOT / "esc_data"  # esc_data/<year>/*.pdf
OUT_DIR = PROJECT_ROOT / "esc_rag_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEC_IDMAP_PATH = OUT_DIR / "esc_sections_idmap.jsonl"
REC_IDMAP_PATH = OUT_DIR / "esc_recommendations_idmap.jsonl"
FIG_IDMAP_PATH = OUT_DIR / "esc_figures_idmap.jsonl"


# =============================================================================
# Chunking configuration
# =============================================================================
SECTION_CHUNK_TARGET_WORDS = 850
SECTION_CHUNK_OVERLAP_WORDS = 120

REC_CHUNK_MIN_WORDS = 35
REC_CHUNK_MAX_WORDS = 240

TABLE_CHUNK_MIN_WORDS = 40
TABLE_CHUNK_MAX_WORDS = 280

FIG_OCR_MIN_WORDS = 18
FIG_OCR_MAX_WORDS = 260
FIG_IMAGE_MIN_AREA = 40_000


# =============================================================================
# Models
# =============================================================================
EMBED_MODEL_NAME = "intfloat/e5-large-v2"
EMBED_BATCH_SIZE = 24

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =============================================================================
# Utilities
# =============================================================================
def sha16(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def normalize_text(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = s.replace("￾", "")
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_for_dedupe(s: str) -> str:
    s = normalize_text(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def safe_json_loads(s: Any) -> Any:
    if isinstance(s, dict):
        return s
    if s is None:
        return {}
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return {}
    try:
        return json.loads(str(s))
    except Exception:
        return {}


def list_pdf_files(data_root: Path) -> List[Path]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    return sorted(data_root.glob("*/*.pdf"))


def infer_year_from_path(pdf_path: Path) -> Optional[int]:
    try:
        y = int(pdf_path.parent.name)
        if 1900 <= y <= 2100:
            return y
    except Exception:
        pass
    return None


def _simple_img_hash(pil_img, size: int = 32) -> str:
    img = pil_img.convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    mean = arr.mean()
    bits = (arr > mean).astype(np.uint8).flatten()
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def _clean_ocr_text(s: str) -> str:
    s = normalize_text(s)
    s = re.sub(r"[|_•·●◦▪︎■]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# =============================================================================
# PDF extraction + cleaning
# =============================================================================
def extract_pdf_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        pages.append({"page_index": i, "page_number": i + 1, "text_raw": text})
    return pages


def get_frequent_lines(
    pages: List[Dict[str, Any]],
    min_frac: float = 0.55,
    max_len: int = 90,
) -> set:
    from collections import Counter

    ctr = Counter()
    for p in pages:
        lines = [normalize_text(l) for l in p["text_raw"].splitlines()]
        lines = [l for l in lines if l and 6 <= len(l) <= max_len]
        for l in set(lines):
            ctr[l] += 1

    threshold = max(2, int(len(pages) * min_frac))
    return {line for line, c in ctr.items() if c >= threshold}


def remove_frequent_lines(text: str, frequent: set) -> str:
    out_lines = []
    for l in text.splitlines():
        nl = normalize_text(l)
        if nl in frequent:
            continue
        out_lines.append(l)
    return "\n".join(out_lines)


def clean_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    frequent = get_frequent_lines(pages, min_frac=0.55)
    out: List[Dict[str, Any]] = []
    for p in pages:
        txt = remove_frequent_lines(p["text_raw"], frequent)
        txt = normalize_text(txt)
        out.append({**p, "text_clean": txt})
    return out


def infer_doc_title_and_year(p0_text: str, fallback_year: Optional[int]) -> Tuple[str, Optional[int]]:
    lines = [l.strip() for l in p0_text.splitlines() if l.strip()]
    title = None
    for l in lines[:80]:
        if "guidelines" in l.lower():
            title = l
            break
    if not title:
        title = lines[0] if lines else "ESC Guideline"

    year = fallback_year
    if year is None:
        m = re.search(r"\b(20\d{2}|19\d{2})\b", p0_text[:2500])
        if m:
            year = int(m.group(1))
    return title, year


# =============================================================================
# Section parsing
# =============================================================================
HEADING_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){0,6})\s+(?P<title>[A-Za-z][^\n]{2,180})$")


def iter_lines_with_page(pages_clean: List[Dict[str, Any]]) -> Iterable[Tuple[int, str]]:
    for p in pages_clean:
        for line in p["text_clean"].splitlines():
            yield p["page_number"], line.strip()


@dataclass
class SectionBlock:
    doc_id: str
    doc_title: str
    year: Optional[int]
    pdf_path: str
    section_num: str
    section_title: str
    page_start: int
    page_end: int
    text: str


def build_sections(
    pages_clean: List[Dict[str, Any]],
    doc_id: str,
    doc_title: str,
    year: Optional[int],
    pdf_path: Path,
) -> List[SectionBlock]:
    current_num = "0"
    current_title = "Front matter"
    buffer_lines: List[str] = []
    page_start = 1
    page_end = 1
    sections: List[SectionBlock] = []

    for page_no, line in iter_lines_with_page(pages_clean):
        if not line:
            continue

        m = HEADING_RE.match(line)
        looks_like_heading = bool(m) and len(line) <= 180 and not line.endswith(".")
        if looks_like_heading:
            text = "\n".join(buffer_lines).strip()
            if text:
                sections.append(
                    SectionBlock(
                        doc_id=doc_id,
                        doc_title=doc_title,
                        year=year,
                        pdf_path=str(pdf_path),
                        section_num=current_num,
                        section_title=current_title,
                        page_start=page_start,
                        page_end=page_end,
                        text=text,
                    )
                )
            current_num = m.group("num")
            current_title = m.group("title").strip()
            buffer_lines = [line]
            page_start = page_no
            page_end = page_no
        else:
            buffer_lines.append(line)
            page_end = page_no

    text = "\n".join(buffer_lines).strip()
    if text:
        sections.append(
            SectionBlock(
                doc_id=doc_id,
                doc_title=doc_title,
                year=year,
                pdf_path=str(pdf_path),
                section_num=current_num,
                section_title=current_title,
                page_start=page_start,
                page_end=page_end,
                text=text,
            )
        )
    return sections


# =============================================================================
# Table detection + fallback extraction
# =============================================================================
TABLE_HINT_RE = re.compile(r"\b(table|recommendations)\b", re.IGNORECASE)


def looks_like_table_block(text: str) -> bool:
    if not text or len(text) < 200:
        return False
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 8:
        return False
    multi_space = sum(1 for l in lines if re.search(r"\S\s{2,}\S", l))
    pipey = sum(1 for l in lines if "|" in l)
    shortish = sum(1 for l in lines if 10 <= len(l) <= 90)
    return (multi_space >= max(4, len(lines) // 4)) or (pipey >= 4) or (shortish >= len(lines) * 0.7)


def tableify_text(text: str) -> str:
    out_lines = []
    for l in text.splitlines():
        l = l.rstrip()
        if not l:
            continue
        l = re.sub(r"\s{2,}", " | ", l)
        out_lines.append(l)
    return "\n".join(out_lines).strip()


# =============================================================================
# Chunking
# =============================================================================
@dataclass
class Chunk:
    chunk_id: str
    row_id: int
    doc_id: str
    doc_title: str
    year: Optional[int]
    pdf_path: str
    chunk_type: str  # "section" | "recommendation" | "figure"
    section_num: str
    section_title: str
    page_start: int
    page_end: int
    text: str
    signals_json: str


# This ensures empty dfs still have these columns (fixes your KeyError)
META_COLUMNS = [
    "chunk_id",
    "row_id",
    "doc_id",
    "doc_title",
    "year",
    "pdf_path",
    "chunk_type",
    "section_num",
    "section_title",
    "page_start",
    "page_end",
    "text",
    "signals_json",
]


def ensure_meta_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee meta DataFrame has the expected columns even when empty.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=META_COLUMNS)
    # If some columns are missing (shouldn't happen but safe), add them
    for c in META_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df[META_COLUMNS]


def word_chunks(text: str, target_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + target_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def split_into_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


REC_SIGNAL_RE = re.compile(
    r"\b("
    r"is recommended|are recommended|recommended that|we recommend|"
    r"should be considered|may be considered|should be performed|is indicated|"
    r"is not recommended|are not recommended"
    r")\b",
    re.IGNORECASE,
)
CLASS_RE = re.compile(r"\bClass\s+(I|IIa|IIb|III)\b", re.IGNORECASE)
LOE_RE = re.compile(r"\b(Level of evidence|LOE)\s*(A|B|C)\b", re.IGNORECASE)


def build_section_chunks(sections: List[SectionBlock], row_id_start: int = 0) -> List[Chunk]:
    out: List[Chunk] = []
    row_id = row_id_start
    for s in sections:
        pieces = word_chunks(s.text, target_words=SECTION_CHUNK_TARGET_WORDS, overlap_words=SECTION_CHUNK_OVERLAP_WORDS)
        for i, piece in enumerate(pieces):
            cid = sha16(f"{s.doc_id}|{s.section_num}|section|p{s.page_start}|{i}")
            signals = {"is_recommendation": False}
            out.append(
                Chunk(
                    chunk_id=cid,
                    row_id=row_id,
                    doc_id=s.doc_id,
                    doc_title=s.doc_title,
                    year=s.year,
                    pdf_path=s.pdf_path,
                    chunk_type="section",
                    section_num=s.section_num,
                    section_title=s.section_title,
                    page_start=s.page_start,
                    page_end=s.page_end,
                    text=piece,
                    signals_json=safe_json_dumps(signals),
                )
            )
            row_id += 1
    return out


def build_recommendation_chunks(sections: List[SectionBlock], row_id_start: int = 0) -> List[Chunk]:
    out: List[Chunk] = []
    row_id = row_id_start
    for s in sections:
        paras = split_into_paragraphs(s.text)
        rec_paras: List[str] = []
        for p in paras:
            w = p.split()
            if len(w) < REC_CHUNK_MIN_WORDS:
                continue
            if (REC_SIGNAL_RE.search(p) or CLASS_RE.search(p) or LOE_RE.search(p)
                or "Recommendations for" in p or "RECOMMENDATIONS" in p):
                rec_paras.append(p)

        for i, rp in enumerate(rec_paras):
            words = rp.split()
            if len(words) > REC_CHUNK_MAX_WORDS:
                rp = " ".join(words[:REC_CHUNK_MAX_WORDS])

            cls = CLASS_RE.search(rp)
            loe = LOE_RE.search(rp)
            signals = {
                "is_recommendation": True,
                "has_class": bool(cls),
                "class": cls.group(1) if cls else None,
                "has_loe": bool(loe),
                "loe": loe.group(2) if loe else None,
            }

            cid = sha16(f"{s.doc_id}|{s.section_num}|rec|p{s.page_start}|{i}")
            out.append(
                Chunk(
                    chunk_id=cid,
                    row_id=row_id,
                    doc_id=s.doc_id,
                    doc_title=s.doc_title,
                    year=s.year,
                    pdf_path=s.pdf_path,
                    chunk_type="recommendation",
                    section_num=s.section_num,
                    section_title=s.section_title,
                    page_start=s.page_start,
                    page_end=s.page_end,
                    text=rp,
                    signals_json=safe_json_dumps(signals),
                )
            )
            row_id += 1
    return out


def build_table_fallback_chunks(
    pages_clean: List[Dict[str, Any]],
    doc_meta: Dict[str, Any],
    row_id_start: int,
) -> List[Chunk]:
    out: List[Chunk] = []
    row_id = row_id_start

    for p in pages_clean:
        txt = p.get("text_clean", "")
        if not txt:
            continue
        if not (looks_like_table_block(txt) or TABLE_HINT_RE.search(txt)):
            continue

        t = tableify_text(txt)
        words = t.split()
        if len(words) < TABLE_CHUNK_MIN_WORDS:
            continue
        if len(words) > TABLE_CHUNK_MAX_WORDS:
            t = " ".join(words[:TABLE_CHUNK_MAX_WORDS])

        cid = sha16(f"{doc_meta['doc_id']}|table|p{p['page_number']}")
        cls_m = CLASS_RE.search(t)
        loe_m = LOE_RE.search(t)
        signals = {
            "is_recommendation": True,
            "table_fallback": True,
            "has_class": bool(cls_m),
            "class": (cls_m.group(1) if cls_m else None),
            "has_loe": bool(loe_m),
            "loe": (loe_m.group(2) if loe_m else None),
        }

        out.append(
            Chunk(
                chunk_id=cid,
                row_id=row_id,
                doc_id=doc_meta["doc_id"],
                doc_title=doc_meta["doc_title"],
                year=doc_meta["year"],
                pdf_path=doc_meta["pdf_path"],
                chunk_type="recommendation",
                section_num="TABLE",
                section_title="Table (fallback extraction)",
                page_start=int(p["page_number"]),
                page_end=int(p["page_number"]),
                text=t,
                signals_json=safe_json_dumps(signals),
            )
        )
        row_id += 1

    return out


# =============================================================================
# Figure OCR extraction (optional)
# =============================================================================
def extract_figure_chunks_from_pdf(
    pdf_path: Path,
    doc_meta: Dict[str, Any],
    row_id_start: int,
) -> List[Chunk]:
    if Image is None or pytesseract is None:
        return []

    fig_chunks: List[Chunk] = []
    row_id = row_id_start

    seen_img_hashes = set()
    seen_text_hashes = set()

    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_no = page_index + 1

        images = page.get_images(full=True)
        if not images:
            continue

        for img in images:
            xref = img[0]
            try:
                img_data = doc.extract_image(xref)
                img_bytes = img_data.get("image")
                if not img_bytes:
                    continue

                pil_img = Image.open(io.BytesIO(img_bytes))
                w, h = pil_img.size
                if w * h < FIG_IMAGE_MIN_AREA:
                    continue

                img_hash = _simple_img_hash(pil_img)
                if img_hash in seen_img_hashes:
                    continue
                seen_img_hashes.add(img_hash)

                raw_text = pytesseract.image_to_string(pil_img.convert("RGB"))
                text = _clean_ocr_text(raw_text)
                if not text:
                    continue
                if len(text.split()) < FIG_OCR_MIN_WORDS:
                    continue
                text = _truncate_words(text, FIG_OCR_MAX_WORDS)

                text_hash = sha16(text.lower())
                if text_hash in seen_text_hashes:
                    continue
                seen_text_hashes.add(text_hash)

                cid = sha16(f"{doc_meta['doc_id']}|figure|p{page_no}|{img_hash[:10]}")
                fig_chunks.append(
                    Chunk(
                        chunk_id=cid,
                        row_id=row_id,
                        doc_id=doc_meta["doc_id"],
                        doc_title=doc_meta["doc_title"],
                        year=doc_meta["year"],
                        pdf_path=doc_meta["pdf_path"],
                        chunk_type="figure",
                        section_num="FIGURE",
                        section_title="Figure / OCR extracted",
                        page_start=page_no,
                        page_end=page_no,
                        text=text,
                        signals_json=safe_json_dumps({"is_figure": True}),
                    )
                )
                row_id += 1
            except Exception:
                continue

    return fig_chunks


# =============================================================================
# Build corpus artifacts
# =============================================================================
def build_corpus_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pdf_files = list_pdf_files(DATA_ROOT)

    manifest_rows: List[Dict[str, Any]] = []
    all_section_chunks: List[Chunk] = []
    all_rec_chunks: List[Chunk] = []
    all_fig_chunks: List[Chunk] = []

    sec_row_id = 0
    rec_row_id = 0
    fig_row_id = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        folder_year = infer_year_from_path(pdf_path)
        try:
            pages = extract_pdf_pages(pdf_path)
            pages_clean = clean_pages(pages)

            title, year = infer_doc_title_and_year(pages_clean[0]["text_clean"], folder_year)
            doc_id = sha16(f"{title}|{year}|{pdf_path.name}")

            doc_meta = {"doc_id": doc_id, "doc_title": title, "year": year, "pdf_path": str(pdf_path)}

            sections = build_sections(pages_clean=pages_clean, doc_id=doc_id, doc_title=title, year=year, pdf_path=pdf_path)

            section_chunks = build_section_chunks(sections, row_id_start=sec_row_id)
            sec_row_id += len(section_chunks)
            all_section_chunks.extend(section_chunks)

            rec_chunks = build_recommendation_chunks(sections, row_id_start=rec_row_id)
            rec_row_id += len(rec_chunks)
            all_rec_chunks.extend(rec_chunks)

            table_chunks = build_table_fallback_chunks(pages_clean, doc_meta=doc_meta, row_id_start=rec_row_id)
            rec_row_id += len(table_chunks)
            all_rec_chunks.extend(table_chunks)

            fig_chunks = extract_figure_chunks_from_pdf(pdf_path=pdf_path, doc_meta=doc_meta, row_id_start=fig_row_id)
            fig_row_id += len(fig_chunks)
            all_fig_chunks.extend(fig_chunks)

            manifest_rows.append(
                {
                    "pdf_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "folder_year": folder_year,
                    "inferred_year": year,
                    "doc_title": title,
                    "doc_id": doc_id,
                    "pages": len(pages),
                    "sections": len(sections),
                    "section_chunks": len(section_chunks),
                    "rec_chunks": len(rec_chunks),
                    "table_fallback_chunks": len(table_chunks),
                    "figure_chunks": len(fig_chunks),
                    "error": None,
                }
            )
        except Exception as e:
            manifest_rows.append({"pdf_path": str(pdf_path), "file_name": pdf_path.name, "folder_year": folder_year, "error": repr(e)})

    manifest = pd.DataFrame(manifest_rows)

    sec_meta = pd.DataFrame([asdict(c) for c in all_section_chunks])
    rec_meta = pd.DataFrame([asdict(c) for c in all_rec_chunks])
    fig_meta = pd.DataFrame([asdict(c) for c in all_fig_chunks])

    sec_meta = ensure_meta_schema(sec_meta)
    rec_meta = ensure_meta_schema(rec_meta)
    fig_meta = ensure_meta_schema(fig_meta)

    if not sec_meta.empty:
        sec_meta = sec_meta.sort_values("row_id").reset_index(drop=True)
    if not rec_meta.empty:
        rec_meta = rec_meta.sort_values("row_id").reset_index(drop=True)
    if not fig_meta.empty:
        fig_meta = fig_meta.sort_values("row_id").reset_index(drop=True)

    return manifest, sec_meta, rec_meta, fig_meta


def save_corpus_artifacts(
    manifest: pd.DataFrame,
    sec_meta: pd.DataFrame,
    rec_meta: pd.DataFrame,
    fig_meta: pd.DataFrame,
) -> None:
    manifest.to_parquet(OUT_DIR / "esc_corpus_manifest.parquet", index=False)
    sec_meta.to_parquet(OUT_DIR / "esc_sections_meta.parquet", index=False)
    rec_meta.to_parquet(OUT_DIR / "esc_recommendations_meta.parquet", index=False)
    fig_meta.to_parquet(OUT_DIR / "esc_figures_meta.parquet", index=False)


def write_idmap_jsonl(meta: pd.DataFrame, path: Path) -> None:
    # write a file even if empty -> alignment checks become simpler
    with path.open("w", encoding="utf-8") as f:
        if meta is None or meta.empty:
            return
        for row in meta[["row_id", "chunk_id"]].itertuples(index=False):
            f.write(safe_json_dumps({"row_id": int(row.row_id), "chunk_id": str(row.chunk_id)}) + "\n")


def load_idmap_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# =============================================================================
# Embeddings + FAISS
# =============================================================================
def embed_passages(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    passages = [f"passage: {t}" for t in texts]
    emb = model.encode(passages, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def embed_query(model: SentenceTransformer, q: str) -> np.ndarray:
    v = model.encode([f"query: {q}"], normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    if vectors.shape[0] > 0:
        index.add(vectors)
    return index


def build_and_save_indexes(sec_meta: pd.DataFrame, rec_meta: pd.DataFrame, fig_meta: pd.DataFrame) -> None:
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # infer embedding dimension safely
    dim = int(model.get_sentence_embedding_dimension())

    sec_meta = ensure_meta_schema(sec_meta)
    rec_meta = ensure_meta_schema(rec_meta)
    fig_meta = ensure_meta_schema(fig_meta)

    sec_meta = sec_meta.sort_values("row_id").reset_index(drop=True) if not sec_meta.empty else sec_meta
    rec_meta = rec_meta.sort_values("row_id").reset_index(drop=True) if not rec_meta.empty else rec_meta
    fig_meta = fig_meta.sort_values("row_id").reset_index(drop=True) if not fig_meta.empty else fig_meta

    write_idmap_jsonl(sec_meta, SEC_IDMAP_PATH)
    write_idmap_jsonl(rec_meta, REC_IDMAP_PATH)
    write_idmap_jsonl(fig_meta, FIG_IDMAP_PATH)

    sec_vecs = embed_passages(model, sec_meta["text"].tolist(), batch_size=EMBED_BATCH_SIZE) if not sec_meta.empty else np.zeros((0, dim), dtype=np.float32)
    rec_vecs = embed_passages(model, rec_meta["text"].tolist(), batch_size=EMBED_BATCH_SIZE) if not rec_meta.empty else np.zeros((0, dim), dtype=np.float32)
    fig_vecs = embed_passages(model, fig_meta["text"].tolist(), batch_size=EMBED_BATCH_SIZE) if not fig_meta.empty else np.zeros((0, dim), dtype=np.float32)

    faiss.write_index(build_faiss_index(sec_vecs), str(OUT_DIR / "esc_sections.faiss"))
    faiss.write_index(build_faiss_index(rec_vecs), str(OUT_DIR / "esc_recommendations.faiss"))
    faiss.write_index(build_faiss_index(fig_vecs), str(OUT_DIR / "esc_figures.faiss"))


# =============================================================================
# Retrieval + reranking
# =============================================================================
def _verify_idmap_alignment(meta: pd.DataFrame, idmap_path: Path, label: str) -> None:
    meta = ensure_meta_schema(meta)
    if meta.empty:
        return
    idmap = load_idmap_jsonl(idmap_path)
    if len(idmap) != len(meta):
        raise RuntimeError(f"{label} idmap length mismatch: {len(idmap)} vs {len(meta)}")
    for pos in (0, len(meta) // 2, len(meta) - 1):
        if meta.iloc[pos]["chunk_id"] != idmap[pos]["chunk_id"]:
            raise RuntimeError(f"{label} FAISS/meta order mismatch. Rebuild artifacts.")


def load_runtime() -> Tuple[
    faiss.Index,
    faiss.Index,
    faiss.Index,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    SentenceTransformer,
    CrossEncoder,
]:
    sec_index = faiss.read_index(str(OUT_DIR / "esc_sections.faiss"))
    rec_index = faiss.read_index(str(OUT_DIR / "esc_recommendations.faiss"))
    fig_index = faiss.read_index(str(OUT_DIR / "esc_figures.faiss"))

    sec_meta = ensure_meta_schema(pd.read_parquet(OUT_DIR / "esc_sections_meta.parquet"))
    rec_meta = ensure_meta_schema(pd.read_parquet(OUT_DIR / "esc_recommendations_meta.parquet"))
    fig_meta = ensure_meta_schema(pd.read_parquet(OUT_DIR / "esc_figures_meta.parquet"))

    if not sec_meta.empty:
        sec_meta = sec_meta.sort_values("row_id").reset_index(drop=True)
    if not rec_meta.empty:
        rec_meta = rec_meta.sort_values("row_id").reset_index(drop=True)
    if not fig_meta.empty:
        fig_meta = fig_meta.sort_values("row_id").reset_index(drop=True)

    _verify_idmap_alignment(sec_meta, SEC_IDMAP_PATH, "Sections")
    _verify_idmap_alignment(rec_meta, REC_IDMAP_PATH, "Recommendations")
    _verify_idmap_alignment(fig_meta, FIG_IDMAP_PATH, "Figures")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    return sec_index, rec_index, fig_index, sec_meta, rec_meta, fig_meta, embed_model, reranker


def faiss_search(
    index: faiss.Index,
    meta: pd.DataFrame,
    embed_model: SentenceTransformer,
    q: str,
    k: int,
) -> pd.DataFrame:
    meta = ensure_meta_schema(meta)
    if index.ntotal == 0 or meta.empty:
        return meta.head(0).copy()

    k_eff = int(min(k, index.ntotal))
    qv = embed_query(embed_model, q)
    scores, idxs = index.search(qv, k_eff)

    idxs_list = idxs[0].tolist()
    scores_list = scores[0].tolist()

    pairs = [(i, s) for i, s in zip(idxs_list, scores_list) if i is not None and int(i) >= 0]
    if not pairs:
        return meta.head(0).copy()

    good_idxs = [int(i) for i, _ in pairs]
    good_scores = [float(s) for _, s in pairs]

    hits = meta.iloc[good_idxs].copy()
    hits["faiss_score"] = good_scores
    return hits.reset_index(drop=True)


def retrieve_candidates(
    query: str,
    sec_index: faiss.Index,
    rec_index: faiss.Index,
    fig_index: faiss.Index,
    sec_meta: pd.DataFrame,
    rec_meta: pd.DataFrame,
    fig_meta: pd.DataFrame,
    embed_model: SentenceTransformer,
    k_sections: int = 30,
    k_recs: int = 40,
    k_figs: int = 12,
) -> pd.DataFrame:
    a = faiss_search(sec_index, sec_meta, embed_model, query, k_sections)
    b = faiss_search(rec_index, rec_meta, embed_model, query, k_recs)
    c = faiss_search(fig_index, fig_meta, embed_model, query, k_figs)

    if not a.empty:
        a["source_index"] = "sections"
    if not b.empty:
        b["source_index"] = "recommendations"
    if not c.empty:
        c["source_index"] = "figures"

    if a.empty and b.empty and c.empty:
        return pd.DataFrame()

    candidates = pd.concat([a, b, c], ignore_index=True)
    candidates = candidates.sort_values("faiss_score", ascending=False).drop_duplicates("chunk_id")
    return candidates.reset_index(drop=True)


def rerank(query: str, candidates: pd.DataFrame, reranker: CrossEncoder, top_k: int = 24) -> pd.DataFrame:
    if candidates is None or len(candidates) == 0:
        return candidates
    pairs = [(query, t) for t in candidates["text"].tolist()]
    scores = reranker.predict(pairs)
    out = candidates.copy()
    out["rerank_score"] = scores
    return out.sort_values("rerank_score", ascending=False).head(top_k).reset_index(drop=True)


def dedupe_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    sim_threshold: float = 0.92,
    max_keep: int = 16,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    kept_rows = []
    seen_hashes = set()
    kept_norm_texts: List[str] = []

    for _, row in df.iterrows():
        txt = str(row.get(text_col, "") or "")
        norm = normalize_for_dedupe(txt)
        if not norm:
            continue

        h = sha16(norm)
        if h in seen_hashes:
            continue

        is_near_dup = False
        for prev in kept_norm_texts:
            if SequenceMatcher(a=norm, b=prev).ratio() >= sim_threshold:
                is_near_dup = True
                break
        if is_near_dup:
            continue

        kept_rows.append(row)
        kept_norm_texts.append(norm)
        seen_hashes.add(h)

        if len(kept_rows) >= max_keep:
            break

    return pd.DataFrame(kept_rows).reset_index(drop=True)


def citation_object(row: pd.Series) -> dict:
    signals = safe_json_loads(row.get("signals_json"))
    return {
        "chunk_id": row.get("chunk_id"),
        "doc_title": row.get("doc_title"),
        "year": row.get("year"),
        "page": int(row.get("page_start", 0)),
        "section_num": row.get("section_num"),
        "section_title": row.get("section_title"),
        "chunk_type": row.get("chunk_type"),
        "class": signals.get("class"),
        "loe": signals.get("loe"),
        "pdf_path": row.get("pdf_path"),
        "signals": signals,
    }


def build_context_bundle(
    query: str,
    sec_index: faiss.Index,
    rec_index: faiss.Index,
    fig_index: faiss.Index,
    sec_meta: pd.DataFrame,
    rec_meta: pd.DataFrame,
    fig_meta: pd.DataFrame,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    top_k_final: int = 12,
    post_rerank_k: int = 24,
    dedupe_similarity: float = 0.92,
    k_sections: int = 30,
    k_recs: int = 40,
    k_figs: int = 12,
) -> dict:
    candidates = retrieve_candidates(
        query=query,
        sec_index=sec_index,
        rec_index=rec_index,
        fig_index=fig_index,
        sec_meta=sec_meta,
        rec_meta=rec_meta,
        fig_meta=fig_meta,
        embed_model=embed_model,
        k_sections=k_sections,
        k_recs=k_recs,
        k_figs=k_figs,
    )

    top = rerank(query, candidates, reranker=reranker, top_k=post_rerank_k)
    top = dedupe_near_duplicates(top, sim_threshold=dedupe_similarity, max_keep=top_k_final)

    citations = [citation_object(top.iloc[i]) for i in range(len(top))]
    context_blocks = [f"[{top.iloc[i]['chunk_id']}]\n{top.iloc[i]['text']}" for i in range(len(top))]

    return {
        "query": query,
        "context_text": "\n\n---\n\n".join(context_blocks),
        "citations": citations,
        "top_table": top,
    }


def build_all_artifacts() -> None:
    manifest, sec_meta, rec_meta, fig_meta = build_corpus_artifacts()
    save_corpus_artifacts(manifest, sec_meta, rec_meta, fig_meta)
    build_and_save_indexes(sec_meta, rec_meta, fig_meta)


def retrieve_context(
    query: str,
    top_k_chunks: int = 12,
    post_rerank_k: int = 24,
    dedupe_similarity: float = 0.92,
    k_sections: int = 30,
    k_recs: int = 40,
    k_figs: int = 12,
) -> dict:
    sec_index, rec_index, fig_index, sec_meta, rec_meta, fig_meta, embed_model, reranker = load_runtime()
    return build_context_bundle(
        query=query,
        sec_index=sec_index,
        rec_index=rec_index,
        fig_index=fig_index,
        sec_meta=sec_meta,
        rec_meta=rec_meta,
        fig_meta=fig_meta,
        embed_model=embed_model,
        reranker=reranker,
        top_k_final=top_k_chunks,
        post_rerank_k=post_rerank_k,
        dedupe_similarity=dedupe_similarity,
        k_sections=k_sections,
        k_recs=k_recs,
        k_figs=k_figs,
    )


if __name__ == "__main__":
    needs_build = not (
        (OUT_DIR / "esc_sections.faiss").exists()
        and (OUT_DIR / "esc_recommendations.faiss").exists()
        and (OUT_DIR / "esc_figures.faiss").exists()
        and (OUT_DIR / "esc_sections_meta.parquet").exists()
        and (OUT_DIR / "esc_recommendations_meta.parquet").exists()
        and (OUT_DIR / "esc_figures_meta.parquet").exists()
        and SEC_IDMAP_PATH.exists()
        and REC_IDMAP_PATH.exists()
        and FIG_IDMAP_PATH.exists()
    )
    if needs_build:
        build_all_artifacts()

    _ = retrieve_context(
        "acute heart failure initial management diuretics oxygen saturation",
        top_k_chunks=10,
        post_rerank_k=22,
        k_sections=30,
        k_recs=40,
        k_figs=12,
    )