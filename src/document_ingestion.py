# Loads FIRs and judgments, OCRs scanned pages, extracts entities, and returns hashed, tagged chunks.

import os
import re
import glob
import json
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    logger, DATA_DIR, CORPUS_DIR, ARTIFACTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, setup_environment,
)

# OCR is optional; only needed for scanned/image PDFs.
try:
    import pytesseract
    from pdf2image import convert_from_path
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False

# Entity patterns for Indian case records.
_PHONE_RE = re.compile(r'\+?\d[\d\s\-]{8,}\d')
_VEHICLE_RE = re.compile(r'[A-Z]{2}[\s\-]?[0-9OlI]{1,2}[\s\-]?[A-Z]{1,2}[\s\-]?[0-9OlI]{4}')
_SECTION_RE = re.compile(r'(?:BNS|BNSS|IPC|POCSO)\s?\d+', re.IGNORECASE)
_NAME_FIELD_RE = re.compile(r'Accused Name\s*:\s*(.+)')
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def detect_language(text):
    # Return 'hi', 'mixed', or 'en' based on script presence.
    has_hindi = bool(_DEVANAGARI_RE.search(text))
    has_latin = bool(re.search(r'[A-Za-z]', text))
    if has_hindi and has_latin:
        return "mixed"
    return "hi" if has_hindi else "en"


def extract_entities(text):
    # Extract FIR-style entities from a chunk; judgment-level fields are parsed once per file, not here.
    names = [n.strip() for n in _NAME_FIELD_RE.findall(text)]
    phones = [p.strip() for p in _PHONE_RE.findall(text) if len(re.sub(r'\D', '', p)) >= 10]
    vehicles = [v.strip() for v in _VEHICLE_RE.findall(text)]
    raw = {re.sub(r"\s+", "", s).upper() for s in _SECTION_RE.findall(text)}
    sections = sorted(s for s in raw if re.match(r"^\d{2,4}[A-Z]{0,2}$", s) and not s.endswith("OF"))
    return {"names": names, "phones": phones, "vehicles": vehicles, "sections": sections}


def _make_chunk_id(text, source_file, page, index):
    # Deterministic SHA-256 id so the same chunk always gets the same id across runs.
    raw = f"{source_file}::{page}::{index}::{text}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    stem = os.path.splitext(source_file)[0]
    return f"{stem}_{digest}"


def _ocr_page(path, page_number):
    # OCR a single PDF page in English + Hindi.
    images = convert_from_path(path, first_page=page_number, last_page=page_number)
    return pytesseract.image_to_string(images[0], lang="eng+hin") if images else ""


def _read_file(path):
    # Return a list of (page_number, text); OCR any PDF page that has almost no extractable text.
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [(1, f.read())]
    if ext == ".pdf":
        pages = []
        try:
            for i, page in enumerate(PyPDFLoader(path).load(), 1):
                text = page.page_content.strip()
                if len(text) < 20 and _OCR_AVAILABLE:
                    text = _ocr_page(path, i)
                pages.append((i, text))
        except Exception as exc:
            logger.error("Failed to read PDF %s: %s", path, exc)
        return pages
    return []


def _load_manifest():
    # Map each synthetic file to its case metadata and ground-truth entity id.
    path = os.path.join(CORPUS_DIR, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {r["file"]: r for r in json.load(f)}


def load_and_chunk_documents():
    # Discover corpus + judgments, chunk them, and attach entity and case metadata to each chunk.
    setup_environment()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    manifest = _load_manifest()
    files = (glob.glob(os.path.join(DATA_DIR, "*.txt")) +
             glob.glob(os.path.join(DATA_DIR, "*.pdf")) +
             glob.glob(os.path.join(DATA_DIR, "*.PDF")) +
             glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if not files:
        raise FileNotFoundError(
            "No documents found. Run synthetic_corpus_generator and/or add judgment PDFs to data/raw_docs."
        )

    all_chunks = []
    for path in files:
        file_name = os.path.basename(path)
        record = manifest.get(file_name, {})
        pages = _read_file(path)

        # Parse the judgment ONCE on its full text (court/year/outcome/repeat need the whole document).
        full_text = " ".join(t for _, t in pages)
        file_judgment_meta = {}
        if (file_name.lower().endswith(".pdf") or file_name.lower().endswith(".txt")) and ("vs" in full_text[:2000].lower() or "versus" in full_text[:2000].lower()):
            from src.judgment_parser import parse_judgment
            file_judgment_meta = parse_judgment(full_text)

        for page_number, page_text in pages:
            if not page_text.strip():
                continue
            for idx, chunk_text in enumerate(splitter.split_text(page_text)):
                metadata = {
                    "source_file": file_name,
                    "case_id": record.get("case_id", file_name),
                    "district": record.get("district", "unknown"),
                    "is_women_safety": record.get("is_women_safety", True),
                    "true_entity_id": record.get("true_entity_id", ""),
                    "arrest_date": record.get("arrest_date"),
                    "page": page_number,
                    "language": detect_language(chunk_text),
                    "entities_json": json.dumps(extract_entities(chunk_text), ensure_ascii=False),
                    "file_judgment_meta": json.dumps(file_judgment_meta, ensure_ascii=False),
                    "chunk_index": idx,
                }
                metadata["chunk_id"] = _make_chunk_id(chunk_text, file_name, page_number, idx)
                all_chunks.append(Document(page_content=chunk_text, metadata=metadata))
        logger.info("Ingested %s", file_name)

    logger.info("Ingestion complete: %d chunks from %d files.", len(all_chunks), len(files))
    return all_chunks


def build_case_entity_index(chunks):
    # Aggregate all entities per case_id and save the index CONNECT will use.
    index = {}
    for chunk in chunks:
        m = chunk.metadata
        cid = m["case_id"]
        ents = json.loads(m.get("entities_json", "{}"))
        jm = json.loads(m.get("file_judgment_meta", "{}"))
        rec = index.setdefault(cid, {
            "names": set(), "phones": set(), "vehicles": set(), "sections": set(),
            "district": m.get("district", "unknown"), "true_entity_id": m.get("true_entity_id", ""),
            "arrest_date": m.get("arrest_date"),
        })
        rec["phones"].update(ents.get("phones", []))
        rec["vehicles"].update(ents.get("vehicles", []))
        rec["sections"].update(ents.get("sections", []))

        # Judgment files: the accused is the single parsed name; also carry court/year/outcome/flag.
        if jm and jm.get("accused"):
            rec["names"].add(jm["accused"])
            rec["judgment_meta"] = {
                "court": jm.get("court"), "year": jm.get("year"), "outcome": jm.get("outcome"),
                "repeat_offender_signal": jm.get("repeat_offender_signal"),
                "repeat_evidence": jm.get("repeat_evidence"),
            }
        else:
            rec["names"].update(ents.get("names", []))

    for rec in index.values():
        for key in ("names", "phones", "vehicles", "sections"):
            rec[key] = sorted(rec[key])

    out_path = os.path.join(ARTIFACTS_DIR, "case_entities.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    logger.info("Case entity index saved: %d cases -> %s", len(index), out_path)
    return index


if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    build_case_entity_index(chunks)