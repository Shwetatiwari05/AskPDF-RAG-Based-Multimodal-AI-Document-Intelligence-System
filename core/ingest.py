"""
ingest.py — Multi-PDF Ingestion Pipeline (per-user isolated)

Directory structure:
    pdf_stores/
        <user_id>/
            <pdf_id>/
                vector_store.index
                chunks.pkl
                metadata.json
                thumbnail.png
                chat_history.json
                summarize_history.json
                redact_history.json

Usage (CLI uses dummy user_id "default_user"):
    python -m core.ingest sample.pdf
    python -m core.ingest sample.pdf --force-ocr
    python -m core.ingest --list
    python -m core.ingest --delete <pdf_id>
    python -m core.ingest --force sample.pdf
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import json
import hashlib
import shutil
import argparse
import fitz
from datetime import datetime
from pathlib import Path

from core.ocr_extractor import extract_text_smart, is_scanned_pdf
from core.preprocess import clean_text, chunk_text
from core.embedder import generate_embeddings
from core.vector_store import create_vector_store, save_vector_store, load_vector_store
from core.db import get_document
from core.supabase_client import supabase

# ── STORE ROOT ────────────────────────────────────────────────────────────────

STORES_DIR = Path("pdf_stores")
STORES_DIR.mkdir(exist_ok=True)


# ── PDF ID GENERATION ─────────────────────────────────────────────────────────

def get_pdf_id(pdf_path: str) -> str:
    """
    Unique ID for a PDF: clean filename stem + first 8 chars of MD5 hash.
    Same file always gets the same ID — prevents duplicate ingestion.
    """
    pdf_path = Path(pdf_path)
    with open(pdf_path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()[:8]
    clean_name = pdf_path.stem.replace(" ", "_").replace("/", "_")[:40]
    return f"{clean_name}_{content_hash}"


# ── STORE PATH ────────────────────────────────────────────────────────────────

def get_store_path(user_id: str, pdf_id: str) -> Path:
    return STORES_DIR / user_id / pdf_id


def recover_document_from_storage(user_id: str, pdf_id: str) -> bool:
    """
    Download a PDF from Supabase Storage and re-ingest it to rebuild
    missing local artifacts (vector store, metadata, thumbnail).

    Returns True on success, False on failure.
    """
    store_path = get_store_path(user_id, pdf_id)

    if (store_path / "vector_store.index").exists():
        print(f"[Recovery] Vector store already exists for {pdf_id}, skipping.")
        return True

    print(f"[Recovery] Starting recovery for pdf_id={pdf_id}, user_id={user_id}")

    doc = get_document(pdf_id, user_id)
    if not doc:
        print(f"[Recovery] Document not found in DB.")
        return False

    storage_path = doc.get("storage_path")
    if not storage_path:
        print(f"[Recovery] No storage_path found in DB record.")
        return False

    pdf_name = doc.get("pdf_name", f"{pdf_id}.pdf")

    print(f"[Recovery] Downloading PDF from Supabase Storage: {storage_path}")
    try:
        pdf_data = supabase.storage.from_("pdfs").download(storage_path)
    except Exception as e:
        print(f"[Recovery] Download failed: {e}")
        return False

    temp_dir = Path("temp_recovery")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / pdf_name

    try:
        with open(temp_path, "wb") as f:
            f.write(pdf_data)
        print(f"[Recovery] Saved temporary PDF: {temp_path}")

        metadata = ingest_pdf(
            str(temp_path),
            user_id,
            force_ocr=doc.get("force_ocr", False),
            used_for=doc.get("used_for", "chat"),
            storage_path=storage_path,
        )
        print(f"[Recovery] Rebuilt vector store successfully for {pdf_id}")
        return True

    except Exception as e:
        print(f"[Recovery] Rebuild failed: {e}")
        return False

    finally:
        if temp_path.exists():
            os.remove(temp_path)
            print(f"[Recovery] Cleaned up temporary PDF")


# ── THUMBNAIL ─────────────────────────────────────────────────────────────────

def generate_thumbnail(pdf_path: Path, store_path: Path) -> None:
    """
    Render page 1 to ~320px-wide PNG. Best-effort — failure never blocks
    ingestion because thumbnails are cosmetic.
    """
    try:
        doc = fitz.open(str(pdf_path))
        if doc.page_count == 0:
            doc.close()
            return
        page = doc.load_page(0)
        zoom = 320 / max(page.rect.width, 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        pix.save(str(store_path / "thumbnail.png"))
        doc.close()
    except Exception as e:
        print(f"   Thumbnail generation failed (non-fatal): {e}")


def get_or_create_thumbnail(user_id: str, pdf_id: str) -> Path | None:
    """
    Return the thumbnail path, generating it on the fly if missing.
    Returns None when the source PDF is no longer accessible.
    Does NOT trigger full document ingestion/recovery — only downloads
    the PDF from Supabase Storage to generate the thumbnail.
    """
    store_path = get_store_path(user_id, pdf_id)
    thumb_path = store_path / "thumbnail.png"
    if thumb_path.exists():
        return thumb_path

    store_path.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(user_id, pdf_id)
    storage_path = meta.get("storage_path") if meta else None

    if not storage_path:
        doc = get_document(pdf_id, user_id)
        if not doc:
            return None
        storage_path = doc.get("storage_path")
        if not storage_path:
            return None
        pdf_name = doc.get("pdf_name", f"{pdf_id}.pdf")
    else:
        pdf_name = meta.get("pdf_name", f"{pdf_id}.pdf")

    try:
        pdf_data = supabase.storage.from_("pdfs").download(storage_path)
    except Exception as e:
        print(f"[Thumbnail] Download failed: {e}")
        return None

    temp_dir = Path("temp_thumbnail")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / pdf_name

    try:
        with open(temp_path, "wb") as f:
            f.write(pdf_data)

        generate_thumbnail(temp_path, store_path)

    except Exception as e:
        print(f"[Thumbnail] Generation failed: {e}")
        return None

    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return thumb_path if thumb_path.exists() else None


# ── METADATA ──────────────────────────────────────────────────────────────────

def save_metadata(user_id: str, pdf_id: str, metadata: dict) -> None:
    store = get_store_path(user_id, pdf_id)
    store.mkdir(parents=True, exist_ok=True)
    path = store / "metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(user_id: str, pdf_id: str) -> dict:
    path = STORES_DIR / user_id / pdf_id / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── HISTORY ───────────────────────────────────────────────────────────────────
# One JSON file per (user, PDF, kind).  Stored inside pdf_stores/user_id/pdf_id/
# so it survives server restarts and is co-located with its vector store.
# Valid kinds: "chat" | "summarize" | "redact"

def _history_path(user_id: str, pdf_id: str, kind: str) -> Path:
    return STORES_DIR / user_id / pdf_id / f"{kind}_history.json"


def load_history(user_id: str, pdf_id: str, kind: str) -> list:
    path = _history_path(user_id, pdf_id, kind)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def append_history(user_id: str, pdf_id: str, kind: str, entry: dict) -> list:
    store = get_store_path(user_id, pdf_id)
    store.mkdir(parents=True, exist_ok=True)
    history = load_history(user_id, pdf_id, kind)
    history.append(entry)
    with open(_history_path(user_id, pdf_id, kind), "w") as f:
        json.dump(history, f, indent=2)
    return history


def clear_history(user_id: str, pdf_id: str, kind: str) -> None:
    path = _history_path(user_id, pdf_id, kind)
    if path.exists():
        path.unlink()


# ── INGEST ────────────────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str,
    user_id: str,
    force_ocr: bool = False,
    used_for: str = "chat",
    storage_path: str | None = None,
) -> dict:
    print("INGEST START")
    """
    Ingest a PDF into its own isolated vector store under the given user.

    Args:
        pdf_path  : path to the PDF file on disk
        user_id   : authenticated user's ID (from Supabase Auth)
        force_ocr : skip text-layer detection, always use Mistral OCR
        used_for  : "chat" | "summarize" | "redact" — stored in metadata
                    for the dashboard badge; updated on every re-upload.

    Returns:
        metadata dict (same shape written to metadata.json)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_id = get_pdf_id(str(pdf_path))
    store_path = get_store_path(user_id, pdf_id)
    print("DEBUG")
    print("user_id =", user_id)
    print("pdf_id =", pdf_id)
    print("store_path =", store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    # ── Already ingested? ─────────────────────────────────────────
    if (store_path / "vector_store.index").exists():
        print(f"Already ingested: {pdf_path.name} (id: {pdf_id})")
        meta = load_metadata(user_id, pdf_id)
        # Keep used_for badge in sync with the latest upload context.
        if meta.get("used_for") != used_for:
            meta["used_for"] = used_for
            save_metadata(user_id, pdf_id, meta)
        return {"pdf_id": pdf_id, "store_path": str(store_path), **meta}

    generate_thumbnail(pdf_path, store_path)

    print("BEFORE EXTRACTION")

    # ── Step 1: Extract ───────────────────────────────────────────
    print(f"\nIngesting: {pdf_path.name}")
    print("─" * 50)
    print("Step 1/5: Extracting text...")
    raw_text = extract_text_smart(str(pdf_path), force_ocr=force_ocr)

    print("AFTER EXTRACTION")
    print("TEXT LENGTH:", len(raw_text))

    # ── Step 2: Clean ─────────────────────────────────────────────
    print("Step 2/5: Cleaning text...")
    if is_scanned_pdf(str(pdf_path)) or force_ocr:
        cleaned_text = re.sub(r'[ \t]+', ' ', raw_text)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
    else:
        cleaned_text = clean_text(raw_text)

    # ── Step 3: Chunk ─────────────────────────────────────────────
    print("Step 3/5: Creating chunks...")
    chunks = chunk_text(cleaned_text)
    print(f"   Total chunks: {len(chunks)}")

    if not chunks:
        raise ValueError(
            "No extractable text found in this PDF — it may be blank "
            "or a scanned image with no OCR-readable content."
        )

    # ── Step 4: Embed ─────────────────────────────────────────────
    print("Step 4/5: Generating embeddings...")

    import gc

    try:
        embeddings = generate_embeddings(chunks)

        print(
            "Embedding shape:",
            embeddings.shape
        )

    except Exception as e:
        print("EMBEDDING ERROR:", repr(e))
        gc.collect()
        raise

    # ── Step 5: Save vector store ─────────────────────────────────
    print("Step 5/5: Building & saving vector store...")
    index = create_vector_store(embeddings)
    save_vector_store(
        index, chunks,
        index_path=str(store_path / "vector_store.index"),
        chunks_path=str(store_path / "chunks.pkl"),
    )

    # ── Metadata ──────────────────────────────────────────────────
    doc        = fitz.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    metadata = {
        "user_id": user_id,
        "pdf_name"   : pdf_path.name,
        "pdf_id"     : pdf_id,
        "page_count" : page_count,
        "chunk_count": len(chunks),
        "word_count" : len(cleaned_text.split()),
        "file_size"  : pdf_path.stat().st_size,
        "used_for"   : used_for,
        "force_ocr"  : force_ocr,
        "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "store_path" : str(store_path),
        "source_path": str(pdf_path),
        "storage_path": storage_path
    }
    save_metadata(user_id, pdf_id, metadata)

    print(f"\nIngestion complete!")
    print(f"   PDF ID  : {pdf_id}")
    print(f"   Pages   : {page_count}")
    print(f"   Chunks  : {len(chunks)}")
    print(f"   Store   : {store_path}")

    return metadata


# ── LIST ──────────────────────────────────────────────────────────────────────

def list_pdfs(user_id: str) -> list:
    """
    Return all ingested PDFs for a specific user, newest first.
    Only iterates pdf_stores/<user_id>/ — never touches other users' data.
    """
    user_dir = STORES_DIR / user_id
    if not user_dir.exists():
        return []

    pdfs = []
    for pdf_dir in user_dir.iterdir():
        if not pdf_dir.is_dir():
            continue
        meta = load_metadata(user_id, pdf_dir.name)
        if not meta:
            continue

        # Back-compat: fill in fields added after the first ingest.
        changed = False
        if "file_size" not in meta and meta.get("source_path"):
            try:
                meta["file_size"] = os.path.getsize(meta["source_path"])
                changed = True
            except OSError:
                meta["file_size"] = None
                changed = True
        if "used_for" not in meta:
            meta["used_for"] = "chat"
            changed = True
        if changed:
            save_metadata(user_id, pdf_dir.name, meta)

        pdfs.append(meta)

    pdfs.sort(key=lambda p: p.get("ingested_at", ""), reverse=True)
    return pdfs


def print_pdf_list(user_id: str) -> None:
    pdfs = list_pdfs(user_id)
    if not pdfs:
        print("No PDFs ingested yet.")
        return
    print(f"\n{'─' * 70}")
    print(f"{'#':<4} {'PDF Name':<35} {'Pages':<7} {'Chunks':<8} {'Ingested'}")
    print(f"{'─' * 70}")
    for i, pdf in enumerate(pdfs, 1):
        print(f"{i:<4} {pdf.get('pdf_name','?')[:34]:<35} "
              f"{pdf.get('page_count','?'):<7} "
              f"{pdf.get('chunk_count','?'):<8} "
              f"{pdf.get('ingested_at','?')}")
    print(f"{'─' * 70}")
    print(f"Total: {len(pdfs)} PDF(s)\n")


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_pdf_store(user_id: str, pdf_id: str):
    """
    Load vector index, chunks, and metadata for a specific user's PDF.
    Raises FileNotFoundError if the store doesn't exist.
    """
    store_path  = get_store_path(user_id, pdf_id)
    index_path  = store_path / "vector_store.index"
    chunks_path = store_path / "chunks.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"No vector store found for pdf_id={pdf_id}, user_id={user_id}"
        )

    index, chunks = load_vector_store(
        index_path=str(index_path),
        chunks_path=str(chunks_path),
    )
    metadata = load_metadata(user_id, pdf_id)
    return index, chunks, metadata


def find_pdf_by_name(user_id: str, name: str) -> str:
    """Find a PDF ID by partial name match within a user's PDFs."""
    pdfs = list_pdfs(user_id)
    name_lower = name.lower()
    matches = [p for p in pdfs if name_lower in p.get("pdf_name", "").lower()]
    if not matches:
        raise ValueError(f"No PDF found matching: {name}")
    if len(matches) > 1:
        print(f"Multiple matches — using first: {matches[0]['pdf_name']}")
    return matches[0]["pdf_id"]


# ── DELETE ────────────────────────────────────────────────────────────────────

def delete_pdf_store(user_id: str, pdf_id: str) -> None:
    """Delete only this user's copy of the PDF store."""
    store = get_store_path(user_id, pdf_id)
    if store.exists():
        shutil.rmtree(store)
        print(f"Deleted store: {store}")
    else:
        print(f"No store found for pdf_id={pdf_id}, user_id={user_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────
# Uses "default_user" as the dummy user_id so the CLI stays usable for
# local testing without a real authentication context.

DEFAULT_USER = "default_user"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
    parser.add_argument("pdf_path", nargs="?", default="sample.pdf")
    parser.add_argument("--force-ocr", action="store_true",
                        help="Force Mistral OCR")
    parser.add_argument("--list",   action="store_true",
                        help="List all ingested PDFs")
    parser.add_argument("--delete", type=str, metavar="PDF_ID",
                        help="Delete a PDF store by ID")
    parser.add_argument("--force",  action="store_true",
                        help="Re-ingest even if already exists")
    parser.add_argument("--user",   type=str, default=DEFAULT_USER,
                        help=f"User ID (default: {DEFAULT_USER})")
    args = parser.parse_args()

    if args.list:
        print_pdf_list(args.user)

    elif args.delete:
        delete_pdf_store(args.user, args.delete)

    else:
        if args.force:
            pdf_id = get_pdf_id(args.pdf_path)
            delete_pdf_store(args.user, pdf_id)
        ingest_pdf(args.pdf_path, user_id=args.user, force_ocr=args.force_ocr)































# """
# ingest.py — Multi-PDF Ingestion Pipeline

# Each PDF gets its own isolated vector store:
#     pdf_stores/
#         <pdf_id>/
#             vector_store.index
#             chunks.pkl
#             metadata.json   ← PDF name, date, page count etc.

# Usage:
#     python -m ingest                          # ingest sample.pdf
#     python -m ingest my_doc.pdf               # ingest specific PDF
#     python -m ingest my_doc.pdf --force-ocr   # force Mistral OCR
#     python -m ingest --list                   # list all ingested PDFs
#     python -m ingest --delete <pdf_id>        # delete a PDF's store
# """

# import os
# import json
# import hashlib
# import argparse
# import fitz
# from datetime import datetime
# from pathlib import Path

# from core.ocr_extractor import extract_text_smart, is_scanned_pdf
# from core.preprocess import clean_text, chunk_text
# from core.embedder import generate_embeddings
# from core.vector_store import create_vector_store, save_vector_store, load_vector_store

# # STORE DIRECTORY

# STORES_DIR = Path("pdf_stores")
# STORES_DIR.mkdir(exist_ok=True)


# # PDF ID GENERATION 

# def get_pdf_id(pdf_path: str) -> str:
#     """
#     Generate a unique ID for a PDF based on its filename + content hash.
#     Same file = same ID (avoids duplicate ingestion).
#     """
#     pdf_path = Path(pdf_path)
#     with open(pdf_path, "rb") as f:
#         content_hash = hashlib.md5(f.read()).hexdigest()[:8]
#     # Clean filename for use as directory name
#     clean_name = pdf_path.stem.replace(" ", "_").replace("/", "_")[:40]
#     return f"{clean_name}_{content_hash}"


# def get_store_path(pdf_id: str) -> Path:
#     """Get the directory path for a PDF's vector store."""
#     return STORES_DIR / pdf_id


# def generate_thumbnail(pdf_path: Path, store_path: Path) -> None:
#     """
#     Render page 1 to a small PNG so the dashboard can show a real preview
#     instead of a generic file icon. Best-effort — a failure here should
#     never block ingestion, since the thumbnail is cosmetic.
#     """
#     try:
#         doc = fitz.open(str(pdf_path))
#         if doc.page_count == 0:
#             doc.close()
#             return
#         page = doc.load_page(0)
#         zoom = 320 / max(page.rect.width, 1)   # ~320px wide is plenty for a card
#         pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
#         pix.save(str(store_path / "thumbnail.png"))
#         doc.close()
#     except Exception as e:
#         print(f"   Thumbnail generation failed: {e}")


# def get_or_create_thumbnail(pdf_id: str) -> Path | None:
#     """
#     Returns the thumbnail path for a PDF, generating it on the fly if it's
#     missing — covers documents ingested before thumbnails existed. Returns
#     None if there's no way to generate one (e.g. source file moved/deleted).
#     """
#     store_path = get_store_path(pdf_id)
#     thumb_path = store_path / "thumbnail.png"
#     if thumb_path.exists():
#         return thumb_path

#     meta = load_metadata(pdf_id)
#     source_path = meta.get("source_path")
#     if not source_path or not Path(source_path).exists():
#         return None

#     generate_thumbnail(Path(source_path), store_path)
#     return thumb_path if thumb_path.exists() else None


# #  MULTI-PDF INGEST

# def ingest_pdf(pdf_path: str, force_ocr: bool = False, used_for: str = "chat") -> dict:
    
#     pdf_path = Path(pdf_path)
#     if not pdf_path.exists():
#         raise FileNotFoundError(f"PDF not found: {pdf_path}")

#     pdf_id = get_pdf_id(str(pdf_path))
#     store_path = get_store_path(pdf_id)

#     # ── Already ingested? ────────────────────────────────────────
#     if store_path.exists() and (store_path / "vector_store.index").exists():
#         print(f"Already ingested: {pdf_path.name} (id: {pdf_id})")
#         print("   Use --force to re-ingest.")
#         meta = load_metadata(pdf_id)
#         # Keep "used_for" in sync with how this file was just (re)uploaded,
#         # so the dashboard badge reflects the most recent context even on
#         # a cache hit.
#         if meta.get("used_for") != used_for:
#             meta["used_for"] = used_for
#             save_metadata(pdf_id, meta)
#         return {"pdf_id": pdf_id, "store_path": str(store_path), **meta}

#     store_path.mkdir(parents=True, exist_ok=True)
#     generate_thumbnail(pdf_path, store_path)

#     # ── Extraction ───────────────────────────────────────────────
#     print(f"\n Ingesting: {pdf_path.name}")
#     print("─" * 50)

#     print("Step 1/5: Extracting text...")
#     raw_text = extract_text_smart(str(pdf_path), force_ocr=force_ocr)

#     # ── Cleaning ─────────────────────────────────────────────────
#     print("Step 2/5: Cleaning text...")

#     import re
#     if is_scanned_pdf(str(pdf_path)) or force_ocr:
#         # Scanned/Mistral output — light cleaning only
#         cleaned_text = re.sub(r'[ \t]+', ' ', raw_text) # normalize whitespace
#         cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()  # remove excessive newlines
#     else:
#         cleaned_text = clean_text(raw_text)

#     # ── Chunking ─────────────────────────────────────────────────
#     print("Step 3/5: Creating chunks...")
#     chunks = chunk_text(cleaned_text)
#     print(f"   Total chunks: {len(chunks)}")

#     if not chunks:
#         raise ValueError(
#             "No extractable text found in this PDF — it may be blank, "
#             "or a scanned image with no OCR-readable content."
#         )

#     # ── Embeddings ───────────────────────────────────────────────
#     print("Step 4/5: Generating embeddings...")
#     embeddings = generate_embeddings(chunks)

#     # ── Vector Store ─────────────────────────────────────────────
#     print("Step 5/5: Building & saving vector store...")
#     index = create_vector_store(embeddings)

#     index_path = str(store_path / "vector_store.index")
#     chunks_path = str(store_path / "chunks.pkl")
#     save_vector_store(index, chunks, index_path=index_path, chunks_path=chunks_path)

#     # ── Metadata ─────────────────────────────────────────────────
#     import fitz
#     doc = fitz.open(str(pdf_path))
#     page_count = len(doc)
#     doc.close()

#     metadata = {
#         "pdf_name": pdf_path.name,
#         "pdf_id": pdf_id,
#         "page_count": page_count,
#         "chunk_count": len(chunks),
#         "word_count": len(cleaned_text.split()),
#         "ingested_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
#         "file_size": pdf_path.stat().st_size,
#         "used_for": used_for,
#         "force_ocr": force_ocr,
#         "store_path": str(store_path),
#         "source_path": str(pdf_path)
#     }
#     save_metadata(pdf_id, metadata)

#     print(f"\nIngestion complete!")
#     print(f"   PDF ID    : {pdf_id}")
#     print(f"   Pages     : {page_count}")
#     print(f"   Chunks    : {len(chunks)}")
#     print(f"   Store     : {store_path}")

#     return metadata


# # METADATA

# def save_metadata(pdf_id: str, metadata: dict):
#     """Save PDF metadata to JSON."""
#     path = get_store_path(pdf_id) / "metadata.json"
#     with open(path, "w") as f:
#         json.dump(metadata, f, indent=2)


# def load_metadata(pdf_id: str) -> dict:
#     """Load PDF metadata from JSON."""
#     path = get_store_path(pdf_id) / "metadata.json"
#     if path.exists():
#         with open(path) as f:
#             return json.load(f)
#     return {}


# # HISTORY (chat / summarize / redact) — one JSON file per PDF per kind,
# # stored alongside its vector store, so it survives server restarts and
# # is exactly "what happened" the next time this document is opened.

# def _history_path(pdf_id: str, user_id: str, kind: str) -> Path:
#     return get_store_path(pdf_id) / f"{user_id}_{kind}_history.json"


# def load_history(pdf_id: str, user_id: str, kind: str) -> list:
#     path = _history_path(pdf_id, user_id, kind)
#     if path.exists():
#         with open(path) as f:
#             return json.load(f)
#     return []


# def append_history(pdf_id: str, user_id: str, kind: str, entry: dict):
#     get_store_path(pdf_id).mkdir(parents=True, exist_ok=True)
#     history = load_history(pdf_id, user_id, kind)
#     history.append(entry)
#     with open(_history_path(pdf_id, user_id, kind), "w") as f:
#         json.dump(history, f, indent=2)
#     return history


# def clear_history(pdf_id: str, user_id: str, kind: str):
#     path = _history_path(pdf_id, user_id, kind)
#     if path.exists():
#         path.unlink()


# # LIST ALL PDFs 

# def list_pdfs() -> list:
#     """List all ingested PDFs with their metadata."""
#     pdfs = []
#     for store_dir in sorted(STORES_DIR.iterdir()):
#         if store_dir.is_dir():
#             meta = load_metadata(store_dir.name)
#             if meta:
#                 # Backward-compat: older ingests (before "file_size" was
#                 # tracked) won't have this field saved. Compute it on the
#                 # fly from the original source file and cache it back to
#                 # metadata.json so we only ever do this once per PDF.
#                 if "file_size" not in meta and meta.get("source_path"):
#                     try:
#                         meta["file_size"] = os.path.getsize(meta["source_path"])
#                         save_metadata(store_dir.name, meta)
#                     except OSError:
#                         meta["file_size"] = None
#                 if "used_for" not in meta:
#                     meta["used_for"] = "chat"
#                     save_metadata(store_dir.name, meta)
#                 pdfs.append(meta)

#     # Newest upload first — "ingested_at" is "YYYY-MM-DD HH:MM", which
#     # sorts correctly as a plain string, so no date parsing needed.
#     pdfs.sort(key=lambda p: p.get("ingested_at", ""), reverse=True)
#     return pdfs


# def print_pdf_list():
#     """Print all ingested PDFs in a nice format."""
#     pdfs = list_pdfs()
#     if not pdfs:
#         print("No PDFs ingested yet.")
#         return

#     print(f"\n{'─' * 70}")
#     print(f"{'#':<4} {'PDF Name':<35} {'Pages':<7} {'Chunks':<8} {'Ingested'}")
#     print(f"{'─' * 70}")
#     for i, pdf in enumerate(pdfs, 1):
#         print(f"{i:<4} {pdf.get('pdf_name','?')[:34]:<35} "
#               f"{pdf.get('page_count','?'):<7} "
#               f"{pdf.get('chunk_count','?'):<8} "
#               f"{pdf.get('ingested_at','?')}")
#     print(f"{'─' * 70}")
#     print(f"Total: {len(pdfs)} PDF(s)\n")
#     return pdfs


# # LOAD BY ID OR NAME

# def load_pdf_store(pdf_id: str):
#     """
#     Load vector store for a specific PDF by its ID.
#     Returns (index, chunks, metadata)
#     """
#     store_path = get_store_path(pdf_id)
#     if not store_path.exists():
#         raise FileNotFoundError(f"No store found for PDF ID: {pdf_id}")

#     index_path = str(store_path / "vector_store.index")
#     chunks_path = str(store_path / "chunks.pkl")
#     index, chunks = load_vector_store(index_path=index_path, chunks_path=chunks_path)
#     metadata = load_metadata(pdf_id)

#     return index, chunks, metadata


# def find_pdf_by_name(name: str) -> str:
#     """Find a PDF ID by partial name match."""
#     pdfs = list_pdfs()
#     name_lower = name.lower()
#     matches = [p for p in pdfs if name_lower in p.get("pdf_name", "").lower()]
#     if not matches:
#         raise ValueError(f"No PDF found matching: {name}")
#     if len(matches) > 1:
#         print(f"Multiple matches found — using first: {matches[0]['pdf_name']}")
#     return matches[0]["pdf_id"]


# # DELETE 

# def delete_pdf_store(pdf_id: str):
#     """Delete a PDF's vector store."""
#     import shutil
#     store_path = get_store_path(pdf_id)
#     if store_path.exists():
#         shutil.rmtree(store_path)
#         print(f"Deleted store for: {pdf_id}")
#     else:
#         print(f"No store found for: {pdf_id}")


# # CLI 

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
#     parser.add_argument("pdf_path", nargs="?", default="sample.pdf",
#                         help="Path to PDF file")
#     parser.add_argument("--force-ocr", action="store_true",
#                         help="Force Mistral OCR (for mixed text+image PDFs)")
#     parser.add_argument("--list", action="store_true",
#                         help="List all ingested PDFs")
#     parser.add_argument("--delete", type=str, metavar="PDF_ID",
#                         help="Delete a PDF's vector store by ID")
#     parser.add_argument("--force", action="store_true",
#                         help="Re-ingest even if already exists")

#     args = parser.parse_args()

#     if args.list:
#         print_pdf_list()

#     elif args.delete:
#         delete_pdf_store(args.delete)

#     else:
#         if args.force:
#             # Delete existing store to force re-ingest
#             pdf_id = get_pdf_id(args.pdf_path)
#             delete_pdf_store(pdf_id)
#         ingest_pdf(args.pdf_path, force_ocr=args.force_ocr)