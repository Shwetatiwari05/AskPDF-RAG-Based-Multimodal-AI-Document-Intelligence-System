"""
main.py — FastAPI Backend for AI Document Intelligence System

Connects all existing modules (core/, features/) to web endpoints.
Run from the project root (same level as core/ and features/):

    uvicorn main:app --reload

Endpoints:
    POST /upload              — upload a PDF, ingest it (returns pdf_id)
    GET  /documents            — list all ingested PDFs
    DELETE /documents/{pdf_id} — delete a PDF's vector store
    POST /chat                 — ask a question about a PDF (with memory)
    GET  /chat/history/{pdf_id} — saved chat history for a PDF
    POST /chat/clear            — clear conversation memory + history for a session
    POST /summarize             — summarize a PDF
    GET  /summarize/history/{pdf_id} — saved summaries for a PDF
    POST /redact                 — redact a PDF, returns downloadable file
    GET  /redact/history/{pdf_id} — saved redaction runs for a PDF
    POST /voice/transcribe        — transcribe uploaded audio (from browser mic)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from typing import TYPE_CHECKING
from core.supabase_client import supabase
from fastapi.responses import Response

if TYPE_CHECKING:
    from core.rag_engine import ConversationMemory

from fastapi import Depends
from core.auth import get_current_user
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel


# from fastapi import Depends
# from core.auth import get_current_user
# import os
# import shutil
# import uuid
# from datetime import datetime
# from pathlib import Path

# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from pydantic import BaseModel

# ─── EXISTING PROJECT MODULES ─────────────────────────────────────────────────
# print("Loading ingest")
# from core.ingest import (
#     ingest_pdf,
#     list_pdfs,
#     load_pdf_store,
#     delete_pdf_store,
#     load_history,
#     append_history,
#     clear_history,
#     load_metadata,
#     get_or_create_thumbnail,
# )
# print("Loading rag")
# from core.rag_engine import generate_answer, ConversationMemory

# print("Loading summarizer")
# from features.summarizer import summarize_plain_text

# print("Loading redactor")
# from features.redactor import redact_pdf

print("Loading db")
from core.db import (
    insert_document,
    get_documents,
    delete_document_db,
    get_document,
)


# ─── APP SETUP ─────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Document Intelligence API")

# Allow frontend (React, running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-document-intelligence-mu.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ─── IN-MEMORY SESSION STORE (conversation memory per chat session) ───────────
# Key: session_id (sent by frontend) → ConversationMemory instance
# NOTE: This resets if the server restarts. Once you add a database,
#       you can persist chat history there instead.

chat_sessions: dict[str, "ConversationMemory"] = {}

def get_or_create_memory(session_id: str):
    from core.rag_engine import ConversationMemory

    if session_id not in chat_sessions:
        chat_sessions[session_id] = ConversationMemory(max_exchanges=12)

    return chat_sessions[session_id]


# ─── REQUEST/RESPONSE MODELS ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    pdf_id: str
    query: str
    session_id: str  # frontend generates a UUID per chat session


class ChatResponse(BaseModel):
    answer: str
    sources: list  # list of {"page_num", "page_range", "text"} for citations later


class SummarizeRequest(BaseModel):
    pdf_id: str | None = None
    text: str | None = None


class RedactRequest(BaseModel):
    pdf_id: str
    redact_types: list[str] = []
    custom_terms: list[str] = []


class ClearMemoryRequest(BaseModel):
    session_id: str


# ─── 1. UPLOAD & INGEST ────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    force_ocr: bool = Form(False),
    used_for: str = Form("chat"),
    current_user = Depends(get_current_user)
):
    print("1. ENTERED UPLOAD API")
    print(file.filename)
    from core.ingest import ingest_pdf
    """
    Upload a PDF — saves it, then ingests it into its own vector store.
    Frontend should show a loading state while this runs (can take a while
    for large/scanned PDFs).

    "used_for" records which page the upload came from (chat/summarize/
    redact) so the dashboard can show what this document was used for.
    """
    print("2. Saving locally")
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    
    # Sanitize filename for local storage and Supabase
    filename = os.path.basename(file.filename)

    filename = re.sub(r"\s+", "_", filename)

    filename = re.sub(r"[^A-Za-z0-9._-]", "_", filename)

    filename = re.sub(r"_+", "_", filename)

    print("Original filename:", file.filename)
    print("Sanitized filename:", filename)

    # Each upload gets its own subfolder (named with a random ID) so that
    # uploading two different files with the same name (e.g. "resume.pdf"
    # twice) never overwrites an earlier upload. The original filename is
    # kept as-is inside that folder so /redact can find it later using the
    # "saved_path" we store in metadata below.
    
    unique_dir = UPLOAD_DIR / uuid.uuid4().hex
    unique_dir.mkdir(parents=True, exist_ok=True)
    save_path = unique_dir / filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    print("3. Saved locally")

    storage_path = f"{current_user.id}/{uuid.uuid4().hex}_{filename}"
    print("4. Uploading to Supabase")

    try:
        with open(save_path, "rb") as f:
            res = supabase.storage.from_("pdfs").upload(
                storage_path,
                f
            )

        print("SUPABASE RESPONSE =", res)
        print("5. Uploaded to Supabase")

    except Exception as e:
        print("SUPABASE UPLOAD FAILED")
        print(repr(e))
        raise
    print("6. Calling ingest")
    print("7. INGEST START")
    try:
        metadata = ingest_pdf(
            str(save_path),
            current_user.id,
            force_ocr=force_ocr,
            used_for=used_for,
            storage_path=storage_path
        )
        print("INGEST COMPLETED")

        existing = get_document(metadata["pdf_id"], current_user.id)
        print("EXISTING =", existing)

        if existing is None:
            response = insert_document(metadata, current_user.id)
            print("INSERT RESPONSE =", response)

    except Exception as e:
        print("DB ERROR =", repr(e))
        raise
    
    return {
        "status": "success",
        "document": metadata
    }


# ─── 2. LIST / DELETE DOCUMENTS ────────────────────────────────────────────────

@app.get("/documents")
async def get_documents_route(
    current_user=Depends(get_current_user)
):
    print("=" * 50)
    print("EMAIL:", current_user.email)
    print("USER ID:", current_user.id)
    print("=" * 50)

    docs = get_documents(current_user.id)

    print("Returned docs:", len(docs))

    return {
        "documents": docs
    }


@app.delete("/documents/{pdf_id}")
async def delete_document(
    pdf_id: str,
    current_user=Depends(get_current_user)
):
    from core.ingest import delete_pdf_store

    delete_pdf_store(current_user.id,pdf_id)

    delete_document_db(
        pdf_id,
        current_user.id
    )

    return {
        "status": "deleted",
        "pdf_id": pdf_id
    }


@app.get("/documents/{pdf_id}/thumbnail")
async def get_document_thumbnail(
    pdf_id: str,
    current_user = Depends(get_current_user)
):
    from core.ingest import get_or_create_thumbnail

    print("PDF ID =", pdf_id)

    thumb_path = get_or_create_thumbnail(current_user.id,pdf_id)

    print("Thumbnail =", thumb_path)

    if not thumb_path:
        raise HTTPException(404, "No thumbnail available")

    return FileResponse(
        thumb_path,
        media_type="image/png",
    )


@app.get("/documents/{pdf_id}/file")
async def get_document_file(
    pdf_id: str,
    current_user=Depends(get_current_user)
):
    from core.ingest import load_metadata

    meta = load_metadata(current_user.id, pdf_id)

    if not meta:
        raise HTTPException(404, "Document not found")

    storage_path = meta.get("storage_path")

    if not storage_path:
        raise HTTPException(404, "Storage path missing")

    try:
        data = (
            supabase.storage
            .from_("pdfs")
            .download(storage_path)
        )

    except Exception as e:
        raise HTTPException(
            404,
            f"PDF not found in storage: {e}"
        )

    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'inline; filename="{meta["pdf_name"]}"'
        }
    )


# ─── 3. CHAT (RAG with memory) ─────────────────────────────────────────────────
@app.post("/chat")
async def chat(
    req: ChatRequest,
    current_user=Depends(get_current_user)
):
    from core.rag_engine import generate_answer
    from core.ingest import (
    load_pdf_store,
    append_history,
    recover_document_from_storage,
    )
    """
    Ask a question about a specific PDF.
    `session_id` lets the same conversation remember earlier turns —
    generate a new UUID on the frontend each time the user starts a fresh chat.
    """
    try:
        from core.db import get_document
        doc = get_document(req.pdf_id, current_user.id)
        if not doc:
            raise HTTPException(404, "Document not found")
        index, chunks, meta = load_pdf_store(current_user.id,req.pdf_id)
    except FileNotFoundError:
        if not recover_document_from_storage(current_user.id, req.pdf_id):
            raise HTTPException(404, f"No document found with id: {req.pdf_id}")
        index, chunks, meta = load_pdf_store(current_user.id,req.pdf_id)

    memory = get_or_create_memory(req.session_id)

    try:
        answer, results = generate_answer(req.query, chunks, index, memory)
    except Exception as e:
        raise HTTPException(500, f"Answer generation failed: {e}")

    sources = [
        {
            "page_num": r.get("page_num"),
            "page_range": r.get("page_range"),
            "text": r["text"][:300],
        }
        for r in results
    ]

    # Persist this exchange so it's still here the next time this document
    # is opened — even in a new tab, or after a server restart.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    append_history(
        current_user.id,
        req.pdf_id,
        "chat",
        {
            "role": "user",
            "content": req.query,
            "at": timestamp
        }
    )

    append_history(
        current_user.id,
        req.pdf_id,
        "chat",
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "at": timestamp
        }
    )

    return ChatResponse(answer=answer, sources=sources)


@app.get("/chat/history/{pdf_id}")
async def get_chat_history(
    pdf_id: str,
    current_user=Depends(get_current_user)
):
    from core.ingest import load_history
    doc = get_document(pdf_id, current_user.id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "messages": load_history(current_user.id,pdf_id,"chat")
    }


@app.post("/chat/clear")
async def clear_chat(req: ClearMemoryRequest,current_user=Depends(get_current_user)):
    from core.ingest import clear_history
    """
    Clear conversation memory for a session AND its saved history.
    The frontend sends the document's pdf_id as session_id, so this wipes
    both the short-term LLM memory and the persisted chat log for that doc.
    """
    doc = get_document(req.session_id, current_user.id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if req.session_id in chat_sessions:
        chat_sessions[req.session_id].clear()
    clear_history(current_user.id,req.session_id,"chat")
    return {"status": "cleared"}


# ─── 4. SUMMARIZE ───────────────────────────────────────────────────────────────
@app.post("/summarize")
async def summarize(
    req: SummarizeRequest,
    current_user=Depends(get_current_user)
):
    from core.ingest import (
    load_pdf_store,
    append_history,
    recover_document_from_storage,
    )
    """
    Summarize either an already-ingested PDF (pass pdf_id)
    or raw pasted text (pass text).
    """
    from features.summarizer import summarize_plain_text

    from core.db import get_document

    if req.pdf_id:
        # Check that this PDF belongs to the logged-in user
        doc = get_document(req.pdf_id, current_user.id)

        if not doc:
            raise HTTPException(404, "Document not found")

        try:
            _, chunks, meta = load_pdf_store(current_user.id,req.pdf_id)
        except FileNotFoundError:
            if not recover_document_from_storage(current_user.id, req.pdf_id):
                raise HTTPException(
                    404,
                    f"No document found with id: {req.pdf_id}"
                )
            _, chunks, meta = load_pdf_store(current_user.id,req.pdf_id)

        full_text = " ".join(c["text"] for c in chunks)

    elif req.text and req.text.strip():
        full_text = req.text

    else:
        raise HTTPException(
            400,
            "Provide either pdf_id or text to summarize"
        )

    try:
        summary = summarize_plain_text(full_text)
    except Exception as e:
        raise HTTPException(
            500,
            f"Summarization failed: {e}"
        )

    if req.pdf_id:
        append_history(
            current_user.id,
            req.pdf_id,
            "summarize",
            {
                "summary": summary,
                "at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        )

    return {
        "pdf_id": req.pdf_id,
        "summary": summary,
    }


@app.get("/summarize/history/{pdf_id}")
async def get_summarize_history(
    pdf_id: str,
    current_user=Depends(get_current_user),
):
    from core.ingest import load_history
    from core.db import get_document

    doc = get_document(pdf_id, current_user.id)

    if not doc:
        raise HTTPException(404, "Document not found")

    return {
        "history": load_history(current_user.id,pdf_id,"summarize")
    }


# ─── 5. REDACT ──────────────────────────────────────────────────────────────────
@app.post("/redact")
async def redact(

    req: RedactRequest,
    current_user=Depends(get_current_user)
):
    
    from features.redactor import redact_pdf
    from core.ingest import append_history
    from core.db import get_document

    # Check ownership
    doc = get_document(req.pdf_id, current_user.id)

    if not doc:
        raise HTTPException(404, "Document not found")

    original_path = Path(doc["source_path"]) if doc.get("source_path") else None

    temp_redact_path = None

    if not original_path or not original_path.exists():
        storage_path = doc.get("storage_path")
        if not storage_path:
            raise HTTPException(404, "Storage path missing")

        print(f"[Redact] source_path missing, downloading from Supabase Storage: {storage_path}")
        try:
            pdf_data = supabase.storage.from_("pdfs").download(storage_path)
        except Exception as e:
            raise HTTPException(404, f"PDF not found in storage: {e}")

        pdf_name = doc.get("pdf_name", f"{req.pdf_id}.pdf")
        temp_redact_path = OUTPUT_DIR / f"temp_redact_{uuid.uuid4().hex[:8]}_{pdf_name}"
        with open(temp_redact_path, "wb") as f:
            f.write(pdf_data)
        original_path = temp_redact_path
        print(f"[Redact] Saved temporary PDF: {original_path}")

    output_filename = f"redacted_{req.pdf_id}_{uuid.uuid4().hex[:8]}.pdf"
    output_path = OUTPUT_DIR / output_filename

    try:
        redact_pdf(
            input_path=str(original_path),
            output_path=str(output_path),
            redact_types=req.redact_types,
            custom_terms=req.custom_terms,
        )
    except Exception as e:
        raise HTTPException(500, f"Redaction failed: {e}")
    finally:
        if temp_redact_path and temp_redact_path.exists():
            os.remove(temp_redact_path)
            print(f"[Redact] Cleaned up temporary PDF")

    download_url = f"/redact/download/{output_filename}"

    append_history(
        current_user.id,
        req.pdf_id,
        "redact",
        {
            "redact_types": req.redact_types,
            "custom_terms": req.custom_terms,
            "download_url": download_url,
            "at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )

    return {
        "status": "success",
        "download_url": download_url,
    }

@app.get("/redact/history/{pdf_id}")
async def get_redact_history(
    pdf_id: str,
    current_user=Depends(get_current_user)
):
    from core.ingest import load_history
    from core.db import get_document

    doc = get_document(pdf_id, current_user.id)

    if not doc:
        raise HTTPException(404, "Document not found")

    return {
        "history": load_history(current_user.id,pdf_id,"redact")
    }


@app.get("/redact/download/{filename}")
async def download_redacted(filename: str):
    """Download endpoint for the redacted PDF produced above."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(file_path, filename=filename, media_type="application/pdf")


# ─── 6. VOICE TRANSCRIPTION ─────────────────────────────────────────────────────

@app.post("/voice/transcribe")
async def transcribe_voice(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """
    Transcribe an audio file recorded in the browser (e.g. webm/wav from
    the mic icon in the frontend) into text.
    """

    from features.voice_handler import transcribe

    original_suffix = Path(file.filename or "").suffix or ".webm"
    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{original_suffix}"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = transcribe(str(temp_path), language="en")
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
    finally:
        if temp_path.exists():
            os.remove(temp_path)

    return {"text": text}
# ─── DIAGNOSTIC: Test exact real chunks for segfault ────────────────────────────

@app.get("/test-segfault")
def test_segfault():
    import os
    import multiprocessing
    import torch
    import joblib

    try:
        import loky
        loky_version = loky.__version__
    except ImportError:
        loky_version = "not installed"

    print("=" * 60)
    print("SEGFAULT TEST: /test-segfault endpoint entered")
    print("PID:", os.getpid())
    print("Parent PID:", os.getppid())
    print("CPU count:", multiprocessing.cpu_count())
    print("Torch threads:", torch.get_num_threads())
    print("Torch interop threads:", torch.get_num_interop_threads())
    print("TOKENIZERS_PARALLELISM:", os.environ.get("TOKENIZERS_PARALLELISM"))
    print("joblib version:", joblib.__version__)
    print("loky version:", loky_version)
    print("=" * 60)

    from core.embedder import get_model, get_last_chunks

    model = get_model()
    print("TEST: model loaded, type =", type(model))

    last_chunks = get_last_chunks()
    if not last_chunks:
        return {"error": "No last chunks available. Upload a PDF first."}

    # Extract raw chunk texts (without "passage: " prefix)
    raw0 = last_chunks[0]["text"]
    raw1 = last_chunks[1]["text"] if len(last_chunks) > 1 else ""

    chunk0 = "passage: " + raw0
    chunk1 = "passage: " + raw1

    print(f"chunk0 len={len(chunk0)}, raw0 len={len(raw0)}")
    print(f"chunk1 len={len(chunk1)}, raw1 len={len(raw1)}")
    print(f"chunk0[:500] = {repr(chunk0[:500])}")
    print(f"chunk1[:500] = {repr(chunk1[:500])}")

    results = {}

    # Test 1: Exact chunk 0 only, batch_size=1
    print("TEST 1: before encode (chunk0 only, batch_size=1)")
    try:
        result1 = model.encode(
            [chunk0],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 1: after encode, shape =", result1.shape)
        results["test1_chunk0_batch1"] = list(result1.shape)
    except Exception as e:
        print("TEST 1: EXCEPTION =", repr(e))
        results["test1_chunk0_batch1"] = f"ERROR: {e}"
        return results

    # Test 2: Exact chunk 1 only, batch_size=1
    if chunk1.strip():
        print("TEST 2: before encode (chunk1 only, batch_size=1)")
        try:
            result2 = model.encode(
                [chunk1],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 2: after encode, shape =", result2.shape)
            results["test2_chunk1_batch1"] = list(result2.shape)
        except Exception as e:
            print("TEST 2: EXCEPTION =", repr(e))
            results["test2_chunk1_batch1"] = f"ERROR: {e}"
            return results

    # Test 3: Both chunks, batch_size=1
    print("TEST 3: before encode (both chunks, batch_size=1)")
    try:
        result3 = model.encode(
            [chunk0, chunk1],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 3: after encode, shape =", result3.shape)
        results["test3_both_batch1"] = list(result3.shape)
    except Exception as e:
        print("TEST 3: EXCEPTION =", repr(e))
        results["test3_both_batch1"] = f"ERROR: {e}"
        return results

    # Test 4: Both chunks, batch_size=8
    print("TEST 4: before encode (both chunks, batch_size=8)")
    try:
        result4 = model.encode(
            [chunk0, chunk1],
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 4: after encode, shape =", result4.shape)
        results["test4_both_batch8"] = list(result4.shape)
    except Exception as e:
        print("TEST 4: EXCEPTION =", repr(e))
        results["test4_both_batch8"] = f"ERROR: {e}"
        return results

    # Test 5: Progressively shorter versions of chunk0 (the longer one)
    for length in [500, 1000, 1500, 2000, 2500, 3000, 3500]:
        if length < len(chunk0):
            test_chunk = chunk0[:length]
            print(f"TEST 5-{length}: before encode (chunk0[:{length}], batch_size=1)")
            try:
                result5 = model.encode(
                    [test_chunk],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                print(f"TEST 5-{length}: after encode, shape =", result5.shape)
                results[f"test5_chunk0_{length}"] = list(result5.shape)
            except Exception as e:
                print(f"TEST 5-{length}: EXCEPTION =", repr(e))
                results[f"test5_chunk0_{length}"] = f"ERROR: {e}"
                return results

    return results


# ─── DIAGNOSTIC: Test exact chunks via query params (survives crashes) ────────────

@app.get("/test-chunks")
def test_chunks(chunk0: str = "", chunk1: str = ""):
    """Test exact chunk texts passed as query params.
    Usage: curl -G "http://127.0.0.1:8000/test-chunks" --data-urlencode "chunk0=..." --data-urlencode "chunk1=..."
    """
    import os
    import multiprocessing
    import torch
    import joblib

    try:
        import loky
        loky_version = loky.__version__
    except ImportError:
        loky_version = "not installed"

    print("=" * 60)
    print("CHUNK TEST: /test-chunks endpoint entered")
    print("PID:", os.getpid())
    print("Parent PID:", os.getppid())
    print("CPU count:", multiprocessing.cpu_count())
    print("Torch threads:", torch.get_num_threads())
    print("Torch interop threads:", torch.get_num_interop_threads())
    print("TOKENIZERS_PARALLELISM:", os.environ.get("TOKENIZERS_PARALLELISM"))
    print("joblib version:", joblib.__version__)
    print("loky version:", loky_version)
    print("=" * 60)

    if not chunk0:
        return {"error": "chunk0 query param required"}

    from core.embedder import get_model

    model = get_model()
    print("TEST: model loaded, type =", type(model))

    # Add passage prefix if not already present
    if not chunk0.startswith("passage: "):
        chunk0 = "passage: " + chunk0
    if chunk1 and not chunk1.startswith("passage: "):
        chunk1 = "passage: " + chunk1

    print(f"chunk0 len={len(chunk0)}")
    print(f"chunk1 len={len(chunk1) if chunk1 else 0}")
    print(f"chunk0[:500] = {repr(chunk0[:500])}")
    if chunk1:
        print(f"chunk1[:500] = {repr(chunk1[:500])}")

    results = {}

    # Test 1: chunk0 only, batch_size=1
    print("TEST 1: before encode (chunk0 only, batch_size=1)")
    try:
        result1 = model.encode(
            [chunk0],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 1: after encode, shape =", result1.shape)
        results["test1_chunk0_batch1"] = list(result1.shape)
    except Exception as e:
        print("TEST 1: EXCEPTION =", repr(e))
        results["test1_chunk0_batch1"] = f"ERROR: {e}"
        return results

    # Test 2: chunk1 only, batch_size=1
    if chunk1 and chunk1.strip():
        print("TEST 2: before encode (chunk1 only, batch_size=1)")
        try:
            result2 = model.encode(
                [chunk1],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 2: after encode, shape =", result2.shape)
            results["test2_chunk1_batch1"] = list(result2.shape)
        except Exception as e:
            print("TEST 2: EXCEPTION =", repr(e))
            results["test2_chunk1_batch1"] = f"ERROR: {e}"
            return results

    # Test 3: both chunks, batch_size=1
    if chunk1 and chunk1.strip():
        print("TEST 3: before encode (both chunks, batch_size=1)")
        try:
            result3 = model.encode(
                [chunk0, chunk1],
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 3: after encode, shape =", result3.shape)
            results["test3_both_batch1"] = list(result3.shape)
        except Exception as e:
            print("TEST 3: EXCEPTION =", repr(e))
            results["test3_both_batch1"] = f"ERROR: {e}"
            return results

    # Test 4: both chunks, batch_size=8
    if chunk1 and chunk1.strip():
        print("TEST 4: before encode (both chunks, batch_size=8)")
        try:
            result4 = model.encode(
                [chunk0, chunk1],
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 4: after encode, shape =", result4.shape)
            results["test4_both_batch8"] = list(result4.shape)
        except Exception as e:
            print("TEST 4: EXCEPTION =", repr(e))
            results["test4_both_batch8"] = f"ERROR: {e}"
            return results

    # Test 5: progressively shorter versions of chunk0
    for length in [500, 1000, 1500, 2000, 2500, 3000, 3500]:
        if length < len(chunk0):
            test_chunk = chunk0[:length]
            print(f"TEST 5-{length}: before encode (chunk0[:{length}], batch_size=1)")
            try:
                result5 = model.encode(
                    [test_chunk],
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                print(f"TEST 5-{length}: after encode, shape =", result5.shape)
                results[f"test5_chunk0_{length}"] = list(result5.shape)
            except Exception as e:
                print(f"TEST 5-{length}: EXCEPTION =", repr(e))
                results[f"test5_chunk0_{length}"] = f"ERROR: {e}"
                return results

    return results


# ─── DIAGNOSTIC: Test embedding in FastAPI process ────────────────────────────────

@app.get("/test-embedding")
def test_embedding(text1: str = "", text2: str = "", use_last_chunks: bool = False):
    import os
    import multiprocessing
    import torch

    print("=" * 50)
    print("TEST: /test-embedding endpoint entered")
    print("PID:", os.getpid())
    print("Parent PID:", os.getppid())
    print("CPU count:", multiprocessing.cpu_count())
    print("Torch threads:", torch.get_num_threads())
    print("Torch interop threads:", torch.get_num_interop_threads())
    print("TOKENIZERS_PARALLELISM:", os.environ.get("TOKENIZERS_PARALLELISM"))
    print("=" * 50)

    from core.embedder import get_model, get_last_chunks

    model = get_model()
    print("TEST: model loaded, type =", type(model))

    # Test 1: Simple encode
    print("TEST 1: before encode (simple)")
    result = model.encode(
        ["hello world"],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    print("TEST 1: after encode, shape =", result.shape)

    # Test 2: With batch
    print("TEST 2: before encode (batch)")
    result2 = model.encode(
        ["hello world", "another test sentence"],
        batch_size=2,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    print("TEST 2: after encode, shape =", result2.shape)

    # Test 3: With normalize_embeddings
    print("TEST 3: before encode (normalize)")
    result3 = model.encode(
        ["hello world", "another test sentence"],
        batch_size=2,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    print("TEST 3: after encode, shape =", result3.shape)

    # Test 4: Use last chunks from actual upload
    if use_last_chunks:
        last_chunks = get_last_chunks()
        if last_chunks:
            batch = ["passage: " + c["text"] for c in last_chunks]
            print(f"TEST 4: using last chunks from upload, count={len(batch)}")
            for i, b in enumerate(batch):
                print(f"  chunk {i}: len={len(b)}, repr[:500]={repr(b[:500])}")
            
            # Test 4a: Full batch
            print("TEST 4a: before encode (full batch from upload)")
            result4a = model.encode(
                batch,
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 4a: after encode, shape =", result4a.shape)

            # Test 4b: First chunk only
            print("TEST 4b: before encode (first chunk only)")
            result4b = model.encode(
                [batch[0]],
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 4b: after encode, shape =", result4b.shape)

            # Test 4c: Second chunk only
            if len(batch) > 1:
                print("TEST 4c: before encode (second chunk only)")
                result4c = model.encode(
                    [batch[1]],
                    batch_size=8,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                print("TEST 4c: after encode, shape =", result4c.shape)

            # Test 4d: Progressively shorter versions of first chunk
            if len(batch[0]) > 1000:
                for length in [1000, 2000, 3000, 4000]:
                    if length < len(batch[0]):
                        text = batch[0].replace("passage: ", "")[:length]
                        print(f"TEST 4d-{length}: before encode (first chunk[:{length}])")
                        result4d = model.encode(
                            ["passage: " + text],
                            batch_size=8,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True
                        )
                        print(f"TEST 4d-{length}: after encode, shape =", result4d.shape)
        else:
            print("TEST 4: no last chunks available")

    # Test 5: With passage prefix from query params
    if text1 or text2:
        t1 = "passage: " + text1 if text1 else "passage: hello world"
        t2 = "passage: " + text2 if text2 else "passage: another test"
        batch = [t1] if not text2 else [t1, t2]
        
        print(f"TEST 5: passage batch from query, len1={len(batch[0])}, len2={len(batch[1]) if len(batch)>1 else 'N/A'}")
        print(f"  repr1[:500] = {repr(batch[0][:500])}")
        if len(batch) > 1:
            print(f"  repr2[:500] = {repr(batch[1][:500])}")
        
        # Test 5a: Full batch
        print("TEST 5a: before encode (passage batch)")
        result5a = model.encode(
            batch,
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 5a: after encode, shape =", result5a.shape)

        # Test 5b: First chunk only
        print("TEST 5b: before encode (first chunk only)")
        result5b = model.encode(
            [batch[0]],
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        print("TEST 5b: after encode, shape =", result5b.shape)

        # Test 5c: Second chunk only
        if len(batch) > 1:
            print("TEST 5c: before encode (second chunk only)")
            result5c = model.encode(
                [batch[1]],
                batch_size=8,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            print("TEST 5c: after encode, shape =", result5c.shape)

        # Test 5d: Progressively shorter versions of first chunk
        if len(batch[0]) > 1000:
            for length in [1000, 2000, 3000, 4000]:
                if length < len(batch[0]):
                    print(f"TEST 5d-{length}: before encode (first chunk[:{length}])")
                    result5d = model.encode(
                        ["passage: " + batch[0].replace("passage: ", "")[:length]],
                        batch_size=8,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True
                    )
                    print(f"TEST 5d-{length}: after encode, shape =", result5d.shape)

    return {
        "test1_shape": list(result.shape),
        "test2_shape": list(result2.shape),
        "test3_shape": list(result3.shape),
    }

# ─── HEALTH CHECK ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Document Intelligence API is running"}


