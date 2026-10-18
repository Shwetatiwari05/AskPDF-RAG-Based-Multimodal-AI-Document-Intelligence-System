# 🚀 AskPDF

### RAG-Based Multimodal AI Document Intelligence System

> An AI-powered document intelligence platform that lets you chat with your PDFs using Retrieval-Augmented Generation (RAG), OCR, semantic search, voice queries, AI summarization, and intelligent redaction.

**Live Demo:** https://ai-document-intelligence-mu.vercel.app

---

## 📖 Overview

AskPDF turns static PDF documents into interactive knowledge sources. It understands both **text-based** and **scanned documents**, and is built as a full-stack product with authentication, per-user document isolation, and cloud storage.

With AskPDF you can:

- 💬 Chat with a PDF using natural language (RAG)
- 🎤 Ask questions with your voice
- 📝 Generate AI-powered summaries
- 🔒 Automatically redact sensitive information
- 🔍 Search documents semantically, not just by keywords
- 📄 Process scanned PDFs with OCR

---

## ✨ Features

### 📂 Document Upload & Ingestion
- Upload text-based or scanned PDFs
- Automatic scanned-vs-text detection
- Automatic chunking, embedding, and per-document FAISS vector index
- Thumbnail generation
- Cloud storage via Supabase Storage
- Per-user isolation — every user only ever sees their own documents

### 💬 AI Document Chat (RAG)
Ask questions in natural language, e.g.:
- "What is this document about?"
- "Explain Chapter 3."
- "What are the eligibility criteria?"
- "Who signed this agreement?"

Powered by semantic search + a cross-encoder reranker + a Groq-hosted LLM, with conversation memory and persisted chat history.

### 🎤 Voice-Based Question Answering
Ask by voice instead of typing:

```
Browser mic → /voice/transcribe → faster-whisper → RAG → Groq LLM → answer
```

### 🧠 AI Summarization
Generate concise summaries of an entire document (or pasted text) using a map-reduce summarization pipeline over the Groq LLM.

### 🔒 Intelligent PDF Redaction
Detect and redact sensitive information, then download a redacted PDF:

- Email addresses
- Phone numbers
- Money / amounts
- Dates
- Aadhaar numbers
- PAN numbers
- Pincodes
- Credit card numbers
- IFSC codes
- Bank account numbers
- Addresses (street / city / state / country)
- Custom user-supplied keywords

Two independent redaction paths:
- **Text PDFs** → PyMuPDF `search_for()` for exact coordinates.
- **Scanned PDFs** → EasyOCR word boxes, then regex / fuzzy / address matching; the redaction is **burned into the page pixels** so it cannot be removed by deleting an overlay.

### 📄 OCR Support
Scanned PDFs are detected automatically and routed through OCR before chunking:

```
PDF pages → render to images → OCR → extracted text → chunking → embeddings → vector store
```

---

## 🏗️ Architecture

```text
                         +----------------------+
                         |      User Upload     |
                         +----------+-----------+
                                    |
                                    v
                      Scanned vs Text Detection
                         (avg chars/page < 50)
                    /                          \
                   /                            \
       Scanned PDF                              Text PDF
   Azure AI Vision Read                     PyMuPDF extraction
                   \                            /
                    \                          /
                     v                        v
                    Text Cleaning  ·  Sentence-aware Chunking (spaCy)
                                    |
                                    v
                   SentenceTransformer Embeddings (MiniLM-L3-v2, CPU)
                                    |
                                    v
                     FAISS Vector Index (per document, cosine/IP)
                                    |
                                    v
                     Semantic Search (top-k = 50)
                                    |
                                    v
                Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)
                                    |
                                    v
                      Groq LLM (openai/gpt-oss-120b)
                                    |
                                    v
                        Context-aware AI response
```

> The redaction engine has its own pipeline (PyMuPDF for text pages, EasyOCR for scanned pages) and does not use the RAG pipeline.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 19, Vite, Tailwind CSS v4, React Router, Supabase JS, react-markdown, lucide-react — deployed on **Vercel** |
| **Backend** | FastAPI, Uvicorn, Pydantic — Python 3.13 |
| **Auth / Database** | Supabase Auth, Supabase PostgreSQL (`documents`, `document_history`), Supabase Storage (`pdfs` bucket) |
| **Retrieval / Embeddings** | Sentence Transformers (`paraphrase-MiniLM-L3-v2`), FAISS (`IndexFlatIP`), Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **LLM** | Groq (`openai/gpt-oss-120b`) |
| **OCR** | Azure AI Vision Read (ingestion), EasyOCR (scanned-page redaction) |
| **PDF Processing** | PyMuPDF (fitz), pdfplumber |
| **NLP** | spaCy (sentence segmentation) |
| **Voice** | faster-whisper, PyAV, SciPy |
| **ML Runtime** | PyTorch (CPU), NumPy |

---

## 📁 Project Structure

```text
.
├── main.py                    # FastAPI app — all HTTP endpoints
├── requirements.txt
├── runtime.txt
├── core/
│   ├── auth.py                # Supabase JWT auth dependency
│   ├── supabase_client.py     # Supabase client (service role)
│   ├── db.py                  # documents + history CRUD
│   ├── ingest.py              # ingest pipeline, per-user stores, thumbnails
│   ├── ocr_extractor.py       # scanned detection + Azure AI Vision OCR
│   ├── pdf_extractor.py       # pdfplumber extraction + boilerplate removal
│   ├── preprocess.py          # clean_text + sentence-aware chunking
│   ├── embedder.py            # SentenceTransformer embeddings
│   ├── vector_store.py        # FAISS build/search + cross-encoder rerank
│   └── rag_engine.py          # retrieval + Groq generation + memory
├── features/
│   ├── redactor.py            # text + scanned PDF redaction engine
│   ├── summarizer.py          # map-reduce summarization (Groq)
│   └── voice_handler.py       # faster-whisper transcription (+ local mic)
├── ocr/
│   └── easyocr_ocr.py         # EasyOCR word boxes for redaction
├── frontend/askpdf-frontend/  # React + Vite SPA
└── tests/
    └── test_redactor_scanned.py
```

At runtime, ingested artifacts live under:

```
pdf_stores/<user_id>/<pdf_id>/
    vector_store.index
    chunks.pkl
    metadata.json
    thumbnail.png
    chat_history.json
    summarize_history.json
    redact_history.json
uploads/<uuid>/<file>.pdf
outputs/redacted_*.pdf
```

---

## 🔌 API Endpoints

All endpoints except `/` and `/redact/download/{filename}` require a valid Supabase session (`Authorization: Bearer <token>`).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload + ingest a PDF (returns `pdf_id`) |
| `GET` | `/documents` | List the current user's documents |
| `DELETE` | `/documents/{pdf_id}` | Delete a document's store, history, and DB record |
| `GET` | `/documents/{pdf_id}/thumbnail` | Page-1 preview PNG |
| `GET` | `/documents/{pdf_id}/file` | Original PDF bytes |
| `POST` | `/chat` | Ask a question about a PDF (with memory) |
| `GET` | `/chat/history/{pdf_id}` | Persisted chat history |
| `POST` | `/chat/clear` | Clear memory + history for a session |
| `POST` | `/summarize` | Summarize a PDF (`pdf_id`) or raw text (`text`) |
| `GET` | `/summarize/history/{pdf_id}` | Saved summaries |
| `POST` | `/redact` | Redact a PDF; returns a download URL + warnings |
| `GET` | `/redact/history/{pdf_id}` | Saved redaction runs |
| `GET` | `/redact/download/{filename}` | Download a redacted PDF |
| `POST` | `/voice/transcribe` | Transcribe browser audio to text |
| `GET` | `/` | Health check |

---

## ⚙️ How the RAG Pipeline Works

1. **Detect** — `is_scanned_pdf` samples the first pages; fewer than ~50 chars/page ⇒ scanned.
2. **Extract** — scanned pages go through Azure AI Vision Read; text pages through PyMuPDF (line-ordered words).
3. **Clean** — text documents use the full cleaner; scanned text gets light whitespace cleanup only.
4. **Chunk** — spaCy sentence segmentation with ~500-word chunks and ~80-word overlap, dropping low-quality (TOC/heading-heavy) chunks.
5. **Embed** — each chunk is embedded with `paraphrase-MiniLM-L3-v2` (normalized, CPU).
6. **Store** — embeddings go into a per-document FAISS `IndexFlatIP` (cosine similarity via inner product).
7. **Retrieve** — on a question, top-50 chunks are fetched, keyword-boosted, then re-ranked with a cross-encoder to the top ~10 (12 for whole-document questions).
8. **Generate** — the reranked context, document info, and recent conversation are passed to `openai/gpt-oss-120b` on Groq.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.13
- Node.js 18+
- A Supabase project (database + storage bucket `pdfs`)
- A Groq API key
- Azure AI Vision key + endpoint (for scanned-PDF OCR)

### 1. Backend

```bash
# from the project root
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# create .env in the project root (see Environment Variables)
uvicorn main:app --reload
```

The API runs at `http://127.0.0.1:8000`.

You can also ingest from the CLI without the API:

```bash
python -m core.ingest sample.pdf            # ingest a PDF
python -m core.ingest sample.pdf --force-ocr
python -m core.ingest --list                # list ingested PDFs
python -m core.ingest --delete <pdf_id>     # delete a store
```

### 2. Frontend

```bash
cd frontend/askpdf-frontend
npm install
npm run dev      # http://localhost:5173
npm run build    # production build
```

---

## 🔑 Environment Variables

### Backend (`.env` in project root)

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Groq LLM access |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Server-side Supabase access (DB + storage) |
| `AZURE_KEY` | Azure AI Vision key |
| `AZURE_ENDPOINT` | Azure AI Vision endpoint |
| `WHISPER_MODEL` | faster-whisper size (default `tiny`) |
| `AZURE_OCR_MAX_PAGES` | OCR page cap (default `50`) |

### Frontend (`frontend/askpdf-frontend/.env`)

| Variable | Purpose |
|----------|---------|
| `VITE_API_URL` | Backend base URL (default `http://127.0.0.1:8000`) |
| `VITE_SUPABASE_URL` | Supabase project URL |
| `VITE_SUPABASE_ANON_KEY` | Supabase anon/public key |

---

## 📊 Highlights

- ✅ Retrieval-Augmented Generation (RAG)
- ✅ Semantic search + cross-encoder re-ranking
- ✅ OCR for scanned PDFs (Azure AI Vision)
- ✅ Pixel-burned redaction for scanned PDFs (EasyOCR)
- ✅ AI summarization
- ✅ Voice-based queries (faster-whisper)
- ✅ Conversation memory + persisted history
- ✅ Supabase Auth with per-user document isolation
- ✅ Cloud storage + production deployment

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 🚀 Future Improvements

- Multi-document querying
- Image understanding inside PDFs
- Table extraction
- Citation-aware responses
- Multi-language document support
- Role-based document access
- Document comparison
- Knowledge graph generation
- Fine-tuned domain-specific models

---

## 📸 Screenshots

**Home Page**
<img width="1145" height="785" alt="Screenshot 2026-07-23 at 11 03 33 PM" src="https://github.com/user-attachments/assets/b43c6abc-14c9-467f-bde2-eadb7061503e" />

**Dashboard**
<img width="1447" height="824" alt="Screenshot 2026-07-23 at 11 06 49 PM" src="https://github.com/user-attachments/assets/d5f45c0b-2946-46c2-9b9a-e85f0d1cdb37" />

**Q&A**
<img width="2940" height="1668" alt="image" src="https://github.com/user-attachments/assets/5c7b8e89-66b9-4bb9-8148-da1cd94cbc45" />

**AI Summary**
<img width="1470" height="841" alt="Screenshot 2026-07-24 at 6 24 20 PM" src="https://github.com/user-attachments/assets/b3d6edb8-6506-49a7-b272-6b3c1542ef5f" />
<img width="1470" height="830" alt="Screenshot 2026-07-24 at 6 25 36 PM" src="https://github.com/user-attachments/assets/94889485-1e99-40ff-a9db-3ebcf491b940" />

**AI Redaction**
<img width="1470" height="827" alt="Screenshot 2026-07-23 at 11 09 22 PM" src="https://github.com/user-attachments/assets/a412bfa0-c7fb-4749-9efa-90726441c393" />
<img width="1470" height="836" alt="Screenshot 2026-07-23 at 11 10 33 PM" src="https://github.com/user-attachments/assets/f6ac9af7-8003-4935-8b28-9f669fc5454b" />

**Voice Search**
<img width="1469" height="869" alt="Screenshot 2026-07-24 at 2 26 39 PM" src="https://github.com/user-attachments/assets/94fc76e5-9213-41d6-98c2-81063c0ed5d3" />
<img width="1465" height="837" alt="Screenshot 2026-07-24 at 2 27 50 PM" src="https://github.com/user-attachments/assets/b534d6d9-6b44-4976-bb47-3d2523989090" />

---

## 👩‍💻 Author

**Shweta Tiwari**

Built with ❤️ using Retrieval-Augmented Generation, NLP, OCR, Semantic Search, and Generative AI.
