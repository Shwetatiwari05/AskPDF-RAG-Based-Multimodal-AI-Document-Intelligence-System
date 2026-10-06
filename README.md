# 🚀 AskPDF

### RAG-Based Multimodal AI Document Intelligence System

> **An AI-powered document intelligence platform that enables users to interact with PDFs using Retrieval-Augmented Generation (RAG), OCR, Semantic Search, Voice Queries, AI Summarization, and Intelligent Redaction.**

---

## 🌐 Live Demo

**https://ai-document-intelligence-mu.vercel.app**

---

# 📖 Overview

AskPDF is an end-to-end AI-powered Document Intelligence Platform designed to transform static PDF documents into interactive knowledge sources.

Unlike traditional PDF readers, AskPDF understands both **text-based** and **scanned documents**, allowing users to:

* 💬 Chat with PDFs using natural language
* 🎤 Ask questions using voice
* 📝 Generate AI-powered summaries
* 🔒 Automatically redact sensitive information
* 🔍 Perform semantic document search
* 📄 Process scanned PDFs using OCR
* ⚡ Retrieve context-aware answers using Retrieval-Augmented Generation (RAG)

The system combines modern NLP, Vector Search, OCR, and Large Language Models to provide accurate, context-aware document understanding.

---

# ✨ Key Features

## 📂 Intelligent PDF Upload

* Upload text-based PDFs
* Upload scanned PDFs
* Automatic document ingestion
* Automatic vector index generation
* Thumbnail generation
* Cloud storage using Supabase Storage

---

## 💬 AI Document Chat

Ask questions in natural language.

Examples:

* What is this document about?
* Explain Chapter 3.
* What are the eligibility criteria?
* Who signed this agreement?

Powered using:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Context Retrieval
* Conversation Memory

---

## 🎤 Voice-Based Question Answering

Users can interact using voice instead of typing.

Pipeline:

Speech → Whisper → RAG → Groq LLM → Response

---

## 🧠 AI Summarization

Generate concise summaries of entire documents using LLMs.

Supports:

* Research Papers
* Reports
* Agreements
* Books
* Technical Documentation

---

## 🔒 Intelligent PDF Redaction

Automatically detect and redact sensitive information.

Supports:

* Email IDs
* Phone Numbers
* Aadhaar Numbers
* PAN Numbers
* Credit Card Numbers
* Dates
* Names
* Custom User Keywords

Produces a downloadable redacted PDF.

---

## 📄 OCR Support

Scanned PDFs are automatically detected.

If text extraction fails:

PDF Images

↓

Mistral OCR

↓

Extracted Text

↓

Chunking

↓

Embeddings

↓

Vector Store

---

## 🔍 Semantic Search

Instead of keyword matching, AskPDF retrieves information using semantic similarity.

Even if the exact wording doesn't exist, the system understands user intent.

Example:

Query:

> "How can I reset my password?"

Document contains:

> "Users may change their credentials through Account Settings."

The system still retrieves the correct section.

---

## 🧩 Retrieval-Augmented Generation (RAG)

The platform follows a production-ready RAG pipeline.

PDF

↓

Text Extraction

↓

Cleaning

↓

Chunking

↓

Sentence Embeddings

↓

FAISS Vector Database

↓

Similarity Search

↓

Cross Encoder Re-ranking

↓

Groq LLM

↓

Final Response

---

# 🏗️ System Architecture

```text
                +----------------------+
                |      User Upload     |
                +----------+-----------+
                           |
                           v
                 PDF Text Extraction
            (pdfplumber / Mistral OCR)
                           |
                           v
                  Text Cleaning
                           |
                           v
                Intelligent Chunking
                           |
                           v
            SentenceTransformer Embeddings
                           |
                           v
                 FAISS Vector Database
                           |
                           v
               Semantic Similarity Search
                           |
                           v
               Cross Encoder Re-ranking
                           |
                           v
                  Groq Large Language Model
                           |
                           v
              Context-Aware AI Response
```

---

# 🛠️ Tech Stack

## Frontend

* React.js
* Tailwind CSS
* Vercel

---

## Backend

* FastAPI
* Railway

---

## Database

* Supabase PostgreSQL
* Supabase Storage

---

## AI / Machine Learning

* Sentence Transformers
* Cross Encoder Re-ranking
* FAISS
* spaCy
* Groq LLM
* Mistral OCR
* Faster Whisper

---

## PDF Processing

* PyMuPDF
* pdfplumber

---

## Deployment

* Frontend → Vercel
* Backend → Railway
* Database → Supabase

---

# ⚙️ AI Pipeline

### 1. Document Upload

* Upload PDF
* Store in Supabase
* Save metadata

---

### 2. Document Processing

* Detect scanned or text PDF
* OCR (if required)
* Clean text
* Chunk document

---

### 3. Embedding Generation

Each chunk is converted into dense vector embeddings using Sentence Transformers.

---

### 4. Vector Storage

Embeddings are stored in a FAISS Vector Index for efficient similarity search.

---

### 5. Retrieval

When a user asks a question:

* Query Embedding
* Similarity Search
* Top-K Retrieval
* Cross Encoder Re-ranking

---

### 6. Response Generation

The retrieved context is passed to Groq LLM to generate a context-aware answer.

---

# 📊 Highlights

✅ Retrieval-Augmented Generation (RAG)

✅ Semantic Search

✅ Cross Encoder Re-ranking

✅ OCR for Scanned PDFs

✅ AI Summarization

✅ AI Redaction

✅ Voice-Based Queries

✅ Conversation Memory

✅ Cloud Storage

✅ Production Deployment

---

# 🚀 Future Improvements

* Multi-document querying
* Image understanding inside PDFs
* Table extraction
* Citation-aware responses
* Multi-language document support
* Role-based document access
* Document comparison
* Knowledge Graph generation
* Fine-tuned domain-specific models

---

# 📸 Screenshots

> Add screenshots of:
>
> * Home Page
<img width="1145" height="785" alt="Screenshot 2026-07-23 at 11 03 33 PM" src="https://github.com/user-attachments/assets/b43c6abc-14c9-467f-bde2-eadb7061503e" />

> * Dashboard
<img width="1447" height="824" alt="Screenshot 2026-07-23 at 11 06 49 PM" src="https://github.com/user-attachments/assets/d5f45c0b-2946-46c2-9b9a-e85f0d1cdb37" />

> * Q&A
<img width="2940" height="1668" alt="image" src="https://github.com/user-attachments/assets/5c7b8e89-66b9-4bb9-8148-da1cd94cbc45" />

> * AI Summary

<img width="1470" height="841" alt="Screenshot 2026-07-24 at 6 24 20 PM" src="https://github.com/user-attachments/assets/b3d6edb8-6506-49a7-b272-6b3c1542ef5f" />
<img width="1470" height="830" alt="Screenshot 2026-07-24 at 6 25 36 PM" src="https://github.com/user-attachments/assets/94889485-1e99-40ff-a9db-3ebcf491b940" />


> * AI Redaction
<img width="1470" height="827" alt="Screenshot 2026-07-23 at 11 09 22 PM" src="https://github.com/user-attachments/assets/a412bfa0-c7fb-4749-9efa-90726441c393" />
<img width="1470" height="836" alt="Screenshot 2026-07-23 at 11 10 33 PM" src="https://github.com/user-attachments/assets/f6ac9af7-8003-4935-8b28-9f669fc5454b" />

> * Voice Search
<img width="1469" height="869" alt="Screenshot 2026-07-24 at 2 26 39 PM" src="https://github.com/user-attachments/assets/94fc76e5-9213-41d6-98c2-81063c0ed5d3" />
<img width="1465" height="837" alt="Screenshot 2026-07-24 at 2 27 50 PM" src="https://github.com/user-attachments/assets/b534d6d9-6b44-4976-bb47-3d2523989090" />

---

# 👩‍💻 Author

**Shweta Tiwari**

Built with ❤️ using Retrieval-Augmented Generation, NLP, OCR, Semantic Search, and Generative AI.
