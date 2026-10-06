"""
summarizer.py — Intelligent Document Summarizer
Uses facebook/bart-large-cnn (BERT-based) for extractive + abstractive summarization.
Detects document type and produces structured, comprehensive summaries.

Compatible with transformers v4.52+ (pipeline("summarization") removed in newer versions,
so we use BartForConditionalGeneration directly — more control, same results).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import re
import textwrap


# ─── MODEL SETUP ──────────────────────────────────────────────────────────────

# facebook/bart-large-cnn is trained on CNN/DailyMail news articles.
# It uses a BERT-style bidirectional encoder + autoregressive decoder.
# Best for long documents that need complete, coherent summaries.
#
# NOTE: We use BartForConditionalGeneration directly instead of pipeline()
# because transformers v4.52+ removed the "summarization" pipeline task.

MODEL_NAME = "facebook/bart-large-cnn"

print("Loading BART summarization model (first run will download ~1.6GB)...")
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
bart_model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
bart_model.eval()  # Inference mode — disables dropout

# Use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model = bart_model.to(DEVICE)

print(f"Model loaded! Running on: {DEVICE}")


def _run_bart(text: str, min_length: int = 40, max_length: int = 200) -> str:
    """
    Core BART inference — tokenize → generate → decode.
    Replaces pipeline("summarization") for full version compatibility.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        summary_ids = bart_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,           # Beam search for better quality
            min_length=min_length,
            max_length=max_length,
            length_penalty=2.0,    # Encourages longer, complete summaries
            early_stopping=True,
            no_repeat_ngram_size=3  # Avoid repetitive phrases
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Wrapper so rest of code stays unchanged
def summarizer_model(text_list, min_length=40, max_length=200, **kwargs):
    """Drop-in replacement for pipeline() output format."""
    results = []
    for text in text_list:
        summary = _run_bart(text, min_length=min_length, max_length=max_length)
        results.append({"summary_text": summary})
    return results


print("Model ready!")


# ─── DOCUMENT TYPE DETECTION ──────────────────────────────────────────────────

DOCUMENT_TYPES = {
    "resume": [
        "work experience", "career objective", "technical skills",
        "cgpa", "gpa", "linkedin", "github", "internship",
        "certifications", "bachelor of", "master of", "b.tech", "m.tech",
    ],
    "invoice": [
        "invoice", "amount due", "subtotal", "gst", "billing address",
        "invoice number", "unit price", "receipt", "payment due",
    ],
    "research_paper": [
        # Research papers have ALL of these together — strict multi-signal check
        "abstract", "methodology", "doi", "journal", "citation",
        "literature review", "hypothesis", "et al.", "ieee", "arxiv",
    ],
    "legal": [
        "whereas", "hereinafter", "indemnification", "jurisdiction",
        "pursuant", "notwithstanding", "herein", "arbitration",
    ],
    "book": [
        "chapter", "preface", "table of contents", "publisher",
        "o'reilly", "packt", "isbn", "all rights reserved",
        "edition", "printed in", "typographical",
    ],
    "report": [
        "executive summary", "findings", "recommendations",
        "key performance", "quarter", "fiscal year", "stakeholder",
    ],
    "medical": [
        "patient", "diagnosis", "prescription", "dosage",
        "clinical", "icd", "contraindication", "prognosis",
    ],
}

def detect_document_type(text: str) -> str:
    """
    Detect document type using keyword frequency + word count heuristics.
    """
    text_lower = text.lower()
    word_count = len(text.split())
    scores = {}

    for doc_type, keywords in DOCUMENT_TYPES.items():
        # Count how many keywords appear — use substring match
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[doc_type] = score

    # Long documents cannot be resumes or invoices
    if word_count > 2000:
        scores.pop("resume", None)
        scores.pop("invoice", None)

    if not scores:
        return "book" if word_count > 5000 else "general"

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Research paper needs strong signal (3+ strict keywords like doi, arxiv, et al.)
    if best_type == "research_paper" and best_score < 3:
        best_type = "book" if word_count > 5000 else "general"

    # Weak signal overall — fallback
    if best_score < 2:
        return "book" if word_count > 5000 else "general"

    return best_type


# ─── TEXT CHUNKING FOR LONG DOCS ──────────────────────────────────────────────

MAX_TOKENS = 1024  # BART max input length

def split_into_token_chunks(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Split text into chunks that fit within BART's token limit."""
    # Encode and split by tokens to be accurate
    tokens = tokenizer.encode(text, truncation=False)
    
    chunks = []
    for i in range(0, len(tokens), max_tokens - 50):  # -50 for safety buffer
        chunk_tokens = tokens[i : i + max_tokens - 50]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    return chunks


def summarize_text_chunks(chunks: list, min_len: int, max_len: int) -> str:
    """Run BART summarization on each chunk and combine results."""
    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 30:  # skip tiny chunks
            continue
        result = summarizer_model([chunk], min_length=min_len, max_length=max_len)
        summaries.append(result[0]["summary_text"])
    return " ".join(summaries)


# ─── SECTION EXTRACTORS (for structured docs) ─────────────────────────────────

def extract_resume_sections(text: str) -> dict:
    """Extract named sections from a resume."""
    section_headers = {
        "contact":        r"^(contact|personal\s+info|about\s+me)\s*$",
        "objective":      r"^(objective|career\s+objective|career\s+goal|profile\s+summary|professional\s+summary|summary)\s*$",
        "education":      r"^(education|academic|qualification|academics)\s*$",
        "experience":     r"^(experience|work\s+experience|work\s+history|employment|internship)\s*$",
        "skills":         r"^(skills|technical\s+skills|core\s+competencies|technologies|skill\s+set)\s*$",
        "projects":       r"^(projects|personal\s+projects|academic\s+projects|key\s+projects)\s*$",
        "certifications": r"^(certifications?|courses?|training|licenses?|achievements?\s+&\s+certifications?)\s*$",
        "achievements":   r"^(achievements?|awards?|honors?|accomplishments?|extra.?curricular)\s*$",
    }

    sections = {}
    lines = text.split("\n")
    current_section = "header"
    current_lines = []

    for line in lines:
        stripped = line.strip()
        line_lower = stripped.lower()
        matched_section = None

        # A section header: matches pattern AND is short AND not a bullet/dash line
        if len(stripped) < 70 and not stripped.startswith(("-", "–", "•", "*")):
            for sec_name, pattern in section_headers.items():
                if re.search(pattern, line_lower):
                    matched_section = sec_name
                    break

        if matched_section:
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = matched_section
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def extract_invoice_fields(text: str) -> dict:
    """Extract key fields from invoices using regex."""
    fields = {}

    patterns_inv = {
        "Invoice Number":   r"invoice\s*#?\s*[:\-]?\s*([A-Z0-9\-/]+)",
        "Date":             r"(?:invoice\s+date|date)[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
        "Due Date":         r"due\s+date[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
        "Total Amount":     r"(?:total|grand\s+total|amount\s+due)[:\s]+([$₹€£]?\s?[\d,]+\.?\d*)",
        "Tax/GST":          r"(?:tax|gst|vat)[:\s]+([$₹€£]?\s?[\d,]+\.?\d*)",
        "Vendor/From":      r"(?:from|vendor|billed\s+by|company)[:\s]+([A-Za-z0-9\s,\.]+)",
        "Client/To":        r"(?:to|bill\s+to|client|customer)[:\s]+([A-Za-z0-9\s,\.]+)",
    }

    text_lower = text.lower()
    for field, pattern in patterns_inv.items():
        match = re.search(pattern, text_lower)
        if match:
            fields[field] = match.group(1).strip()

    return fields


# ─── STRUCTURED SUMMARY BUILDERS ──────────────────────────────────────────────

def split_projects(proj_text: str) -> list:
    """
    Split projects section into individual project blocks.
    
    Strategy: A project TITLE line is identified by these signals:
      1. Not a bullet/dash line
      2. Not a tech-stack line (starts with known tech words)
      3. Not a continuation sentence (starts with lowercase or action verbs)
      4. Contains parentheses OR dash (—) which is common in project titles
         OR is an ALL-CAPS-WORDS title pattern
    """
    lines = proj_text.split("\n")
    projects = []
    current = []

    CONTENT_STARTERS = (
        "–", "-", "•", "*", "built", "developed", "applied", "performed",
        "identified", "achieving", "scoring", "transformer", "deployed",
        "computed", "coordinated", "completed", "practiced", "led",
        "python,", "java,", "using", "the ", "a ", "an ", "cosine",
        "this ", "we ", "i ", "it ", "in ", "for ", "with ",
    )

    def is_project_title(line: str) -> bool:
        s = line.strip()
        if not s or len(s) < 5:
            return False
        low = s.lower()
        # Must not start with content/bullet starters
        if any(low.startswith(x) for x in CONTENT_STARTERS):
            return False
        # Must have capital letters
        if not re.search(r'[A-Z]', s):
            return False
        # Strong signals: has parentheses (like "WasteAI (CNN-Based)")
        # OR has em-dash (like "WasteAI — Smart Waste")
        has_paren = "(" in s and ")" in s
        has_emdash = "—" in s or " - " in s
        # OR: title-like capitalization (most words capitalized)
        words = s.split()
        cap_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
        is_title_case = cap_ratio >= 0.5 and len(words) >= 3
        return (has_paren or has_emdash or is_title_case) and len(s) < 150

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if is_project_title(stripped) and current:
            block = "\n".join(current).strip()
            if len(block.split()) > 8:
                projects.append(block)
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        block = "\n".join(current).strip()
        if len(block.split()) > 8:
            projects.append(block)

    return projects if projects else [proj_text]


def build_resume_summary(text: str) -> str:
    sections = extract_resume_sections(text)
    output_parts = []

    output_parts.append("=" * 60)
    output_parts.append("RESUME SUMMARY")
    output_parts.append("=" * 60)

    # ── Candidate Header ──────────────────────────────────────
    if "header" in sections and sections["header"].strip():
        output_parts.append("\n👤  CANDIDATE")
        output_parts.append("-" * 40)
        # Show first 3 meaningful lines (name, contact, etc.)
        header_lines = [l.strip() for l in sections["header"].split("\n") if l.strip()][:4]
        output_parts.append("\n".join(header_lines))

    # ── Career Objective ──────────────────────────────────────
    if "objective" in sections and sections["objective"].strip():
        output_parts.append("\n CAREER OBJECTIVE")
        output_parts.append("-" * 40)
        obj_text = sections["objective"].strip()
        # Always show directly — objective is usually 2-3 lines, no need to summarize
        output_parts.append(obj_text)

    # ── Technical Skills ──────────────────────────────────────
    if "skills" in sections and sections["skills"].strip():
        output_parts.append("\n TECHNICAL SKILLS")
        output_parts.append("-" * 40)
        output_parts.append(sections["skills"].strip())

    # ── Projects (each project summarized separately) ─────────
    if "projects" in sections and sections["projects"].strip():
        output_parts.append("\n PROJECTS")
        output_parts.append("-" * 40)
        project_blocks = split_projects(sections["projects"])
        for i, proj_block in enumerate(project_blocks, 1):
            proj_lines = proj_block.split("\n")
            title_line = proj_lines[0].strip() if proj_lines else f"Project {i}"
            body = "\n".join(proj_lines[1:]).strip() if len(proj_lines) > 1 else proj_block

            output_parts.append(f"\n  [{i}] {title_line}")

            if not body.strip():
                pass  # title only, nothing to show
            elif len(body.split()) < 15:
                # Short body — show directly, skip BART
                for bl in body.split("\n"):
                    bl = bl.strip()
                    if bl and not bl.startswith(title_line[:20]):  # skip if repeat of title
                        output_parts.append(f"      {bl}")
            else:
                # Long enough — BART summarize (use body only, not title again)
                chunks = split_into_token_chunks(body)
                summary = summarize_text_chunks(chunks, min_len=30, max_len=120)
                if summary:
                    for s_line in summary.split(". "):
                        s_line = s_line.strip().rstrip(".")
                        if s_line and s_line.lower() not in title_line.lower():
                            output_parts.append(f"      • {s_line}.")
                else:
                    for bl in body.split("\n"):
                        if bl.strip():
                            output_parts.append(f"      {bl.strip()}")

    # ── Work Experience ───────────────────────────────────────
    if "experience" in sections and sections["experience"].strip():
        output_parts.append("\n WORK EXPERIENCE")
        output_parts.append("-" * 40)
        exp_text = sections["experience"].strip()
        exp_lines = [l for l in exp_text.split("\n") if l.strip()]

        # Detect job header lines:
        # Pattern: line contains a year (2020, 2024 etc.) AND is not a bullet
        # e.g. "The Marine Robotics Team Co-Marketing Head July 2024 – Apr 2025"
        import re as _re
        YEAR_PAT = _re.compile(r'\b(19|20)\d{2}\b')
        BULLET = ("-", "–", "•", "*")

        jobs = []
        current_job = []
        for line in exp_lines:
            s = line.strip()
            is_header = (
                YEAR_PAT.search(s)
                and not s.startswith(BULLET)
                and len(s) < 120
            )
            if is_header and current_job:
                jobs.append(current_job)
                current_job = [s]
            elif is_header:
                current_job = [s]
            else:
                current_job.append(s)
        if current_job:
            jobs.append(current_job)

        for job in jobs:
            if not job:
                continue
            header = job[0]
            bullets = [l for l in job[1:] if l.strip()]

            output_parts.append(f"\n {header}")
            for b in bullets:
                b = b.strip()
                if b and not b.startswith(header[:15]):
                    # Normalize bullet symbol
                    if not b.startswith(("–", "-", "•")):
                        b = "– " + b
                    output_parts.append(f"      {b}")

    # ── Education ─────────────────────────────────────────────
    if "education" in sections and sections["education"].strip():
        output_parts.append("\n EDUCATION")
        output_parts.append("-" * 40)
        for line in sections["education"].strip().split("\n"):
            if line.strip():
                output_parts.append(f"  {line.strip()}")

    # ── Certifications ────────────────────────────────────────
    if "certifications" in sections and sections["certifications"].strip():
        output_parts.append("\n CERTIFICATIONS")
        output_parts.append("-" * 40)
        cert_text = sections["certifications"].strip()
        # Join bullet-only lines with next line (PDF splits "•" from text)
        cert_lines = [l.strip() for l in cert_text.split("\n") if l.strip()]
        merged = []
        i = 0
        while i < len(cert_lines):
            if cert_lines[i] in ("•", "-", "–", "*") and i + 1 < len(cert_lines):
                merged.append("• " + cert_lines[i + 1])
                i += 2
            else:
                line = cert_lines[i]
                if not line.startswith(("•", "-", "–")):
                    line = "• " + line
                merged.append(line)
                i += 1
        for m in merged:
            output_parts.append(f"  {m}")

    # ── Achievements ──────────────────────────────────────────
    if "achievements" in sections and sections["achievements"].strip():
        output_parts.append("\n🏆  ACHIEVEMENTS")
        output_parts.append("-" * 40)
        for line in sections["achievements"].strip().split("\n"):
            if line.strip():
                output_parts.append(f"  {line.strip()}")

    output_parts.append("\n" + "=" * 60)
    return "\n".join(output_parts)


def build_invoice_summary(text: str) -> str:
    fields = extract_invoice_fields(text)
    output_parts = []

    output_parts.append("=" * 60)
    output_parts.append(" INVOICE SUMMARY")
    output_parts.append("=" * 60)

    if fields:
        output_parts.append("\n KEY INVOICE DETAILS")
        output_parts.append("-" * 40)
        for field, value in fields.items():
            output_parts.append(f"  {field:<20}: {value}")

    # Summarize full invoice text for line items / description
    output_parts.append("\n CONTENT SUMMARY")
    output_parts.append("-" * 40)
    chunks = split_into_token_chunks(text)
    summary = summarize_text_chunks(chunks, min_len=50, max_len=200)
    output_parts.append(summary)

    output_parts.append("\n" + "=" * 60)
    return '\n'.join(output_parts)


def build_research_summary(text: str) -> str:
    output_parts = []

    output_parts.append("=" * 60)
    output_parts.append(" RESEARCH PAPER SUMMARY")
    output_parts.append("=" * 60)

    word_count = len(text.split())
    output_parts.append(f"\n Document Info")
    output_parts.append("-" * 40)
    output_parts.append(f"  Word count: ~{word_count:,}")

    # ── Abstract: extract and show directly (no BART needed) ────
    abstract_match = re.search(
        r'(?:^|\n)abstract[:\s]*\n?(.*?)(?=\n\n|\n(?:introduction|keywords|1\.|i\.))',
        text, flags=re.IGNORECASE | re.DOTALL
    )
    if abstract_match:
        abstract_text = abstract_match.group(1).strip()
        if len(abstract_text.split()) > 10:
            output_parts.append("\n📋  ABSTRACT")
            output_parts.append("-" * 40)
            output_parts.append(abstract_text[:800])

    # ── One-paragraph overall summary (hierarchical) ────────────
    chunks = split_into_token_chunks(text)
    total_chunks = len(chunks)

    output_parts.append("\n OVERALL SUMMARY")
    output_parts.append("-" * 40)

    # Pass 1: mini-summary per chunk
    mini_summaries = []
    for chunk in chunks:
        if len(chunk.split()) < 40:
            continue
        result = summarizer_model([chunk], min_length=20, max_length=70)
        mini_summaries.append(result[0]["summary_text"].strip())

    # Pass 2: combine into one final summary
    if mini_summaries:
        combined = " ".join(mini_summaries)
        result = summarizer_model([combined[:3000]], min_length=80, max_length=250)
        output_parts.append(result[0]["summary_text"])

    # ── Section-wise breakdown (named sections if found) ────────
    # Try to find named sections like Introduction, Methodology, etc.
    SECTION_NAMES = [
        "introduction", "related work", "background",
        "methodology", "method", "approach", "model",
        "experiment", "results", "evaluation",
        "discussion", "conclusion", "future work"
    ]

    named_sections = {}
    text_lower = text.lower()
    for sec in SECTION_NAMES:
        pat = re.search(
            rf'(?:^|\n)(?:\d+\.?\s*)?{sec}[:\s]*\n(.*?)(?=\n(?:\d+\.?\s*)?(?:{"| ".join(SECTION_NAMES)})|$)',
            text, flags=re.IGNORECASE | re.DOTALL
        )
        if pat:
            sec_text = pat.group(1).strip()
            if len(sec_text.split()) > 30:
                named_sections[sec.title()] = sec_text

    if named_sections:
        output_parts.append("\n SECTION-WISE BREAKDOWN")
        output_parts.append("-" * 40)
        for sec_name, sec_text in named_sections.items():
            chunks_s = split_into_token_chunks(sec_text)
            summaries_s = []
            for chunk in chunks_s[:3]:
                if len(chunk.split()) < 30:
                    continue
                result = summarizer_model([chunk], min_length=20, max_length=100)
                summaries_s.append(result[0]["summary_text"].strip())
            if summaries_s:
                output_parts.append(f"\n  {sec_name}:")
                output_parts.append("  " + " ".join(summaries_s))

    output_parts.append("\n" + "=" * 60)
    return "\n".join(output_parts)


def build_legal_summary(text: str) -> str:
    output_parts = []

    output_parts.append("=" * 60)
    output_parts.append(" LEGAL DOCUMENT SUMMARY")
    output_parts.append("=" * 60)

    # Extract parties
    party_match = re.findall(
        r'(?:between|party|parties)[:\s]+([A-Z][A-Za-z\s,\.]+(?:Ltd|Inc|LLP|Corp|Company|LLC)?)',
        text[:2000]
    )
    if party_match:
        output_parts.append("\n👥  PARTIES INVOLVED")
        output_parts.append("-" * 40)
        for p in party_match[:4]:
            output_parts.append(f"  • {p.strip()}")

    output_parts.append("\n KEY CLAUSES SUMMARY")
    output_parts.append("-" * 40)
    chunks = split_into_token_chunks(text)
    for i, chunk in enumerate(chunks[:5], 1):
        if len(chunk.split()) < 50:
            continue
        result = summarizer_model([chunk], min_length=50, max_length=150)
        output_parts.append(f"\n[Clause Block {i}]")
        output_parts.append(result[0]["summary_text"])

    output_parts.append("\n" + "=" * 60)
    return '\n'.join(output_parts)


def hierarchical_summarize(chunks: list, doc_type: str) -> str:
    """
    Two-pass summarization for large documents (books, long reports):
    Pass 1 — Summarize every chunk into a short sentence (~60 words)
    Pass 2 — Group those mini-summaries into batches, summarize each batch
    Pass 3 — Combine batch summaries into one final flowing summary

    This way ALL 189 chunks are covered, not just the first 12.
    """
    is_book = doc_type in ("book", "book_chapter")
    total = len(chunks)

    print(f"  Pass 1: Summarizing {total} chunks...")

    # ── Pass 1: mini-summary per chunk ──────────────────────────
    mini_summaries = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunk.split()) < 30:
            continue
        if i % 20 == 0:
            print(f"  ... processed {i}/{total} chunks")
        result = summarizer_model([chunk], min_length=20, max_length=60)
        mini_summaries.append(result[0]["summary_text"].strip())

    print(f"  Pass 2: Grouping {len(mini_summaries)} mini-summaries into batches...")

    # ── Pass 2: summarize groups of mini-summaries ───────────────
    # Combine 15 mini-summaries into one batch text, then summarize
    BATCH_SIZE = 15
    batch_summaries = []
    for i in range(0, len(mini_summaries), BATCH_SIZE):
        batch_text = " ".join(mini_summaries[i:i + BATCH_SIZE])
        if len(batch_text.split()) < 20:
            continue
        result = summarizer_model([batch_text], min_length=40, max_length=150)
        batch_summaries.append(result[0]["summary_text"].strip())

    print(f"  Pass 3: Creating final summary from {len(batch_summaries)} batches...")

    # ── Pass 3: final summary from all batch summaries ───────────
    if len(batch_summaries) > 1:
        combined = " ".join(batch_summaries)
        result = summarizer_model([combined[:3000]], min_length=80, max_length=300)
        final_summary = result[0]["summary_text"].strip()
    elif batch_summaries:
        final_summary = batch_summaries[0]
    else:
        final_summary = "Could not generate summary — document may be too short or unreadable."

    return final_summary, batch_summaries


def build_general_summary(text: str, doc_type: str = "Document") -> str:
    output_parts = []

    emoji_map = {"book": "", "book_chapter": "", "report": "", "medical": "", "general": ""}
    emoji = emoji_map.get(doc_type,)
    label = doc_type.upper()

    output_parts.append("=" * 60)
    output_parts.append(f"{emoji}  {label} SUMMARY")
    output_parts.append("=" * 60)

    word_count = len(text.split())
    chunks = split_into_token_chunks(text)
    total_chunks = len(chunks)

    output_parts.append(f"\n Document Info")
    output_parts.append("-" * 40)
    output_parts.append(f"  Word count     : ~{word_count:,}")
    output_parts.append(f"  Chunks processed: {total_chunks}")

    is_book = doc_type in ("book", "book_chapter")

    if word_count > 3000:
        # ── Long doc: use hierarchical 3-pass summarization ──────
        output_parts.append(f"\n{'COMPLETE BOOK SUMMARY' if is_book else 'COMPLETE SUMMARY'}")
        output_parts.append("-" * 40)
        output_parts.append("  (Processing all sections — this may take a few minutes...)")
        print()

        final_summary, batch_summaries = hierarchical_summarize(chunks, doc_type)

        output_parts.append(f"\n{final_summary}")

        # Also show section-level breakdown
        if batch_summaries and len(batch_summaries) > 1:
            output_parts.append(f"\n\n{'─' * 40}")
            output_parts.append(f"SECTION BREAKDOWN")
            output_parts.append("─" * 40)
            for i, bs in enumerate(batch_summaries, 1):
                output_parts.append(f"\n  Part {i}:  {bs}")

    else:
        # ── Short doc: direct summarization ──────────────────────
        output_parts.append(f"\nSUMMARY")
        output_parts.append("-" * 40)
        summaries = []
        for chunk in chunks[:6]:
            if len(chunk.split()) < 30:
                continue
            result = summarizer_model([chunk], min_length=40, max_length=150)
            summaries.append(result[0]["summary_text"].strip())
        output_parts.append("\n" + " ".join(summaries))

    output_parts.append("\n" + "=" * 60)
    return "\n".join(output_parts)


# ─── MAIN SUMMARIZE FUNCTION ──────────────────────────────────────────────────

def summarize_document(text: str, doc_type: str = None) -> str:
    """
    Main entry point. Auto-detects document type and generates
    a comprehensive, structured summary.

    Args:
        text     : Full document text (cleaned)
        doc_type : Optional override. If None, auto-detected.
                   Options: 'resume', 'invoice', 'research_paper',
                            'legal', 'book_chapter', 'report', 'medical', 'general'

    Returns:
        Formatted summary string.
    """
    if not text or len(text.strip()) < 50:
        return "Document is too short or empty to summarize."

    # Auto-detect if not provided
    if not doc_type:
        doc_type = detect_document_type(text)
        print(f"Detected document type: {doc_type.upper()}")

    # Route to the right summarizer
    if doc_type == "resume":
        return build_resume_summary(text)
    elif doc_type == "invoice":
        return build_invoice_summary(text)
    elif doc_type == "research_paper":
        return build_research_summary(text)
    elif doc_type == "legal":
        return build_legal_summary(text)
    else:
        return build_general_summary(text, doc_type)  # book, report, medical, general


# ─── INTEGRATION WITH YOUR RAG PIPELINE ──────────────────────────────────────

def summarize_pdf(pdf_path: str, doc_type: str = None) -> str:
    """
    Full pipeline: PDF → Extract → Clean → Summarize.
    Plugs directly into your existing core modules.
    
    Usage:
        from summarizer import summarize_pdf
        summary = summarize_pdf("my_resume.pdf")
        print(summary)
    """
    from core.ocr_extractor import extract_text_smart
    cleaned = extract_text_smart(pdf_path)
    return summarize_document(cleaned, doc_type) 

    print(f"Extracting text from: {pdf_path}")

    doc_fitz = fitz.open(pdf_path)
    pages_text = []

    for page in doc_fitz:
        words = page.get_text("words")
        if not words:
            continue
        # Group words by y-coordinate into lines
        lines_dict = {}
        for w in words:
            y_key = round(w[1], 1)
            if y_key not in lines_dict:
                lines_dict[y_key] = []
            lines_dict[y_key].append((w[0], w[4]))

        page_lines = []
        for y in sorted(lines_dict.keys()):
            line_words = sorted(lines_dict[y], key=lambda x: x[0])
            line_text = " ".join(ww[1] for ww in line_words)
            page_lines.append(line_text)
        pages_text.append("\n".join(page_lines))

    doc_fitz.close()
    raw_text = "\n\n".join(pages_text)

    cleaned = re.sub(r'[ \t]+', ' ', raw_text)
    # Remove lines that are ONLY pipe characters and spaces (PDF column separators)
    cleaned = re.sub(r'^[\s|]+$', '', cleaned, flags=re.MULTILINE)
    # Remove lone | at start of line
    cleaned = re.sub(r'^\|\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'^ +', '', cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    print("Generating summary...")
    return summarize_document(cleaned, doc_type)


def summarize_chunks(chunks: list, doc_type: str = None) -> str:
    """
    Summarize from already-processed chunks (from your ingest pipeline).
    
    Usage:
        from summarizer import summarize_chunks
        summary = summarize_chunks(chunks)  # chunks from chunk_text()
    """
    full_text = " ".join(c["text"] for c in chunks)
    return summarize_document(full_text, doc_type)


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os

    print("\n" + "=" * 60)
    print(" DOCUMENT SUMMARIZER")
    print("=" * 60)

    # ── Mode 1: PDF path passed as argument ───────────────────
    # Usage: python -m features.summarizerrr.summarizerr my_resume.pdf
    # Usage: python -m features.summarizerrr.summarizerr report.pdf report
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        doc_type = sys.argv[2] if len(sys.argv) > 2 else None

        if not os.path.exists(pdf_path):
            print(f"File not found: {pdf_path}")
            sys.exit(1)

        print(f"\nFile     : {pdf_path}")
        print(f"Doc type : {doc_type or 'auto-detect'}\n")

        summary = summarize_pdf(pdf_path, doc_type)
        print(summary)

        out_path = pdf_path.replace(".pdf", "_summary.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"\nSummary saved to: {out_path}")

    # ── Mode 2: Interactive — ask user for input ───────────────
    else:
        print("\nHow do you want to provide the document?")
        print("  1 → PDF file path")
        print("  2 → Paste text directly")
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "1":
            pdf_path = input("Enter full PDF path: ").strip().strip('"').strip("'")
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                sys.exit(1)
            doc_type = input("Doc type (leave blank to auto-detect) [resume/invoice/research_paper/legal/report/medical]: ").strip() or None
            summary = summarize_pdf(pdf_path, doc_type)

            print("\n" + summary)

            save = input("\nSave summary to file? (y/n): ").strip().lower()
            if save == "y":
                out_path = pdf_path.replace(".pdf", "_summary.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"Saved to: {out_path}")

        elif choice == "2":
            print("\nPaste your text below.")
            print("When done, type  ---END---  on a new line and press Enter:\n")
            lines = []
            while True:
                line = input()
                if line.strip() == "---END---":
                    break
                lines.append(line)
            text = "\n".join(lines)

            if not text.strip():
                print("No text provided.")
                sys.exit(1)

            doc_type = input("\nDoc type (leave blank to auto-detect) [resume/invoice/research_paper/legal/report/medical]: ").strip() or None
            summary = summarize_document(text, doc_type)
            print("\n" + summary)

            save = input("\n💾 Save summary to file? (y/n): ").strip().lower()
            if save == "y":
                out_path = "summary_output.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"Saved to: {out_path}")

        else:
            print("Invalid choice.")