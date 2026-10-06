import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pdfplumber

from dotenv import load_dotenv
from groq import Groq

# LOAD ENV
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# PDF EXTRACTION
from core.ocr_extractor import extract_text_smart

def extract_text_from_pdf(pdf_path):
    return extract_text_smart(pdf_path)

# CLEAN TEXT
def clean_text(text):
    return " ".join(text.split())

# CHUNK TEXT
def chunk_text(
    text,
    chunk_size=12000,
    overlap=500
):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(
            text[start:end]
        )
        start += chunk_size - overlap
    return chunks

# GROQ CALL

def call_groq(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            temperature=0.2,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert Document Intelligence AI.

You analyze PDFs of any type including:
- Research Papers
- Books
- Resumes
- Invoices
- Contracts
- Reports
- Meeting Notes
- Educational Material
- Financial Documents

Generate accurate summaries.

Do not hallucinate.

Only use information present in the document.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

def summarize_chunk(chunk):
    prompt = f"""Summarize the following text concisely in a short paragraph.
Keep it simple and clear.
Do not add document type, headings, or extra sections.
Just write a clean summary paragraph.

Text:
{chunk}

Summary:"""
    return call_groq(prompt)

# FULL DOCUMENT SUMMARY
def summarize_document(text):
    # text = text[:20000]
    chunks = chunk_text(text)
    print(
        f"\nTotal summary chunks: {len(chunks)}"
    )
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(
            f"Summarizing chunk {i+1}/{len(chunks)}..."
        )
        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)
    combined = "\n\n".join(
    chunk_summaries)

    if len(combined) > 30000:
        combined = combined[:30000]
    print("\nGenerating final summary...")

    final_prompt = f"""Summarize the following text clearly and concisely.
Include important points, facts (if any), names, numbers if present.
Write as many sentences as needed based on the content.
Do not add headings or document type.
Do not invent or add information not present in the text.
Just write a natural, clean summary paragraph.

Text:
{combined}

Summary:"""
    return call_groq(final_prompt)
def summarize_plain_text(text):
    text = clean_text(text)
    if not text.strip():
        return "No text provided."
    return summarize_document(text)

# MAIN
if __name__ == "__main__":

    print("\nChoose Input Type:")
    print("1. PDF")
    print("2. Text")

    choice = input("\nEnter choice: ").strip()

    # PDF MODE
    if choice == "1":
        pdf_path = input(
            "\nEnter PDF path: "
        ).strip().strip("'").strip('"')
        if not os.path.exists(pdf_path):
            print(
                f"\nFile not found: {pdf_path}"
            )
            exit()

        print("\nExtracting PDF...")

        raw_text = extract_text_from_pdf(
            pdf_path
        )

        cleaned_text = clean_text(
            raw_text
        )

        if not cleaned_text.strip():

            print(
                "\nNo readable text found in PDF."
            )

            exit()

        print(
            f"\nCharacters extracted: {len(cleaned_text)}"
        )

        print(
            "\nGenerating summary..."
        )

        final_summary = summarize_document(
            cleaned_text
        )

    # TEXT MODE
    elif choice == "2":

        print(
            "\nPaste your text below."
        )

        print(
            "Press Enter twice when finished.\n"
        )

        lines = []

        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        user_text = "\n".join(lines)
        final_summary = summarize_plain_text(
            user_text
        )

    else:

        print("Invalid choice.")
        exit()

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80 + "\n")

    print(final_summary)

    with open(
        "summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(final_summary)

    print(
        "\nSummary saved to summary.txt"
    )