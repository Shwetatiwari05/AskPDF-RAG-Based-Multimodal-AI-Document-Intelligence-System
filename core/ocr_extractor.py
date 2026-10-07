import fitz
import re
import os
import time
import io
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

load_dotenv()

# AZURE AI VISION CLIENT
_azure_client = None


def get_azure_client():
    global _azure_client
    if _azure_client is None:
        key = os.getenv("AZURE_KEY")
        endpoint = os.getenv("AZURE_ENDPOINT")
        if not key:
            raise ValueError("AZURE_KEY not found in .env file!")
        if not endpoint:
            raise ValueError("AZURE_ENDPOINT not found in .env file!")
        _azure_client = ImageAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
    return _azure_client


# SCANNED PDF DETECTION


def is_scanned_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Detect if PDF is scanned (image-based) or text-based.
    Strategy: extract text from first few pages — if avg < 50 chars, it's scanned.
    """
    doc = fitz.open(pdf_path)
    pages_to_check = min(sample_pages, len(doc))
    total_chars = sum(
        len(doc[i].get_text("text").strip()) for i in range(pages_to_check)
    )
    doc.close()

    avg_chars = total_chars / pages_to_check if pages_to_check > 0 else 0

    if avg_chars < 50:
        print(f"Scanned PDF detected (avg {avg_chars:.0f} chars/page) → using Azure OCR")
        return True
    else:
        print(
            f"Text-based PDF detected (avg {avg_chars:.0f} chars/page) → using normal extraction"
        )
        return False


# AZURE AI VISION OCR


def _render_page_to_jpeg(page, dpi: int = 150, quality: int = 85) -> bytes:
    """Render a PyMuPDF page to JPEG bytes at given DPI and quality."""
    pix = page.get_pixmap(dpi=dpi)
    img = fitz.Pixmap(pix, 0) if pix.alpha else pix
    return img.tobytes("jpg", jpg_quality=quality)


def _page_fits_limit(image_bytes: bytes, max_bytes: int = 3_500_000) -> bool:
    return len(image_bytes) <= max_bytes


def _analyze_with_retry(client, image_bytes: bytes, max_retries: int = 3) -> str:
    """Call Azure Read with exponential backoff on 429. Returns page text or raises."""
    for attempt in range(max_retries + 1):
        try:
            result = client.analyze(
                image_data=image_bytes, visual_features=[VisualFeatures.READ]
            )
            if result.read is None or result.read.blocks is None:
                return ""
            lines = []
            for block in result.read.blocks:
                for line in block.lines:
                    lines.append(line.text)
            return "\n".join(lines)
        except HttpResponseError as e:
            if e.status_code == 429:
                if attempt == max_retries:
                    raise
                retry_after = e.response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else (
                    2 ** attempt
                )
                print(f"  Rate limited (429), waiting {wait}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait)
                continue
            if e.status_code == 400 and "not supported" in str(e).lower():
                raise RuntimeError(
                    "Azure Read feature not supported in this region"
                ) from e
            raise
    raise RuntimeError("Max retries exceeded")


def extract_text_ocr(pdf_path: str, **kwargs) -> str:
    """
    Extract text from scanned PDF using Azure AI Vision Read API.
    Renders each page to JPEG and sends separately (free tier limit).
    """
    client = get_azure_client()

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    max_pages = int(os.getenv("AZURE_OCR_MAX_PAGES", "50"))
    if total_pages > max_pages:
        doc.close()
        raise ValueError(
            f"PDF has {total_pages} pages, exceeds AZURE_OCR_MAX_PAGES={max_pages}. "
            f"Split the PDF or increase the limit."
        )

    pages_text = []

    for page_num in range(total_pages):
        page = doc[page_num]
        print(f"OCR page {page_num + 1}/{total_pages} (Azure)")

        # Render to JPEG, downscale if over size limit
        dpi = 150
        quality = 85
        image_bytes = _render_page_to_jpeg(page, dpi, quality)

        while not _page_fits_limit(image_bytes) and dpi > 100:
            dpi -= 10
            if dpi < 100:
                dpi = 100
            image_bytes = _render_page_to_jpeg(page, dpi, quality)

        if not _page_fits_limit(image_bytes):
            doc.close()
            raise ValueError(
                f"Page {page_num + 1} JPEG exceeds 3.5 MB even at minimum DPI. "
                f"Cannot process with Azure free tier."
            )

        page_text = _analyze_with_retry(client, image_bytes)
        if page_text.strip():
            pages_text.append(page_text)

        # Rate limit: ~3.2 seconds between calls (20 calls/min)
        if page_num < total_pages - 1:
            time.sleep(3.2)

    doc.close()

    full_text = "\n\n".join(pages_text)
    full_text = clean_ocr_text(full_text)

    print(f" Azure OCR complete — {len(pages_text)} page(s) processed")
    return full_text


# OCR TEXT CLEANING


def clean_ocr_text(text: str) -> str:
    """Light cleanup of Azure OCR output — it's already very clean."""
    # Remove markdown image tags (sometimes present)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# OCR TEXT WITH BOUNDING BOXES (for redaction on scanned PDFs)


def extract_blocks_with_boxes(pdf_path: str):
    """
    Run OCR on a scanned PDF and return each page's text blocks with bbox.
    Uses pytesseract instead of Azure (which doesn't support bounding boxes in this SDK).
    """
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(pdf_path)
    pages_blocks = []

    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))

        scale_x = page.rect.width / img.width
        scale_y = page.rect.height / img.height

        ocr_data = pytesseract.image_to_data(
            img, output_type=pytesseract.Output.DICT
        )

        n = len(ocr_data["text"])
        blocks = []
        for i in range(n):
            w = ocr_data["text"][i].strip()
            if not w:
                continue
            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            ww = ocr_data["width"][i]
            hh = ocr_data["height"][i]
            blocks.append(
                {
                    "text": w,
                    "bbox": (
                        x * scale_x,
                        y * scale_y,
                        (x + ww) * scale_x,
                        (y + hh) * scale_y,
                    ),
                    "page_width": img.width,
                    "page_height": img.height,
                }
            )
        pages_blocks.append(blocks)

    doc.close()
    return pages_blocks


# NORMAL PDF EXTRACTION


def extract_text_normal(pdf_path: str) -> str:
    """Fast text extraction for normal (non-scanned) PDFs using fitz."""
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        words = page.get_text("words")
        if not words:
            continue
        lines_dict = {}
        for w in words:
            y_key = round(w[1], 1)
            if y_key not in lines_dict:
                lines_dict[y_key] = []
            lines_dict[y_key].append((w[0], w[4]))

        page_lines = []
        for y in sorted(lines_dict.keys()):
            line_words = sorted(lines_dict[y], key=lambda x: x[0])
            page_lines.append(" ".join(ww[1] for ww in line_words))
        pages_text.append("\n".join(page_lines))

    doc.close()
    raw = "\n\n".join(pages_text)
    raw = re.sub(r"^[\s|]+$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"^\|\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


# SMART ENTRY POINT


def extract_text_smart(pdf_path: str, force_ocr: bool = False, **kwargs) -> str:
    """
    MAIN FUNCTION — auto-detects scanned vs text-based PDF.
    force_ocr=True → always use Azure (for mixed text+image PDFs)
    """
    print(f"Analyzing PDF: {pdf_path}")

    if force_ocr or is_scanned_pdf(pdf_path):
        return extract_text_ocr(pdf_path)
    else:
        return extract_text_normal(pdf_path)








# import fitz
# import re
# import os
# import base64
# from pathlib import Path
# from dotenv import load_dotenv
# from mistralai.client.sdk import Mistral

# load_dotenv()

# # MISTRAL CLIENT 
# _client = None

# def get_client():
#     global _client
#     if _client is None:
#         api_key = os.getenv("MISTRAL_API_KEY")
#         if not api_key:
#             raise ValueError("MISTRAL_API_KEY not found in .env file!")
#         _client = Mistral(api_key=api_key)
#     return _client


# # SCANNED PDF DETECTION 

# def is_scanned_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
#     """
#     Detect if PDF is scanned (image-based) or text-based.
#     Strategy: extract text from first few pages — if avg < 50 chars, it's scanned.
#     """
#     doc = fitz.open(pdf_path)
#     pages_to_check = min(sample_pages, len(doc))
#     total_chars = sum(len(doc[i].get_text("text").strip()) for i in range(pages_to_check))
#     doc.close()

#     avg_chars = total_chars / pages_to_check if pages_to_check > 0 else 0

#     if avg_chars < 50:
#         print(f"Scanned PDF detected (avg {avg_chars:.0f} chars/page) → using Mistral OCR")
#         return True
#     else:
#         print(f"Text-based PDF detected (avg {avg_chars:.0f} chars/page) → using normal extraction")
#         return False


# # MISTRAL OCR 

# def extract_text_ocr(pdf_path: str, **kwargs) -> str:
#     """
#     Extract text from scanned PDF using Mistral OCR API.
#     Sends PDF as base64 — no need to convert to images.
#     """
#     client = get_client()

#     print(f"Sending to Mistral OCR...")

#     # Read PDF as base64
#     with open(pdf_path, "rb") as f:
#         pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

#     # Call Mistral OCR API
#     response = client.ocr.process(
#         model="mistral-ocr-latest",
#         document={
#             "type": "document_url",
#             "document_url": f"data:application/pdf;base64,{pdf_data}",
#         }
#     )

#     # Extract text from all pages
#     pages_text = []
#     for page in response.pages:
#         if page.markdown:
#             pages_text.append(page.markdown)

#     full_text = "\n\n".join(pages_text)
#     full_text = clean_ocr_text(full_text)

#     print(f" Mistral OCR complete — {len(response.pages)} page(s) processed")
#     return full_text

# # OCR TEXT CLEANING 

# def clean_ocr_text(text: str) -> str:
#     """Light cleanup of Mistral OCR output — it's already very clean."""
#     # Remove markdown image tags (Mistral sometimes adds these)
#     text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
#     # Normalize whitespace
#     text = re.sub(r'[ \t]+', ' ', text)
#     text = re.sub(r'\n{3,}', '\n\n', text)
#     return text.strip()

# # NORMAL PDF EXTRACTION 

# def extract_text_normal(pdf_path: str) -> str:
#     """Fast text extraction for normal (non-scanned) PDFs using fitz."""
#     doc = fitz.open(pdf_path)
#     pages_text = []

#     for page in doc:
#         words = page.get_text("words")
#         if not words:
#             continue
#         lines_dict = {}
#         for w in words:
#             y_key = round(w[1], 1)
#             if y_key not in lines_dict:
#                 lines_dict[y_key] = []
#             lines_dict[y_key].append((w[0], w[4]))

#         page_lines = []
#         for y in sorted(lines_dict.keys()):
#             line_words = sorted(lines_dict[y], key=lambda x: x[0])
#             page_lines.append(" ".join(ww[1] for ww in line_words))
#         pages_text.append("\n".join(page_lines))

#     doc.close()
#     raw = "\n\n".join(pages_text)
#     raw = re.sub(r'^[\s|]+$', '', raw, flags=re.MULTILINE)
#     raw = re.sub(r'^\|\s*', '', raw, flags=re.MULTILINE)
#     raw = re.sub(r'[ \t]+', ' ', raw)
#     raw = re.sub(r'\n{3,}', '\n\n', raw)
#     return raw.strip()

# # SMART ENTRY POINT 

# def extract_text_smart(pdf_path: str, force_ocr: bool = False, **kwargs) -> str:
#     """
#     MAIN FUNCTION — auto-detects scanned vs text-based PDF.
#     force_ocr=True → always use Mistral (for mixed text+image PDFs)
#     """
#     print(f"Analyzing PDF: {pdf_path}")

#     if force_ocr or is_scanned_pdf(pdf_path):
#         return extract_text_ocr(pdf_path)
#     else:
#         return extract_text_normal(pdf_path)


