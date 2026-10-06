"""
redactor.py — PDF Redaction Engine

TWO PATHS:
  Text PDF    → PyMuPDF search_for() → exact coords → redact
  Scanned PDF → EasyOCR → word bboxes → fuzzy match → redact

WHAT GETS REDACTED:
  email, phone, money, date, aadhaar, pan, credit_card, ifsc,
  address (street + city + state + country + pincode)
  via regex + keyword dictionaries
"""

import os, re, warnings
from difflib import SequenceMatcher
import fitz
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── REGEX PATTERNS ────────────────────────────────────────────────────────────
PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\(91[\-\s]?\d{2}\)[\s\-]?\d{8}|\+91[\-\s]?\d{2}[\-\s]?\d{8}|0\d{2,4}[\-\s]?\d{6,8}|(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",
    "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?\s+\d{1,2}(?:\s*(?:st|nd|rd|th))?,?\s+\d{4}|\d{1,2}(?:\s*(?:st|nd|rd|th))?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?,?\s+\d{4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}",
    "aadhaar": r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
    "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    "pincode":r"\b\d{6}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ifsc":r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "money":r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",
    "account_number": r"\bBSB\s*#?\s*\d[\d\s\-]{4,8}\b|\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",
    # address regex — catches "123 Somewhere St", "Suite 5A-1204", "Flat 4B", "Plot No. 7"
    "address":r"(?i)\b(?:suite|apt|apartment|flat|floor|unit|plot|house|room|no\.?)\s+[\w\-]+|\b\d+[A-Za-z]?\s+[\w\s]{2,30}(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|blvd|boulevard|way|court|ct|place|pl|nagar|marg|colony|sector)\b",
}
ADDRESS_KEYWORDS = {
    # Structural address words
    'COLONY','MARG','ROAD','LANE','NAGAR','WEST','EAST','NORTH','SOUTH',
    'SECTOR','BLOCK','STREET','PLOT','FLAT','HOUSE','APARTMENT','APT',
    'BUILDING','BLDG','NEAR','OPP','LAYOUT','PHASE','CROSS','MAIN','PARK',
    'SOCIETY','COMPLEX','TOWER','VTC','CINEMA','EXTENSION',
    # Address labels
    'STATE','DISTRICT','SUBDISTRICT','TALUKA','TEHSIL','VILLAGE','PO',
    'SUB DIVISION','SUBDIVISION',
    # Indian cities / towns
    'MUMBAI','DELHI','THANE','PUNE','ANDHERI','BORIVALI','KURLA','NAGPUR',
    'NASHIK','AURANGABAD','KALYAN','DOMBIVLI','VASAI','VIRAR','VASHI',
    'CHEMBUR','DADAR','GOREGAON','MALAD','KANDIVALI','MIRA','BHAYANDER',
    'DAHISAR','MULUND','ULHASNAGAR','KANJURMARG','BHANDUP','WADALA',
    'AHMEDABAD','SURAT','RAJKOT','VADODARA','JAIPUR','JODHPUR',
    'UDAIPUR','KOTA','LUCKNOW','KANPUR','AGRA','VARANASI','PATNA','GAYA',
    'BHUBANESWAR','CUTTACK','PURI','RANCHI','JAMSHEDPUR','GUWAHATI',
    'SHILLONG','AGARTALA','IMPHAL','KOHIMA','AIZAWL','ITANAGAR','PANAJI',
    'BENGALURU','BANGALORE','MYSORE','MANGALORE','HUBLI','CHENNAI',
    'COIMBATORE','MADURAI','TRICHY','SALEM','VELLORE','ERODE',
    'HYDERABAD','VIJAYAWADA','GUNTUR','VISAKHAPATNAM','TIRUPATI',
    'KOLKATA','HOWRAH','DURGAPUR','SILIGURI',
    # Indian states / UTs
    'MAHARASHTRA','GUJARAT','RAJASTHAN','KARNATAKA','TAMILNADU','KERALA',
    'TELANGANA','ANDHRA','MADHYA PRADESH','UTTARAKHAND','PUNJAB','HARYANA',
    'UTTAR PRADESH','BIHAR','ODISHA','JHARKHAND','CHHATTISGARH',
    'HIMACHAL','JAMMU','KASHMIR','LADAKH','ANDAMAN','CHANDIGARH',
    # International countries / cities
    'INDIA','USA','UK','ENGLAND','SCOTLAND','WALES','AUSTRALIA','CANADA',
    'GERMANY','FRANCE','CHINA','JAPAN','SINGAPORE','DUBAI','UAE','LONDON',
    'CALIFORNIA','TEXAS','FLORIDA','NEW YORK',
}


# ── PAGE TYPE DETECTION ───────────────────────────────────────────────────────

def is_page_image_based(page: fitz.Page) -> bool:
    if len(page.get_text("words")) > 20:
        return False
    raw = page.get_text().strip()
    area = page.rect.width * page.rect.height
    img_area = sum(
        r.width * r.height
        for img in page.get_images(full=True)
        for r in page.get_image_rects(img[0])
    )
    img_ratio = img_area / area if area else 0
    if img_ratio > 0.25 and len(raw) < 250:
        return True
    return len(raw) < 100 and img_ratio > 0.5


# ── TEXT HELPERS ──────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())).strip()

def tokenize(text: str) -> list:
    return [t for t in normalize_text(text).split() if t]

def clean(t: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(t))).lower().strip()

def is_address_line(line: str) -> bool:
    lu = re.sub(r'[^A-Za-z\s]', ' ', line).upper()
    for tok in lu.split():
        t = re.sub(r'[^A-Za-z]', '', tok)
        if t in ADDRESS_KEYWORDS:
            return True
    return False


# ── SENSITIVE STRING COLLECTION ───────────────────────────────────────────────

def collect_sensitive_strings(text: str, redact_types: list, custom_terms: list) -> list:
    found = []

    for rtype in (redact_types or []):
        # Regex-based
        if rtype in PATTERNS:
            for m in re.finditer(PATTERNS[rtype], text, re.IGNORECASE):
                found.append(m.group(0).strip())

        # Keyword-based address/location detection
        if rtype == "address":
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                for word in line.upper().split():
                    clean_word = re.sub(r'[^A-Za-z]', '', word)
                    if clean_word in ADDRESS_KEYWORDS:
                        found.append(clean_word)

    for term in (custom_terms or []):
        if term.strip():
            found.append(term.strip())

    # Deduplicate exact matches only
    seen, result = set(), []
    for s in found:
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            result.append(s)
    return result


# ── EASYOCR ───────────────────────────────────────────────────────────────────

_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


def get_easyocr_words(page: fitz.Page) -> list:
    import numpy as np
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    sx = page.rect.width  / pix.width
    sy = page.rect.height / pix.height

    reader = get_easyocr_reader()
    results = reader.readtext(img)

    words = []
    for (bbox_pts, text, conf) in results:
        if conf < 0.3 or not text.strip():
            continue
        xs = [pt[0] for pt in bbox_pts]
        ys = [pt[1] for pt in bbox_pts]
        words.append({
            "text": text,
            "bbox": (min(xs)*sx, min(ys)*sy, max(xs)*sx, max(ys)*sy),
            "conf": conf,
        })
    return words


# ── PHRASE MATCHING (EasyOCR words → fitz.Rect) ──────────────────────────────

def phrase_rects_from_easyocr(phrase: str, easy_words: list, padding: int = 2) -> list:
    phrase_tokens = tokenize(phrase)
    if not phrase_tokens:
        return []

    matches = []

    # Single word
    if len(phrase_tokens) == 1:
        target = phrase_tokens[0]
        for word in easy_words:
            for tok in tokenize(word["text"]):
                score = SequenceMatcher(None, target, tok).ratio()
                if score >= 0.88:
                    x0, y0, x1, y1 = word["bbox"]
                    matches.append(fitz.Rect(x0-padding, y0-padding, x1+padding, y1+padding))
                    break

    # Multi-word: sliding window over EasyOCR word list
    else:
        target = " ".join(phrase_tokens)
        word_entries = []
        for w in easy_words:
            toks = tokenize(w["text"])
            if toks:
                word_entries.append({"tokens": toks, "bbox": w["bbox"]})

        n = len(phrase_tokens)
        for i in range(len(word_entries) - n + 1):
            candidate_tokens = []
            for j in range(n):
                candidate_tokens.extend(word_entries[i+j]["tokens"])
            candidate = " ".join(candidate_tokens)

            score = SequenceMatcher(None, target, candidate).ratio()
            if score < 0.82:
                continue

            xs0, ys0, xs1, ys1 = [], [], [], []
            for j in range(n):
                x0, y0, x1, y1 = word_entries[i+j]["bbox"]
                xs0.append(x0); ys0.append(y0)
                xs1.append(x1); ys1.append(y1)

            matches.append(fitz.Rect(
                min(xs0)-padding, min(ys0)-padding,
                max(xs1)+padding, max(ys1)+padding,
            ))

    # Deduplicate
    unique = []
    for r in matches:
        if not any(abs(r.x0-u.x0) < 3 and abs(r.y0-u.y0) < 3 for u in unique):
            unique.append(r)
    return unique


# ── TEXT PAGE REDACTION ───────────────────────────────────────────────────────

def redact_text_page(page: fitz.Page, page_num: int,
                     redact_types: list, custom_terms: list, audit_log: list):
    page_text = page.get_text("text")
    words = page.get_text("words")
    sensitives = collect_sensitive_strings(page_text, redact_types, custom_terms)

    for phrase in sensitives:
        phrase = phrase.strip()
        if not phrase:
            continue

        # For address keywords, always use word-list matching instead of search_for()
        if phrase.upper() in ADDRESS_KEYWORDS:
            p = clean(phrase)
            for w in words:
                if clean(w[4]) == p:
                    rect = fitz.Rect(w[0]-1, w[1]-1, w[2]+1, w[3]+1)
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    audit_log.append({"page": page_num+1, "type": "text", "text": phrase})
                    print(f"  ✓ [addr] {phrase[:60]}")
            continue

        # search_for on very short strings (<=3 chars) causes false positives
        # e.g. "AZ", "VIC" matching inside other words like "Service"
        # For short strings, use whole-word match via word list instead
        if len(phrase) <= 3:
            p = clean(phrase)
            for w in words:
                if clean(w[4]) == p:
                    rect = fitz.Rect(w[0]-1, w[1]-1, w[2]+1, w[3]+1)
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    audit_log.append({"page": page_num+1, "type": "text", "text": phrase})
                    print(f"  ✓ [word] {phrase[:60]}")
        else:
            rects = page.search_for(phrase)
            if rects:
                for rect in rects:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    audit_log.append({"page": page_num+1, "type": "text", "text": phrase})
                    print(f"  ✓ [text] {phrase[:60]}")
            else:
                p = clean(phrase)
                for w in words:
                    if clean(w[4]) == p:
                        rect = fitz.Rect(w[0]-1, w[1]-1, w[2]+1, w[3]+1)
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                        audit_log.append({"page": page_num+1, "type": "text", "text": phrase})
                        print(f"  ✓ [word] {phrase[:60]}")
    page.apply_redactions()


# ── IMAGE PAGE REDACTION ──────────────────────────────────────────────────────

def redact_image_page(page: fitz.Page, page_num: int,
                      redact_types: list, custom_terms: list, audit_log: list):
    print(f"  Running EasyOCR on page {page_num+1}...")
    try:
        easy_words = get_easyocr_words(page)
    except Exception as e:
        print(f"  [EasyOCR ERROR] {e}")
        return

    if not easy_words:
        print(f"  [WARN] No words detected on page {page_num+1}")
        return

    ocr_text = "\n".join(w["text"] for w in easy_words)
    sensitives = collect_sensitive_strings(ocr_text, redact_types, custom_terms)
    print(f"  Words: {len(easy_words)} | Sensitives: {len(sensitives)}")

    applied = []

    for phrase in sensitives:
        rects = phrase_rects_from_easyocr(phrase, easy_words)
        for rect in rects:
            if any(abs(r.x0-rect.x0) < 3 and abs(r.y0-rect.y0) < 3 for r in applied):
                continue
            page.add_redact_annot(rect, fill=(0, 0, 0))
            applied.append(rect)
            audit_log.append({"page": page_num+1, "type": "easyocr", "text": phrase})
            print(f"  ✓ [easyocr] {phrase[:60]}")
        if not rects:
            print(f"  ~ [no match] {phrase[:50]}")

    page.apply_redactions()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def redact_pdf(input_path: str, output_path: str,
               redact_types: list = None, custom_terms: list = None) -> str:
    doc          = fitz.open(input_path)
    audit_log    = []
    redact_types = list(redact_types or [])
    custom_terms = list(custom_terms or [])

    # address always implies pincode
    if "address" in redact_types and "pincode" not in redact_types:
        redact_types.append("pincode")

    image_pages = [i for i, p in enumerate(doc) if is_page_image_based(p)]
    text_pages  = [i for i in range(len(doc)) if i not in image_pages]

    print(f"\nPDF: {len(doc)} page(s) | "
          f"{len(image_pages)} scanned (EasyOCR) | "
          f"{len(text_pages)} text (search_for)")

    for page_num, page in enumerate(doc):
        if page_num in image_pages:
            print(f"\nPage {page_num+1}: [SCANNED → EasyOCR]")
            redact_image_page(page, page_num, redact_types, custom_terms, audit_log)
        else:
            print(f"\nPage {page_num+1}: [TEXT → PyMuPDF]")
            redact_text_page(page, page_num, redact_types, custom_terms, audit_log)

    doc.save(output_path, garbage=4, clean=True)
    doc.close()

    print(f"\n✓ Saved: {output_path} | {len(audit_log)} redaction(s)")
    for item in audit_log:
        print(f"  Page {item['page']} | {item['type']:8} | {item['text']}")

    return output_path




















# import fitz
# import re
# import spacy
# import warnings
 
# from core.ocr_extractor import is_scanned_pdf, extract_blocks_with_boxes
 
# warnings.filterwarnings("ignore") # Ignore warnings
 
# # LOAD SPACY MODEL
 
# nlp = spacy.load("en_core_web_sm")
 
# # REGEX PATTERNS
 
# patterns = {
#     "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
 
#     # Indian + International phone formats
#     "phone": r"(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",
 
#     # $, ₹, Rs, EUR, GBP formats with decimals
#     "amount": r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|€\s?\d+[\d,]*\.?\d*|£\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",
 
#     # DD/MM/YYYY, MM-DD-YYYY, January 25 2016, 25 Jan 2016, 2016-01-25
#     "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}",
 
#     # Aadhaar formats: 1234 5678 9012 or 123456789012
#     "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b",
 
#     # PAN with word boundary
#     "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
 
#     # Credit card with/without spaces/dashes
#     "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
 
#     # IFSC code
#     "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
 
#     # Account numbers (8-18 digits)
#     "account_number": r"\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",
 
#     # BSB (Australian bank)
#     "bsb": r"\bBSB\s*#?\s*\d{3}[\s-]?\d{3}\b",
 
#     # Street address patterns
#     "street_address": r"\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b",
# }
 
# # Whitelist of words to skip during NER redaction
# technical_words = set() 
 
# # REDACTION FUNCTION
 
#  # This function takes a PDF page, the text to be redacted, and an optional label for the type of information being redacted.
# def add_redaction(page, text, label=None):
#     areas = page.search_for(text)
#     for area in areas:
#         if label:
#             page.add_redact_annot(
#                 area,
#                 text=f"[REDACTED {label.upper()}]",
#                 fill=(0, 0, 0),
#                 text_color=(1, 1, 1)
#             )
 
#         else:
 
#             page.add_redact_annot(
#                 area,
#                 fill=(0, 0, 0)
#             )
 
# # SCANNED PDF REDACTION (via local OCR bounding boxes)
 
# def _group_words_into_lines(words, y_tolerance=10):
#     """
#     Tesseract gives back individual words with their own bounding box.
#     A pattern like an email or phone number is usually split across
#     several words (e.g. "john" "@" "email.com" might even be 1-3
#     separate detections) — checking one word at a time would almost
#     never match the email/phone/date regex patterns.
 
#     This groups words into LINES (by similar y-coordinate, left to
#     right), so we can run the same regex/NER checks against a full
#     line of text like the text-based path does — then map any match
#     back to just the words that made up the matched text.
#     """
#     if not words:
#         return []
 
#     sorted_words = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))
 
#     lines = []
#     current_line = [sorted_words[0]]
#     current_y = sorted_words[0]["bbox"][1]
 
#     for word in sorted_words[1:]:
#         y = word["bbox"][1]
#         if abs(y - current_y) <= y_tolerance:
#             current_line.append(word)
#         else:
#             lines.append(current_line)
#             current_line = [word]
#             current_y = y
#     lines.append(current_line)
 
#     # Within each line, keep words left-to-right (already true from the
#     # initial sort, but re-sort defensively in case of tied y-values).
#     for line in lines:
#         line.sort(key=lambda w: w["bbox"][0])
 
#     return lines
 
 
# def redact_scanned_pdf(
#     input_path,
#     output_path,
#     redact_types=None,
#     custom_terms=None
# ):
#     """
#     Redact a scanned PDF that has no real text layer.
 
#     Strategy:
#       1. OCR the page locally (Tesseract, via extract_blocks_with_boxes)
#          to get every word's text + pixel position.
#       2. Group words into lines, and rebuild each line's full text
#          with a position->word map, so a regex match spanning multiple
#          words (e.g. "john@email.com" or "9876543210") can be traced
#          back to exactly which word boxes to redact.
#       3. Run the same regex/NER/custom-term checks used for normal
#          PDFs against each line's text.
#       4. For every match, black out only the word(s) whose character
#          range overlaps the match — not the whole line — by unioning
#          their bounding boxes into one redaction rectangle.
#     """
#     pages_words = extract_blocks_with_boxes(input_path)
 
#     doc = fitz.open(input_path)
#     audit_log = []
 
#     for page_num, page in enumerate(doc):
#         if page_num >= len(pages_words):
#             continue
 
#         words = pages_words[page_num]
#         if not words:
#             continue
 
#         page_rect = page.rect
#         page_width = words[0]["page_width"]
#         page_height = words[0]["page_height"]
#         scale_x = page_rect.width / page_width
#         scale_y = page_rect.height / page_height
 
#         lines = _group_words_into_lines(words)
#         already_redacted_boxes = []
 
#         for line in lines:
#             # Rebuild the line's text, remembering each word's
#             # character start/end offset within that joined string so
#             # a regex match's span can be mapped back to specific words.
#             line_text = ""
#             word_spans = []  # (start_char, end_char, word_dict)
#             for w in line:
#                 start = len(line_text)
#                 line_text += w["text"]
#                 end = len(line_text)
#                 word_spans.append((start, end, w))
#                 line_text += " "
 
#             def words_overlapping(span_start, span_end):
#                 return [
#                     w for (s, e, w) in word_spans
#                     if s < span_end and e > span_start
#                 ]
 
#             matched_spans = []  # list of (start, end, type)
 
#             # REGEX CHECK
#             if redact_types:
#                 for rtype in redact_types:
#                     if rtype not in patterns:
#                         continue
#                     for m in re.finditer(patterns[rtype], line_text):
#                         matched_spans.append((m.start(), m.end(), rtype))
 
#             # NER CHECK
#             if redact_types and ("name" in redact_types or "address" in redact_types):
#                 doc_nlp = nlp(line_text[:2000])
#                 for ent in doc_nlp.ents:
#                     if ent.label_ == "PERSON" and "name" in redact_types:
#                         matched_spans.append((ent.start_char, ent.end_char, "name"))
#                     elif ent.label_ in ["GPE", "LOC", "FAC"] and "address" in redact_types:
#                         matched_spans.append((ent.start_char, ent.end_char, "address"))
 
#             # CUSTOM TERMS CHECK (case-insensitive substring)
#             if custom_terms:
#                 lower_line = line_text.lower()
#                 for term in custom_terms:
#                     term = term.strip()
#                     if not term:
#                         continue
#                     start_search = 0
#                     term_lower = term.lower()
#                     while True:
#                         idx = lower_line.find(term_lower, start_search)
#                         if idx == -1:
#                             break
#                         matched_spans.append((idx, idx + len(term), "custom"))
#                         start_search = idx + len(term)
 
#             # Redact each match's underlying word boxes
#             for span_start, span_end, rtype in matched_spans:
#                 hit_words = words_overlapping(span_start, span_end)
#                 if not hit_words:
#                     continue
 
#                 x0 = min(w["bbox"][0] for w in hit_words)
#                 y0 = min(w["bbox"][1] for w in hit_words)
#                 x1 = max(w["bbox"][2] for w in hit_words)
#                 y1 = max(w["bbox"][3] for w in hit_words)
 
#                 box_key = (round(x0), round(y0), round(x1), round(y1))
#                 if box_key in already_redacted_boxes:
#                     continue
#                 already_redacted_boxes.append(box_key)
 
#                 pdf_rect = fitz.Rect(
#                     x0 * scale_x,
#                     y0 * scale_y,
#                     x1 * scale_x,
#                     y1 * scale_y,
#                 )
#                 page.add_redact_annot(pdf_rect, fill=(0, 0, 0))
 
#                 matched_text = line_text[span_start:span_end]
#                 audit_log.append({
#                     "page": page_num + 1,
#                     "type": rtype,
#                     "text": matched_text,
#                 })
 
#         page.apply_redactions()
 
#     doc.save(output_path, garbage=4, clean=True)
#     doc.close()
 
#     print("\nRedacted scanned PDF saved successfully!")
#     print(f"Output File: {output_path}")
#     print("\nAudit Log:")
#     for item in audit_log:
#         print(item)
 
#     return output_path
 
 
# # MAIN PDF REDACTION (text-based PDFs)
 
# def redact_pdf(
#     input_path,
#     output_path,
#     redact_types=None,
#     custom_terms=None
# ):
#     # Scanned PDFs (handwritten notes, photographed documents, old
#     # letters) have no real text layer — page.get_text() returns an
#     # empty string for them, so every regex/NER check below silently
#     # matches nothing and the "redacted" file comes back unchanged.
#     # Route those through OCR-based redaction instead, which finds text
#     # via Mistral's bounding boxes and blacks out those pixel regions
#     # directly rather than searching for a string that doesn't exist
#     # in the PDF's text layer.
#     if is_scanned_pdf(input_path):
#         print("Scanned PDF detected — using OCR-based redaction instead of text search.")
#         return redact_scanned_pdf(
#             input_path=input_path,
#             output_path=output_path,
#             redact_types=redact_types,
#             custom_terms=custom_terms,
#         )
 
#     doc = fitz.open(input_path)   
#     audit_log = []
#     for page_num, page in enumerate(doc):
#         already_redacted = set()
#         full_text = page.get_text()
 
#         # REGEX REDACTION
 
#         if redact_types:
#             for rtype in redact_types:
#                 if rtype in patterns:
#                     matches = re.finditer(    # Use regex to find matches of the current redact type in the page text based on the defined patterns for each type
#                         patterns[rtype],
#                         full_text
#                     )
#     # For each regex match found, check if the matched text is valid and has not already been redacted, then apply the redaction annotation to the page using the add_redaction function.
#                     for match in matches:
#                         matched_text = match.group().strip()  # Get the matched text from the regex search and strip any extra whitespace for accurate redaction.
#                         if (
#                             not matched_text
#                             or matched_text in already_redacted
#                         ):
#                             continue
#                         add_redaction(
#                             page,
#                             matched_text,
#                             label=rtype
#                         )
 
#                         already_redacted.add(matched_text)
 
#                         audit_log.append({
#                             "page": page_num + 1,
#                             "type": rtype,
#                             "text": matched_text
#                         })
 
#         # NER REDACTION
 
#         if redact_types and (
#             "name" in redact_types
#             or "address" in redact_types
#         ):
 
#             doc_nlp = nlp(full_text[:10000])   # Process the page text with spaCy's NLP model to detect named entities. Limiting to first 10k characters for performance on large pages.
 
#             for ent in doc_nlp.ents:
#                 entity_text = ent.text.strip()
 
#                 # FILTERS
#                 if (
#                     len(entity_text) < 3
#                     or entity_text in already_redacted
#                     or entity_text in technical_words
#                 ):
#                     continue
 
#                 # Skip weird technical entities
#                 if re.search(r"[\d_=(){}[\]]", entity_text):   # If the entity text contains digits or special characters that are common in technical terms, skip redaction 
#                     continue
 
#                 text = entity_text.strip()
 
#                 # skip only obvious noise
#                 if len(text) < 2:
#                     continue
 
#                 # PERSON
#                 if (
#                     ent.label_ == "PERSON"
#                     and "name" in redact_types
#                 ):
 
#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="name"
#                     )
 
#                     already_redacted.add(entity_text)
 
#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "name",
#                         "text": entity_text
#                     })
 
#                 # ADDRESS
#                 elif (
#                     ent.label_ in ["GPE", "LOC", "FAC"]
#                     and "address" in redact_types
#                 ):
 
#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="address"
#                     )
 
#                     already_redacted.add(entity_text)
 
#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "address",
#                         "text": entity_text
#                     })
 
#         # CUSTOM TERMS
 
#         if custom_terms:
#             for term in custom_terms:
#                 term = term.strip()
#         # Skip empty terms or terms that have already been redacted to avoid unnecessary processing and redundant redactions
#                 if (
#                     not term
#                     or term in already_redacted
#                 ):
#                     continue
 
#                 add_redaction(
#                     page,
#                     term,
#                     label="custom"
#                 )
 
#                 already_redacted.add(term)
 
#                 audit_log.append({
#                     "page": page_num + 1,
#                     "type": "custom",
#                     "text": term
#                 })
 
#         # APPLY REDACTIONS
#         page.apply_redactions()
 
#     # SAVE PDF
#     doc.save(
#         output_path,
#         garbage=4,    # Optimize the PDF by removing unused objects and compressing the file, which can help reduce the file size after redaction.
#         clean=True
#     )
 
#     doc.close()
 
#     print("\nRedacted PDF saved successfully!")
#     print(f"Output File: {output_path}")
 
#     # AUDIT LOG
 
#     print("\nAudit Log:")
 
#     for item in audit_log:
#         print(item)
 
#     return output_path
 
 
# if __name__ == "__main__":
 
#     redact_pdf(
#         input_path="invoice.pdf",
 
#         output_path="redacted_output.pdf",
 
#         redact_types=[
#             "email",
#             "phone",
#             "amount",
#             "date",
#             "aadhaar",
#             "pan",
#             "credit_card",
#             "ifsc",
#             "name",
#             "address",
#             "street_address"
#         ],
 
#         custom_terms=[
#             "confidential",
#             "secret"
#         ]
#     )



















# """
# redactor.py — PDF Redaction Engine

# TWO PATHS:
#   Printed/text PDF     → PyMuPDF search_for() → exact coords → redact
#   Scanned/handwritten  → Mistral OCR 4 (include_blocks=True)
#                          → block bbox OR word-level proportional bbox → redact
# """

# import os, re, base64, warnings
# from difflib import SequenceMatcher
# import fitz
# from dotenv import load_dotenv
# from ocr.easyocr_helper import (
#     get_easyocr_words,
#     phrase_rects_from_easyocr,
# )

# warnings.filterwarnings("ignore")
# load_dotenv()

# # ── spaCy NER (optional) ─────────────────────────────────────────────────────
# try:
#     import spacy
#     nlp = spacy.load("en_core_web_sm")
#     SPACY_AVAILABLE = True
# except Exception:
#     SPACY_AVAILABLE = False

# # ── REGEX PATTERNS ────────────────────────────────────────────────────────────
# PATTERNS = {
#     "email":       r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
#     "phone":       r"(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",
#     "date":        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}",
#     "aadhaar":     r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
#     "pan":         r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
#     "pincode":     r"\b\d{6}\b",
#     "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
#     "ifsc":        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
#     "amount":      r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",
#     "account_number": r"\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",
# }

# SPACY_TYPE_MAP = {
#     "name":    ["PERSON"],
#     "org":     ["ORG"],
#     "address": ["GPE", "LOC", "FAC"],
#     "money":   ["MONEY"],
#     "time":    ["TIME"],
#     "date":    ["DATE"],
# }

# ADDRESS_KEYWORDS = {
#     'COLONY','MARG','ROAD','LANE','NAGAR','WEST','EAST','NORTH','SOUTH',
#     'SECTOR','BLOCK','STREET','PLOT','FLAT','HOUSE','APARTMENT','APT',
#     'BUILDING','BLDG','NEAR','OPP','LAYOUT','PHASE','CROSS','MAIN','PARK','MARG',
#     'SOCIETY','COMPLEX','TOWER','MUMBAI','DELHI','THANE','PUNE','ANDHERI',
#     'BORIVALI','KURLA','MAHARASHTRA','GUJARAT','RAJASTHAN','KARNATAKA',
#     'TAMILNADU','KERALA','TELANGANA','ANDHRA','PRADESH','UTTARAKHAND',
#     'PUNJAB','HARYANA','BENGAL','ODISHA','BIHAR',# Mumbai-specific localities
#     'KANJURMARG', 'BHANDUP', 'KURLA', 'ANDHERI', 'BORIVALI', 'THANE', 'MULUND',
#     'CHEMBUR', 'WADALA', 'DADAR', 'GOREGAON', 'MALAD', 'KANDIVALI', 'VASAI',
#     'VIRAR', 'MIRA', 'BHAYANDER', 'DAHISAR', 'MUMBAI', 'SUBURBAN',
#    # Address label prefixes (e.g. "State: Maharashtra", "Sub District: Kurla")
#     'STATE', 'DISTRICT', 'SUBDISTRICT', 'TALUKA', 'TEHSIL', 'VILLAGE', 'PO', 'VTC',
#     'SUBDIVISION','SUB-DIVISION'
# }


# # ── MISTRAL CLIENT ────────────────────────────────────────────────────────────

# _client = None

# def get_client():
#     global _client
#     if _client is None:
#         from mistralai.client.sdk import Mistral
#         api_key = os.getenv("MISTRAL_API_KEY")
#         if not api_key:
#             raise ValueError("MISTRAL_API_KEY not set in .env")
#         _client = Mistral(api_key=api_key)
#     return _client


# # ── PAGE TYPE DETECTION ───────────────────────────────────────────────────────

# def is_page_image_based(page):
#     # If search_for() can find text, treat it as text.
#     if len(page.get_text("words")) > 20:
#         return False

#     raw = page.get_text().strip()

#     area = page.rect.width * page.rect.height

#     img_area = sum(
#         r.width * r.height
#         for img in page.get_images(full=True)
#         for r in page.get_image_rects(img[0])
#     )

#     img_ratio = img_area / area if area else 0

#     if img_ratio > 0.25 and len(raw) < 250:
#         return True

#     return len(raw) < 100 and img_ratio > 0.5


# # ── HELPERS ───────────────────────────────────────────────────────────────────
# from difflib import SequenceMatcher
# import fitz


# def normalize_text(text: str) -> str:
#     return " ".join(
#         "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
#     )


# def tokenize(text: str):
#     return normalize_text(text).split()


# def phrase_rects_from_easyocr(
#     phrase: str,
#     easy_words: list,
#     padding: int = 2,
# ):
#     """
#     easy_words:
#     [
#         {
#             "text": "...",
#             "bbox": (x0,y0,x1,y1)
#         },
#         ...
#     ]

#     Returns:
#         [fitz.Rect]
#     """

#     phrase_tokens = tokenize(phrase)

#     if not phrase_tokens:
#         return []

#     word_tokens = [
#         normalize_text(w["text"])
#         for w in easy_words
#     ]

#     best = None
#     best_ratio = 0

#     min_width = max(1, len(phrase_tokens) - 1)
#     max_width = min(len(word_tokens), len(phrase_tokens) + 2)

#     target = " ".join(phrase_tokens)

#     for width in range(min_width, max_width + 1):

#         for start in range(len(word_tokens) - width + 1):

#             candidate = " ".join(
#                 word_tokens[start:start + width]
#             )

#             if candidate == target:
#                 best = (start, start + width)
#                 best_ratio = 1.0
#                 break

#             ratio = SequenceMatcher(
#                 None,
#                 target,
#                 candidate
#             ).ratio()

#             if ratio > best_ratio:
#                 best_ratio = ratio
#                 best = (start, start + width)

#         if best_ratio == 1.0:
#             break

#     if best is None or best_ratio < 0.7:
#         return []

#     start, end = best

#     xs0 = []
#     ys0 = []
#     xs1 = []
#     ys1 = []

#     for w in easy_words[start:end]:
#         x0, y0, x1, y1 = w["bbox"]

#         xs0.append(x0)
#         ys0.append(y0)
#         xs1.append(x1)
#         ys1.append(y1)

#     rect = fitz.Rect(
#         min(xs0) - padding,
#         min(ys0) - padding,
#         max(xs1) + padding,
#         max(ys1) + padding,
#     )

#     return [rect]


# def is_address_line(line: str) -> bool:
#     lu = re.sub(r'[^A-Za-z\s]', ' ', line).upper()
#     for tok in lu.split():
#         t = re.sub(r'[^A-Za-z]', '', tok)
#         if t in ADDRESS_KEYWORDS:
#             return True
#     return False

# def clean(t: str) -> str:
#     return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', str(t))).lower().strip()


# def normalize_text(text: str) -> str:
#     return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', str(text).lower())).strip()


# def tokenize(text: str) -> list[str]:
#     return [token for token in normalize_text(text).split() if token]


# _printed_first_ocr_block = False

# def _g(obj, key):
#     """Get attr from dict or object."""
#     return obj[key] if isinstance(obj, dict) else getattr(obj, key, 0)


# def _ocr_word_text(word) -> str:
#     return str(
#         _g(word, "text")
#         or _g(word, "content")
#         or _g(word, "word")
#         or ""
#     )


# def _ocr_word_rect(word, page: fitz.Page, img_w: float, img_h: float,
#                    padding: int = 4) -> fitz.Rect | None:
#     sx = page.rect.width / img_w if img_w else 1
#     sy = page.rect.height / img_h if img_h else 1

#     if isinstance(word, dict):
#         if {"x0", "y0", "x1", "y1"}.issubset(word.keys()):
#             return fitz.Rect(
#                 word["x0"] * sx - padding,
#                 word["y0"] * sy - padding,
#                 word["x1"] * sx + padding,
#                 word["y1"] * sy + padding,
#             )
#         if {"left", "top", "right", "bottom"}.issubset(word.keys()):
#             return fitz.Rect(
#                 word["left"] * sx - padding,
#                 word["top"] * sy - padding,
#                 word["right"] * sx + padding,
#                 word["bottom"] * sy + padding,
#             )
#         bbox = word.get("bbox")
#         if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
#             return fitz.Rect(
#                 bbox[0] * sx - padding,
#                 bbox[1] * sy - padding,
#                 bbox[2] * sx + padding,
#                 bbox[3] * sy + padding,
#             )

#     top_left_x = getattr(word, "top_left_x", None)
#     top_left_y = getattr(word, "top_left_y", None)
#     bottom_right_x = getattr(word, "bottom_right_x", None)
#     bottom_right_y = getattr(word, "bottom_right_y", None)

#     if any(v is None for v in (top_left_x, top_left_y, bottom_right_x, bottom_right_y)):
#         return None

#     return fitz.Rect(
#         top_left_x * sx - padding,
#         top_left_y * sy - padding,
#         bottom_right_x * sx + padding,
#         bottom_right_y * sy + padding,
#     )


# def _collect_ocr_words(block) -> list:
#     words = _g(block, "words") or []
#     if words:
#         return words

#     collected = []
#     for line in (_g(block, "lines") or []):
#         line_words = _g(line, "words") or []
#         if line_words:
#             collected.extend(line_words)
#     return collected


# def _block_text(block) -> str:
#     parts = []
#     content = _g(block, "content")
#     if content:
#         parts.append(str(content))

#     markdown = _g(block, "markdown")
#     if markdown:
#         parts.append(str(markdown))

#     words = _collect_ocr_words(block)
#     if words:
#         word_text = " ".join(_ocr_word_text(word) for word in words if _ocr_word_text(word))
#         if word_text:
#             parts.append(word_text)

#     return "\n".join(part for part in parts if part).strip()


# def _print_first_ocr_block(block) -> None:
#     global _printed_first_ocr_block
#     if _printed_first_ocr_block:
#         return

#     _printed_first_ocr_block = True
#     print("FIRST OCR BLOCK STRUCTURE")
#     print("TYPE:", type(block).__name__)
#     print("FIELDS:", list(getattr(block, "__dict__", {}).keys()))
#     print("CONTENT:", repr(getattr(block, "content", None)))
#     print("HAS WORDS:", hasattr(block, "words"))
#     print("HAS LINES:", hasattr(block, "lines"))
#     print("HAS WORD BBOXES:", hasattr(block, "word_bboxes"))


# def find_matching_window(phrase: str, block_text: str, window_width: int) -> tuple[int, int, float] | None:
#     phrase_tokens = tokenize(phrase)
#     block_tokens = tokenize(block_text)

#     if not phrase_tokens or not block_tokens:
#         return None

#     phrase_text = " ".join(phrase_tokens)
#     if window_width <= 0:
#         window_width = len(phrase_tokens)

#     best_match = None
#     best_ratio = 0.0

#     min_width = max(1, window_width - 2)
#     max_width = min(len(block_tokens), window_width + 2)

#     for candidate_width in range(min_width, max_width + 1):
#         for start in range(0, len(block_tokens) - candidate_width + 1):
#             window_tokens = block_tokens[start:start + candidate_width]
#             window_text = " ".join(window_tokens)

#             if phrase_text == window_text:
#                 return start, start + candidate_width, 1.0

#             ratio = SequenceMatcher(None, phrase_text, window_text).ratio()

#             if ratio > best_ratio:
#                 best_ratio = ratio
#                 best_match = (start, start + candidate_width, ratio)

#     if best_match and best_match[2] > 0.7:
#         return best_match

#     return None


# def estimate_rect_from_words(words: list, page: fitz.Page, img_w: float, img_h: float,
#                              padding: int = 5) -> fitz.Rect | None:
#     rects = []
#     for word in words:
#         rect = _ocr_word_rect(word, page, img_w, img_h, padding=padding)
#         if rect is not None:
#             rects.append(rect)

#     if not rects:
#         return None

#     return fitz.Rect(
#         min(rect.x0 for rect in rects),
#         min(rect.y0 for rect in rects),
#         max(rect.x1 for rect in rects),
#         max(rect.y1 for rect in rects),
#     )


# def estimate_rect_from_block(block, page: fitz.Page, img_w: float, img_h: float,
#                              matched_text: str, padding: int = 2) -> fitz.Rect | None:
#     block_text = _block_text(block)
#     normalized_block = normalize_text(block_text)
#     normalized_match = normalize_text(matched_text)

#     if not normalized_block or not normalized_match:
#         return None

#     start = normalized_block.find(normalized_match)
#     if start < 0:
#         return None

#     end = start + len(normalized_match)
#     block_chars = [char for char in block_text if not char.isspace()]
#     if not block_chars:
#         return None

#     total_chars = len(block_chars)
#     if total_chars == 0:
#         return None

#     block_left = _g(block, "top_left_x")
#     block_top = _g(block, "top_left_y")
#     block_right = _g(block, "bottom_right_x")
#     block_bottom = _g(block, "bottom_right_y")

#     if any(value is None for value in (block_left, block_top, block_right, block_bottom)):
#         return None

#     sx = page.rect.width / img_w if img_w else 1
#     sy = page.rect.height / img_h if img_h else 1
#     width = (block_right - block_left) * sx
#     height = (block_bottom - block_top) * sy

#     if width <= 0 or height <= 0:
#         return None

#     start_ratio = max(0.0, min(1.0, start / max(len(normalized_block), 1)))
#     end_ratio = max(start_ratio, min(1.0, end / max(len(normalized_block), 1)))

#     return fitz.Rect(
#         block_left * sx + width * start_ratio - padding,
#         block_top * sy - padding,
#         block_left * sx + width * end_ratio + padding,
#         block_bottom * sy + padding,
#     )


# # ── SENSITIVE STRING COLLECTION ───────────────────────────────────────────────

# def collect_sensitive_strings(text: str, redact_types: list, custom_terms: list) -> list[str]:
#     found = []
#     for rtype in (redact_types or []):
#         if rtype in PATTERNS:
#             for m in re.finditer(PATTERNS[rtype], text, re.IGNORECASE):
#                 found.append(m.group(0).strip())
#         if SPACY_AVAILABLE and rtype in SPACY_TYPE_MAP:
#             doc = nlp(text)
#             for ent in doc.ents:
#                 if ent.label_ in SPACY_TYPE_MAP[rtype]:
#                     found.append(ent.text.strip())
#         if rtype == "address":
#             for line in text.splitlines():
#                 line = line.strip()
#                 if line and is_address_line(line):
#                     found.append(line)

#     for term in (custom_terms or []):
#         if term.strip():
#             found.append(term.strip())

#     seen, result = set(), []
#     for s in found:
#         if s and s not in seen:
#             seen.add(s)
#             result.append(s)
#     return result


# # ── TEXT PAGE ─────────────────────────────────────────────────────────────────

# def redact_text_page(page, page_num,
#                      redact_types, custom_terms, audit_log):

#     words = page.get_text("words")
#     page_text = page.get_text("text")

#     sensitives = collect_sensitive_strings(
#         page_text,
#         redact_types,
#         custom_terms
#     )

#     for phrase in sensitives:
#         phrase = phrase.strip()
#         if not phrase:
#             continue

#         # ---------- First try exact search ----------
#         rects = page.search_for(
#             phrase
#         )

#         if rects:
#             for rect in rects:
#                 page.add_redact_annot(rect, fill=(0,0,0))
#                 audit_log.append({
#                     "page": page_num+1,
#                     "type": "text",
#                     "text": phrase
#                 })
#                 print("✓", phrase)

#         # ---------- Fallback word match ----------
#         else:
#             p = clean(phrase)

#             for w in words:
#                 if clean(w[4]) == p:
#                     rect = fitz.Rect(
#                         w[0]-1,
#                         w[1]-1,
#                         w[2]+1,
#                         w[3]+1
#                     )

#                     page.add_redact_annot(rect, fill=(0,0,0))

#                     audit_log.append({
#                         "page": page_num+1,
#                         "type":"text",
#                         "text":phrase
#                     })

#                     print("✓ word:", phrase)

#     page.apply_redactions()

# # ── MISTRAL OCR ───────────────────────────────────────────────────────────────

# def mistral_ocr_blocks(pdf_path: str) -> list[dict]:
#     client = get_client()
#     with open(pdf_path, "rb") as f:
#         b64 = base64.standard_b64encode(f.read()).decode()

#     print("  Calling Mistral OCR 4...")
#     resp = client.ocr.process(
#         model="mistral-ocr-latest",
#         document={
#             "type": "document_url",
#             "document_url": f"data:application/pdf;base64,{b64}",
#         },
#         include_blocks=True,
#     )
#     print(f"  Done — {len(resp.pages)} page(s)")

#     pages = []
#     for p in resp.pages:
#         d = p.dimensions
#         pages.append({
#             "width":    _g(d, "width")  if d else 1,
#             "height":   _g(d, "height") if d else 1,
#             "blocks":   p.blocks or [],
#             "markdown": p.markdown or "",
#         })
#     return pages


# def block_to_pdf_rect(block, page: fitz.Page, img_w: float, img_h: float,
#                       padding: int = 2) -> fitz.Rect:
#     """Convert Mistral pixel bbox → PDF point rect."""
#     sx = page.rect.width  / img_w
#     sy = page.rect.height / img_h
#     return fitz.Rect(
#         _g(block, "top_left_x")     * sx - padding,
#         _g(block, "top_left_y")     * sy - padding,
#         _g(block, "bottom_right_x") * sx + padding,
#         _g(block, "bottom_right_y") * sy + padding,
#     )


# def phrase_rects_in_block(phrase, block, page, img_w, img_h):
#     content = _block_text(block)
#     if not content:
#         return []

#     phrase_c = normalize_text(phrase)
#     if not phrase_c:
#         return []

#     words = _collect_ocr_words(block)
#     if words:
#         word_texts = [normalize_text(_ocr_word_text(word)) for word in words]
#         word_stream = " ".join(word_texts)
#         match = find_matching_window(phrase, word_stream, len(tokenize(phrase)))

#         if match is not None:
#             matched_start, matched_end, _ = match
#             matched_words = words[matched_start:matched_end]
#             matched_rect = estimate_rect_from_words(
#                 matched_words,
#                 page,
#                 img_w,
#                 img_h,
#                 padding=2,
#             )
#             if matched_rect is not None:
#                 return [matched_rect]

#     estimated_rect = estimate_rect_from_block(
#         block,
#         page,
#         img_w,
#         img_h,
#         phrase,
#         padding=2,
#     )
#     if estimated_rect is not None:
#         return [estimated_rect]

#     return []

# # ── IMAGE PAGE ────────────────────────────────────────────────────────────────

# def redact_image_page(page: fitz.Page, page_num: int, page_ocr: dict,
#                       redact_types: list, custom_terms: list, audit_log: list):

#     img_w = page_ocr["width"]
#     img_h = page_ocr["height"]
#     blocks = page_ocr["blocks"]
#     md = page_ocr["markdown"]

#     if not blocks:
#         print(f"[WARN] No OCR blocks on page {page_num+1}")
#         return

#     # ---------- EasyOCR pass ----------
#     try:
#         easy_words = get_easyocr_words(page)
#         print(f"[EasyOCR] Words detected: {len(easy_words)}")
#     except Exception as e:
#         print("[EasyOCR ERROR]", e)
#         easy_words = []

#     sensitives = collect_sensitive_strings(md, redact_types, custom_terms)
#     print(f"Blocks: {len(blocks)} | Sensitives: {len(sensitives)}")

#     applied = []

#     for phrase in sensitives:
#         found = False

#         # ===============================
#         # 1. FIRST TRY EASYOCR
#         # ===============================
#         if easy_words:
#             rects = phrase_rects_from_easyocr(phrase, easy_words)

#             for rect in rects:

#                 if any(
#                     abs(r.x0 - rect.x0) < 3 and
#                     abs(r.y0 - rect.y0) < 3
#                     for r in applied
#                 ):
#                     continue

#                 page.add_redact_annot(rect, fill=(0, 0, 0))
#                 applied.append(rect)

#                 audit_log.append({
#                     "page": page_num + 1,
#                     "type": "easyocr",
#                     "text": phrase
#                 })

#                 print("✓ EasyOCR:", phrase)
#                 found = True

#         # ===============================
#         # 2. FALLBACK → Mistral block logic
#         # ===============================
#         if not found:

#             for block in blocks:

#                 rects = phrase_rects_in_block(
#                     phrase,
#                     block,
#                     page,
#                     img_w,
#                     img_h,
#                 )

#                 if not rects:
#                     continue

#                 for rect in rects:

#                     if any(
#                         abs(r.x0 - rect.x0) < 3 and
#                         abs(r.y0 - rect.y0) < 3
#                         for r in applied
#                     ):
#                         continue

#                     page.add_redact_annot(rect, fill=(0, 0, 0))
#                     applied.append(rect)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "mistral",
#                         "text": phrase
#                     })

#                     print("✓ Mistral:", phrase)

#                     found = True

#         if not found:
#             print("~", phrase)

#     page.apply_redactions()


# # ── MAIN ──────────────────────────────────────────────────────────────────────

# def redact_pdf(input_path: str, output_path: str,
#                redact_types: list = None, custom_terms: list = None):
#     doc          = fitz.open(input_path)
#     audit_log    = []
#     redact_types = list(redact_types or [])
#     custom_terms = list(custom_terms or [])

#     image_pages = [i for i, p in enumerate(doc) if is_page_image_based(p)]
#     text_pages = [i for i in range(len(doc)) if i not in image_pages]

#     if "address" in redact_types and "pincode" not in redact_types:
#         redact_types.append("pincode")

#     print(f"\nPDF: {len(doc)} page(s) | "
#           f"{len(image_pages)} image-based (Mistral OCR 4) | "
#           f"{len(text_pages)} text-based (search_for)")

#     ocr_data = {}
#     if image_pages:
#         try:
#             all_pages = mistral_ocr_blocks(input_path)
#             for i in image_pages:
#                 ocr_data[i] = all_pages[i] if i < len(all_pages) else {
#                     "width": 1, "height": 1, "blocks": [], "markdown": ""
#                 }
#         except Exception as e:
#             print(f"[ERROR] Mistral OCR failed: {e}")
#             text_pages += image_pages
#             image_pages.clear()

#     for page_num, page in enumerate(doc):
#         if page_num in image_pages:
#             print(f"\nPage {page_num + 1}: [IMAGE → Mistral OCR 4]")
#             redact_image_page(page, page_num, ocr_data[page_num],
#                               redact_types, custom_terms, audit_log)
#         else:
#             print(f"\nPage {page_num + 1}: [TEXT → PyMuPDF search_for]")
#             redact_text_page(page, page_num, redact_types, custom_terms, audit_log)

#     doc.save(output_path, garbage=4, clean=True)
#     doc.close()

#     print(f"\n✓ Saved: {output_path} | {len(audit_log)} redaction(s)")
#     for item in audit_log:
#         print(f"  Page {item['page']} | {item['type']:6} | {item['text']}")

#     return output_path







# import fitz
# import re
# import spacy
# import warnings

# warnings.filterwarnings("ignore") # Ignore warnings

# # LOAD SPACY MODEL

# nlp = spacy.load("en_core_web_sm")    

# # REGEX PATTERNS

# patterns = {
#     "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

#     # Indian + International phone formats
#     "phone": r"(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",

#     # $, ₹, Rs, EUR, GBP formats with decimals
#     "amount": r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|€\s?\d+[\d,]*\.?\d*|£\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",

#     # DD/MM/YYYY, MM-DD-YYYY, January 25 2016, 25 Jan 2016, 2016-01-25
#     "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}",

#     # Aadhaar formats: 1234 5678 9012 or 123456789012
#     "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b",

#     # PAN with word boundary
#     "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

#     # Credit card with/without spaces/dashes
#     "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",

#     # IFSC code
#     "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

#     # Account numbers (8-18 digits)
#     "account_number": r"\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",

#     # BSB (Australian bank)
#     "bsb": r"\bBSB\s*#?\s*\d{3}[\s-]?\d{3}\b",

#     # Street address patterns
#     "street_address": r"\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b",
# }

# # Whitelist of words to skip during NER redaction
# technical_words = set() 

# # REDACTION FUNCTION

#  # This function takes a PDF page, the text to be redacted, and an optional label for the type of information being redacted.
# def add_redaction(page, text, label=None):
#     areas = page.search_for(text)
#     for area in areas:
#         if label:
#             page.add_redact_annot(
#                 area,
#                 text=f"[REDACTED {label.upper()}]",
#                 fill=(0, 0, 0),
#                 text_color=(1, 1, 1)
#             )

#         else:

#             page.add_redact_annot(
#                 area,
#                 fill=(0, 0, 0)
#             )

# # MAIN PDF REDACTION

# def redact_pdf(
#     input_path,
#     output_path,
#     redact_types=None,
#     custom_terms=None
# ):

#     doc = fitz.open(input_path)   
#     audit_log = []
#     for page_num, page in enumerate(doc):
#         already_redacted = set()
#         full_text = page.get_text()

#         # REGEX REDACTION

#         if redact_types:
#             for rtype in redact_types:
#                 if rtype in patterns:
#                     matches = re.finditer(    # Use regex to find matches of the current redact type in the page text based on the defined patterns for each type
#                         patterns[rtype],
#                         full_text
#                     )
#     # For each regex match found, check if the matched text is valid and has not already been redacted, then apply the redaction annotation to the page using the add_redaction function.
#                     for match in matches:
#                         matched_text = match.group().strip()  # Get the matched text from the regex search and strip any extra whitespace for accurate redaction.
#                         if (
#                             not matched_text
#                             or matched_text in already_redacted
#                         ):
#                             continue
#                         add_redaction(
#                             page,
#                             matched_text,
#                             label=rtype
#                         )

#                         already_redacted.add(matched_text)

#                         audit_log.append({
#                             "page": page_num + 1,
#                             "type": rtype,
#                             "text": matched_text
#                         })

#         # NER REDACTION

#         if redact_types and (
#             "name" in redact_types
#             or "address" in redact_types
#         ):

#             doc_nlp = nlp(full_text[:10000])   # Process the page text with spaCy's NLP model to detect named entities. Limiting to first 10k characters for performance on large pages.

#             for ent in doc_nlp.ents:
#                 entity_text = ent.text.strip()

#                 # FILTERS
#                 if (
#                     len(entity_text) < 3
#                     or entity_text in already_redacted
#                     or entity_text in technical_words
#                 ):
#                     continue

#                 # Skip weird technical entities
#                 if re.search(r"[\d_=(){}[\]]", entity_text):   # If the entity text contains digits or special characters that are common in technical terms, skip redaction 
#                     continue

#                 text = entity_text.strip()

#                 # skip only obvious noise
#                 if len(text) < 2:
#                     continue

#                 # PERSON
#                 if (
#                     ent.label_ == "PERSON"
#                     and "name" in redact_types
#                 ):

#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="name"
#                     )

#                     already_redacted.add(entity_text)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "name",
#                         "text": entity_text
#                     })

#                 # ADDRESS
#                 elif (
#                     ent.label_ in ["GPE", "LOC", "FAC"]
#                     and "address" in redact_types
#                 ):

#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="address"
#                     )

#                     already_redacted.add(entity_text)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "address",
#                         "text": entity_text
#                     })

#         #CUSTOM TERMS

#         if custom_terms:
#             for term in custom_terms:
#                 term = term.strip()
#         # Skip empty terms or terms that have already been redacted to avoid unnecessary processing and redundant redactions
#                 if (
#                     not term
#                     or term in already_redacted
#                 ):
#                     continue

#                 add_redaction(
#                     page,
#                     term,
#                     label="custom"
#                 )

#                 already_redacted.add(term)

#                 audit_log.append({
#                     "page": page_num + 1,
#                     "type": "custom",
#                     "text": term
#                 })

#         # APPLY REDACTIONS
#         page.apply_redactions()

#     # SAVE PDF
#     doc.save(
#         output_path,
#         garbage=4,    # Optimize the PDF by removing unused objects and compressing the file, which can help reduce the file size after redaction.
#         clean=True
#     )

#     doc.close()

#     print("\nRedacted PDF saved successfully!")
#     print(f"Output File: {output_path}")

#     # AUDIT LOG

#     print("\nAudit Log:")

#     for item in audit_log:
#         print(item)

#     return output_path


# if __name__ == "__main__":

#     redact_pdf(
#         input_path="invoice.pdf",

#         output_path="redacted_output.pdf",

#         redact_types=[
#             "email",
#             "phone",
#             "amount",
#             "date",
#             "aadhaar",
#             "pan",
#             "credit_card",
#             "ifsc",
#             "name",
#             "address",
#             "street_address"
#         ],

#         custom_terms=[
#             "confidential",
#             "secret"
#         ]
#     )



# import fitz
# import re
# import warnings
# from difflib import SequenceMatcher

# warnings.filterwarnings("ignore")

# # ─── REGEX PATTERNS ────────────────────────────────────────────────────────────

# patterns = {
#     "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

#     "amount": (
#         r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*"
#         r"|\$\s?\d+[\d,]*\.?\d*|€\s?\d+[\d,]*\.?\d*|£\s?\d+[\d,]*\.?\d*"
#         r"|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)"
#     ),

#     "phone": (
#         r"(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}"
#         r"|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}"
#     ),

#     "date": (
#         r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
#         r"|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
#         r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
#         r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4}"
#         r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
#     ),

#     # FIX: OCR adds noise digit to first group ("16533" instead of "6533")
#     # So match \d{3,5} for first group, strict \d{4} for remaining two
#     "aadhaar": r"\d{3,5}[\s\-\.]?\d{4}[\s\-\.]?\d{4}|\b\d{12}\b",

#     "pincode": r"\bPIN\s*(?:Code:?\s*)?\d{6}\b|\b\d{6}\b",
#     "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
#     "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
#     "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
#     "account_number": r"\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",
# }

# # Keywords that indicate a line is an address line
# # Strip non-alpha from OCR tokens before matching (handles "COLONY," → "COLONY")
# ADDRESS_KEYWORDS = {
#     # Structural address words
#     'COLONY', 'MARG', 'ROAD', 'LANE', 'NAGAR', 'WEST', 'EAST', 'NORTH', 'SOUTH',
#     'CINEMA', 'SECTOR', 'BLOCK', 'STREET', 'PLOT', 'FLAT', 'HOUSE', 'APARTMENT',
#     'APT', 'BUILDING', 'BLDG', 'NEAR', 'OPP', 'LAYOUT', 'EXTENSION', 'PHASE',
#     'CROSS', 'MAIN', 'PARK', 'SOCIETY', 'COMPLEX', 'TOWER', 'VTC', 'NCH',
#     # Mumbai-specific localities
#     'KANJURMARG', 'BHANDUP', 'KURLA', 'ANDHERI', 'BORIVALI', 'THANE', 'MULUND',
#     'CHEMBUR', 'WADALA', 'DADAR', 'GOREGAON', 'MALAD', 'KANDIVALI', 'VASAI',
#     'VIRAR', 'MIRA', 'BHAYANDER', 'DAHISAR', 'MUMBAI', 'SUBURBAN',
#     # Address label prefixes (e.g. "State: Maharashtra", "Sub District: Kurla")
#     'STATE', 'DISTRICT', 'SUBDISTRICT', 'TALUKA', 'TEHSIL', 'VILLAGE', 'PO', 'VTC',
#     # Indian states — so "State: Maharashtra" line gets caught
#     'MAHARASHTRA', 'GUJARAT', 'RAJASTHAN', 'KARNATAKA', 'TAMILNADU', 'KERALA',
#     'TELANGANA', 'ANDHRA', 'PRADESH', 'UTTARAKHAND', 'PUNJAB', 'HARYANA',
#     'BENGAL', 'ODISHA', 'JHARKHAND', 'CHHATTISGARH', 'UTTARPRADESH', 'DELHI',
#     'BIHAR', 'ASSAM', 'HIMACHAL', 'GOA', 'MANIPUR', 'MEGHALAYA', 'TRIPURA',
# }


# # ─── TEXT-BASED PDF HELPERS ────────────────────────────────────────────────────

# def add_redaction(page, text):
#     print("SEARCHING:", repr(text))

#     areas = page.search_for(text)

#     print("FOUND:", len(areas))

#     for area in areas:
#         page.add_redact_annot(area, fill=(0,0,0))


# def get_normalized_page_text(page):
#     blocks = page.get_text("dict")["blocks"]
#     lines_text = []
#     for block in blocks:
#         if block.get("type") != 0:
#             continue
#         for line in block.get("lines", []):
#             spans = line.get("spans", [])
#             merged = ""
#             prev_size = None
#             for span in spans:
#                 text = span["text"]
#                 size = span["size"]
#                 if prev_size and size < prev_size * 0.8:
#                     merged = merged.rstrip() + text.strip()
#                 else:
#                     merged += text
#                 prev_size = size
#             lines_text.append(merged)
#     return "\n".join(lines_text)


# # ─── SCANNED PAGE DETECTION ────────────────────────────────────────────────────

# def is_page_scanned(page):
#     """
#     True if page is image-based (scanned).
#     Checks both text density AND image area coverage — Aadhaar PDFs
#     have header text but are still fundamentally scanned images.
#     """
#     raw_text = page.get_text().strip()
#     page_area = page.rect.width * page.rect.height

#     # Count image coverage
#     image_area = 0
#     for img in page.get_images(full=True):
#         for rect in page.get_image_rects(img[0]):
#             image_area += rect.width * rect.height

#     image_ratio = image_area / page_area if page_area > 0 else 0
#     text_density = len(raw_text) / page_area if page_area > 0 else 0

#     # Scanned = large image footprint + sparse text
#     return (len(raw_text) < 300 and image_ratio > 0.25) or text_density < 0.003


# # ─── OCR LINE GROUPING ─────────────────────────────────────────────────────────

# def group_words_into_lines(words, y_tolerance=6):
#     """
#     Group pytesseract word-level results into lines by Y-coordinate proximity.
#     Returns list of lines, each line = list of word dicts.

#     Why this matters: OCR gives individual tokens. "6533 3040 0575" comes as
#     3 separate words. We join them per-line so regex can match across tokens.
#     """
#     if not words:
#         return []

#     sorted_words = sorted(words, key=lambda w: (w["y0"], w["x0"]))
#     lines = [[sorted_words[0]]]

#     for word in sorted_words[1:]:
#         avg_y = sum(w["y0"] for w in lines[-1]) / len(lines[-1])
#         if abs(word["y0"] - avg_y) <= y_tolerance:
#             lines[-1].append(word)
#         else:
#             lines.append([word])

#     return lines


# def line_bbox(line_words):
#     return fitz.Rect(
#         min(w["x0"] for w in line_words),
#         min(w["y0"] for w in line_words),
#         max(w["x1"] for w in line_words),
#         max(w["y1"] for w in line_words),
#     )


# def line_has_address_keyword(line_words):
#     """Check if a line contains any address keyword (strips OCR noise chars first)."""
#     for w in line_words:
#         clean = re.sub(r'[^A-Za-z]', '', w["text"]).upper()
#         if clean in ADDRESS_KEYWORDS:
#             return True
#         if len(clean) >= 5 and any(kw in clean for kw in ADDRESS_KEYWORDS):
#             return True

#     # Also check joined line for multi-word label prefixes
#     # e.g. "Sub District: Kurla" — "SUB" and "DISTRICT" are separate tokens
#     joined = " ".join(re.sub(r'[^A-Za-z]', '', w["text"]) for w in line_words).upper()
#     address_label_patterns = [
#         r'\bSUB\s*DISTRICT\b', r'\bSUB\s*DIST\b', r'\bPIN\s*CODE\b',
#         r'\bPOST\s*OFFICE\b', r'\bPO\s*BOX\b',
#     ]
#     for pat in address_label_patterns:
#         if re.search(pat, joined):
#             return True

#     return False


# # ─── SCANNED PAGE REDACTION ────────────────────────────────────────────────────

# def redact_scanned_page_with_ocr(page, redact_types, custom_terms, already_redacted, audit_log, page_num):
#     """
#     Redact a scanned PDF page using Tesseract OCR.

#     Strategy:
#     - Run pytesseract at 300 DPI to get word-level bounding boxes
#     - Group words into lines by Y-coordinate
#     - Per line: run regex on joined line text → if match, redact matched words' bbox
#     - For address lines: keyword detection → redact entire line bbox
#     - For custom terms: fuzzy match across line tokens
#     """
#     try:
#         import pytesseract
#         from PIL import Image
#         import io

#         pix = page.get_pixmap(dpi=300)
#         img = Image.open(io.BytesIO(pix.tobytes("png")))

#         # Scale factors: OCR coords are in image pixels, fitz needs PDF points
#         scale_x = page.rect.width / img.width
#         scale_y = page.rect.height / img.height

#         ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

#         # Build word list with scaled PDF coordinates
#         words = []
#         for i in range(len(ocr_data["text"])):
#             token = ocr_data["text"][i].strip()
#             if not token:
#                 continue
#             x, y, w, h = (
#                 ocr_data["left"][i], ocr_data["top"][i],
#                 ocr_data["width"][i], ocr_data["height"][i],
#             )

#             # ── JUNK TOKEN FILTERS ──────────────────────────────────────────
#             # 1. Skip mostly-non-alphanumeric tokens (barcode bars "||||||||")
#             alnum_ratio = sum(c.isalnum() for c in token) / len(token)
#             if alnum_ratio < 0.4:
#                 continue

#             # 2. Skip single-char tokens — usually vertical text fragments
#             if len(token) == 1:
#                 continue

#             # 3. Skip tokens with extremely tall/narrow bounding boxes
#             # (vertical text like "48531196" printed sideways gets read as
#             #  thin tall boxes — these cause giant line bboxes when grouped)
#             if h > 0 and w > 0 and (h / w) > 4:
#                 continue
#             # ────────────────────────────────────────────────────────────────

#             words.append({
#                 "text": token,
#                 "x0": x * scale_x,
#                 "y0": y * scale_y,
#                 "x1": (x + w) * scale_x,
#                 "y1": (y + h) * scale_y,
#             })

#         lines = group_words_into_lines(words)
#         redacted_rects = []  # track to avoid double-annotating same area

#         def already_covered(rect):
#             for r in redacted_rects:
#                 if abs(r.x0 - rect.x0) < 5 and abs(r.y0 - rect.y0) < 5:
#                     return True
#             return False

#         def apply_rect(rect, rtype, matched_text):
#             if already_covered(rect):
#                 return
#             page.add_redact_annot(rect, fill=(0, 0, 0))
#             redacted_rects.append(rect)
#             audit_log.append({"page": page_num + 1, "type": rtype, "text": matched_text})
#             already_redacted.add(matched_text)

#         for line_words in lines:
#             line_text = " ".join(w["text"] for w in line_words)

#             # ── REGEX PATTERNS ──────────────────────────────────────────────
#             if redact_types:
#                 for rtype in redact_types:
#                     if rtype == "address":
#                         continue  # handled separately below
#                     if rtype not in patterns:
#                         continue

#                     for m in re.finditer(patterns[rtype], line_text, re.IGNORECASE):
#                         matched = m.group().strip()
#                         if not matched or matched in already_redacted:
#                             continue

#                         # Find which words in this line are part of the match
#                         match_start, match_end = m.start(), m.end()
#                         pos = 0
#                         matched_words = []
#                         for word in line_words:
#                             wlen = len(word["text"])
#                             word_start = pos
#                             word_end = pos + wlen
#                             # +1 for the space separator
#                             if word_start < match_end and word_end > match_start:
#                                 matched_words.append(word)
#                             pos += wlen + 1  # +1 for space

#                         if matched_words:
#                             rect = fitz.Rect(
#                                 min(w["x0"] for w in matched_words) - 1,
#                                 min(w["y0"] for w in matched_words) - 1,
#                                 max(w["x1"] for w in matched_words) + 1,
#                                 max(w["y1"] for w in matched_words) + 1,
#                             )
#                             apply_rect(rect, rtype, matched)

#             # ── ADDRESS: keyword-based whole-line redaction ─────────────────
#             if redact_types and "address" in redact_types:
#                 if line_has_address_keyword(line_words):
#                     rect = line_bbox(line_words)
#                     apply_rect(rect, "address", line_text.strip())

#             # ── CUSTOM TERMS: fuzzy match ───────────────────────────────────
#             if custom_terms:
#                 for term in custom_terms:
#                     term = term.strip()
#                     if not term or term in already_redacted:
#                         continue
#                     term_tokens = term.lower().split()
#                     nw = len(term_tokens)
#                     for i in range(len(line_words) - nw + 1):
#                         chunk = line_words[i:i + nw]
#                         chunk_text = " ".join(c["text"] for c in chunk)
#                         chunk_clean = re.sub(r'[^\w\s]', '', chunk_text).strip().lower()
#                         term_clean = re.sub(r'[^\w\s]', '', term).strip().lower()
#                         if SequenceMatcher(None, chunk_clean, term_clean).ratio() >= 0.82:
#                             rect = line_bbox(chunk)
#                             apply_rect(rect, "custom", f"{term} (OCR: {chunk_text})")
#                             break

#     except Exception as e:
#         print(f"  [ERROR] Scanned redaction failed on page {page_num + 1}: {e}")
#         import traceback
#         traceback.print_exc()


# # ─── TEXT-BASED PAGE REDACTION ─────────────────────────────────────────────────

# def redact_normal_page(page, redact_types, custom_terms, already_redacted, audit_log, page_num):
#     full_text = get_normalized_page_text(page)

#     if redact_types:
#         for rtype in redact_types:
#             if rtype in patterns:
#                 for match in re.finditer(patterns[rtype], full_text, re.IGNORECASE):
#                     matched_text = match.group().strip()
#                     if not matched_text or matched_text in already_redacted:
#                         continue
#                     add_redaction(page, matched_text)
#                     already_redacted.add(matched_text)
#                     audit_log.append({"page": page_num + 1, "type": rtype, "text": matched_text})

#     if custom_terms:
#         for term in custom_terms:
#             term = term.strip()
#             if not term or term in already_redacted:
#                 continue
#             add_redaction(page, term)
#             already_redacted.add(term)
#             audit_log.append({"page": page_num + 1, "type": "custom", "text": term})


# # ─── MAIN ENTRY POINT ──────────────────────────────────────────────────────────

# def redact_pdf(input_path, output_path, redact_types=None, custom_terms=None):
#     doc = fitz.open(input_path)
#     audit_log = []
#     redact_types = list(redact_types) if redact_types else []

#     # Auto-add pincode when address is selected
#     if "address" in redact_types and "pincode" not in redact_types:
#         redact_types.append("pincode")

#     for page_num, page in enumerate(doc):
#         print("\n====================")
#         print("PAGE:", page_num + 1)
#         print("SCANNED:", is_page_scanned(page))
#         print("CUSTOM TERMS:", custom_terms)
#         print("TEXT LENGTH:", len(page.get_text()))
#         print(page.get_text()[:500])
#         print("====================")

#         already_redacted = set()
#         scanned = is_page_scanned(page)
#         print(f"Page {page_num + 1}: {'[SCANNED → OCR]' if scanned else '[TEXT-BASED]'}")

#         if scanned:
#             redact_scanned_page_with_ocr(
#                 page, redact_types, custom_terms,
#                 already_redacted, audit_log, page_num
#             )
#         else:
#             redact_normal_page(
#                 page, redact_types, custom_terms,
#                 already_redacted, audit_log, page_num
#             )

#         page.apply_redactions()

#     doc.save(output_path, garbage=4, clean=True)
#     doc.close()

#     print("\nRedacted PDF saved!")
#     print(f"Output: {output_path}")
#     print("\nAudit Log:")
#     for item in audit_log:
#         print(f"  Page {item['page']} | {item['type']:15} | {item['text']}")

#     return output_path


# if __name__ == "__main__":
#     redact_pdf(
#         input_path="invoice.pdf",
#         output_path="redacted_output.pdf",
#         redact_types=["email", "phone", "date", "aadhaar", "pan", "address"],
#         custom_terms=[]
#     )














# import fitz
# import re
# import spacy
# import warnings

# warnings.filterwarnings("ignore") # Ignore warnings

# # LOAD SPACY MODEL

# nlp = spacy.load("en_core_web_sm")    

# # REGEX PATTERNS

# patterns = {
#     "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

#     # Indian + International phone formats
#     "phone": r"(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",

#     # $, ₹, Rs, EUR, GBP formats with decimals
#     "amount": r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|€\s?\d+[\d,]*\.?\d*|£\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",

#     # DD/MM/YYYY, MM-DD-YYYY, January 25 2016, 25 Jan 2016, 2016-01-25
#     "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}",

#     # Aadhaar formats: 1234 5678 9012 or 123456789012
#     "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b",

#     # PAN with word boundary
#     "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

#     # Credit card with/without spaces/dashes
#     "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",

#     # IFSC code
#     "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

#     # Account numbers (8-18 digits)
#     "account_number": r"\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",

#     # BSB (Australian bank)
#     "bsb": r"\bBSB\s*#?\s*\d{3}[\s-]?\d{3}\b",

#     # Street address patterns
#     "street_address": r"\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b",
# }

# # Whitelist of words to skip during NER redaction
# technical_words = set() 

# # REDACTION FUNCTION

#  # This function takes a PDF page, the text to be redacted, and an optional label for the type of information being redacted.
# def add_redaction(page, text, label=None):
#     areas = page.search_for(text)
#     for area in areas:
#         if label:
#             page.add_redact_annot(
#                 area,
#                 text=f"[REDACTED {label.upper()}]",
#                 fill=(0, 0, 0),
#                 text_color=(1, 1, 1)
#             )

#         else:

#             page.add_redact_annot(
#                 area,
#                 fill=(0, 0, 0)
#             )

# # MAIN PDF REDACTION

# def redact_pdf(
#     input_path,
#     output_path,
#     redact_types=None,
#     custom_terms=None
# ):

#     doc = fitz.open(input_path)   
#     audit_log = []
#     for page_num, page in enumerate(doc):
#         already_redacted = set()
#         full_text = page.get_text()

#         # REGEX REDACTION

#         if redact_types:
#             for rtype in redact_types:
#                 if rtype in patterns:
#                     matches = re.finditer(    # Use regex to find matches of the current redact type in the page text based on the defined patterns for each type
#                         patterns[rtype],
#                         full_text
#                     )
#     # For each regex match found, check if the matched text is valid and has not already been redacted, then apply the redaction annotation to the page using the add_redaction function.
#                     for match in matches:
#                         matched_text = match.group().strip()  # Get the matched text from the regex search and strip any extra whitespace for accurate redaction.
#                         if (
#                             not matched_text
#                             or matched_text in already_redacted
#                         ):
#                             continue
#                         add_redaction(
#                             page,
#                             matched_text,
#                             label=rtype
#                         )

#                         already_redacted.add(matched_text)

#                         audit_log.append({
#                             "page": page_num + 1,
#                             "type": rtype,
#                             "text": matched_text
#                         })

#         # NER REDACTION

#         if redact_types and (
#             "name" in redact_types
#             or "address" in redact_types
#         ):

#             doc_nlp = nlp(full_text[:10000])   # Process the page text with spaCy's NLP model to detect named entities. Limiting to first 10k characters for performance on large pages.

#             for ent in doc_nlp.ents:
#                 entity_text = ent.text.strip()

#                 # FILTERS
#                 if (
#                     len(entity_text) < 3
#                     or entity_text in already_redacted
#                     or entity_text in technical_words
#                 ):
#                     continue

#                 # Skip weird technical entities
#                 if re.search(r"[\d_=(){}[\]]", entity_text):   # If the entity text contains digits or special characters that are common in technical terms, skip redaction 
#                     continue

#                 text = entity_text.strip()

#                 # skip only obvious noise
#                 if len(text) < 2:
#                     continue

#                 # PERSON
#                 if (
#                     ent.label_ == "PERSON"
#                     and "name" in redact_types
#                 ):

#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="name"
#                     )

#                     already_redacted.add(entity_text)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "name",
#                         "text": entity_text
#                     })

#                 # ADDRESS
#                 elif (
#                     ent.label_ in ["GPE", "LOC", "FAC"]
#                     and "address" in redact_types
#                 ):

#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="address"
#                     )

#                     already_redacted.add(entity_text)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "address",
#                         "text": entity_text
#                     })

#         # CUSTOM TERMS

#         if custom_terms:
#             for term in custom_terms:
#                 term = term.strip()
#         # Skip empty terms or terms that have already been redacted to avoid unnecessary processing and redundant redactions
#                 if (
#                     not term
#                     or term in already_redacted
#                 ):
#                     continue

#                 add_redaction(
#                     page,
#                     term,
#                     label="custom"
#                 )

#                 already_redacted.add(term)

#                 audit_log.append({
#                     "page": page_num + 1,
#                     "type": "custom",
#                     "text": term
#                 })

#         # APPLY REDACTIONS
#         page.apply_redactions()

#     # SAVE PDF
#     doc.save(
#         output_path,
#         garbage=4,    # Optimize the PDF by removing unused objects and compressing the file, which can help reduce the file size after redaction.
#         clean=True
#     )

#     doc.close()

#     print("\nRedacted PDF saved successfully!")
#     print(f"Output File: {output_path}")

#     # AUDIT LOG

#     print("\nAudit Log:")

#     for item in audit_log:
#         print(item)

#     return output_path


# if __name__ == "__main__":

#     redact_pdf(
#         input_path="invoice.pdf",

#         output_path="redacted_output.pdf",

#         redact_types=[
#             "email",
#             "phone",
#             "amount",
#             "date",
#             "aadhaar",
#             "pan",
#             "credit_card",
#             "ifsc",
#             "name",
#             "address",
#             "street_address"
#         ],

#         custom_terms=[
#             "confidential",
#             "secret"
#         ]
#     )





# import fitz
# import re
# import spacy
# import warnings

# warnings.filterwarnings("ignore") # Ignore warnings

# # LOAD SPACY MODEL

# nlp = spacy.load("en_core_web_sm")    

# # REGEX PATTERNS

# patterns = {
#     "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

#     # Indian + International phone formats
#     "phone": r"(\+91[\-\s]?)?[6-9]\d{4}[\-\s]?\d{5}|\+?1?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}",

#     # $, ₹, Rs, EUR, GBP formats with decimals
#     "amount": r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|€\s?\d+[\d,]*\.?\d*|£\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",

#     # DD/MM/YYYY, MM-DD-YYYY, January 25 2016, 25 Jan 2016, 2016-01-25
#     "date": r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}",

#     # Aadhaar formats: 1234 5678 9012 or 123456789012
#     "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b|\b\d{12}\b",

#     # PAN with word boundary
#     "pan": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

#     # Credit card with/without spaces/dashes
#     "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",

#     # IFSC code
#     "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

#     # Account numbers (8-18 digits)
#     "account_number": r"\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",

#     # BSB (Australian bank)
#     "bsb": r"\bBSB\s*#?\s*\d{3}[\s-]?\d{3}\b",

#     # Street address patterns
#     "street_address": r"\d+\s+[A-Za-z]+(?:\s+[A-Za-z]+)*\s+(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way|Court|Ct|Place|Pl)\b",
# }

# # Whitelist of words to skip during NER redaction
# technical_words = set() 

# # REDACTION FUNCTION

#  # This function takes a PDF page, the text to be redacted, and an optional label for the type of information being redacted.
# def add_redaction(page, text, label=None):
#     areas = page.search_for(text)
#     for area in areas:
#         if label:
#             page.add_redact_annot(
#                 area,
#                 text=f"[REDACTED {label.upper()}]",
#                 fill=(0, 0, 0),
#                 text_color=(1, 1, 1)
#             )

#         else:

#             page.add_redact_annot(
#                 area,
#                 fill=(0, 0, 0)
#             )

# # MAIN PDF REDACTION

# def redact_pdf(
#     input_path,
#     output_path,
#     redact_types=None,
#     custom_terms=None
# ):

#     doc = fitz.open(input_path)   
#     audit_log = []
#     for page_num, page in enumerate(doc):
#         already_redacted = set()
#         full_text = page.get_text()

#         # REGEX REDACTION

#         if redact_types:
#             for rtype in redact_types:
#                 if rtype in patterns:
#                     matches = re.finditer(    # Use regex to find matches of the current redact type in the page text based on the defined patterns for each type
#                         patterns[rtype],
#                         full_text
#                     )
#     # For each regex match found, check if the matched text is valid and has not already been redacted, then apply the redaction annotation to the page using the add_redaction function.
#                     for match in matches:
#                         matched_text = match.group().strip()  # Get the matched text from the regex search and strip any extra whitespace for accurate redaction.
#                         if (
#                             not matched_text
#                             or matched_text in already_redacted
#                         ):
#                             continue
#                         add_redaction(
#                             page,
#                             matched_text,
#                             label=rtype
#                         )

#                         already_redacted.add(matched_text)

#                         audit_log.append({
#                             "page": page_num + 1,
#                             "type": rtype,
#                             "text": matched_text
#                         })

#         # NER REDACTION

#         if redact_types and (
#             "name" in redact_types
#             or "address" in redact_types
#         ):

#             doc_nlp = nlp(full_text[:10000])   # Process the page text with spaCy's NLP model to detect named entities. Limiting to first 10k characters for performance on large pages.

#             for ent in doc_nlp.ents:
#                 entity_text = ent.text.strip()

#                 # FILTERS
#                 if (
#                     len(entity_text) < 3
#                     or entity_text in already_redacted
#                     or entity_text in technical_words
#                 ):
#                     continue

#                 # Skip weird technical entities
#                 if re.search(r"[\d_=(){}[\]]", entity_text):   # If the entity text contains digits or special characters that are common in technical terms, skip redaction 
#                     continue

#                 text = entity_text.strip()

#                 # skip only obvious noise
#                 if len(text) < 2:
#                     continue

#                 # PERSON
#                 if (
#                     ent.label_ == "PERSON"
#                     and "name" in redact_types
#                 ):

#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="name"
#                     )

#                     already_redacted.add(entity_text)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "name",
#                         "text": entity_text
#                     })

#                 # ADDRESS
#                 elif (
#                     ent.label_ in ["GPE", "LOC", "FAC"]
#                     and "address" in redact_types
#                 ):

#                     add_redaction(
#                         page,
#                         entity_text,
#                         label="address"
#                     )

#                     already_redacted.add(entity_text)

#                     audit_log.append({
#                         "page": page_num + 1,
#                         "type": "address",
#                         "text": entity_text
#                     })

#         # CUSTOM TERMS

#         if custom_terms:
#             for term in custom_terms:
#                 term = term.strip()
#         # Skip empty terms or terms that have already been redacted to avoid unnecessary processing and redundant redactions
#                 if (
#                     not term
#                     or term in already_redacted
#                 ):
#                     continue

#                 add_redaction(
#                     page,
#                     term,
#                     label="custom"
#                 )

#                 already_redacted.add(term)

#                 audit_log.append({
#                     "page": page_num + 1,
#                     "type": "custom",
#                     "text": term
#                 })

#         # APPLY REDACTIONS
#         page.apply_redactions()

#     # SAVE PDF
#     doc.save(
#         output_path,
#         garbage=4,    # Optimize the PDF by removing unused objects and compressing the file, which can help reduce the file size after redaction.
#         clean=True
#     )

#     doc.close()

#     print("\nRedacted PDF saved successfully!")
#     print(f"Output File: {output_path}")

#     # AUDIT LOG

#     print("\nAudit Log:")

#     for item in audit_log:
#         print(item)

#     return output_path


# if __name__ == "__main__":

#     redact_pdf(
#         input_path="invoice.pdf",

#         output_path="redacted_output.pdf",

#         redact_types=[
#             "email",
#             "phone",
#             "amount",
#             "date",
#             "aadhaar",
#             "pan",
#             "credit_card",
#             "ifsc",
#             "name",
#             "address",
#             "street_address"
#         ],

#         custom_terms=[
#             "confidential",
#             "secret"
#         ]
#     )