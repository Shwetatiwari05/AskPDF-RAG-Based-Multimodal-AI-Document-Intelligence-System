"""
redactor.py — PDF Redaction Engine

TWO PATHS:
  Text PDF    → PyMuPDF search_for() → exact coords → redact          (unchanged)
  Scanned PDF → EasyOCR segments → split into WORD boxes (pixel gaps)
                → regex / custom-term / address matching on the words
                → black boxes burned into the page image → small JPEG

WHAT GETS REDACTED:
  email, phone, money, date, aadhaar, pan, credit_card, ifsc, account_number,
  address (street + city + state + country + pincode), custom terms

SCANNED-PAGE RULES:
  * Only the matched words are covered, never a whole OCR line / segment.
  * A match that spans several lines gets one small box per line (no big union box).
  * Boxes are tight to the ink, so neighbouring lines are not clipped.
  * Lines that are clearly an address (2+ address cues) are covered fully.
  * The redaction is burned into the pixels; the original scan is not kept in
    the output file, so nothing can be recovered by removing an overlay.
  * If OCR fails on a scanned page, an error is raised (the page is never
    silently left unredacted).
"""

import io
import os
import re
import warnings
from difflib import SequenceMatcher

import fitz
import numpy as np
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
    "pincode": r"\b\d{6}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ifsc": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
    "money": r"₹\s?\d+[\d,]*\.?\d*|Rs\.?\s?\d+[\d,]*|\$\s?\d+[\d,]*\.?\d*|\d+[\d,]*\.?\d*\s?(?:USD|INR|EUR|GBP)",
    "account_number": r"\bBSB\s*#?\s*\d[\d\s\-]{4,8}\b|\bACC\s*#?\s*\d[\d\s]{6,17}\b|\bA/C\s*#?\s*\d[\d\s]{6,17}\b",
    # address regex — catches "123 Somewhere St", "Suite 5A-1204", "Flat 4B", "Plot No. 7"
    "address": r"(?i)\b(?:suite|apt|apartment|flat|floor|unit|plot|house|room|no\.?)\s+[\w\-]+|\b\d+[A-Za-z]?\s+[\w\s]{2,30}(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|blvd|boulevard|way|court|ct|place|pl|nagar|marg|colony|sector)\b",
}

# Scanned pages only: Indian numbers are often printed as "+91 9898 989898",
# which the generic phone pattern above does not catch.
IMAGE_PATTERN_OVERRIDES = {
    "phone": PATTERNS["phone"]
    + r"|(?<!\d)(?:\+\s?91[\s\-]?)?[6-9]\d{3,4}[\s\-]?\d{5,6}(?!\d)",
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

# Scanned pages only. Structural words (ROAD, STATE, MAIN ...) only COUNT as
# address cues, they are never redacted on their own. Place names (cities,
# states, countries) are redacted word-by-word, or the whole line when the line
# is clearly an address.
_STRUCTURAL = {
    'COLONY','MARG','ROAD','LANE','NAGAR','WEST','EAST','NORTH','SOUTH',
    'SECTOR','BLOCK','STREET','PLOT','FLAT','HOUSE','APARTMENT','APT',
    'BUILDING','BLDG','NEAR','OPP','LAYOUT','PHASE','CROSS','MAIN','PARK',
    'SOCIETY','COMPLEX','TOWER','VTC','CINEMA','EXTENSION','STATE','DISTRICT',
    'SUBDISTRICT','TALUKA','TEHSIL','VILLAGE','PO','SUB DIVISION','SUBDIVISION',
}
_EXTRA_STRUCTURAL = {'UNIT', 'ESTATE', 'INDUSTRIAL'}   # cues only, image path
IMG_STRUCTURAL_CUES = _STRUCTURAL | _EXTRA_STRUCTURAL
IMG_PLACE_WORDS = ADDRESS_KEYWORDS - _STRUCTURAL
# A country on its own ("Government of India") is not redacted; it is covered
# only when it is part of an address line. Cities and states are redacted alone.
_COUNTRY_WORDS = {'INDIA','USA','UK','ENGLAND','SCOTLAND','WALES','AUSTRALIA',
                  'CANADA','GERMANY','FRANCE','CHINA','JAPAN','SINGAPORE','UAE'}
IMG_PLACE_SINGLE = IMG_PLACE_WORDS - _COUNTRY_WORDS

# ── SCANNED-PAGE SETTINGS ─────────────────────────────────────────────────────
OUT_DPI = 200            # resolution of the burned-in page image
OUT_MAX_PX = 3200        # longest side of that image, in pixels
JPEG_QUALITY = 82
OCR_MIN_CONF = 0.20      # keep low-confidence text too: better to over-redact
WORD_GAP_FACTOR = 0.25   # gap (x text height) that separates two words
ADDRESS_LINE_MIN_CUES = 2


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


# ── SENSITIVE STRING COLLECTION (text pages) ──────────────────────────────────

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
        langs = [l.strip() for l in os.getenv("REDACT_OCR_LANGS", "en").split(",") if l.strip()]
        _easyocr_reader = easyocr.Reader(langs or ['en'], gpu=False)
    return _easyocr_reader


def _ocr_segments(img: np.ndarray) -> list:
    """
    Run EasyOCR on an RGB uint8 image (H x W x 3).
    Returns segments: [{"text": str, "bbox": (x0, y0, x1, y1) in image pixels, "conf": float}]
    EasyOCR returns phrase/line-level segments, not single words; the words are
    recovered later in _split_segment().
    """
    h, w = img.shape[:2]
    segs = []
    for (poly, text, conf) in get_easyocr_reader().readtext(img):
        if conf < OCR_MIN_CONF or not str(text).strip():
            continue
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        segs.append({
            "text": str(text),
            "bbox": (max(0.0, min(xs)), max(0.0, min(ys)),
                     min(float(w), max(xs)), min(float(h), max(ys))),
            "conf": float(conf),
        })
    return segs


# ── SEGMENT → WORD BOXES ──────────────────────────────────────────────────────

def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = sum_b = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best:
            best, thr = var, t
    return int(thr)


def _runs(mask: np.ndarray) -> list:
    """Runs of True in a 1-D boolean array -> [(start, end_exclusive), ...]"""
    if mask.size == 0 or not mask.any():
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _merge_close_runs(runs: list, max_gap: int) -> list:
    if not runs:
        return []
    merged = [list(runs[0])]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(r) for r in merged]


def _proportional_spans(tokens: list, xa: int, xb: int) -> list:
    """Fallback: split [xa, xb] between the tokens by character count."""
    weights = [max(1, len(t)) for t in tokens]
    space = 0.5
    total = sum(weights) + space * (len(tokens) - 1)
    unit = (xb - xa) / total if total else 0
    spans, pos = [], float(xa)
    for wgt in weights:
        spans.append((int(round(pos)), int(round(pos + wgt * unit))))
        pos += wgt * unit + space * unit
    return spans


def _split_segment(gray: np.ndarray, seg: dict) -> list:
    """
    Split one OCR segment into word boxes with TIGHT ink bounds.
    Uses the white gaps between words in the pixels; falls back to a
    character-proportional split if the gap count does not match the text.
    """
    H, W = gray.shape
    x0, y0, x1, y1 = [int(round(v)) for v in seg["bbox"]]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    tokens = seg["text"].split()
    if not tokens or x1 - x0 < 2 or y1 - y0 < 2:
        return []

    crop = gray[y0:y1, x0:x1]
    box_fallback = lambda spans: [
        {"text": t, "bbox": (x0 + a, y0, x0 + b, y1)} for t, (a, b) in zip(tokens, spans)
    ]

    if float(crop.std()) < 10:          # blank / flat area: no ink to measure
        return box_fallback(_proportional_spans(tokens, 0, x1 - x0))

    ink = crop < _otsu_threshold(crop)

    # keep the main text band (drops neighbouring lines that the box overlaps)
    row_runs = _merge_close_runs(_runs(ink.sum(axis=1) >= 2), 3)
    if not row_runs:
        return box_fallback(_proportional_spans(tokens, 0, x1 - x0))
    ra, rb = max(row_runs, key=lambda r: int(ink[r[0]:r[1]].sum()))
    band = ink[ra:rb]
    h_text = rb - ra

    col_thr = max(1, int(0.05 * h_text))
    col_runs = _runs(band.sum(axis=0) >= col_thr)
    if not col_runs:
        return box_fallback(_proportional_spans(tokens, 0, x1 - x0))

    gap_thr = max(3, WORD_GAP_FACTOR * h_text)
    groups = [[col_runs[0]]]
    for prev, cur in zip(col_runs, col_runs[1:]):
        if cur[0] - prev[1] >= gap_thr:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    spans = [(g[0][0], g[-1][1]) for g in groups]

    n = len(tokens)
    if len(spans) > n:                  # too many pieces: merge across the smallest gaps
        while len(spans) > n:
            gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
            j = int(np.argmin(gaps))
            spans[j] = (spans[j][0], spans[j + 1][1])
            del spans[j + 1]
    elif len(spans) < n:                # too few pieces: split by character count
        spans = _proportional_spans(tokens, spans[0][0], spans[-1][1])

    words = []
    for tok, (a, b) in zip(tokens, spans):
        a, b = max(0, a), max(a + 1, b)
        sub = band[:, a:b]
        rr = np.where(sub.sum(axis=1) >= 1)[0]
        wy0, wy1 = (ra + int(rr[0]), ra + int(rr[-1]) + 1) if rr.size else (ra, rb)
        words.append({"text": tok, "bbox": (x0 + a, y0 + wy0, x0 + b, y0 + wy1)})
    return words


def _collect_words(gray: np.ndarray, segs: list) -> list:
    """All words in reading order, each tagged with its segment id and row id."""
    order = sorted(range(len(segs)),
                   key=lambda i: (segs[i]["bbox"][1] + segs[i]["bbox"][3]) / 2)
    rows = []
    for i in order:
        x0, y0, x1, y1 = segs[i]["bbox"]
        cy, h = (y0 + y1) / 2, max(1.0, y1 - y0)
        for r in rows:
            if abs(cy - r["cy"]) <= 0.6 * min(h, r["h"]):
                r["segs"].append(i)
                k = len(r["segs"])
                r["cy"] += (cy - r["cy"]) / k
                r["h"] += (h - r["h"]) / k
                break
        else:
            rows.append({"cy": cy, "h": h, "segs": [i]})
    rows.sort(key=lambda r: r["cy"])

    words = []
    for ridx, r in enumerate(rows):
        for si in sorted(r["segs"], key=lambda i: segs[i]["bbox"][0]):
            for w in _split_segment(gray, segs[si]):
                w["seg"], w["row"] = si, ridx
                words.append(w)
    return words


def _match_text(t: str) -> str:
    """Text used for pattern matching. A day number followed by OCR junk
    (superscript 'th' read as quotes/marks, e.g. 28'") becomes just the number."""
    m = re.fullmatch(r'(\d{1,2})[^\w\s]{1,3}', t)
    return m.group(1) if m else t


def _build_stream(words: list):
    """
    Join the words into one string for regex matching and remember each word's
    character span. Words from far-apart columns on the same row are separated
    by ' ¦ ' so a pattern cannot run across two columns.
    """
    parts, spans, pos = [], [], 0
    for i, w in enumerate(words):
        if i:
            p = words[i - 1]
            if w["row"] != p["row"]:
                sep = "\n"
            elif w["seg"] == p["seg"]:
                sep = " "
            else:
                gap = w["bbox"][0] - p["bbox"][2]
                h = max(1, p["bbox"][3] - p["bbox"][1])
                sep = " " if gap <= 1.2 * h else " ¦ "
            parts.append(sep)
            pos += len(sep)
        mt = _match_text(w["text"])
        spans.append((pos, pos + len(mt)))
        parts.append(mt)
        pos += len(mt)
    return "".join(parts), spans


# ── MATCHING ON WORDS ─────────────────────────────────────────────────────────

def _clean_tok(t: str) -> str:
    # strips spaces/punctuation only, so non-English scripts (e.g. Hindi) still compare
    return re.sub(r"[\s.,;:!?\-_/\\()\[\]{}\"'|<>*#@&+=~`^%$]", "", t.lower())


def _words_for_span(spans: list, start: int, end: int) -> list:
    return [i for i, (s, e) in enumerate(spans) if s < end and e > start]


def _match_words(words: list, stream: str, spans: list,
                 redact_types: list, custom_terms: list) -> dict:
    """Returns {word_index: label} for every word that must be covered."""
    hit = {}

    # 1. regex types (on the joined text, so "28th May 2026", e-mails and phone
    #    numbers are found even when OCR split them into several words)
    for rtype in redact_types:
        if rtype == "address":
            continue
        pat = IMAGE_PATTERN_OVERRIDES.get(rtype) or PATTERNS.get(rtype)
        if not pat:
            continue
        for m in re.finditer(pat, stream, re.IGNORECASE):
            if not m.group(0).strip():
                continue
            for i in _words_for_span(spans, m.start(), m.end()):
                hit.setdefault(i, rtype)

    # 2. address: a line that is clearly an address is covered fully, and so are
    #    the weaker lines directly above/below it (e.g. "E/13, NCH COLONY," above
    #    "KANJURMARG WEST,"). Other lines: only city / state names.
    if "address" in redact_types:
        by_seg = {}
        for i, w in enumerate(words):
            by_seg.setdefault(w["seg"], []).append(i)
        cues, kw_cues, box = {}, {}, {}
        for sid, idxs in by_seg.items():
            seg_text = " ".join(words[i]["text"] for i in idxs)
            c = set()
            for i in idxs:
                tok = re.sub(r'[^A-Za-z]', '', words[i]["text"]).upper()
                if tok in IMG_STRUCTURAL_CUES or tok in IMG_PLACE_WORDS:
                    c.add(tok)
            if re.search(PATTERNS["address"], seg_text):
                c.add("<addr>")
            kw_cues[sid] = len(c)
            if re.search(PATTERNS["pincode"], seg_text):
                c.add("<pin>")
            cues[sid] = len(c)
            box[sid] = (min(words[i]["bbox"][0] for i in idxs), min(words[i]["bbox"][1] for i in idxs),
                        max(words[i]["bbox"][2] for i in idxs), max(words[i]["bbox"][3] for i in idxs))

        def adjacent(a, b):
            ax0, ay0, ax1, ay1 = box[a]
            bx0, by0, bx1, by1 = box[b]
            overlap = min(ax1, bx1) - max(ax0, bx0)
            if overlap < 0.3 * min(ax1 - ax0, bx1 - bx0):
                return False
            gap = max(0, max(ay0, by0) - min(ay1, by1))
            return gap <= 1.6 * max(ay1 - ay0, by1 - by0)

        addr = {sid for sid, c in cues.items() if c >= ADDRESS_LINE_MIN_CUES}
        grew = True
        while grew:
            grew = False
            for sid, c in cues.items():
                if sid not in addr and kw_cues[sid] >= 1 and any(adjacent(sid, a) for a in addr):
                    addr.add(sid)
                    grew = True

        for sid, idxs in by_seg.items():
            if sid in addr:
                for i in idxs:
                    hit.setdefault(i, "address")
            else:
                for i in idxs:
                    tok = re.sub(r'[^A-Za-z]', '', words[i]["text"]).upper()
                    if tok in IMG_PLACE_SINGLE:
                        hit.setdefault(i, "address")

    # 3. custom terms (names, codenames): case-insensitive, tolerant to OCR typos,
    #    may continue on the next line
    norm = [_clean_tok(w["text"]) for w in words]
    for term in custom_terms:
        toks = [t for t in (_clean_tok(x) for x in term.split()) if t]
        n = len(toks)
        if not n:
            continue
        for i in range(len(words) - n + 1):
            ok = True
            for j, tk in enumerate(toks):
                cand = norm[i + j]
                if not cand:
                    ok = False
                    break
                if len(tk) <= 3:
                    good = cand == tk
                else:
                    need = 0.90 if len(tk) == 4 else 0.84
                    good = SequenceMatcher(None, tk, cand).ratio() >= need
                if not good:
                    ok = False
                    break
            if ok:
                for j in range(n):
                    hit.setdefault(i + j, "custom")
    return hit


def _boxes_from_hits(words: list, hit: dict) -> list:
    """
    Group hit words into small boxes: one per segment piece, split again when
    two hit words are far apart. Returns [(x0, y0, x1, y1, label, text)].
    """
    boxes, cur, prev_i = [], None, None
    for i in sorted(hit):
        w = words[i]
        x0, y0, x1, y1 = w["bbox"]
        h = max(1, y1 - y0)
        if (cur and prev_i == i - 1 and cur["seg"] == w["seg"]
                and x0 - cur["x1"] <= 3 * h and cur["row"] == w["row"]):
            cur["x0"], cur["y0"] = min(cur["x0"], x0), min(cur["y0"], y0)
            cur["x1"], cur["y1"] = max(cur["x1"], x1), max(cur["y1"], y1)
            cur["text"].append(w["text"])
            prev_i = i
        else:
            prev_i = i
            if cur:
                boxes.append(cur)
            cur = {"seg": w["seg"], "row": w["row"], "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                   "label": hit[i], "text": [w["text"]]}
    if cur:
        boxes.append(cur)
    return [(b["x0"], b["y0"], b["x1"], b["y1"], b["label"], " ".join(b["text"])) for b in boxes]


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


# ── IMAGE (SCANNED) PAGE REDACTION ────────────────────────────────────────────

def _render_page(page: fitz.Page) -> np.ndarray:
    """Render the page to an RGB uint8 array (the same image is used for OCR and for the output)."""
    zoom = min(OUT_DPI / 72.0, OUT_MAX_PX / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def redact_image_page(page: fitz.Page, page_num: int,
                      redact_types: list, custom_terms: list, audit_log: list):
    """
    Redact one scanned page.
    Returns JPEG bytes of the page with the black boxes burned in, or None if
    nothing had to be redacted (the caller then keeps the original page).
    Raises RuntimeError if OCR fails, so a scanned page is never left unredacted by accident.
    """
    print(f"  Running EasyOCR on page {page_num+1}...")
    img = _render_page(page)
    try:
        segs = _ocr_segments(img)
    except Exception as e:
        raise RuntimeError(f"OCR failed on scanned page {page_num+1}: {e}") from e

    if not segs:
        print(f"  [WARN] No text detected on page {page_num+1}")
        return None

    gray = np.asarray(
        0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2], dtype=np.uint8
    )
    words = _collect_words(gray, segs)
    stream, spans = _build_stream(words)
    hit = _match_words(words, stream, spans, redact_types, custom_terms)
    boxes = _boxes_from_hits(words, hit)
    print(f"  Segments: {len(segs)} | Words: {len(words)} | Boxes: {len(boxes)}")

    if not boxes:
        return None

    H, W = img.shape[:2]
    for (x0, y0, x1, y1, label, text) in boxes:
        h = max(1, y1 - y0)
        pad_x, pad_y = max(3, round(0.12 * h)), max(1, round(0.03 * h))
        a, b = max(0, x0 - pad_x), max(0, y0 - pad_y)
        c, d = min(W, x1 + pad_x), min(H, y1 + pad_y)
        img[b:d, a:c] = 0
        audit_log.append({"page": page_num+1, "type": label, "text": text})
        print(f"  ✓ [{label}] {text[:60]}")

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=JPEG_QUALITY,
                              optimize=True, dpi=(OUT_DPI, OUT_DPI))
    return buf.getvalue()


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

    out = fitz.open()
    for page_num, page in enumerate(doc):
        if page_num in image_pages:
            print(f"\nPage {page_num+1}: [SCANNED → EasyOCR]")
            jpeg = redact_image_page(page, page_num, redact_types, custom_terms, audit_log)
            if jpeg is None:
                out.insert_pdf(doc, from_page=page_num, to_page=page_num)
            else:
                # New page with the original size that holds ONLY the redacted image,
                # so the original (unredacted) scan is not carried into the output.
                new_page = out.new_page(width=page.rect.width, height=page.rect.height)
                new_page.insert_image(new_page.rect, stream=jpeg)
        else:
            print(f"\nPage {page_num+1}: [TEXT → PyMuPDF]")
            redact_text_page(page, page_num, redact_types, custom_terms, audit_log)
            out.insert_pdf(doc, from_page=page_num, to_page=page_num)

    out.set_metadata({})
    out.save(output_path, garbage=4, deflate=True)
    out.close()
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