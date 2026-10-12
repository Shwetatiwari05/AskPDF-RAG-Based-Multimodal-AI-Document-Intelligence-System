"""
Scanned-page redaction tests — synthetic pages built with Pillow
(no network except EasyOCR's own model download).

Covers: dates in several formats, a +91 phone, a multi-line address
block, a person name on its own line and inside a sentence, a
bilingual (Devanagari) name line, small text, and a heading that
must stay readable.
"""
import io
import os
import re
import sys

os.environ.setdefault("REDACT_OCR_LANGS", "en,hi")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from PIL import Image, ImageDraw, ImageFont

from features import redactor as R

ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
DEV = "/System/Library/Fonts/Supplemental/DevanagariMT.ttc"
W, H = 1240, 1754          # A4 at 150 dpi

REDACT_TYPES = ["phone", "date", "aadhaar", "address"]
CUSTOM_TERMS = ["Shweta tiwari", "श्वेता तिवारी"]


def _font(size, dev=False):
    return ImageFont.truetype(DEV if dev else ARIAL, size)


def _make_page_image():
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    y = [110]

    def line(text, size=26, gap=58, dev=False):
        dr.text((110, y[0]), text, font=_font(size, dev), fill="black")
        y[0] += gap

    line("Government of India", 40, 90)                    # stays readable
    line("Confidential", 20, 46)
    line("Shweta Tiwari", 30)                             # name, own line
    line("Dear Shweta Tiwari, your application was selected.", 24)
    line("DOB 31/12/2005", 24)
    line("Joined 31-12-2005", 24)
    line("Start 31.12.2005", 24)
    line("Pin 3112/2005", 24)
    line("Phone +91 98989 89898", 24)
    line("E/13, NCH COLONY,", 24, 34)                  # address block
    line("L B S MARG,", 24, 34)
    line("NEAR HUMA CINEMA,", 24, 34)
    line("KANJURMARG WEST,", 24, 34)
    line("MUMBAI, MAHARASHTRA 400078", 24, 34)
    line("Small print: Aadhaar 6533 3040 0575", 12, 34)   # small text
    line("श्वेता तिवारी", 26, dev=True)                   # bilingual line
    return img


def _save_pdf(img, path):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(path)
    doc.close()


def _ocr_text(img):
    res = R.get_easyocr_reader().readtext(img)
    return " ".join(t for _p, t, c in res if c >= R.OCR_MIN_CONF and t.strip())


def test_synthetic_scanned_page(tmp_path):
    src = str(tmp_path / "scan.pdf")
    out = str(tmp_path / "redacted.pdf")
    _save_pdf(_make_page_image(), src)

    R.redact_pdf(src, out, redact_types=REDACT_TYPES,
                 custom_terms=CUSTOM_TERMS)

    doc = fitz.open(out)
    page = doc[0]

    # the redacted page is one full-page image (the original scan
    # is not carried into the output)
    imgs = page.get_images(full=True)
    assert imgs, "output page is not an image page"
    r = page.get_image_rects(imgs[0][0])[0]
    assert abs(r.width - page.rect.width) < 1
    assert abs(r.height - page.rect.height) < 1

    # every redaction box is inside the page
    srcdoc = fitz.open(src)
    img = R._render_page(srcdoc[0])
    segs = R._ocr_segments(img, rotation=False)
    gray = R._render_gray(img)
    words = R._collect_words(gray, segs)
    stream, spans = R._build_stream(words)
    hit, match_groups = R._match_words(words, stream, spans, REDACT_TYPES, CUSTOM_TERMS)
    boxes = R._boxes_from_hits(words, hit, match_groups)
    assert boxes, "no boxes found on the synthetic page"
    ih, iw = img.shape[:2]
    for box in boxes:
        x0, y0, x1, y1 = box[:4]
        assert 0 <= x0 < x1 <= iw, f"box x out of page: {x0},{x1}"
        assert 0 <= y0 < y1 <= ih, f"box y out of page: {y0},{y1}"
    srcdoc.close()

    # fresh OCR of the redacted output
    out_img = R._render_page(page)
    text = _ocr_text(out_img)
    norm = re.sub(r"[^a-z0-9]", "", text.lower())

    # non-sensitive text is still readable
    assert "government" in norm, f"heading lost: {text!r}"
    assert "selected" in norm, f"sentence lost: {text!r}"

    # sensitive strings cannot be read any more
    for s in ("shweta", "tiwari", "श्वेता", "colony", "marg", "nch",
              "kanjurmarg", "cinema", "mumbai", "maharashtra",
              "2005", "98989", "89898", "6533", "3040", "0575"):
        assert s not in norm, f"still readable: {s!r}"
    assert "श्वेता" not in text, "devanagari name still readable"
    assert not re.search(R.IMAGE_PATTERN_OVERRIDES["date"], text), \
        f"date still readable: {text!r}"
    assert not re.search(R.IMAGE_PATTERN_OVERRIDES["phone"], text), \
        f"phone still readable: {text!r}"

    # the output stays small (burned-in JPEG page)
    assert os.path.getsize(out) < 2 * 1024 * 1024, \
        f"output too large: {os.path.getsize(out)}"
    doc.close()
