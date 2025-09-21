import io
import re
import sys
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional

import pandas as pd
import pytesseract
import requests
from ics import Calendar, Event
from PIL import Image, ImageOps

# Configuration
IMAGE_URL = "https://telugucalendar.org/calendar/2025/chicago/chicago-2025-9.png"
# If you already have a local file, set LOCAL_IMAGE_PATH to that path and leave IMAGE_URL = None
LOCAL_IMAGE_PATH: Optional[str] = None  # e.g. "./calendar.png"
# process both languages
OCR_LANGS = ["eng", "tel"]  # runs OCR for English and Telugu
OUTPUT_ICS = "output_calendar.ics"
OUTPUT_CSV = "output_calendar.csv"


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


def open_local_image(path: str) -> Image.Image:
    return Image.open(path)


def preprocess_image(img: Image.Image) -> Image.Image:
    # Convert to grayscale and increase contrast for better OCR
    gray = ImageOps.grayscale(img)
    # Optional binarization
    bw = gray.point(lambda x: 0 if x < 200 else 255, "1")
    # Return a higher-resolution image to improve OCR accuracy
    return bw.resize((bw.width * 2, bw.height * 2), Image.LANCZOS)


def run_ocr(img: Image.Image, langs: List[str] = None) -> Dict[str, str]:
    """
    Run Tesseract for each language in langs and return a dict:
      { 'eng': '...', 'tel': '...', 'combined': 'eng_text\\n\\ntel_text' }
    """
    if langs is None:
        langs = ["eng"]

    results: Dict[str, str] = {}
    for l in langs:
        try:
            text = pytesseract.image_to_string(img, lang=l)
        except pytesseract.TesseractError as e:
            text = ""
            print(f"Warning: Tesseract failed for lang='{l}': {e}", file=sys.stderr)
        results[l] = text or ""

    # combined: prefer Telugu text if present, but include both for debugging
    combined = []
    if results.get("eng"):
        combined.append(f"[ENGLISH OCR]\n{results['eng'].strip()}")
    if results.get("tel"):
        combined.append(f"[TELUGU OCR]\n{results['tel'].strip()}")
    results["combined"] = "\n\n".join(combined).strip()
    return results


def split_into_date_blocks(raw_text: str) -> List[Dict]:
    """
    Heuristic: split OCR text into blocks per date.
    We look for lines that start with a day number (1-31) or contain a standalone day number.
    If no headers found, fallback to splitting into 30 equal blocks.
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() != ""]
    date_header_re = re.compile(r"^(?:Day\s*)?(\b[1-9]\b|\b[1-2][0-9]\b|\b3[01]\b)\b", re.IGNORECASE)
    blocks: List[Dict] = []
    current_block_lines: List[str] = []
    current_day: Optional[int] = None

    for ln in lines:
        m = date_header_re.match(ln)
        if m:
            # start new block
            if current_block_lines:
                blocks.append({"day": current_day, "text": "\n".join(current_block_lines)})
            current_day = int(m.group(1))
            # include the header line but remove the day number for cleaner text
            rest = date_header_re.sub("", ln).strip()
            current_block_lines = [rest] if rest else []
        else:
            current_block_lines.append(ln)

    if current_block_lines:
        blocks.append({"day": current_day, "text": "\n".join(current_block_lines)})

    # If no day numbers detected, fallback: split into 30 parts
    if all(b["day"] is None for b in blocks) or not blocks:
        raw_lines = [ln for ln in raw_text.splitlines() if ln.strip() != ""]
        n = 30
        chunk_size = max(1, len(raw_lines) // n)
        blocks = []
        day = 1
        for i in range(0, len(raw_lines), chunk_size):
            chunk = raw_lines[i : i + chunk_size]
            blocks.append({"day": day if day <= 31 else None, "text": "\n".join(chunk)})
            day += 1

    # Fill missing day numbers by sequence where possible
    last_day = None
    for b in blocks:
        if b["day"] is None and last_day is not None:
            b["day"] = last_day + 1
        last_day = b["day"]

    return blocks


def extract_fields_from_block(block_text: str) -> Dict:
    """
    Try to extract tithi, nakshatra, and times using simple regex heuristics.
    Keep everything flexible — if not found, leave empty and keep raw text.
    """
    # common time formats
    time_re = re.compile(r"(\d{1,2}[:.]\d{2})\s*(AM|PM|am|pm)?")
    # keywords for tithi/nakshatra (English heuristics)
    tithi_re = re.compile(r"(Tithi[:\s-]*([A-Za-z0-9 \-]+))", re.IGNORECASE)
    nak_re = re.compile(r"(Nakshatra[:\s-]*([A-Za-z0-9 \-]+))", re.IGNORECASE)

    times = time_re.findall(block_text)
    times_extracted = ["".join(t).strip() for t in times] if times else []

    tithi_m = tithi_re.search(block_text)
    nak_m = nak_re.search(block_text)

    tithi = tithi_m.group(2).strip() if tithi_m else ""
    nakshatra = nak_m.group(2).strip() if nak_m else ""

    return {
        "raw": block_text,
        "tithi": tithi,
        "nakshatra": nakshatra,
        "times": "; ".join(times_extracted),
    }


def build_calendar_and_csv(blocks: List[Dict], year: int, month: int) -> None:
    rows = []
    cal = Calendar()

    for b in blocks:
        day = b.get("day")
        if not isinstance(day, int) or day < 1 or day > 31:
            continue
        fields = extract_fields_from_block(b["text"])
        try:
            evt_date = date(year, month, day)
        except Exception:
            # skip invalid dates
            continue

        # create ICS event (all-day event with details in description)
        e = Event()
        e.name = f"Telugu Panchangam {evt_date.isoformat()}"
        e.begin = evt_date.isoformat()
        desc_lines = []
        if fields["tithi"]:
            desc_lines.append(f"Tithi: {fields['tithi']}")
        if fields["nakshatra"]:
            desc_lines.append(f"Nakshatra: {fields['nakshatra']}")
        if fields["times"]:
            desc_lines.append(f"Times: {fields['times']}")
        desc_lines.append("Raw OCR:\n" + fields["raw"])
        e.description = "\n".join(desc_lines)
        cal.events.add(e)

        rows.append(
            {
                "date": evt_date.isoformat(),
                "tithi": fields["tithi"],
                "nakshatra": fields["nakshatra"],
                "times": fields["times"],
                "raw_text": fields["raw"],
            }
        )

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    # save ICS
    with open(OUTPUT_ICS, "w", encoding="utf-8") as f:
        f.writelines(cal)


def main():
    try:
        if LOCAL_IMAGE_PATH:
            img = open_local_image(LOCAL_IMAGE_PATH)
        else:
            img = download_image(IMAGE_URL)

        proc = preprocess_image(img)
        print("Running OCR for languages:", OCR_LANGS)
        ocr_results = run_ocr(proc, langs=OCR_LANGS)

        # prefer Telugu combined if you want Telugu-first parsing; use 'combined' for full output
        raw_text = ocr_results.get("combined", "")
        if not raw_text.strip():
            print("Warning: OCR returned empty text.", file=sys.stderr)

        # optional: debug print of per-language outputs
        print("\n--- OCR per language (short preview) ---")
        for lang in OCR_LANGS:
            preview = (ocr_results.get(lang) or "").strip()[:1000]
            print(f"\n[{lang}] preview:\n{preview}\n")
        print("--- end preview ---\n")

        blocks = split_into_date_blocks(raw_text)
        # Guess month/year from image URL if possible (simple heuristic)
        m = re.search(r"(\d{4}).*?[-_/](\d{1,2})", IMAGE_URL or "") if IMAGE_URL else None
        if m:
            year = int(m.group(1))
            month = int(m.group(2))
        else:
            # fallback: ask user or default to current month
            today = datetime.now()
            year = today.year
            month = today.month

        # Print parsed output for each block before building calendar/CSV
        print("\nParsed OCR blocks (preview):\n")
        for b in blocks:
            day = b.get("day")
            print(f"--- Day: {day} ---")
            print("OCR text:")
            print(b["text"])
            fields = extract_fields_from_block(b["text"])
            print("Parsed fields:")
            for k, v in fields.items():
                print(f"  {k}: {v}")
            print("-" * 60)

        build_calendar_and_csv(blocks, year, month)
        print(f"\nDone. CSV -> {OUTPUT_CSV}  ICS -> {OUTPUT_ICS}")
    except Exception as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
