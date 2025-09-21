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
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Configuration
IMAGE_URLS = [
    f"https://telugucalendar.org/calendar/2025/chicago/chicago-2025-{month}.png"
    for month in range(1, 13)
]
# If you already have a local file, set LOCAL_IMAGE_PATH to that path and leave IMAGE_URLS = None
LOCAL_IMAGE_PATH: Optional[str] = None  # e.g. "./calendar.png"
# process both languages
OCR_LANGS = ["eng", "tel"]  # runs OCR for English and Telugu
OUTPUT_ICS = "output_calendar.ics"
OUTPUT_CSV = "output_calendar.csv"


def download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))

def preprocess_image(img: Image.Image) -> Image.Image:
    # Convert to grayscale and increase contrast for better OCR
    gray = ImageOps.grayscale(img)
    # Optional binarization
    bw = gray.point(lambda x: 0 if x < 200 else 255, "1")
    # Return a higher-resolution image to improve OCR accuracy
    return bw.resize((bw.width * 2, bw.height * 2), Image.LANCZOS)

def detect_calendar_cells_via_contours(pil_image, min_area=1000, debug=False):
    open_cv_image = np.array(pil_image.convert("L"))
    _, thresh = cv2.threshold(open_cv_image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > min_area and w > 50 and h > 50:
            boxes.append((y, x, w, h))

    boxes.sort()
    sorted_cells = sorted(boxes, key=lambda b: (b[0] // 50, b[1]))
    blocks = []

    for idx, (y, x, w, h) in enumerate(sorted_cells):
        cell = pil_image.crop((x, y, x + w, y + h))
        row = idx // 7 + 1
        col = idx % 7 + 1
        blocks.append(((row, col), cell))
        if debug:
            plt.figure(figsize=(2, 2))
            plt.imshow(cell, cmap='gray')
            plt.title(f"Block {row}-{col}")
            plt.axis('off')
            plt.show()

    return blocks


def main():
    try:
            for url in IMAGE_URLS:
                print(f"Processing: {url}")
                
                img = download_image(url)
                proc = preprocess_image(img)
                blocks = detect_calendar_cells_via_contours(proc, debug=True)
                
                for (row, col), cell in blocks:
                    if row == 5 and col == 5:
                        width, height = cell.size
                        num_rows = 5
                        num_cols = 7
                        cell_width = width // num_cols
                        cell_height = height // num_rows

                        for sub_row in range(num_rows):
                            for sub_col in range(num_cols):
                                left = sub_col * cell_width
                                upper = sub_row * cell_height
                                right = (sub_col + 1) * cell_width
                                lower = (sub_row + 1) * cell_height
                                sub_cell = cell.crop((left, upper, right, lower))

                                plt.figure(figsize=(2.5, 2.5))
                                plt.imshow(sub_cell, cmap='gray')
                                plt.title(f"Block 5-5 → Cell ({sub_row+1},{sub_col+1})", fontsize=8)
                                plt.axis('off')
                                plt.tight_layout()
                                plt.show()

                                eng_text = pytesseract.image_to_string(sub_cell, lang="eng").strip()
                                tel_text = pytesseract.image_to_string(sub_cell, lang="tel").strip()
                                tel_lines = [line.strip() for line in tel_text.split("\n") if line.strip()]
                                for idx, tel_line in enumerate(tel_lines, 1):
                                    print(f"Telugu Line {idx}: {tel_line}")

                                # Attempt to extract a date number from English OCR
                                date_match = re.search(r'\b([1-9]|[12][0-9]|3[01])\b', eng_text)
                                if date_match:
                                    calendar_date = int(date_match.group(1))
                                    month_match = re.search(r'(\d{4})-(\d{1,2})', url)
                                    year, month = month_match.groups()
                                    print(f"Date: {calendar_date}-{month}-{year}")
                                    print(f"Tithi & Nakshatram: {tel_text}")
                                    print("-" * 40)

                                

    except Exception as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
