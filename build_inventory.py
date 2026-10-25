"""
Inventory the downloaded NIH bulletins and test the Table-1 dengue row pattern on every PDF.

Run (from the same folder as download_nih_bulletins.py):  python build_inventory.py
Output: data/inventory.csv  + a short summary in the terminal
"""

import csv
import re
import sys
from pathlib import Path

import pdfplumber

RAW = Path("data/raw")
OUT = Path("data/inventory.csv")

NUM = r"(?:NR|\d[\d,]*)"
ROW = re.compile(
    rf"^Dengue\s+((?:{NUM}\s+){{7}}{NUM})\s*$", re.M
)  # province row: 7 provinces + Total
WEEK = re.compile(r"\bWeek\s*(?:No\.?|#)?\s*[:\-]?\s*(\d{1,2})\b", re.I)
YEAR = re.compile(r"\b(20[12]\d)\b")
COLS = ["AJK", "Balochistan", "GB", "ICT", "KP", "Punjab", "Sindh", "Total"]


def read_pdf(path):
    with pdfplumber.open(path) as pdf:
        pages = [(p.extract_text() or "") for p in pdf.pages]
    return pages


def inspect(path):
    pages = read_pdf(path)
    text = "\n".join(pages)
    first = pages[0] if pages else ""
    week = WEEK.search(first)
    year = YEAR.search(first)
    m = ROW.search(text)
    t1_values, t1_ok = "", False
    if m:
        vals = [
            None if v == "NR" else int(v.replace(",", "")) for v in m.group(1).split()
        ]
        t1_values = "|".join("NR" if v is None else str(v) for v in vals)
        t1_ok = sum(v or 0 for v in vals[:-1]) == vals[-1]
    dengue_lines = [
        l.strip() for l in text.splitlines() if l.strip().lower().startswith("dengue")
    ]
    return {
        "file": path.name,
        "pages": len(pages),
        "week_guess": week.group(1) if week else "",
        "year_guess": year.group(1) if year else "",
        "t1_match": bool(m),
        "t1_sum_ok": t1_ok,
        "t1_values(" + ",".join(COLS) + ")": t1_values,
        "dengue_lines": " || ".join(dengue_lines)[:400],
        "page1_head": first[:160].replace("\n", " / "),
        "error": "",
    }


def main():
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {RAW.resolve()}")
    rows = []
    for i, p in enumerate(pdfs, 1):
        try:
            rows.append(inspect(p))
        except Exception as e:  # keep going; record the problem
            rows.append({"file": p.name, "error": f"{type(e).__name__}: {e}"})
        if i % 20 == 0:
            print(f"  {i}/{len(pdfs)} done")

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("t1_match")]
    bad_sum = [r for r in ok if not r.get("t1_sum_ok")]
    miss = [r for r in rows if not r.get("t1_match")]
    print(
        f"\nPDFs: {len(rows)} | Table-1 dengue row matched: {len(ok)} | sum check failed: {len(bad_sum)}"
    )
    print(f"Not matched (layout differs or error): {len(miss)}")
    for r in miss[:15]:
        print("  -", r["file"], r.get("error", ""))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
