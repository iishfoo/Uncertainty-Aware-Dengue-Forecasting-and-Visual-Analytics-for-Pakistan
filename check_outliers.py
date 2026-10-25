"""
Check the most extreme weekly values against the source PDFs (so you can confirm by eye that they are real
and not a parsing mistake).

    python check_outliers.py                  # KP, Sindh, Balochistan
    python check_outliers.py Balochistan      # one region
    python check_outliers.py Balochistan 12   # one region, top 12

Reads data/weekly_panel.csv and the PDFs in data/raw. For each extreme week it prints the value we parsed
and the raw 'Diseases ...' header and 'Dengue ...' row of Table 1 exactly as read from the PDF.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

RAW = Path("data/raw")
HEAD = re.compile(r"^Diseases[ \t]+.*Total", re.M)
ROW = re.compile(r"^Dengue[ \t]+(?:(?:NR|\d[\d,]*)[ \t]+){5,}(?:NR|\d[\d,]*)[ \t]*$", re.M)


def raw_lines(pdf_name):
    path = RAW / pdf_name
    if not path.exists():
        return "(PDF not found)", ""
    with pdfplumber.open(path) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages).replace("(cid:415)", "ti")
    i = re.search(r"Table\s*1\s*:", text)
    t = text[i.start():] if i else text
    h, r = HEAD.search(t), ROW.search(t)
    return (h.group(0) if h else "(header not found)"), (r.group(0) if r else "(dengue row not found)")


def main():
    regions = [sys.argv[1]] if len(sys.argv) > 1 else ["KP", "Sindh", "Balochistan"]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    df = pd.read_csv("data/weekly_panel.csv", parse_dates=["week_start"]).sort_values(["region", "week_start"])
    for reg in regions:
        g = df[df.region == reg].reset_index(drop=True)
        c = g.cases
        neigh = pd.concat([c.shift(k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)], axis=1).median(axis=1)
        g["typical_nearby"] = neigh
        g["ratio"] = (c + 1) / (neigh.fillna(c.median()) + 1)
        print(f"\n{'=' * 78}\n{reg}: {int(c.notna().sum())} weeks with a value; median {c.median():.0f}, max {c.max():.0f}\n{'=' * 78}")
        for _, r in g.sort_values("ratio", ascending=False).head(top).iterrows():
            print(f"\n{int(r.year)}-W{int(r.week):02d} ({r.week_start.date()}): cases = {r.cases:.0f}  | typical nearby weeks = {r.typical_nearby:.0f}"
                  f"  | file: {r.file}")
            h, row = raw_lines(r.file)
            print("   PDF header:", h)
            print("   PDF row   :", row)
    print("\nRead the header and the row together: the number under the region's column must be the one we parsed.")


if __name__ == "__main__":
    main()
