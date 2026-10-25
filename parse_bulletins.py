"""
Parse NIH IDSR weekly bulletins (dengue only) into tidy CSVs.   -- v2.2

Run (same folder as download_nih_bulletins.py):
    python parse_bulletins.py              # reads data/raw
    python parse_bulletins.py some_folder  # reads another folder

Outputs in data/:
    weekly_dengue_province.csv    suspected dengue cases per region/week (Table 1; NR kept as missing)
    weekly_compliance_region.csv  expected vs received IDSR reports per region/week (+ the bulletin's own National row)
    weekly_lab_dengue.csv         dengue lab tests / positives per region/week (Table 5)
    weekly_panel.csv              the three above merged (region x week) - the modelling table
    parse_log.csv                 one row per PDF: keys, what was parsed, warnings
"""

import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pdfplumber

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw")
OUT = Path("data")
START = (2023, 22)  # first week of the continuous archive (ISO year, week)

NUM = r"(?:NR|\d[\d,]*)"
COLMAP = {
    "AJK": "AJK",
    "Balochistan": "Balochistan",
    "GB": "GB",
    "Gilgit": "GB",
    "ICT": "ICT",
    "ISL": "ICT",
    "KP": "KP",
    "KPK": "KP",
    "Punjab": "Punjab",
    "Sindh": "Sindh",
}

# ---------- week / year -------------------------------------------------------------------------
TITLE = re.compile(r"Pakistan,\s*Week\s*(\d{1,2})\s*,\s*(20\d{2})")
TITLE_EARLY = re.compile(
    r"Epi\s*Week\s*-?\s*(\d{1,2})\s*\(([^)]*?)(20\d{2})\s*\)", re.S
)  # 2023 W22-W37 layout
FILE_KEYS = [
    re.compile(r"Report[-_](\d{1,2})[-_](20\d{2})"),
    re.compile(r"Week[-_](\d{1,2})[-_]Bulletin[-_(]*(20\d{2})"),
]


def title_key(text):
    m = TITLE.search(text)
    if m:
        return (int(m.group(2)), int(m.group(1))), "title"
    m = TITLE_EARLY.search(text)
    if m:
        return (int(m.group(3)), int(m.group(1))), "early-title"
    return None, ""


def file_key(name):
    for rx in FILE_KEYS:
        m = rx.search(name)
        if m:
            return (int(m.group(2)), int(m.group(1)))
    return None


# ---------- Table 1 (province-level suspected cases) ----------------------------------------------
def toi(tok):
    return None if tok in ("NR", "-") else int(tok.replace(",", ""))


def parse_table1(text):
    m = re.search(r"Table\s*1\s*:", text)
    sec = text[m.start() :] if m else text
    h = re.search(r"^Diseases[ \t]+([^\n]*)\n?([^\n]*)", sec, re.M)
    if not h:
        return None, "Table 1 header not found"
    toks = h.group(1).split()
    if "Total" not in toks:  # header wrapped onto the next line
        toks += h.group(2).split()
    toks = (
        " ".join(toks).replace("IC T", "ICT").replace("*", "").split()
    )  # 'Punjab*' and a split 'IC T'
    if "Total" not in toks:
        return None, "Table 1 header has no Total column"
    cols = toks[: toks.index("Total")]
    unknown = [c for c in cols if c not in COLMAP]
    if unknown or not cols:
        return None, f"Table 1 header has unknown columns: {cols}"
    row = re.search(rf"^Dengue\s+((?:{NUM}\s+){{{len(cols)}}}{NUM})\s*$", sec, re.M)
    if not row:
        return None, "Dengue row of Table 1 not found"
    vals = [toi(v) for v in row.group(1).split()]
    d = {COLMAP[c]: v for c, v in zip(cols, vals)}
    ok = sum(v or 0 for v in vals[:-1]) == vals[-1]
    return (d, vals[-1], ok), ""


# ---------- region compliance (overview table) -------------------------------------------------------
N3 = re.compile(r"^(\d[\d,]*)\s+(\d[\d,]*)\s+(\d{1,3})$")
NAME_N3 = re.compile(
    r"^(Khyber Pakhtunkhwa|Azad Jammu Kashmir|Islamabad Capital Territory|Balochistan|"
    r"Gilgit Baltistan|Sindh|National)\s+(\d[\d,]*)\s+(\d[\d,]*)\s+(\d{1,3})$"
)
REG_START = re.compile(r"^(Khyber|Azad|Islamabad|Balochistan|Gilgit|Sindh|National)")
REG_KEY = [
    ("khyber", "KP"),
    ("azad", "AJK"),
    ("islamabad", "ICT"),
    ("balochistan", "Balochistan"),
    ("gilgit", "GB"),
    ("sindh", "Sindh"),
    ("national", "National"),
]


def reg_key(name):
    n = name.lower()
    return next(k for p, k in REG_KEY if n.startswith(p))


def parse_compliance(text):
    j = text.find("Expected Reports")
    if j < 0:
        return None, "no region compliance table (early layout?)"
    lines = [l.strip() for l in text[j : j + 1200].splitlines()][1:]
    rows, i = {}, 0
    while i < len(lines) and len(rows) < 7:
        l = lines[i]
        m = NAME_N3.match(l)
        if m:  # "Sindh 2086 1963 94"
            rows[reg_key(m.group(1))] = (
                toi(m.group(2)),
                toi(m.group(3)),
                int(m.group(4)),
            )
            i += 1
            continue
        if (
            re.fullmatch(r"Islamabad\s+Capital", l)
            and i + 1 < len(lines)
            and N3.match(lines[i + 1])
        ):
            m = N3.match(lines[i + 1])  # "Islamabad Capital" / "27 21 78" / "Territory"
            rows["ICT"] = (toi(m.group(1)), toi(m.group(2)), int(m.group(3)))
            i += 2
            continue
        m = N3.match(l)
        if (
            m and i + 1 < len(lines) and REG_START.match(lines[i + 1])
        ):  # "1857 1699 91" / "Sindh"
            rows[reg_key(lines[i + 1])] = (
                toi(m.group(1)),
                toi(m.group(2)),
                int(m.group(3)),
            )
            i += 2
            continue
        i += 1
    if len(rows) != 7:
        return None, f"could not read the 7 compliance rows (got {len(rows)})"
    return rows, ""


# ---------- lab table (position based, robust to garbled text order) ---------------------------------
TOKEN = re.compile(r"^(?:\d[\d,]*|-|NR)$")


def parse_labs(pdf):
    for page in pdf.pages:
        if not re.search(
            r"Table\s*5\s*:\s*Public Health Lab",
            (page.extract_text() or "").replace("(cid:415)", "ti"),
        ):
            continue
        words = page.extract_words()
        for w in words:
            w["text"] = w["text"].replace("(cid:415)", "ti")
        reg_words = [w for w in words if w["text"] in COLMAP]
        tops = sorted({round(w["top"]) for w in reg_words})
        header = None
        for t in tops:  # first line carrying >= 5 region names
            row = [w for w in reg_words if abs(w["top"] - t) <= 3]
            if len({w["text"] for w in row}) >= 5:
                header = sorted(row, key=lambda w: w["x0"])
                break
        if not header:
            return None, "lab header not found"
        h_top = header[0]["top"]
        regs = [COLMAP[w["text"]] for w in header]
        subs = sorted(
            [
                w
                for w in words
                if (w["text"] == "Test" or w["text"].startswith("Pos"))
                and h_top < w["top"] < h_top + 40
            ],
            key=lambda w: w["x0"],
        )
        centers = [
            (w["x0"] + w["x1"]) / 2
            for w in (subs if len(subs) == 2 * len(regs) else header)
        ]
        paired = len(subs) == 2 * len(regs)
        first_x = min(centers) - 12
        labels = [w for w in words if w["top"] > h_top and w["x1"] < first_x]
        dg = next((w for w in labels if w["text"] == "Dengue"), None)
        if dg is None:
            return None, "Dengue label not found in lab table"
        others = [
            abs(w["top"] - dg["top"]) for w in labels if abs(w["top"] - dg["top"]) > 1
        ]
        tol = min(5.0, 0.49 * min(others)) if others else 5.0
        cells = [
            w
            for w in words
            if abs(w["top"] - dg["top"]) <= tol
            and w["x0"] > dg["x1"]
            and TOKEN.match(w["text"])
        ]
        pitch = (
            min(b - a for a, b in zip(sorted(centers), sorted(centers)[1:]))
            if len(centers) > 1
            else 40
        )
        slot = {}
        for w in cells:
            c = (w["x0"] + w["x1"]) / 2
            k = min(range(len(centers)), key=lambda i: abs(centers[i] - c))
            if abs(centers[k] - c) > 0.9 * pitch:
                continue
            if k in slot:
                return None, "two values fall in one lab column (layout shifted)"
            slot[k] = toi(w["text"])
        if paired:
            return {
                r: (slot.get(2 * i), slot.get(2 * i + 1)) for i, r in enumerate(regs)
            }, "tests_pos"
        return {r: (None, slot.get(i)) for i, r in enumerate(regs)}, "pos_only"
    return None, "Table 5 (laboratories) not found"


# ---------- one PDF ---------------------------------------------------------------------------------
def parse_pdf(path):
    with pdfplumber.open(path) as pdf:
        text = "\n".join((p.extract_text() or "") for p in pdf.pages).replace(
            "(cid:415)", "ti"
        )
        rec = {"file": path.name, "warnings": []}
        tk, src = title_key(text)
        fk = file_key(path.name)
        if (
            tk is None and fk
        ):  # no readable title: trust the filename (week-year in its name)
            tk, src = fk, "filename-fallback"
        rec["title_key"], rec["file_key"], rec["key_source"] = tk, fk, src
        if tk is None:
            rec["status"] = "no week/year title"
            return rec
        rec["status"] = "ok"

        t1, msg = parse_table1(text)
        if t1:
            rec["t1"], rec["t1_total"], rec["t1_ok"] = t1
            if not rec["t1_ok"]:
                rec["warnings"].append("Table 1: provinces do not add up to Total")
        else:
            rec["warnings"].append(msg)

        comp, msg = parse_compliance(text)
        if comp:
            rec["comp"] = comp
            reg = [v for k, v in comp.items() if k != "National"]
            nat = comp["National"]
            de, dr = sum(v[0] for v in reg) - nat[0], sum(v[1] for v in reg) - nat[1]
            if de or dr:
                rec["warnings"].append(
                    f"compliance: regions minus National = expected {de:+d}, received {dr:+d}"
                )
            for k, (e, r, p) in comp.items():
                if e and abs(round(100 * r / e) - p) > 1:
                    rec["warnings"].append(
                        f"compliance: stated {p}% vs computed {round(100 * r / e)}% ({k})"
                    )
        else:
            rec["warnings"].append(msg)

        labs, fmt = parse_labs(pdf)
        if labs:
            rec["labs"], rec["lab_format"] = labs, fmt
        else:
            rec["warnings"].append(fmt)
    return rec


def main():
    OUT.mkdir(exist_ok=True)
    pdfs = sorted(RAW.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {RAW.resolve()}")
    recs = []
    for i, p in enumerate(pdfs, 1):
        try:
            recs.append(parse_pdf(p))
        except Exception as e:
            recs.append(
                {
                    "file": p.name,
                    "status": f"error: {type(e).__name__}: {e}",
                    "warnings": [],
                }
            )
        if i % 20 == 0:
            print(f"  {i}/{len(pdfs)}")

    # final (year, week) key: the title, unless it collides with another file and the filename disagrees
    good = [r for r in recs if r.get("status") == "ok"]
    count = {}
    for r in good:
        count[r["title_key"]] = count.get(r["title_key"], 0) + 1
    for r in good:
        r["key"] = r["title_key"]
        fk = r["file_key"]
        if fk and fk != r["title_key"]:
            if count[r["title_key"]] > 1:
                r["key"] = fk
                r["warnings"].append(
                    f"title says {r['title_key']} but filename says {fk}: used the filename"
                )
            else:
                r["warnings"].append(
                    f"title says {r['title_key']} but filename says {fk}: kept the title"
                )

    usable = [r for r in good if r["key"] >= START]
    best = {}
    for r in sorted(
        usable,
        key=lambda r: (
            any(
                k in r["file"].lower() for k in ("updated", "reviewed", "final", "new")
            ),
            r["file"],
        ),
    ):
        best[r["key"]] = r
    kept = {r["file"] for r in best.values()}

    prov, comp, lab = [], [], []
    for (y, w), r in sorted(best.items()):
        base = {
            "year": y,
            "week": w,
            "week_start": date.fromisocalendar(y, w, 1),
            "file": r["file"],
        }
        for reg, v in r.get("t1", {}).items():
            prov.append(
                {
                    **base,
                    "region": reg,
                    "cases": v,
                    "is_nr": v is None,
                    "national_total": r["t1_total"],
                    "t1_sum_ok": r["t1_ok"],
                }
            )
        nat = r.get("comp", {}).get("National", (None, None, None))
        for reg, (e, rc, p) in r.get("comp", {}).items():
            comp.append(
                {
                    **base,
                    "region": reg,
                    "expected": e,
                    "received": rc,
                    "stated_pct": p,
                    "compliance": round(rc / e, 4) if e else None,
                    "national_expected": nat[0],
                    "national_received": nat[1],
                }
            )
        for reg, (t, pos) in r.get("labs", {}).items():
            lab.append(
                {
                    **base,
                    "region": reg,
                    "tests": t,
                    "positives": pos,
                    "lab_format": r["lab_format"],
                }
            )

    dp, dc, dl = pd.DataFrame(prov), pd.DataFrame(comp), pd.DataFrame(lab)
    dp.to_csv(OUT / "weekly_dengue_province.csv", index=False)
    dc.to_csv(OUT / "weekly_compliance_region.csv", index=False)
    dl.to_csv(OUT / "weekly_lab_dengue.csv", index=False)
    keys = ["year", "week", "week_start", "region"]
    dcr = dc[dc.region != "National"]
    panel = dp.merge(
        dcr[keys + ["expected", "received", "compliance"]], on=keys, how="left"
    ).merge(dl[keys + ["tests", "positives", "lab_format"]], on=keys, how="left")
    panel.to_csv(OUT / "weekly_panel.csv", index=False)

    log = pd.DataFrame(
        [
            {
                "file": r["file"],
                "status": r.get("status"),
                "title_key": r.get("title_key"),
                "file_key": r.get("file_key"),
                "final_key": r.get("key"),
                "kept": r["file"] in kept,
                "warnings": "; ".join(r.get("warnings", [])),
            }
            for r in recs
        ]
    )
    log.to_csv(OUT / "parse_log.csv", index=False, encoding="utf-8-sig")

    last = max(best)
    d, missing = date.fromisocalendar(*START, 1), []
    while d <= date.fromisocalendar(*last, 1):
        iso = d.isocalendar()
        if (iso[0], iso[1]) not in best:
            missing.append(f"{iso[0]}-W{iso[1]:02d}")
        d += timedelta(days=7)

    def weeks_with(text):
        return [
            f"{y}-W{w:02d}"
            for (y, w), r in sorted(best.items())
            if any(text in x for x in r["warnings"])
        ]

    print(
        f"\nPDFs read: {len(recs)} | usable weeks (2023-W22 onward): {len(best)} | not used as weekly bulletins: {len(recs) - len(best)}"
    )
    print(
        f"Window: {START[0]}-W{START[1]:02d} -> {last[0]}-W{last[1]:02d} | missing weeks ({len(missing)}): {missing}"
    )
    print(
        f"Table 1 missing or failed: {len([r for r in best.values() if 't1' not in r])} | sum check failed: {int((~dp.drop_duplicates(['year', 'week'])['t1_sum_ok']).sum())}"
    )
    pj = dp[dp.region == "Punjab"].drop_duplicates(["year", "week"])
    print(
        "Punjab NR by year:",
        {int(y): f"{int(g.is_nr.sum())}/{len(g)}" for y, g in pj.groupby("year")},
    )
    print(
        f"Compliance table not read: {len([r for r in best.values() if 'comp' not in r])} weeks | lab row not read: {len([r for r in best.values() if 'labs' not in r])} weeks"
    )
    for label, key in (
        ("no region compliance table (early layout?)", "no region compliance"),
        ("compliance rows unreadable", "could not read the 7"),
        ("Table 1 problems", "Table 1"),
        ("lab problems", "lab"),
        ("Table 5 not found", "Table 5"),
    ):
        w = weeks_with(key)
        if w:
            print(f"  {label}: {w}")
    from collections import Counter

    reasons = Counter(
        re.sub(r"[+-]?\d+", "N", x)[:90]
        for r in best.values()
        for x in r["warnings"]
        if not x.startswith("compliance: ")
    )
    if reasons:
        print("  most common problems:")
        for txt, n in reasons.most_common(8):
            print(f"    {n:3d} x {txt}")
    fb = [
        f"{y}-W{w:02d}"
        for (y, w), r in sorted(best.items())
        if r["key_source"] == "filename-fallback"
    ]
    if fb:
        print(
            f"  week taken from the filename (no title in the PDF): {len(fb)} weeks, first {fb[:3]}"
        )
    for label, key in (
        ("Table 1", "Table 1"),
        ("Table 1", "Dengue row"),
        ("compliance", "could not read"),
        ("lab", "lab column"),
    ):
        for (y, w), r in sorted(best.items()):
            for x in r["warnings"]:
                if key in x:
                    print(f"    [{label}] {y}-W{w:02d}: {x[:110]}")
    for (y, w), r in sorted(best.items()):
        if "t1_ok" in r and not r["t1_ok"]:
            print(
                f"    [Table 1 sum check failed] {y}-W{w:02d}: {r['t1']} total {r['t1_total']}"
            )
    sw = weeks_with("regions minus National")
    print(
        f"  bulletin's own regions != National: {len(sw)} weeks (small source inconsistency, kept)"
    )
    for r in best.values():
        for x in r["warnings"]:
            if x.startswith("title says"):
                print("  KEY FIX:", r["file"], "|", x)
    print("Details for every file: data/parse_log.csv")


if __name__ == "__main__":
    main()
