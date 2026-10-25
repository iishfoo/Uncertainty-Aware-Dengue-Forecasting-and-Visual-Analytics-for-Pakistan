"""
Download NIH Pakistan IDSR weekly bulletin PDFs (public data).

Run:  python download_nih_bulletins.py
Output: data/raw/*.pdf  (+ data/page.html for debugging, data/failed.txt if any fail)
"""

import html as htmllib
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote, urljoin

import requests

PAGE = "https://www.nih.org.pk/phb/weekly-bulletin"
OUT = Path("data/raw")
VERIFY_SSL = (
    True  # set to False ONLY if you get an SSLError (public data, no login involved)
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,*/*",
}


def extract_links(page_html):
    """Return sorted unique absolute PDF URLs found under /wp-content/uploads/."""
    text = htmllib.unescape(page_html)
    # hrefs can contain spaces/brackets, so match up to the closing quote
    found = re.findall(
        r'href=["\']([^"\']+?\.pdf)(?:\?[^"\']*)?["\']', text, flags=re.I
    )
    links = {urljoin(PAGE, f.strip()) for f in found if "/wp-content/uploads/" in f}
    return sorted(links)


def local_name(url):
    """2026/10/Weekly Report-37-2026.pdf -> 2026_10_Weekly_Report-37-2026.pdf (no collisions)."""
    path = unquote(url).split("/wp-content/uploads/", 1)[1]
    name = path.replace("/", "_").replace(" ", "_")
    return re.sub(r'[<>:"\\|?*]', "", name)


def fetch(session, url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=90, verify=VERIFY_SSL)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"   attempt {attempt}/{retries} failed: {e}")
            time.sleep(2 * attempt)
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(HEADERS)

    page = fetch(session, PAGE)
    if page is None:
        sys.exit(
            "Could not open the bulletin page. Copy the error above and send it to me."
        )
    Path("data/page.html").write_text(page.text, encoding="utf-8")

    links = extract_links(page.text)
    print(f"Found {len(links)} PDF links on the page")
    if not links:
        print(
            "No PDF links matched. Check data/page.html (page may load links with JavaScript)."
        )
        sys.exit(1)

    ok = skipped = 0
    failed = []
    for i, url in enumerate(links, 1):
        dest = OUT / local_name(url)
        if dest.exists() and dest.stat().st_size > 10_000:
            skipped += 1
            continue
        print(f"[{i}/{len(links)}] {dest.name}")
        r = fetch(session, quote(unquote(url), safe=":/"))
        if r is None or not r.content.startswith(b"%PDF"):
            failed.append(url)
            continue
        dest.write_bytes(r.content)
        ok += 1
        time.sleep(0.5)  # be polite to the server

    print(f"\nDownloaded: {ok} | already had: {skipped} | failed: {len(failed)}")
    if failed:
        Path("data/failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print("Failed URLs saved to data/failed.txt")


if __name__ == "__main__":
    main()
