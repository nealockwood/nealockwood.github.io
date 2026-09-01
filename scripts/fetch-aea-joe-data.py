#!/usr/bin/env python3
"""Download AEA JOE native XLS result sets for one or more issue years."""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://www.aeaweb.org"
USER_AGENT = "nealockwood-pages-joe-tracker/1.0"


def issue_ids(start_year: int, end_year: int) -> list[str]:
    return [f"{year}-{half:02d}" for year in range(start_year, end_year + 1) for half in (1, 2)]


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def resultset_link(issue: str) -> str | None:
    query = urllib.parse.urlencode(
        {
            "ListingsForm[issue]": issue,
            "ListingsForm[in_active]": "1",
        }
    )
    page_url = f"{BASE_URL}/joe/listings?{query}"
    page = fetch(page_url).decode("utf-8", errors="replace")
    match = re.search(r'href="([^"]*resultset_xls_output\.php[^"]*)"', page)
    if not match:
        return None
    link = html.unescape(match.group(1))
    return urllib.parse.urljoin(BASE_URL, link)


def download_issue(issue: str, output_dir: Path) -> int | None:
    link = resultset_link(issue)
    if not link:
        print(f"{issue}: no XLS export link", file=sys.stderr)
        return None

    output = output_dir / f"joe_resultset_{issue}.xlsx"
    output.write_bytes(fetch(link))
    rows = len(pd.read_excel(output))
    print(f"{issue}: {rows} rows")
    if rows == 0:
        output.unlink(missing_ok=True)
        return 0
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--output-dir", default="joe-data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    downloaded = 0
    for issue in issue_ids(args.start_year, args.end_year):
        rows = download_issue(issue, output_dir)
        if rows is None:
            continue
        downloaded += 1
        total += rows

    if downloaded == 0:
        raise SystemExit("No AEA JOE result sets were downloaded")
    print(f"Downloaded {downloaded} result sets with {total} rows")


if __name__ == "__main__":
    main()
