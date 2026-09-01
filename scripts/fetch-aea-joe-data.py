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


def resultset_link(issue: str, archived: bool) -> str | None:
    params = {"ListingsForm[issue]": issue}
    if archived:
        params["ListingsForm[in_active]"] = "1"
    query = urllib.parse.urlencode(params)
    page_url = f"{BASE_URL}/joe/listings?{query}"
    page = fetch(page_url).decode("utf-8", errors="replace")
    match = re.search(r'href="([^"]*resultset_xls_output\.php[^"]*)"', page)
    if not match:
        return None
    link = html.unescape(match.group(1))
    return urllib.parse.urljoin(BASE_URL, link)


def read_export(issue: str, archived: bool, output: Path) -> pd.DataFrame | None:
    link = resultset_link(issue, archived=archived)
    if not link:
        label = "archived" if archived else "active"
        print(f"{issue}: no {label} XLS export link", file=sys.stderr)
        return None

    output.write_bytes(fetch(link))
    frame = pd.read_excel(output)
    frame["aea_resultset"] = "archived" if archived else "active"
    return frame


def download_issue(issue: str, output_dir: Path) -> int | None:
    active_path = output_dir / f".joe_resultset_{issue}_active.xlsx"
    archived_path = output_dir / f".joe_resultset_{issue}_archived.xlsx"
    frames = [
        frame
        for frame in (
            read_export(issue, archived=False, output=active_path),
            read_export(issue, archived=True, output=archived_path),
        )
        if frame is not None and not frame.empty
    ]
    active_path.unlink(missing_ok=True)
    archived_path.unlink(missing_ok=True)

    if not frames:
        print(f"{issue}: no rows")
        return 0

    combined = pd.concat(frames, ignore_index=True)
    if "jp_id" in combined:
        combined = combined.drop_duplicates(subset=["jp_id"], keep="first")
    else:
        combined = combined.drop_duplicates(keep="first")

    output = output_dir / f"joe_resultset_{issue}.xlsx"
    combined.to_excel(output, index=False)
    rows = len(combined)
    print(f"{issue}: {rows} rows")
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
