#!/usr/bin/env python3
"""Build a compact JSON feed for the JOE tracker Pages view."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


START_WEEK = 31
MAX_WEEK = 54


CATEGORIES = [
    ("all", "All postings"),
    ("us_tenure", "US tenure track"),
    ("finance", "Finance"),
    ("jel_e", "JEL E"),
    ("jel_c", "JEL C"),
    ("fed", "Fed / regulators"),
    ("us", "United States"),
    ("non_us", "Non-US"),
    ("tenure", "Tenure track"),
    ("non_tenure", "Non-tenure academic"),
    ("industry", "Industry"),
    ("custom", "Custom"),
]


def read_source(data_dir: Path, min_academic_year: int) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {data_dir}")

    frames = []
    for file in files:
        frame = pd.read_excel(file)
        frame["source_file"] = file.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["Date_Active"] = pd.to_datetime(df["Date_Active"], errors="coerce")
    df = df.dropna(subset=["Date_Active"]).copy()
    df["calendar_year"] = df["Date_Active"].dt.year.astype(int)
    df["iso_week"] = df["Date_Active"].dt.isocalendar().week.astype(int)
    df["academic_year"] = df.apply(
        lambda row: row["calendar_year"] - 1
        if row["iso_week"] < START_WEEK
        else row["calendar_year"],
        axis=1,
    ).astype(int)
    df["market_week"] = df["iso_week"].apply(
        lambda week: week - START_WEEK + 1 if week >= START_WEEK else week + (52 - START_WEEK + 1)
    )
    df["plot_week"] = df["market_week"] + START_WEEK - 1
    df = df[df["academic_year"] >= min_academic_year].copy()
    return df


def column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df:
        return df[name]
    return pd.Series("", index=df.index)


def contains(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def jel_code(df: pd.DataFrame, code: str) -> pd.Series:
    return contains(column(df, "JEL_Classifications"), rf"(?:^|\n)\s*{code}(?:\b|\d| -)")


def category_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    text = (
        column(df, "jp_institution").fillna("").astype(str)
        + " "
        + column(df, "jp_department").fillna("").astype(str)
        + " "
        + column(df, "jp_full_text").fillna("").astype(str)
    )
    locations = column(df, "locations").fillna("").astype(str)
    sections = column(df, "jp_section").fillna("").astype(str)

    finance = jel_code(df, "G")
    jel_e = jel_code(df, "E")
    jel_c = jel_code(df, "C")
    fed = text.str.contains(
        r"Federal Reserve|Board of Governors|FDIC|Federal Deposit Insurance|"
        r"Office of the Comptroller|Comptroller of the Currency|bank regulator",
        case=False,
        regex=True,
    )
    us = locations.str.contains(r"\bUNITED STATES\b", case=False, regex=True)
    tenure = sections.str.contains(r"tenure", case=False, regex=True)
    us_tenure = us & sections.str.contains(
        r"full-time academic.*tenure", case=False, regex=True
    )
    non_tenure = sections.str.contains(
        r"visiting|temporary|part-time|part time|adjunct|post[- ]?doc|postdoctoral",
        case=False,
        regex=True,
    ) & ~sections.str.contains(r"nonacademic", case=False, regex=True)
    industry = sections.str.contains(r"nonacademic", case=False, regex=True)
    custom = (
        sections.str.strip().ne("")
        & column(df, "JEL_Classifications").fillna("").astype(str).str.strip().ne("")
    )

    return {
        "all": df,
        "us_tenure": df[us_tenure],
        "finance": df[finance],
        "jel_e": df[jel_e],
        "jel_c": df[jel_c],
        "fed": df[fed],
        "us": df[us],
        "non_us": df[~us],
        "tenure": df[tenure],
        "non_tenure": df[non_tenure],
        "industry": df[industry],
        "custom": df[custom],
    }


def weekly_series(df: pd.DataFrame) -> dict[str, object]:
    grouped = (
        df.groupby(["academic_year", "plot_week"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["academic_year", "plot_week"])
    )

    years = []
    for year, year_data in grouped.groupby("academic_year"):
        year_data = year_data.copy().sort_values("plot_week")
        year_data["cumulative"] = year_data["count"].cumsum()
        year_data["rolling4"] = year_data["count"].rolling(4, min_periods=1).sum()
        rows = [
            {
                "week": int(row.plot_week),
                "count": float(row.count),
                "cumulative": float(row.cumulative),
                "rolling4": float(row.rolling4),
            }
            for row in year_data.itertuples()
            if START_WEEK <= int(row.plot_week) <= MAX_WEEK
        ]
        total = rows[-1]["cumulative"] if rows else 0
        latest_week = rows[-1]["week"] if rows else None
        years.append(
            {
                "year": int(year),
                "total": int(round(total)),
                "latestWeek": latest_week,
                "rows": rows,
            }
        )

    years.sort(key=lambda item: item["year"])
    return {
        "years": years,
        "latestYear": years[-1]["year"] if years else None,
    }


def clean_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data-dir", default="joe-data", help="Directory of AEA JOE .xlsx exports")
    parser.add_argument("--min-academic-year", type=int, default=2015)
    parser.add_argument("--output", default="pages/joe-tracker-data.json")
    args = parser.parse_args()

    data_dir = Path(args.source_data_dir).resolve()
    df = read_source(data_dir, args.min_academic_year)
    frames = category_frames(df)

    categories = []
    for key, title in CATEGORIES:
        frame = frames[key]
        series = weekly_series(frame)
        latest_year = series["latestYear"]
        latest = next((item for item in series["years"] if item["year"] == latest_year), None)
        categories.append(
            {
                "id": key,
                "title": title,
                "observations": int(len(frame)),
                "latestYearTotal": latest["total"] if latest else 0,
                "latestWeek": latest["latestWeek"] if latest else None,
                **series,
            }
        )

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "startWeek": START_WEEK,
        "maxWeek": MAX_WEEK,
        "source": "AEA JOE native XLS result sets",
        "sourceUrl": "https://www.aeaweb.org/joe/listings",
        "sourceFiles": sorted(path.name for path in data_dir.glob("*.xlsx")),
        "categories": categories,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_json(payload), indent=2) + "\n")
    print(f"Wrote {output} with {len(categories)} categories from {len(df)} rows")


if __name__ == "__main__":
    main()
