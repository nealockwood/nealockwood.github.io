#!/usr/bin/env python3
"""Build the static data snapshot for pages/irfs.html.

Run from this repository root:

    uv run --with pandas --with numpy --with openpyxl --with pyreadstat scripts/build-irfs-data.py

By default this reads ../mps_rep. Override with:

    MPS_REP_DIR=/path/to/mps_rep uv run ...
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


SITE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("MPS_REP_DIR", SITE_ROOT.parent / "mps_rep")).resolve()
OUTPUT_PATH = SITE_ROOT / "pages" / "irfs-data.json"
RAW_EVENT_URL = os.environ.get(
    "JK_SOURCE_URL",
    "https://raw.githubusercontent.com/paulbousquet/GBMPSurprise/main/jk_source_old.csv",
)

MARKET_CONTROLS = [
    "nfp_surp",
    "nfp_12m",
    "sp500_3m",
    "slope_3m",
    "bcom_3m",
    "tr_skew",
]

GREENBOOK_CONTROLS = [
    "gRGDPB1",
    "gRGDPF0",
    "gRGDPF1",
    "gRGDPF2",
    "gRGDPF3",
    "DgRGDPB1",
    "DgRGDPF0",
    "DgRGDPF1",
    "DgRGDPF2",
    "gPGDPB1",
    "gPGDPF0",
    "gPGDPF1",
    "gPGDPF2",
    "gPGDPF3",
    "DgPGDPB1",
    "DgPGDPF0",
    "DgPGDPF1",
    "DgPGDPF2",
    "UNEMPF0",
]

FRED_MACRO_CONTROLS = [
    "ffr",
    "ebp",
    "GS1",
    "dgdp",
    "dunemp",
    "dlip",
    "dlcpi",
    "dlpce",
    "dlsent",
    "dlnas",
    "dlmpu",
]

RAW_SHOCK_COLUMNS = [
    "mp1",
    "mp2",
    "ff1",
    "ff2",
    "ff3",
    "ff4",
    "ff5",
    "ff6",
    "ed1",
    "ed2",
    "ed3",
    "ed4",
    "ed5",
    "ed6",
    "ed7",
    "ed8",
]

OUTCOMES = {
    "unrate": {"label": "Unemployment", "source": "UNRATE", "transform": "diff"},
    "cpi": {"label": "CPI", "source": "CPIAUCSL", "transform": "logdiff100"},
    "pce": {"label": "PCE", "source": "PCEPI", "transform": "logdiff100"},
    "ip": {"label": "Industrial Production", "source": "INDPRO", "transform": "logdiff100"},
    "ffr": {"label": "Federal Funds Rate", "source": "ffr", "transform": "diff"},
    "ebp": {"label": "Excess Bond Premium", "source": "ebp", "transform": "diff"},
}


def value(x):
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return float(x)
    if pd.isna(x):
        return None
    return x


def month_key(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m")


def date_key(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def compute_mps_from_pca(prep: pd.DataFrame) -> pd.Series:
    cols = ["ed1", "ed2", "ed3", "ed4"]
    clean = prep[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = clean.notna().all(axis=1)
    x = clean.loc[valid, cols].astype(float).to_numpy()
    if len(x) < 4:
        return pd.Series(np.nan, index=prep.index, dtype=float)
    std = x.std(axis=0, ddof=0)
    if np.any(std < 1e-12):
        return pd.Series(np.nan, index=prep.index, dtype=float)
    x_std = (x - x.mean(axis=0)) / std
    _, _, vt = np.linalg.svd(x_std, full_matrices=False)
    pc1 = np.dot(x_std, vt[0])
    y = clean.loc[valid, "ed4"].astype(float).to_numpy()
    design = np.column_stack([np.ones(len(pc1)), pc1])
    coef = np.linalg.lstsq(design, y, rcond=None)[0][1]
    out = pd.Series(np.nan, index=prep.index, dtype=float)
    out.loc[valid] = 0.01 * pc1 / coef
    return out


def read_macro() -> pd.DataFrame:
    mondat, _ = pyreadstat.read_dta(SOURCE_ROOT / "test" / "mondat.dta")
    mondat["daten"] = pd.to_datetime(mondat["daten"])
    mondat["month"] = mondat["daten"].dt.strftime("%Y-%m")
    return mondat


def read_external_shocks() -> tuple[pd.DataFrame, pd.DataFrame]:
    brw = pd.read_excel(SOURCE_ROOT / "BJMW-BRW-shocks-updated-1.xlsx", sheet_name="Data")
    brw["date"] = pd.to_datetime(brw["date"]).dt.normalize()
    brw = brw.rename(columns={"brw": "brw", "scheduled_meeting": "scheduled"})

    ns = pd.read_excel(SOURCE_ROOT / "BJMW-2025-monetary-policy-shocks-series.xlsx", sheet_name="Data")
    ns["date"] = pd.to_datetime(ns["date"]).dt.normalize()
    ns = ns.rename(
        columns={
            "NSmethod_Nsdata": "ns",
            "Scheduled_FOMC_announcement": "scheduled",
        }
    )
    ns = ns[["date", "scheduled", "ns"]]
    return brw[["date", "scheduled", "brw"]], ns


def read_raw_events() -> pd.DataFrame:
    source = SOURCE_ROOT / "jk_source_old.csv"
    if not source.exists():
        source = RAW_EVENT_URL
    raw = pd.read_csv(source)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    raw = raw[raw["date"].notna()].copy()
    raw["month"] = raw["date"].dt.strftime("%Y-%m")
    if "fomc_latest" in raw:
        raw["fomc_latest_date"] = pd.to_datetime(raw["fomc_latest"], errors="coerce").dt.normalize()
    else:
        raw["fomc_latest_date"] = pd.NaT

    numeric_cols = sorted(
        set(RAW_SHOCK_COLUMNS + MARKET_CONTROLS + ["unscheduled", "main", "ff4_mr"])
    )
    for col in numeric_cols:
        if col in raw:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # The monthly shock file used by fredup keeps 9/17/2001 as a possible
    # event month for controls, but blanks out the high-frequency shock series.
    raw.loc[raw["date"] == pd.Timestamp("2001-09-17"), RAW_SHOCK_COLUMNS] = np.nan

    raw["scheduled"] = 1 - raw.get("unscheduled", 0).fillna(0)
    raw["possible"] = 1
    raw["mps"] = compute_mps_from_pca(raw)
    raw["bs"] = raw["mps"]
    return raw


def prep_events(macro: pd.DataFrame) -> list[dict]:
    prep = pd.read_csv(SOURCE_ROOT / "prep.csv")
    prep.columns = [c.strip() for c in prep.columns]
    prep["date"] = pd.to_datetime(prep["date"]).dt.normalize()
    prep["month"] = prep["date"].dt.strftime("%Y-%m")
    if "fomc_latest" in prep:
        prep["fomc_latest_date"] = pd.to_datetime(prep["fomc_latest"], errors="coerce").dt.normalize()

    events = read_raw_events()

    brw, ns = read_external_shocks()
    events = events.merge(brw, on="date", how="left", suffixes=("", "_brw"))
    events = events.merge(ns, on="date", how="left", suffixes=("", "_ns"))
    if "scheduled" in events:
        events["scheduled"] = pd.to_numeric(events["scheduled"], errors="coerce")
    if "scheduled_ns" in events:
        events["scheduled_ns"] = pd.to_numeric(events["scheduled_ns"], errors="coerce")
    if "scheduled" in events and "scheduled_ns" in events:
        missing_scheduled = events["scheduled"].isna()
        events.loc[missing_scheduled, "scheduled"] = events.loc[missing_scheduled, "scheduled_ns"]
    elif "scheduled_ns" in events:
        events["scheduled"] = events["scheduled_ns"]
    events["scheduled"] = events["scheduled"].fillna(1 - events.get("unscheduled", 0))

    by_date: dict[str, dict] = {}
    control_cols = sorted(set(MARKET_CONTROLS + GREENBOOK_CONTROLS))
    base_cols = [
        "date",
        "month",
        "main",
        "unscheduled",
        "nzlb",
        "possible",
        "scheduled",
        *RAW_SHOCK_COLUMNS,
        "mps",
        "bs",
        "ns",
        "brw",
    ]

    macro_controls = macro.set_index("month", drop=False)
    prep_by_date = {date_key(row["date"]): row for _, row in prep.iterrows()}
    prep_by_fomc = {}
    if "fomc_latest_date" in prep:
        prep_by_fomc = {
            date_key(row["fomc_latest_date"]): row
            for _, row in prep.iterrows()
            if pd.notna(row.get("fomc_latest_date"))
        }

    for _, row in events.iterrows():
        rec = {"source": "jk_source_old"}
        for col in base_cols + control_cols:
            if col == "date":
                rec[col] = date_key(row[col])
            elif col in row:
                rec[col] = value(row[col])

        control_row = prep_by_date.get(rec["date"])
        if control_row is None and pd.notna(row.get("fomc_latest_date")):
            control_row = prep_by_fomc.get(date_key(row["fomc_latest_date"]))
        if control_row is not None:
            for col in control_cols:
                if rec.get(col) is None and col in control_row:
                    rec[col] = value(control_row[col])

        mrow = macro_controls.loc[row["month"]] if row["month"] in macro_controls.index else None
        if mrow is not None:
            rec["nzlb"] = 0 if value(mrow.get("zlb")) == 1 else 1
            for col in MARKET_CONTROLS:
                if rec.get(col) is None and col in mrow:
                    rec[col] = value(mrow[col])
        by_date[rec["date"]] = rec

    external = brw.merge(ns, on="date", how="outer", suffixes=("_brw", "_ns"))
    for _, row in external.iterrows():
        dkey = date_key(row["date"])
        if dkey in by_date:
            if value(row.get("brw")) is not None:
                by_date[dkey]["brw"] = value(row.get("brw"))
            if value(row.get("ns")) is not None:
                by_date[dkey]["ns"] = value(row.get("ns"))
            continue

        scheduled = value(row.get("scheduled_brw"))
        if scheduled is None:
            scheduled = value(row.get("scheduled_ns"))
        rec = {
            "date": dkey,
            "month": month_key(row["date"]),
            "source": "external",
            "main": 1 if scheduled == 1 else 0,
            "unscheduled": 0 if scheduled == 1 else 1,
            "scheduled": scheduled,
            "possible": 1,
            "brw": value(row.get("brw")),
            "ns": value(row.get("ns")),
        }
        mrow = macro_controls.loc[rec["month"]] if rec["month"] in macro_controls.index else None
        if mrow is not None:
            rec["nzlb"] = 0 if value(mrow.get("zlb")) == 1 else 1
            for col in MARKET_CONTROLS:
                if col in mrow:
                    rec[col] = value(mrow[col])
        by_date[dkey] = rec

    return sorted(by_date.values(), key=lambda x: x["date"])


def macro_rows(macro: pd.DataFrame) -> list[dict]:
    cols = ["month", "daten", "zlb"] + [spec["source"] for spec in OUTCOMES.values()]
    cols += MARKET_CONTROLS + FRED_MACRO_CONTROLS
    rows = []
    for _, row in macro.sort_values("daten").iterrows():
        rec = {"month": row["month"], "date": date_key(row["daten"])}
        for col in cols:
            if col in ("month", "daten"):
                continue
            if col in row:
                rec[col] = value(row[col])
        rows.append(rec)
    return rows


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Source repo not found: {SOURCE_ROOT}")

    macro = read_macro()
    events = prep_events(macro)
    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": [
                "jk_source_old.csv",
                "prep.csv",
                "test/mondat.dta",
                "BJMW-BRW-shocks-updated-1.xlsx",
                "BJMW-2025-monetary-policy-shocks-series.xlsx",
            ],
            "defaults": {
                "shock": "mp1",
                "horizon": 24,
                "shock_lags": 12,
                "dependent_lags": 12,
                "macro_lags": 12,
                "aggregation": "main",
                "include_zlb": False,
                "exclude_future_zlb_leads": True,
                "impute_zeros": False,
                "exclude_unscheduled": False,
                "scale_to_ffr_h0_bp": 50,
                "ci": 0.9,
            },
        },
        "shocks": {
            "mp1": {"label": "MP1", "source": "jk_source_old.csv event rows"},
            "ff4": {"label": "FF4", "source": "jk_source_old.csv event rows"},
            "ed4": {"label": "ED4", "source": "jk_source_old.csv event rows"},
            "mps": {"label": "MPS", "source": "PCA(ed1, ed2, ed3, ed4), normalized by ED4"},
            "bs": {"label": "BS", "source": "MPS with mandatory Bauer-Swanson controls"},
            "ns": {"label": "NS", "source": "BJMW NSmethod_Nsdata"},
            "brw": {"label": "BRW", "source": "BJMW latest BRW file"},
        },
        "outcomes": OUTCOMES,
        "controls": {
            "market": MARKET_CONTROLS,
            "greenbook": GREENBOOK_CONTROLS,
            "fred_macro": FRED_MACRO_CONTROLS,
            "bs_mandatory": MARKET_CONTROLS,
        },
        "macro": macro_rows(macro),
        "events": events,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(events)} events, {len(payload['macro'])} monthly rows)")


if __name__ == "__main__":
    main()
