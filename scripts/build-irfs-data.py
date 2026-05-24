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

EVENT_SCALE_COLUMNS = ["ust2y"]
TARGET_FFR_COLUMNS = ["tffr", "dtffr"]

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
    mondat = add_target_ffr_change(mondat)
    return mondat


def add_target_ffr_change(macro: pd.DataFrame) -> pd.DataFrame:
    macro = macro.copy()
    if "tffr" not in macro.columns:
        target_path = SOURCE_ROOT / "codex" / "dff_fedfunds_tffr_monthly.csv"
        if target_path.exists():
            target = pd.read_csv(target_path)
            target["month"] = pd.to_datetime(target["DATE"]).dt.strftime("%Y-%m")
            target["tffr"] = pd.to_numeric(target["TFFR"], errors="coerce")
            macro = macro.merge(target[["month", "tffr"]], on="month", how="left")
    if "tffr" in macro.columns:
        macro["tffr"] = pd.to_numeric(macro["tffr"], errors="coerce")
        if "dtffr" not in macro.columns:
            macro = macro.sort_values("daten")
            macro["dtffr"] = macro["tffr"].diff()
    if "dtffr" in macro.columns:
        macro["dtffr"] = pd.to_numeric(macro["dtffr"], errors="coerce")
    return macro


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


def read_greenbook_monthly_controls() -> dict[str, dict[str, float]]:
    mapping_path = SOURCE_ROOT / "test" / "intermediates" / "GBFOMCmapping.csv"
    workbook_path = SOURCE_ROOT / "test" / "intermediates" / "gbweb_row_format.xlsx"
    if not mapping_path.exists() or not workbook_path.exists():
        return {}

    mapping = pd.read_csv(mapping_path, dtype=str)
    if not {"FOMCdate", "GBdate"}.issubset(mapping.columns):
        return {}
    gb = mapping[["FOMCdate", "GBdate"]].copy()
    gb["fomc"] = pd.to_datetime(gb["FOMCdate"], format="%Y%m%d", errors="coerce")
    gb["gbdate"] = pd.to_datetime(gb["GBdate"], format="%Y%m%d", errors="coerce")
    gb = gb.dropna(subset=["fomc", "gbdate", "GBdate"]).sort_values("fomc")

    for sheet in ["gRGDP", "gPGDP", "UNEMP"]:
        sheet_df = pd.read_excel(workbook_path, sheet_name=sheet)
        sheet_df["GBdate"] = sheet_df["GBdate"].astype(str)
        keep = ["GBdate"] + [col for col in sheet_df.columns if str(col).startswith(sheet)]
        gb = gb.merge(sheet_df[keep], on="GBdate", how="left")

    gb["gb_yq"] = gb["gbdate"].dt.year * 4 + gb["gbdate"].dt.quarter
    same_quarter = gb["gb_yq"].eq(gb["gb_yq"].shift(1))
    next_quarter = gb["gb_yq"].gt(gb["gb_yq"].shift(1))

    for prefix in ["gRGDP", "gPGDP", "UNEMP"]:
        gb[f"D{prefix}B1"] = np.where(
            same_quarter,
            gb[f"{prefix}B1"] - gb[f"{prefix}B1"].shift(1),
            np.where(next_quarter, gb[f"{prefix}B1"] - gb[f"{prefix}F0"].shift(1), np.nan),
        )
        for horizon in range(4):
            current = f"{prefix}F{horizon}"
            previous_next = f"{prefix}F{horizon + 1}"
            gb[f"D{prefix}F{horizon}"] = np.where(
                same_quarter,
                gb[current] - gb[current].shift(1),
                np.where(next_quarter, gb[current] - gb[previous_next].shift(1), np.nan),
            )

    rows: dict[str, dict[str, float]] = {}
    for _, row in gb.sort_values("fomc").iterrows():
        month = month_key(row["fomc"])
        dest = rows.setdefault(month, {})
        for col in GREENBOOK_CONTROLS:
            val = row.get(col)
            if col not in dest and isinstance(val, (int, float, np.integer, np.floating)):
                val = float(val)
                if math.isfinite(val):
                    dest[col] = val
    return rows


def prep_events(macro: pd.DataFrame) -> list[dict]:
    prep = pd.read_csv(SOURCE_ROOT / "prep.csv")
    prep.columns = [c.strip() for c in prep.columns]
    prep["date"] = pd.to_datetime(prep["date"]).dt.normalize()
    prep = prep[prep["date"].notna()].copy()
    prep["month"] = prep["date"].dt.strftime("%Y-%m")

    brw, ns = read_external_shocks()
    events = prep.merge(brw[["date", "brw"]], on="date", how="left")
    events = events.merge(ns[["date", "ns"]], on="date", how="left")

    numeric_cols = sorted(
        set(
            RAW_SHOCK_COLUMNS
            + EVENT_SCALE_COLUMNS
            + MARKET_CONTROLS
            + GREENBOOK_CONTROLS
            + ["unscheduled", "main", "nzlb", "possible", "scheduled", "ff4_mr", "brw", "ns"]
        )
    )
    for col in numeric_cols:
        if col in events:
            events[col] = pd.to_numeric(events[col], errors="coerce")

    if "scheduled" not in events:
        unscheduled = events.get("unscheduled", pd.Series(0, index=events.index)).fillna(0)
        events["scheduled"] = 1 - unscheduled
    else:
        events["scheduled"] = events["scheduled"].fillna(
            1 - events.get("unscheduled", pd.Series(0, index=events.index)).fillna(0)
        )
    if "possible" not in events:
        events["possible"] = 1
    else:
        events["possible"] = events["possible"].fillna(1)
    if "mps" not in events or events["mps"].notna().sum() == 0:
        events["mps"] = compute_mps_from_pca(events)
    if "bs" not in events or events["bs"].notna().sum() == 0:
        events["bs"] = events["mps"]

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
        *EVENT_SCALE_COLUMNS,
        *RAW_SHOCK_COLUMNS,
        "mps",
        "bs",
        "ns",
        "brw",
    ]

    for _, row in events.iterrows():
        rec = {"source": "prep"}
        for col in base_cols + control_cols:
            if col == "date":
                rec[col] = date_key(row[col])
            elif col in row:
                rec[col] = value(row[col])

        by_date[rec["date"]] = rec

    return sorted(by_date.values(), key=lambda x: x["date"])


def macro_rows(macro: pd.DataFrame, events: list[dict]) -> list[dict]:
    cols = ["month", "daten", "zlb"] + [spec["source"] for spec in OUTCOMES.values()]
    cols += MARKET_CONTROLS + FRED_MACRO_CONTROLS + TARGET_FFR_COLUMNS + EVENT_SCALE_COLUMNS
    monthly_greenbook_controls = read_greenbook_monthly_controls()
    monthly_event_values: dict[str, dict[str, float]] = {}
    monthly_event_controls: dict[str, dict[str, float]] = {}
    for row in sorted(events, key=lambda x: x["date"]):
        month = row.get("month")
        if not month:
            continue
        dest = monthly_event_values.setdefault(month, {})
        for col in EVENT_SCALE_COLUMNS:
            val = row.get(col)
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                dest[col] = dest.get(col, 0.0) + float(val)
        control_dest = monthly_event_controls.setdefault(month, {})
        for col in MARKET_CONTROLS + GREENBOOK_CONTROLS:
            val = row.get(col)
            if col not in control_dest and isinstance(val, (int, float)) and math.isfinite(float(val)):
                control_dest[col] = float(val)
    rows = []
    for _, row in macro.sort_values("daten").iterrows():
        rec = {"month": row["month"], "date": date_key(row["daten"])}
        for col in cols:
            if col in ("month", "daten"):
                continue
            if col in row:
                rec[col] = value(row[col])
        if row["month"] in monthly_event_values:
            for col, val in monthly_event_values[row["month"]].items():
                rec[col] = value(val)
        if row["month"] in monthly_event_controls:
            for col, val in monthly_event_controls[row["month"]].items():
                if not isinstance(rec.get(col), (int, float)):
                    rec[col] = value(val)
        if row["month"] in monthly_greenbook_controls:
            for col, val in monthly_greenbook_controls[row["month"]].items():
                if not isinstance(rec.get(col), (int, float)):
                    rec[col] = value(val)
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
                "prep.csv",
                "test/mondat.dta",
                "codex/dff_fedfunds_tffr_monthly.csv",
                "test/intermediates/GBFOMCmapping.csv",
                "test/intermediates/gbweb_row_format.xlsx",
                "BJMW-BRW-shocks-updated-1.xlsx",
                "BJMW-2025-monetary-policy-shocks-series.xlsx",
            ],
            "defaults": {
                "series": "mp1",
                "shock": "mp1",
                "horizon": 24,
                "shock_lags": 12,
                "use_whitening": False,
                "whitening_lags": 8,
                "dependent_lags": 12,
                "macro_lags": 12,
                "aggregation": "main",
                "included_years": "1993-2025",
                "include_zlb": False,
                "include_covid": False,
                "impute_zeros": False,
                "exclude_unscheduled": False,
                "scale_mode": "ust2y-impact",
                "scale_to_ffr_h0_bp": 50,
                "scale_to_ust2y_bp": 10,
                "scale_to_shock_sd": 1,
                "ci": 0.9,
            },
        },
        "series": {
            "mp1": {"label": "MP1", "source": "prep.csv"},
            "ff4": {"label": "FF4", "source": "prep.csv"},
            "ed4": {"label": "ED4", "source": "prep.csv"},
            "mps": {"label": "MPS", "source": "PCA(ed1, ed2, ed3, ed4), normalized by ED4"},
            "bs": {"label": "BS", "source": "MPS with mandatory Bauer-Swanson controls"},
            "ns": {"label": "NS", "source": "BJMW NSmethod_Nsdata"},
            "brw": {"label": "BRW", "source": "BJMW latest BRW file"},
            "dtffr": {
                "label": "Target FFR Change",
                "source": "Fredup target FFR monthly change",
                "frequency": "monthly",
            },
        },
        "outcomes": OUTCOMES,
        "controls": {
            "market": MARKET_CONTROLS,
            "greenbook": GREENBOOK_CONTROLS,
            "fred_macro": FRED_MACRO_CONTROLS,
            "bs_mandatory": MARKET_CONTROLS,
        },
        "macro": macro_rows(macro, events),
        "events": events,
    }
    payload["shocks"] = payload["series"]
    OUTPUT_PATH.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({len(events)} events, {len(payload['macro'])} monthly rows)")


if __name__ == "__main__":
    main()
