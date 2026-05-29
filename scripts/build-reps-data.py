#!/usr/bin/env python3
"""Build the static data snapshot for pages/reps.html.

ALL inputs are pulled from the (private) GitHub repo `mindcraft1997/misc` at
build time -- nothing is read from a local working tree. Authentication uses a
GitHub token from $GITHUB_TOKEN, else `gh auth token`.

    python3 scripts/build-reps-data.py            # writes pages/reps-data.json

Replicates the forecast-disagreement construction in misc/romer/monthlydisag.do
and misc/romer/stablediag.do:

  dp_h = gPCPIF_h(Greenbook CPI)  - cpi_h (Blue Chip)      "CBI"  (inflation gap)
  dg_h = gRGDPF_h(Greenbook RGDP) - rgdp_h(Blue Chip)      "RTN"  (growth gap)
  df_h = ffr_h  (Greenbook FFR)   - ff_h (Blue Chip)              (FFR gap)
  du_h = UNEMPF_h(Greenbook UR)   - un_h (Blue Chip, BCEI only)   (UR gap)

Greenbook forecast levels: misc/romer/intermediates/gbweb_row_format.xlsx
Greenbook FFR path:        misc/romer/rpred.csv
GB date -> FOMC mapping:   misc/romer/intermediates/GBFOMCmapping.csv
HF surprises + BS controls:misc/romer/prep.csv  (mp1, ed1-4, ff4, nfp_surp, ...)
GSS surprise PCs:          misc/ForecastErrors/fedfunds_forecast_errors_gss.csv
Blue Chip consensus:       misc/romer/{bcff,bcei,combined}_consensus.csv, bcff_med.csv
Blue Chip panel:           misc/romer/bcff_panel.csv  (individual forecasters)
Timed GB-vs-BCFF FFR:      misc/ForecastErrors/gb_vs_bcff_fedfunds_timed.csv
"""
from __future__ import annotations
import io, json, math, os, subprocess, urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO = "mindcraft1997/misc"
OUT  = os.environ.get("REPS_OUT", "reps-data.json")
H    = range(5)                      # horizons 0..4
BS   = ["nfp_surp","nfp_12m","sp500_3m","slope_3m","bcom_3m","tr_skew"]


def token() -> str:
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if t: return t
    return subprocess.check_output(["gh","auth","token"], text=True).strip()

_TOK = token()

def fetch(path: str) -> bytes:
    """Download a file from the misc repo (any size) via the contents API."""
    api = f"https://api.github.com/repos/{REPO}/contents/{path}"
    req = urllib.request.Request(api, headers={
        "Authorization": f"Bearer {_TOK}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "build-reps-data",
    })
    meta = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dl = urllib.request.Request(meta["download_url"], headers={"User-Agent": "build-reps-data"})
    return urllib.request.urlopen(dl, timeout=120).read()

def csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(fetch(path)), **kw)

def gbdate_dt(s):
    return pd.to_datetime(s.astype(str).str.slice(0,8), format="%Y%m%d", errors="coerce")

def num(s):
    return pd.to_numeric(s, errors="coerce")

def rnd(x, d=5):
    if x is None: return None
    try: x = float(x)
    except (TypeError, ValueError): return None
    if not math.isfinite(x): return None
    return round(x, d)


# ---------------------------------------------------------------- Greenbook
def greenbook() -> pd.DataFrame:
    """Greenbook forecast levels by FOMC month, with quarter-mismatch rollover."""
    xls = pd.ExcelFile(io.BytesIO(fetch("romer/intermediates/gbweb_row_format.xlsx")))
    gb = None
    for sheet in ["gRGDP","gPCPI","UNEMP"]:
        df = xls.parse(sheet)
        df["GBdate"] = df["GBdate"].astype(str)
        keep = ["GBdate"] + [c for c in df.columns if str(c).startswith(sheet)]
        df = df[keep]
        gb = df if gb is None else gb.merge(df, on="GBdate", how="outer")
    rp = csv("romer/rpred.csv")
    rp["GBdate"] = rp["GBdate"].astype(str)
    ffr_cols = [f"ffr{i}" for i in range(10) if f"ffr{i}" in rp.columns]
    gb = gb.merge(rp[["GBdate"]+ffr_cols], on="GBdate", how="left")
    mp = csv("romer/intermediates/GBFOMCmapping.csv", dtype=str)
    gb = gb.merge(mp[["GBdate","FOMCdate"]], on="GBdate", how="left")
    gb["gbdt"]   = gbdate_dt(gb["GBdate"])
    gb["fomcdt"] = gbdate_dt(gb["FOMCdate"])
    gb = gb.dropna(subset=["fomcdt"]).sort_values("gbdt").reset_index(drop=True)
    gb["month"] = gb["fomcdt"].dt.strftime("%Y-%m")
    gbYQ = gb["gbdt"].dt.year*4 + gb["gbdt"].dt.quarter
    # NOTE: monthlydisag.do / stablediag.do apply a quarter-mismatch rollover gated on
    # `mismatch` built from the surprise file (scheduled-FOMC quarter vs surprise-release
    # quarter). On the analysis sample that flag is identically 0, so the rollover never
    # fires on any kept row. We therefore do NOT roll horizons here: GB forecast levels
    # (and the FFR path, which the .do files never roll) are used as published for the
    # FOMC month. This keeps dp/dg/df exactly matching the .do files on the sample.
    # GB forecast revisions (same-quarter vs next-quarter), retained for optional "news"
    for pre in ["gRGDP","gPCPI","UNEMP"]:
        same = gbYQ.eq(gbYQ.shift(1))
        nxt  = gbYQ.gt(gbYQ.shift(1))
        for i in range(5):
            cur = num(gb[f"{pre}F{i}"])
            gb[f"D{pre}F{i}"] = np.where(same, cur-cur.shift(1),
                                np.where(nxt, cur-num(gb[f"{pre}F{i+1}"]).shift(1), np.nan))
    # one GB row per FOMC month
    return gb.drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)


# ---------------------------------------------------------------- consensus
def consensus(path: str) -> pd.DataFrame:
    c = csv(path)
    c["month"] = pd.to_datetime(c["date"], errors="coerce").dt.strftime("%Y-%m")
    return c.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="first")


# ---------------------------------------------------------------- surprises
def surprises() -> pd.DataFrame:
    prep = csv("romer/prep.csv")
    prep["dt"] = pd.to_datetime(prep["daten"], format="%d%b%Y", errors="coerce")
    prep["month"] = prep["dt"].dt.strftime("%Y-%m")
    cols = ["mp1","ff4","ed1","ed2","ed3","ed4"] + BS + ["unscheduled","main"]
    for c in cols:
        if c in prep: prep[c] = num(prep[c])
    prep = prep.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="first")
    # GSS PCs (a distinct surprise measure) from ForecastErrors
    gss = csv("ForecastErrors/fedfunds_forecast_errors_gss.csv")
    gss["month"] = pd.to_datetime(gss["date"], errors="coerce").dt.strftime("%Y-%m")
    gss = gss.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="first")
    prep = prep.merge(gss[["month","gss_pc1","gss_pc2"]], on="month", how="left")
    return prep


def standshock(s: pd.Series) -> pd.Series:
    """Replicate romer/standshock.ado: mean/sd over NON-ZERO values, zeros stay zero."""
    s = num(s)
    mask = s.notna() & (s != 0)
    mu, sd = s[mask].mean(), s[mask].std()
    out = s.copy()
    out[mask] = (s[mask] - mu) / sd
    out[s == 0] = 0.0
    return out


def add_mps_and_std(d: pd.DataFrame) -> pd.DataFrame:
    """mps = PCA(ed1..ed4) normalised by ED4 loading; standardise key shocks (standshock)."""
    sub = d.dropna(subset=["ed1","ed2","ed3","ed4"]).copy()
    if len(sub) >= 8:
        X = sub[["ed1","ed2","ed3","ed4"]].astype(float).values
        Xs = (X - X.mean(0)) / X.std(0)
        _,_,vt = np.linalg.svd(Xs, full_matrices=False)
        pc1 = Xs @ vt[0]
        b = np.polyfit(pc1, sub["ed4"].astype(float).values, 1)[0]
        d.loc[sub.index, "mps"] = 0.01 * pc1 / b
    for c in ["mp1","ff4","ed4","mps"]:
        d[c+"_std"] = standshock(d[c])
    return d


# ---------------------------------------------------------------- gaps
def gaps(d: pd.DataFrame, gb_pre: str, bc_pre: str) -> list[str]:
    """Create gap columns <out>_h = GB level - BC level for h in 0..4; return names."""
    names = []
    for i in H:
        gbcol = f"{gb_pre}{i}"; bccol = f"{bc_pre}_{i}"
        out = f"{bc_pre}gap_{i}"
        if gbcol in d and bccol in d:
            d[out] = num(d[gbcol]) - num(d[bccol]); names.append(out)
    return names


# ---------------------------------------------------------------- main
def main() -> None:
    gb = greenbook()
    surp = add_mps_and_std(surprises())

    SERIES = {  # forecast-series label -> consensus file (Blue Chip side)
        "bcff":     "romer/bcff_consensus.csv",
        "bcei":     "romer/bcei_consensus.csv",
        "combined": "romer/combined_consensus.csv",
        "med":      "romer/bcff_med.csv",
    }
    GBMAP = {"cpi":"gPCPIF","rgdp":"gRGDPF","ff":"ffr","un":"UNEMPF"}  # BC var -> GB col prefix

    # assemble per-series monthly gap frames, all keyed by month
    series_frames = {}
    for name, path in SERIES.items():
        bc = consensus(path)
        merged = bc.merge(gb, on="month", how="inner", suffixes=("","_gb"))
        block = {"month": merged["month"]}
        for bcvar, gbpre in GBMAP.items():
            for i in H:
                gbcol, bccol = f"{gbpre}{i}", f"{bcvar}_{i}"
                if gbcol in merged and bccol in merged:
                    key = {"cpi":"dp","rgdp":"dg","ff":"df","un":"du"}[bcvar]
                    block[f"{key}_{i}"] = num(merged[gbcol]) - num(merged[bccol])
        series_frames[name] = pd.DataFrame(block)

    # master monthly frame = union of months that appear in any series, + surprises
    all_months = sorted(set().union(*[set(f["month"]) for f in series_frames.values()]))
    base = pd.DataFrame({"month": all_months})
    base = base.merge(surp[["month","mp1","ff4","ed4","mps","mp1_std","ff4_std","ed4_std",
                            "mps_std","gss_pc1","gss_pc2"]+BS+["unscheduled","main"]],
                      on="month", how="left")
    base["year"] = pd.to_datetime(base["month"]+"-01").dt.year
    m = base["month"]
    base["nzlb"] = ~(((m>="2008-11")&(m<="2015-11"))|((m>="2020-03")&(m<="2022-02")))

    monthly = []
    sf = {n: f.set_index("month") for n,f in series_frames.items()}
    for _, r in base.iterrows():
        mo = r["month"]
        rec = {
            "month": mo, "year": int(r["year"]),
            "nzlb": bool(r["nzlb"]),
            "scheduled": (None if pd.isna(r.get("unscheduled")) else int(r["unscheduled"]) == 0),
            "shocks": {k: rnd(r.get(k)) for k in
                       ["mp1","ff4","ed4","mps","mp1_std","ff4_std","ed4_std","mps_std","gss_pc1","gss_pc2"]},
            "bs": {k: rnd(r.get(k)) for k in BS},
            "series": {},
        }
        for name, frame in sf.items():
            if mo in frame.index:
                row = frame.loc[mo]
                blk = {}
                for key in ["dp","dg","df","du"]:
                    vals = [rnd(row.get(f"{key}_{i}")) for i in H]
                    if any(v is not None for v in vals): blk[key] = vals
                if blk: rec["series"][name] = blk
        monthly.append(rec)

    # --------------------------- panel (individual BCFF forecasters) ---------
    pan = csv("romer/bcff_panel.csv")
    pan["month"] = pd.to_datetime(pan["date"], errors="coerce").dt.strftime("%Y-%m")
    gb_idx = gb.set_index("month")
    nzlb_by_month = dict(zip(base["month"], base["nzlb"]))
    panel = []
    gb_have = set(gb_idx.index)
    for _, r in pan.iterrows():
        mo = r["month"]
        if mo not in gb_have: continue
        g = gb_idx.loc[mo]
        if isinstance(g, pd.DataFrame): g = g.iloc[0]
        rec = {"id": int(r["IDn"]) if pd.notna(r["IDn"]) else None, "month": mo,
               "nzlb": bool(nzlb_by_month.get(mo, ((mo<"2008-11")or(mo>"2015-11")) and ((mo<"2020-03")or(mo>"2022-02"))))}
        any_gap = False
        for key, gbpre, bcv in [("dp","gPCPIF","cpi"),("dg","gRGDPF","rgdp"),("df","ffr","ff")]:
            vals = []
            for i in H:
                gv = g.get(f"{gbpre}{i}"); bv = r.get(f"{bcv}_{i}")
                v = rnd((float(gv)-float(bv)) if pd.notna(gv) and pd.notna(bv) else None)
                vals.append(v)
                if v is not None: any_gap = True
            rec[key] = vals
        if rec["id"] is not None and any_gap:
            panel.append(rec)

    # --------------------------- timed GB-vs-BCFF FFR (aux for ex.2) ---------
    timed = csv("ForecastErrors/gb_vs_bcff_fedfunds_timed.csv")
    timed["year"] = pd.to_datetime(timed["pub_date"], errors="coerce").dt.year
    aux = []
    for _, r in timed.iterrows():
        if pd.isna(r.get("gb_fcst")) or pd.isna(r.get("bc_fcst")): continue
        aux.append({"pub": str(r["pub_date"])[:10], "year": int(r["year"]),
                    "q": int(r["Qahead"]) if pd.notna(r["Qahead"]) else None,
                    "gap": rnd(float(r["gb_fcst"])-float(r["bc_fcst"])),
                    "gb_err": rnd(r.get("gb_err")), "bc_err": rnd(r.get("bc_err"))})

    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo": REPO,
            "sources": [
                "romer/intermediates/gbweb_row_format.xlsx (Greenbook gRGDP/gPCPI/UNEMP)",
                "romer/rpred.csv (Greenbook FFR path)",
                "romer/intermediates/GBFOMCmapping.csv",
                "romer/prep.csv (HF surprises + Bauer-Swanson controls)",
                "ForecastErrors/fedfunds_forecast_errors_gss.csv (GSS PCs)",
                "romer/bcff_consensus.csv, bcei_consensus.csv, combined_consensus.csv, bcff_med.csv",
                "romer/bcff_panel.csv (individual BCFF forecasters)",
                "ForecastErrors/gb_vs_bcff_fedfunds_timed.csv",
            ],
            "construction": {
                "dp": "gPCPIF_h - cpi_h  (Greenbook CPI - Blue Chip)   [CBI / inflation gap]",
                "dg": "gRGDPF_h - rgdp_h (Greenbook RGDP - Blue Chip)  [RTN / growth gap]",
                "df": "ffr_h - ff_h      (Greenbook FFR - Blue Chip)   [FFR gap]",
                "du": "UNEMPF_h - un_h   (Greenbook UR - Blue Chip; BCEI only)",
                "mps": "0.01 * PC1(ed1..ed4) / loading(ED4)  (Nakamura-Steinsson style)",
                "note": "Faithful to misc/romer/monthlydisag.do & stablediag.do; bcff_time intra-quarter shift omitted.",
            },
            "defaults": {"series": "bcff", "min_year": 1993, "exclude_zlb": True},
        },
        "shocks": {
            "mp1_std": {"label": "MP1 (std)"}, "ff4_std": {"label": "FF4 (std)"},
            "ed4_std": {"label": "ED4 (std)"}, "mps_std": {"label": "MPS (std)"},
            "mp1": {"label": "MP1 (raw)"}, "ff4": {"label": "FF4 (raw)"},
            "ed4": {"label": "ED4 (raw)"}, "mps": {"label": "MPS (raw)"},
            "gss_pc1": {"label": "GSS PC1"}, "gss_pc2": {"label": "GSS PC2"},
        },
        "forecast_series": {
            "bcff": {"label": "BCFF consensus"}, "bcei": {"label": "BCEI consensus"},
            "combined": {"label": "Combined consensus"}, "med": {"label": "BCFF median"},
        },
        "bs_controls": BS,
        "gap_labels": {"dp": "CBI (CPI gap)", "dg": "RTN (RGDP gap)",
                       "df": "FFR gap", "du": "UR gap"},
        "monthly": monthly,
        "panel": panel,
        "timed_ffr": aux,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    n_panel_ids = len({p["id"] for p in panel})
    print(f"Wrote {OUT}: {len(monthly)} monthly rows, {len(panel)} panel rows "
          f"({n_panel_ids} forecasters), {len(aux)} timed-FFR rows.")


if __name__ == "__main__":
    main()
