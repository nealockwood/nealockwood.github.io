#!/usr/bin/env python3
"""Build the static data snapshot for pages/reps.html.

ALL inputs are pulled from the (private) GitHub repo `mindcraft1997/misc` at
build time -- nothing is read from a local working tree. Authentication uses a
GitHub token from $GITHUB_TOKEN, else `gh auth token`.

    python3 scripts/build-reps-data.py            # writes reps-data.json

Replicates the forecast-disagreement construction in misc/romer/monthlydisag.do,
stablediag.do and disaglp.do.

Terminology (as used by the project):
  RTN = "real-time news"          = the Bauer-Swanson (2023) pre-meeting controls
                                    (nfp_surp, nfp_12m, sp500_3m, slope_3m, bcom_3m, tr_skew).
  CBI = "central-bank information" = differences between the Fed (Greenbook) and a
                                    private (Blue Chip) forecast, per macro variable:
        cpi  gap dp_h = gPCPIF_h - cpi_h
        rgdp gap dg_h = gRGDPF_h - rgdp_h
        un   gap du_h = UNEMPF_h - un_h        (private UR forecast: BCEI, the only source)
        ffr  gap df_h = E^Fed[i] - E^Mkt[i]    (Greenbook FFR path - Blue Chip FFR)

Timing: BCFF surveys are sometimes timed a month off from the Greenbook. For BCFF
the cpi/rgdp gaps apply the monthlydisag.do `bcff_time==1` forward shift, and the
FFR gap is taken from the already-timed ForecastErrors/gb_vs_bcff_fedfunds_timed.csv.
The combined consensus is aligned by construction (no shift).

Sources (all in misc):
  romer/intermediates/gbweb_row_format.xlsx   Greenbook gRGDP/gPCPI/UNEMP levels
  romer/rpred.csv                             Greenbook FFR path
  romer/intermediates/GBFOMCmapping.csv       GBdate <-> FOMCdate
  romer/prep.csv                              HF surprises + RTN controls + scheduling
  ForecastErrors/fedfunds_forecast_errors_gss.csv   GSS PCs
  romer/bcff_consensus.csv, bcei_consensus.csv, combined_consensus.csv, bcff_med.csv
  romer/bcff_panel.csv                        individual BCFF forecasters
  ForecastErrors/gb_vs_bcff_fedfunds_timed.csv  timing-correct GB-vs-BCFF FFR + bcff_time
"""
from __future__ import annotations
import io, json, math, os, subprocess, urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO = "mindcraft1997/misc"
OUT  = os.environ.get("REPS_OUT", "reps-data.json")
H    = range(5)
BS   = ["nfp_surp","nfp_12m","sp500_3m","slope_3m","bcom_3m","tr_skew"]   # RTN
GB_PRE   = {"cpi":"gPCPIF","rgdp":"gRGDPF","un":"UNEMPF","ffr":"ffr"}      # Greenbook column prefix
PRIV_PRE = {"cpi":"cpi","rgdp":"rgdp","un":"un","ffr":"ff"}                # Blue Chip column base (FFR is ff_)


def token() -> str:
    t = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return t or subprocess.check_output(["gh","auth","token"], text=True).strip()
_TOK = token()

def fetch(path: str) -> bytes:
    api = f"https://api.github.com/repos/{REPO}/contents/{path}"
    req = urllib.request.Request(api, headers={
        "Authorization": f"Bearer {_TOK}", "Accept": "application/vnd.github+json",
        "User-Agent": "build-reps-data"})
    meta = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dl = urllib.request.Request(meta["download_url"], headers={"User-Agent": "build-reps-data"})
    return urllib.request.urlopen(dl, timeout=120).read()

def csv(path, **kw): return pd.read_csv(io.BytesIO(fetch(path)), **kw)
def gbdate_dt(s): return pd.to_datetime(s.astype(str).str.slice(0,8), format="%Y%m%d", errors="coerce")
def num(s): return pd.to_numeric(s, errors="coerce")
def rnd(x, d=5):
    if x is None: return None
    try: x = float(x)
    except (TypeError, ValueError): return None
    return round(x, d) if math.isfinite(x) else None


def greenbook() -> pd.DataFrame:
    """Greenbook forecast levels by FOMC month (no rollover; .do mismatch flag is 0 on sample)."""
    xls = pd.ExcelFile(io.BytesIO(fetch("romer/intermediates/gbweb_row_format.xlsx")))
    gb = None
    for sheet in ["gRGDP","gPCPI","UNEMP"]:
        df = xls.parse(sheet); df["GBdate"] = df["GBdate"].astype(str)
        keep = ["GBdate"] + [c for c in df.columns if str(c).startswith(sheet)]
        gb = df[keep] if gb is None else gb.merge(df[keep], on="GBdate", how="outer")
    rp = csv("romer/rpred.csv"); rp["GBdate"] = rp["GBdate"].astype(str)
    ffr = [f"ffr{i}" for i in range(10) if f"ffr{i}" in rp.columns]
    gb = gb.merge(rp[["GBdate"]+ffr], on="GBdate", how="left")
    mp = csv("romer/intermediates/GBFOMCmapping.csv", dtype=str)
    gb = gb.merge(mp[["GBdate","FOMCdate"]], on="GBdate", how="left")
    gb["fomcdt"] = gbdate_dt(gb["FOMCdate"])
    gb = gb.dropna(subset=["fomcdt"]).sort_values("fomcdt")
    gb["month"] = gb["fomcdt"].dt.strftime("%Y-%m")
    return gb.drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)


def consensus(path):
    c = csv(path); c["month"] = pd.to_datetime(c["date"], errors="coerce").dt.strftime("%Y-%m")
    return c.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="first").sort_values("month").reset_index(drop=True)


def standshock(s):
    """romer/standshock.ado: mean/sd over NON-ZERO values; zeros stay zero."""
    s = num(s); mask = s.notna() & (s != 0); mu, sd = s[mask].mean(), s[mask].std()
    out = s.copy(); out[mask] = (s[mask]-mu)/sd; out[s == 0] = 0.0
    return out


def surprises():
    prep = csv("romer/prep.csv")
    prep["dt"] = pd.to_datetime(prep["daten"], format="%d%b%Y", errors="coerce")
    prep["month"] = prep["dt"].dt.strftime("%Y-%m")
    for c in ["mp1","ff4","ed1","ed2","ed3","ed4","main"]+BS:
        if c in prep: prep[c] = num(prep[c])
    prep = prep.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="first")
    gss = csv("ForecastErrors/fedfunds_forecast_errors_gss.csv")
    gss["month"] = pd.to_datetime(gss["date"], errors="coerce").dt.strftime("%Y-%m")
    gss = gss.dropna(subset=["month"]).drop_duplicates(subset=["month"], keep="first")
    prep = prep.merge(gss[["month","gss_pc1","gss_pc2"]], on="month", how="left")
    # mps = 0.01 * PC1(ed1..ed4) / loading(ED4)
    sub = prep.dropna(subset=["ed1","ed2","ed3","ed4"]).copy()
    if len(sub) >= 8:
        X = sub[["ed1","ed2","ed3","ed4"]].astype(float).values
        Xs = (X - X.mean(0))/X.std(0); _,_,vt = np.linalg.svd(Xs, full_matrices=False)
        pc1 = Xs @ vt[0]; b = np.polyfit(pc1, sub["ed4"].astype(float).values, 1)[0]
        prep.loc[sub.index, "mps"] = 0.01*pc1/b
    for c in ["mp1","ff4","ed4","mps"]: prep[c+"_std"] = standshock(prep[c])
    return prep


def bcff_time_flag():
    """Per-month bcff_time flag (the only misc source). Used for the monthlydisag.do
    dp/dg forward shift. NOTE: df (the FFR gap) is NOT shifted in monthlydisag --
    it is plain ffr_i - ff_i -- so we do not use this file's timed FFR values."""
    t = csv("ForecastErrors/gb_vs_bcff_fedfunds_timed.csv")
    t["month"] = pd.to_datetime(t["pub_date"], errors="coerce").dt.strftime("%Y-%m")
    return t.groupby("month")["bcff_time"].first().to_dict()


def shift_bcff(bc, bcff_time, varbases):
    """Apply monthlydisag.do bcff_time==1 forward shift: use next month's BC level."""
    bc = bc.copy()
    flag = bc["month"].map(lambda m: bcff_time.get(m, 0) == 1)
    for v in varbases:
        for i in H:
            col = f"{v}_{i}"
            if col in bc: bc.loc[flag, col] = bc[col].shift(-1)[flag]
    return bc


def main():
    gb = greenbook(); surp = surprises()
    bcff_time = bcff_time_flag()
    gb_idx = gb.set_index("month")

    SERIES = {"bcff":"romer/bcff_consensus.csv", "bcei":"romer/bcei_consensus.csv",
              "combined":"romer/combined_consensus.csv", "med":"romer/bcff_med.csv"}
    # BCEI unemployment is the only private UR forecast -> du source for every series
    bcei = consensus("romer/bcei_consensus.csv").set_index("month")
    un_by_month = {m: [rnd(num(bcei.loc[m]).get(f"un_{i}")) for i in H] for m in bcei.index}

    def gap(gbrow, bcrow, var):
        out = []
        for i in H:
            gv = gbrow.get(f"{GB_PRE[var]}{i}"); bv = bcrow.get(f"{PRIV_PRE[var]}_{i}")
            out.append(rnd((float(gv)-float(bv)) if pd.notna(gv) and pd.notna(bv) else None))
        return out

    series_data = {}
    for name, path in SERIES.items():
        bc = consensus(path)
        if name == "bcff":
            bc = shift_bcff(bc, bcff_time, ["cpi","rgdp"])     # timing shift for cpi/rgdp
        bc = bc.set_index("month")
        rows = {}
        for m in bc.index:
            if m not in gb_idx.index: continue
            g = gb_idx.loc[m]; r = bc.loc[m]
            cbi, raw = {}, {}
            if "cpi_0" in bc.columns:  cbi["cpi"]  = gap(g, r, "cpi");  raw["cpi"]  = [rnd(r.get(f"cpi_{i}")) for i in H]
            if "rgdp_0" in bc.columns: cbi["rgdp"] = gap(g, r, "rgdp"); raw["rgdp"] = [rnd(r.get(f"rgdp_{i}")) for i in H]
            # FFR gap df_h = ffr_h - ff_h (Greenbook FFR path - Blue Chip), NO shift (per monthlydisag.do)
            if "ff_0" in bc.columns:
                cbi["ffr"] = gap(g, r, "ffr"); raw["ffr"] = [rnd(r.get(f"ff_{i}")) for i in H]
            # unemployment gap from BCEI (or own if BCEI series)
            if m in un_by_month:
                un = num(bcei.loc[m]) if name == "bcei" else None
                gbun = [g.get(f"UNEMPF{i}") for i in H]
                cbi["un"] = [rnd((float(gbun[i])-un_by_month[m][i]) if gbun[i] is not None
                                 and pd.notna(gbun[i]) and un_by_month[m][i] is not None else None) for i in H]
                raw["un"] = un_by_month[m]
            rows[m] = {"cbi": {k:v for k,v in cbi.items() if any(x is not None for x in v)},
                       "raw": {k:v for k,v in raw.items() if any(x is not None for x in v)}}
        series_data[name] = rows

    all_months = sorted(set().union(*[set(r) for r in series_data.values()]))
    base = surp.set_index("month")
    monthly = []
    for m in all_months:
        s = base.loc[m] if m in base.index else None
        mainflag = (None if s is None or pd.isna(s.get("main")) else int(s["main"]) == 1)
        nzlb = ((m<"2008-11") or (m>"2015-11")) and ((m<"2020-03") or (m>"2022-02"))
        g = gb_idx.loc[m] if m in gb_idx.index else None
        rec = {"month": m, "year": int(m[:4]), "nzlb": bool(nzlb), "main": mainflag,
               "shocks": {k: rnd(None if s is None else s.get(k)) for k in
                          ["mp1","ff4","ed4","mps","mp1_std","ff4_std","ed4_std","mps_std","gss_pc1","gss_pc2"]},
               "rtn": {k: rnd(None if s is None else s.get(k)) for k in BS},
               "gbraw": ({v: [rnd(g.get(f"{GB_PRE[v]}{i}")) for i in H] for v in ["cpi","rgdp","un","ffr"]} if g is not None else {}),
               "series": {name: series_data[name][m] for name in SERIES if m in series_data[name]}}
        monthly.append(rec)

    # --------------- panel (individual BCFF forecasters) ----------------
    pan = csv("romer/bcff_panel.csv")
    pan["month"] = pd.to_datetime(pan["date"], errors="coerce").dt.strftime("%Y-%m")
    nzlb_by = {r["month"]: r["nzlb"] for r in monthly}
    main_by = {r["month"]: r["main"] for r in monthly}
    panel = []
    for _, r in pan.iterrows():
        m = r["month"]
        if m not in gb_idx.index or pd.isna(r["IDn"]): continue
        g = gb_idx.loc[m]
        cbi = {}
        for var, pre in [("cpi","gPCPIF"),("rgdp","gRGDPF")]:
            vals = [rnd((float(g.get(f"{pre}{i}"))-float(r.get(f"{var}_{i}")))
                        if pd.notna(g.get(f"{pre}{i}")) and pd.notna(r.get(f"{var}_{i}")) else None) for i in H]
            if any(v is not None for v in vals): cbi[var] = vals
        ffr = [rnd((float(g.get(f"ffr{i}"))-float(r.get(f"ff_{i}")))
                   if pd.notna(g.get(f"ffr{i}")) and pd.notna(r.get(f"ff_{i}")) else None) for i in H]
        if any(v is not None for v in ffr): cbi["ffr"] = ffr
        if cbi:
            panel.append({"id": int(r["IDn"]), "month": m, "nzlb": bool(nzlb_by.get(m, True)),
                          "main": main_by.get(m), "bcff_time": int(bcff_time.get(m, 0)), "cbi": cbi})

    payload = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo": REPO,
            "labels": {
                "RTN": "Real-time news (Bauer-Swanson pre-meeting controls)",
                "CBI": "Central-bank information (Fed - private forecast gaps)",
                "df":  "FFR forecast gap = E^Fed[i] - E^Mkt[i] (Greenbook - Blue Chip)",
            },
            "cbi_vars": {"cpi":"CPI inflation gap (dp)","rgdp":"real GDP growth gap (dg)",
                         "un":"unemployment gap (du; private = BCEI)","ffr":"FFR gap (df)"},
            "construction_note": ("Faithful to romer/monthlydisag.do, stablediag.do, disaglp.do. "
                "BCFF cpi/rgdp gaps apply the bcff_time==1 forward shift; BCFF FFR gap uses the "
                "timing-correct gb_vs_bcff_fedfunds_timed.csv; combined consensus aligned by construction."),
            "defaults": {"series":"bcff","min_year":1993,"exclude_zlb":True,"main_only":True},
        },
        "shocks": {"mp1_std":{"label":"MP1"},"ff4_std":{"label":"FF4"},"ed4_std":{"label":"ED4"},
                   "mps_std":{"label":"MPS"},"gss_pc1":{"label":"GSS PC1"},"gss_pc2":{"label":"GSS PC2"},
                   "mp1":{"label":"MP1 (raw)"},"ff4":{"label":"FF4 (raw)"},"ed4":{"label":"ED4 (raw)"},"mps":{"label":"MPS (raw)"}},
        "forecast_series": {"bcff":{"label":"BCFF consensus"},"bcei":{"label":"BCEI consensus"},
                            "combined":{"label":"Combined consensus"},"med":{"label":"BCFF median"}},
        "rtn_controls": BS,
        "cbi_vars": ["cpi","rgdp","un","ffr"],
        "monthly": monthly,
        "panel": panel,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    print(f"Wrote {OUT}: {len(monthly)} monthly rows, {len(panel)} panel rows "
          f"({len({p['id'] for p in panel})} forecasters); bcff_time==1 months: {sum(1 for v in bcff_time.values() if v==1)}")


if __name__ == "__main__":
    main()
