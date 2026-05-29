#!/usr/bin/env python3
"""Build the static data snapshot for pages/reps.html.

Shock + Bauer-Swanson controls come from jk_source_old.csv (paulbousquet/GBMPSurprise)
-- the same source the romer local-projection do-files use. Forecast data and the
realized-macro (FRED) controls come from misc (mindcraft1997). Auth: $GITHUB_TOKEN
else `gh auth token`.

    python3 scripts/build-reps-data.py            # writes reps-data.json

Concepts:
  RTN = response to news  (predictors of the surprise; like irfs.html control groups):
        - market : Bauer-Swanson controls (nfp_surp, nfp_12m, sp500_3m, slope_3m, bcom_3m, tr_skew)
        - gb     : Greenbook forecaster controls (gPCPIF/gRGDPF/UNEMPF/ffr levels)
        - fred   : realized FRED macro (ffr, ebp, GS1, dgdp, dunemp, dlip, dlcpi, dlpce, dlsent, dlnas, dlmpu)
  CBI = central-bank information (Fed - private forecast gaps), per macro variable:
        cpi dp=gPCPIF-cpi ; rgdp dg=gRGDPF-rgdp ; un du=UNEMPF-un(BCEI) ; ffr df=ffr-ff (E^F-E^M)
  BCFF cpi/rgdp gaps apply the monthlydisag.do bcff_time==1 forward shift; df is plain ffr-ff
  (no shift, per the .do); combined consensus is aligned by construction.
"""
from __future__ import annotations
import io, json, math, os, subprocess, urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO = "mindcraft1997/misc"
JK_URL = "https://raw.githubusercontent.com/paulbousquet/GBMPSurprise/main/jk_source_old.csv"
OUT  = os.environ.get("REPS_OUT", "reps-data.json")
H    = range(5)
BS   = ["nfp_surp","nfp_12m","sp500_3m","slope_3m","bcom_3m","tr_skew"]
FRED = ["ffr","ebp","GS1","dgdp","dunemp","dlip","dlcpi","dlpce","dlsent","dlnas","dlmpu"]
GB_PRE   = {"cpi":"gPCPIF","rgdp":"gRGDPF","un":"UNEMPF","ffr":"ffr"}
PRIV_PRE = {"cpi":"cpi","rgdp":"rgdp","un":"un","ffr":"ff"}


def token():
    return (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
            or subprocess.check_output(["gh","auth","token"], text=True).strip())
_TOK = token()

def fetch(path):
    api = f"https://api.github.com/repos/{REPO}/contents/{path}"
    req = urllib.request.Request(api, headers={"Authorization": f"Bearer {_TOK}",
        "Accept": "application/vnd.github+json", "User-Agent": "build-reps-data"})
    meta = json.loads(urllib.request.urlopen(req, timeout=60).read())
    dl = urllib.request.Request(meta["download_url"], headers={"User-Agent": "build-reps-data"})
    return urllib.request.urlopen(dl, timeout=120).read()

def get_url(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "build-reps-data"}), timeout=120).read()

def csv(path, **kw): return pd.read_csv(io.BytesIO(fetch(path)), **kw)
def gbdate_dt(s): return pd.to_datetime(s.astype(str).str.slice(0,8), format="%Y%m%d", errors="coerce")
def num(s): return pd.to_numeric(s, errors="coerce")
def rnd(x, d=5):
    if x is None: return None
    try: x = float(x)
    except (TypeError, ValueError): return None
    return round(x, d) if math.isfinite(x) else None
def standshock(s):
    s = num(s); mask = s.notna() & (s != 0); mu, sd = s[mask].mean(), s[mask].std()
    out = s.copy(); out[mask] = (s[mask]-mu)/sd; out[s == 0] = 0.0
    return out


def greenbook():
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

def shocks_jk():
    """Surprises + Bauer-Swanson controls + scheduling flags from jk_source_old.csv (one row/month)."""
    jk = pd.read_csv(io.BytesIO(get_url(JK_URL)))
    jk["dt"] = pd.to_datetime(jk["Date"], errors="coerce")
    jk = jk.dropna(subset=["dt"])
    jk["month"] = jk["dt"].dt.strftime("%Y-%m")
    jk["scheduled"] = (num(jk["Unscheduled"]).fillna(0) == 0).astype(int)
    ren = {"MP1":"mp1","FF4":"ff4","ED1":"ed1","ED2":"ed2","ED3":"ed3","ED4":"ed4",
           "NFP_SURP":"nfp_surp","NFP_12M":"nfp_12m","SP500_3M":"sp500_3m",
           "SLOPE_3M":"slope_3m","BCOM_3M":"bcom_3m","TR_SKEW":"tr_skew"}
    jk = jk.rename(columns=ren)
    for c in list(ren.values()): jk[c] = num(jk[c])
    # one row per FOMC month: prefer the scheduled meeting
    jk = jk.sort_values(["month","scheduled"], ascending=[True,False]).drop_duplicates("month", keep="first")
    # mps = 0.01 * PC1(ed1..ed4) / loading(ed4)
    sub = jk[["ed1","ed2","ed3","ed4"]].apply(num).replace([np.inf,-np.inf], np.nan).dropna()
    sub = jk.loc[sub.index].copy()
    if len(sub) >= 8:
        X = sub[["ed1","ed2","ed3","ed4"]].astype(float).values
        sd = X.std(0); sd[sd == 0] = 1.0
        Xs = (X - X.mean(0))/sd; _,_,vt = np.linalg.svd(Xs, full_matrices=False)
        pc1 = Xs @ vt[0]; b = np.polyfit(pc1, sub["ed4"].astype(float).values, 1)[0]
        jk.loc[sub.index, "mps"] = 0.01*pc1/b
    for c in ["mp1","ff4","ed4","mps"]: jk[c+"_std"] = standshock(jk[c])
    gss = csv("ForecastErrors/fedfunds_forecast_errors_gss.csv")
    gss["month"] = pd.to_datetime(gss["date"], errors="coerce").dt.strftime("%Y-%m")
    gss = gss.dropna(subset=["month"]).drop_duplicates("month", keep="first")
    jk = jk.merge(gss[["month","gss_pc1","gss_pc2"]], on="month", how="left")
    return jk.set_index("month")

def fred_macro():
    m = pd.read_stata(io.BytesIO(fetch("romer/mondat.dta")))
    dcol = "daten" if "daten" in m.columns else "date"
    m["month"] = pd.to_datetime(m[dcol], errors="coerce").dt.strftime("%Y-%m")
    keep = ["month"] + [c for c in FRED if c in m.columns]
    m = m.dropna(subset=["month"]).drop_duplicates("month", keep="last")
    return m[keep].set_index("month")

def bcff_time_flag():
    t = csv("ForecastErrors/gb_vs_bcff_fedfunds_timed.csv")
    t["month"] = pd.to_datetime(t["pub_date"], errors="coerce").dt.strftime("%Y-%m")
    return t.groupby("month")["bcff_time"].first().to_dict()

def shift_bcff(bc, bcff_time, varbases):
    bc = bc.copy(); flag = bc["month"].map(lambda m: bcff_time.get(m, 0) == 1)
    for v in varbases:
        for i in H:
            col = f"{v}_{i}"
            if col in bc: bc.loc[flag, col] = bc[col].shift(-1)[flag]
    return bc


def main():
    gb = greenbook(); jk = shocks_jk(); fred = fred_macro()
    bcff_time = bcff_time_flag(); gb_idx = gb.set_index("month")
    SERIES = {"bcff":"romer/bcff_consensus.csv", "bcei":"romer/bcei_consensus.csv",
              "combined":"romer/combined_consensus.csv"}
    bcei = consensus("romer/bcei_consensus.csv").set_index("month")
    un_by = {m: [rnd(num(bcei.loc[m]).get(f"un_{i}")) for i in H] for m in bcei.index}

    def gap(g, r, var):
        return [rnd((float(g.get(f"{GB_PRE[var]}{i}"))-float(r.get(f"{PRIV_PRE[var]}_{i}")))
                    if pd.notna(g.get(f"{GB_PRE[var]}{i}")) and pd.notna(r.get(f"{PRIV_PRE[var]}_{i}")) else None) for i in H]

    series_data = {}
    for name, path in SERIES.items():
        bc = consensus(path)
        if name == "bcff": bc = shift_bcff(bc, bcff_time, ["cpi","rgdp"])
        bc = bc.set_index("month"); rows = {}
        for m in bc.index:
            if m not in gb_idx.index: continue
            g, r = gb_idx.loc[m], bc.loc[m]; cbi, rawp = {}, {}
            if "cpi_0" in bc.columns:  cbi["cpi"]  = gap(g, r, "cpi");  rawp["cpi"]  = [rnd(r.get(f"cpi_{i}")) for i in H]
            if "rgdp_0" in bc.columns: cbi["rgdp"] = gap(g, r, "rgdp"); rawp["rgdp"] = [rnd(r.get(f"rgdp_{i}")) for i in H]
            if "ff_0" in bc.columns:   cbi["ffr"]  = gap(g, r, "ffr");  rawp["ffr"]  = [rnd(r.get(f"ff_{i}")) for i in H]
            if m in un_by:
                gbun = [g.get(f"UNEMPF{i}") for i in H]
                cbi["un"] = [rnd((float(gbun[i])-un_by[m][i]) if gbun[i] is not None and pd.notna(gbun[i]) and un_by[m][i] is not None else None) for i in H]
                rawp["un"] = un_by[m]
            rows[m] = {"cbi": {k:v for k,v in cbi.items() if any(x is not None for x in v)},
                       "raw": {k:v for k,v in rawp.items() if any(x is not None for x in v)}}
        series_data[name] = rows

    all_months = sorted(set().union(*[set(r) for r in series_data.values()]) | set(jk.index))
    monthly = []
    for m in all_months:
        s = jk.loc[m] if m in jk.index else None
        sched = (None if s is None or pd.isna(s.get("scheduled")) else int(s["scheduled"]) == 1)
        nzlb = ((m<"2008-11") or (m>"2015-11")) and ((m<"2020-03") or (m>"2022-02"))
        g = gb_idx.loc[m] if m in gb_idx.index else None
        f = fred.loc[m] if m in fred.index else None
        has_shock = s is not None and any(pd.notna(s.get(k)) for k in ["mp1","ed4","ff4"])
        has_series = any(m in series_data[name] for name in SERIES)
        if not has_shock and not has_series: continue   # drop empty future/gap months
        monthly.append({
            "month": m, "year": int(m[:4]), "nzlb": bool(nzlb), "scheduled": sched,
            "shocks": {k: rnd(None if s is None else s.get(k)) for k in
                       ["mp1","ff4","ed4","mps","mp1_std","ff4_std","ed4_std","mps_std","gss_pc1","gss_pc2"]},
            "rtn": {k: rnd(None if s is None else s.get(k)) for k in BS},
            "fred": ({k: rnd(f.get(k)) for k in FRED if k in fred.columns} if f is not None else {}),
            "gbraw": ({v: [rnd(g.get(f"{GB_PRE[v]}{i}")) for i in H] for v in ["cpi","rgdp","un","ffr"]} if g is not None else {}),
            "series": {name: series_data[name][m] for name in SERIES if m in series_data[name]},
        })

    # panel (individual BCFF forecasters)
    pan = csv("romer/bcff_panel.csv"); pan["month"] = pd.to_datetime(pan["date"], errors="coerce").dt.strftime("%Y-%m")
    sched_by = {r["month"]: r["scheduled"] for r in monthly}; nzlb_by = {r["month"]: r["nzlb"] for r in monthly}
    panel = []
    for _, r in pan.iterrows():
        m = r["month"]
        if m not in gb_idx.index or pd.isna(r["IDn"]): continue
        g = gb_idx.loc[m]; cbi = {}
        for var, pre in [("cpi","gPCPIF"),("rgdp","gRGDPF"),("ffr","ffr")]:
            base = "ff" if var == "ffr" else var
            vals = [rnd((float(g.get(f"{pre}{i}"))-float(r.get(f"{base}_{i}")))
                        if pd.notna(g.get(f"{pre}{i}")) and pd.notna(r.get(f"{base}_{i}")) else None) for i in H]
            if any(v is not None for v in vals): cbi[var] = vals
        if cbi:
            panel.append({"id": int(r["IDn"]), "month": m, "year": int(m[:4]),
                          "nzlb": bool(nzlb_by.get(m, True)), "scheduled": sched_by.get(m),
                          "bcff_time": int(bcff_time.get(m, 0)), "cbi": cbi})

    payload = {
        "meta": {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repo": REPO,
            "shock_source": "paulbousquet/GBMPSurprise jk_source_old.csv (MP1/FF4/ED1-4 + BS controls + scheduling)",
            "construction_note": ("Shocks/BS controls from jk_source_old; Greenbook from gbweb_row_format.xlsx + rpred; "
                "FRED macro from mondat.dta; consensus from misc. BCFF cpi/rgdp gaps apply the monthlydisag.do bcff_time "
                "forward shift; df=ffr-ff with no shift; combined aligned by construction; shocks standardized via standshock.ado."),
            "defaults": {"series":"bcff","min_year":1993,"exclude_zlb":True}},
        "shocks": {"mp1_std":{"label":"MP1"},"ff4_std":{"label":"FF4"},"ed4_std":{"label":"ED4"},
                   "mps_std":{"label":"MPS"},"gss_pc1":{"label":"GSS PC1"},"gss_pc2":{"label":"GSS PC2"}},
        "forecast_series": {"bcff":{"label":"BCFF consensus"},"bcei":{"label":"BCEI consensus"},"combined":{"label":"Combined consensus"}},
        "rtn_controls": BS, "fred_controls": [c for c in FRED if c in fred.columns],
        "cbi_vars": ["cpi","rgdp","un","ffr"],
        "monthly": monthly, "panel": panel,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    print(f"Wrote {OUT}: {len(monthly)} months, {len(panel)} panel obs "
          f"({len({p['id'] for p in panel})} forecasters); scheduled months: "
          f"{sum(1 for r in monthly if r['scheduled'])}, unscheduled: {sum(1 for r in monthly if r['scheduled'] is False)}")


if __name__ == "__main__":
    main()
