#!/usr/bin/env python3
"""Build pages/reps-data.json by running the ACTUAL romer Stata code, so the webapp
matches the Stata files exactly (zero ambiguity).

Pipeline:
  1. Fetch the romer inputs + ados + do-files from the private repo mindcraft1997/misc.
  2. Patch them minimally so they run headless:
       - flpdecomp.ado: add the missing `tempvar h_range` (committed ado omits it).
       - monthlydisag.do -> build_consensus.do: parameterise `filn` ($FILN), add an
         MDY date fallback (bcei/combined use M/D/Y), and append an export of the
         per-month gaps/shocks/controls.
       - stablediag.do -> build_panel.do: cut before the reghdfe calls and append a
         per-(forecaster,month) export of the gaps.
  3. Run Stata (stata-mp) for each consensus series (bcff/bcei/combined) and the panel.
  4. Assemble the Stata exports + FRED macro (mondat.dta) + GSS PCs into reps-data.json.

Requires: Stata (stata-mp on PATH or $STATA), a GitHub token ($GITHUB_TOKEN or `gh auth token`).
Shocks/BS controls come from mpsur/mp1.csv (what monthlydisag.do imports); Greenbook from
gbweb_row_format.xlsx+rpred; everything standardised via standshock.ado.
"""
from __future__ import annotations
import io, json, math, os, re, subprocess, tempfile, urllib.request
from datetime import datetime, timezone
import numpy as np, pandas as pd

REPO="mindcraft1997/misc"; H=range(5)
BS=["nfp_surp","nfp_12m","sp500_3m","slope_3m","bcom_3m","tr_skew"]
FRED=["ffr","ebp","GS1","dgdp","dunemp","dlip","dlcpi","dlpce","dlsent","dlnas","dlmpu"]
SERIES=["bcff","bcei","combined"]
STATA=os.environ.get("STATA","stata-mp")
WORK=os.environ.get("REPS_WORK", tempfile.mkdtemp(prefix="reps_"))
OUT=os.environ.get("REPS_OUT","reps-data.json")

def tok(): return (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
                   or subprocess.check_output(["gh","auth","token"],text=True).strip())
_T=tok()
def fetch(path):
    api=f"https://api.github.com/repos/{REPO}/contents/{path}"
    req=urllib.request.Request(api,headers={"Authorization":f"Bearer {_T}","Accept":"application/vnd.github+json","User-Agent":"x"})
    meta=json.loads(urllib.request.urlopen(req,timeout=60).read())
    return urllib.request.urlopen(urllib.request.Request(meta["download_url"],headers={"User-Agent":"x"}),timeout=180).read()
def save(path,data,mode="wb"):
    full=os.path.join(WORK,path); os.makedirs(os.path.dirname(full),exist_ok=True)
    open(full,mode).write(data); return full
def rnd(x,d=5):
    if x is None: return None
    try: x=float(x)
    except (TypeError,ValueError): return None
    return round(x,d) if math.isfinite(x) else None
def arr(row,base): return [rnd(row.get(f"{base}_{i}")) for i in H]

EXPORT_CONS = r'''
* ===================== EXPORT (reps build) =====================
gen ym = string(year(raw_daten))+"-"+string(month(raw_daten),"%02.0f")
keep if !missing(rgdp_0)
local duv ""
cap unab duv : du_*
local unv ""
cap unab unv : un_*
export delimited ym dp_* dg_* df_* `duv' mp1 ff4 ed4 mps mp1_std ff4_std ed4_std mps_std ///
  nfp_surp nfp_12m sp500_3m slope_3m bcom_3m tr_skew possible nzlb cpi_* rgdp_* ff_* `unv' ///
  gPCPIF0 gPCPIF1 gPCPIF2 gPCPIF3 gPCPIF4 gRGDPF0 gRGDPF1 gRGDPF2 gRGDPF3 gRGDPF4 ///
  UNEMPF0 UNEMPF1 UNEMPF2 UNEMPF3 UNEMPF4 ffr0 ffr1 ffr2 ffr3 ffr4 using "out_${FILN}.csv", replace
'''
EXPORT_PANEL = r'''
* ===================== EXPORT panel (reps build) =====================
cap drop ym
gen ym = string(year(raw_date))+"-"+string(month(raw_date),"%02.0f")
keep if !missing(df_4) | !missing(dp_0) | !missing(dg_0)
export delimited idn ym dp_* dg_* df_* bcff_time using "out_panel.csv", replace
'''

def prep_stata():
    # inputs
    for p in ["romer/rpred.csv","romer/bcff_consensus.csv","romer/bcei_consensus.csv",
              "romer/combined_consensus.csv","romer/bcff_panel.csv",
              "romer/intermediates/GBFOMCmapping.csv","romer/intermediates/gbweb_row_format.xlsx",
              "romer/standshock.ado"]:
        save(os.path.basename(p) if not p.endswith(("GBFOMCmapping.csv","gbweb_row_format.xlsx"))
             else os.path.join("intermediates",os.path.basename(p)), fetch(p))
    # flpdecomp.ado + tempvar h_range fix
    fd=fetch("romer/flpdecomp.ado").decode()
    fd=fd.replace("\tif (`H' == `h1') {","\ttempvar h_range\n\tif (`H' == `h1') {",1)
    save("flpdecomp.ado",fd.encode())
    # build_consensus.do from monthlydisag.do
    md=fetch("romer/monthlydisag.do").decode()
    md=md.replace("local filn bcff",'local filn "$FILN"')
    md=md.replace('gen raw_daten = date(date, "YMD")',
                  'gen raw_daten = date(date, "YMD")\nreplace raw_daten = date(date, "MDY") if missing(raw_daten)',1)
    save("build_consensus.do",(md+EXPORT_CONS).encode())
    # build_panel.do from stablediag.do (cut before reghdfe)
    sd=fetch("romer/stablediag.do").decode()
    cut=sd.find("reghdfe F.dff_4"); assert cut>0,"reghdfe anchor not found in stablediag.do"
    save("build_panel.do",(sd[:cut]+EXPORT_PANEL).encode())
    # master
    master=('adopath ++ "."\nset more off\n'
            'foreach f in bcff bcei combined {\n  global FILN "`f\'"\n  do build_consensus.do\n}\n'
            'do build_panel.do\n')
    save("master.do",master.encode())

def run_stata():
    subprocess.run([STATA,"-b","do","master.do"],cwd=WORK,check=False)
    for f in [f"out_{s}.csv" for s in SERIES]+["out_panel.csv"]:
        if not os.path.exists(os.path.join(WORK,f)):
            raise SystemExit(f"Stata did not produce {f}; see {WORK}/master.log")

def assemble():
    m=pd.read_stata(io.BytesIO(fetch("romer/mondat.dta")))
    dc="daten" if "daten" in m.columns else "date"
    m["month"]=pd.to_datetime(m[dc],errors="coerce").dt.strftime("%Y-%m")
    fred=m.dropna(subset=["month"]).drop_duplicates("month",keep="last").set_index("month")
    g=pd.read_csv(io.BytesIO(fetch("ForecastErrors/fedfunds_forecast_errors_gss.csv")))
    g["month"]=pd.to_datetime(g["date"],errors="coerce").dt.strftime("%Y-%m")
    gss=g.dropna(subset=["month"]).drop_duplicates("month",keep="first").set_index("month")
    con={s:pd.read_csv(os.path.join(WORK,f"out_{s}.csv")).drop_duplicates("ym",keep="first").set_index("ym") for s in SERIES}
    bcei=con["bcei"]
    du_by={mo:arr(bcei.loc[mo],"du") for mo in bcei.index if "du_0" in bcei.columns}
    un_by={mo:arr(bcei.loc[mo],"un") for mo in bcei.index if "un_0" in bcei.columns}
    months=sorted(set().union(*[set(con[s].index) for s in SERIES]))
    canon=con["bcff"]
    monthly=[]
    for mo in months:
        c=canon.loc[mo] if mo in canon.index else None
        f=fred.loc[mo] if mo in fred.index else None; gs=gss.loc[mo] if mo in gss.index else None
        rec={"month":mo,"year":int(mo[:4]),
             "nzlb":bool(int(c["nzlb"])) if c is not None and pd.notna(c.get("nzlb")) else True,
             "scheduled":(None if c is None or pd.isna(c.get("possible")) else int(c["possible"])==1),
             "shocks":{"gss_pc1":rnd(None if gs is None else gs.get("gss_pc1")),"gss_pc2":rnd(None if gs is None else gs.get("gss_pc2"))},
             "rtn":{k:rnd(None if c is None else c.get(k)) for k in BS},
             "fred":({k:rnd(f.get(k)) for k in FRED if k in fred.columns} if f is not None else {}),
             "gbraw":({"cpi":[rnd(c.get(f"gPCPIF{i}")) for i in H],"rgdp":[rnd(c.get(f"gRGDPF{i}")) for i in H],
                       "un":[rnd(c.get(f"UNEMPF{i}")) for i in H],"ffr":[rnd(c.get(f"ffr{i}")) for i in H]} if c is not None else {}),
             "series":{}}
        for s in SERIES:
            if mo not in con[s].index: continue
            r=con[s].loc[mo]
            cbi={"cpi":arr(r,"dp"),"rgdp":arr(r,"dg"),"ffr":arr(r,"df")}
            if mo in du_by: cbi["un"]=du_by[mo]
            raw={"cpi":arr(r,"cpi"),"rgdp":arr(r,"rgdp"),"ffr":arr(r,"ff")}
            if mo in un_by: raw["un"]=un_by[mo]
            rec["series"][s]={"cbi":{k:v for k,v in cbi.items() if any(x is not None for x in v)},
                              "raw":{k:v for k,v in raw.items() if any(x is not None for x in v)},
                              "shocks":{k:rnd(r.get(k)) for k in ["mp1_std","ff4_std","ed4_std","mps_std","mp1","ff4","ed4","mps"]}}
        monthly.append(rec)
    pan=pd.read_csv(os.path.join(WORK,"out_panel.csv")); pan["month"]=pan["ym"]
    nz={r["month"]:r["nzlb"] for r in monthly}; sc={r["month"]:r["scheduled"] for r in monthly}
    panel=[]
    for _,r in pan.iterrows():
        cbi={"cpi":arr(r,"dp"),"rgdp":arr(r,"dg"),"ffr":arr(r,"df")}
        cbi={k:v for k,v in cbi.items() if any(x is not None for x in v)}
        if not cbi or pd.isna(r.get("idn")): continue
        mo=str(r["month"])
        panel.append({"id":int(r["idn"]),"month":mo,"year":int(mo[:4]),"nzlb":bool(nz.get(mo,True)),
                      "scheduled":sc.get(mo),"bcff_time":int(r["bcff_time"]) if pd.notna(r.get("bcff_time")) else 0,"cbi":cbi})
    payload={"meta":{"built_at":datetime.now(timezone.utc).isoformat(timespec="seconds"),"repo":REPO,
        "generator":"Stata romer/monthlydisag.do (per series) + stablediag.do (panel), exported then assembled",
        "construction_note":("Generated by running the actual romer Stata code; shocks/BS controls from mpsur mp1.csv, "
          "Greenbook from gbweb_row_format.xlsx+rpred, standardized via standshock.ado; FRED from mondat.dta; GSS from "
          "ForecastErrors. Exercise-2 IRF reproduces disaglp.do; Exercise-1 first stage reproduces monthlydisag.do."),
        "defaults":{"series":"bcff","min_year":1988,"exclude_zlb":False}},
        "shocks":{"mp1_std":{"label":"MP1"},"ff4_std":{"label":"FF4"},"ed4_std":{"label":"ED4"},"mps_std":{"label":"MPS"},"gss_pc1":{"label":"GSS PC1"},"gss_pc2":{"label":"GSS PC2"}},
        "forecast_series":{"bcff":{"label":"BCFF consensus"},"bcei":{"label":"BCEI consensus"},"combined":{"label":"Combined consensus"}},
        "rtn_controls":BS,"fred_controls":[c for c in FRED if c in fred.columns],"cbi_vars":["cpi","rgdp","un","ffr"],
        "monthly":monthly,"panel":panel}
    json.dump(payload,open(OUT,"w"),separators=(",",":"),allow_nan=False)
    print(f"Wrote {OUT}: {len(monthly)} months, {len(panel)} panel obs ({len({p['id'] for p in panel})} forecasters). Work dir: {WORK}")

if __name__=="__main__":
    print("work dir:",WORK); prep_stata(); run_stata(); assemble()
