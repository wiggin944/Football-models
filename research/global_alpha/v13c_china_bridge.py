#!/usr/bin/env python3
"""v13C: Chinese cross-index price history continuity audit and exploratory rerun."""
import concurrent.futures as cf
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import v13b_global_pilot as v

REGIONS=["csi300","csi500","csi1000"]
OUTPUT=Path("results")
OUTPUT.mkdir(exist_ok=True)

def symbol_set(region):
    base=next(x["sha"] for x in v.tree(v.PIN) if x["path"]=="data")
    root=next(x["sha"] for x in v.tree(base) if x["path"]==region)
    descendants=next(x["sha"] for x in v.tree(root) if x["path"]=="constituents")
    return {x["path"] for x in v.tree(descendants) if x["type"]=="tree"}

def get_daily(region,symbol):
    path=f"constituents/{symbol}/prices_daily.csv"
    raw=v.get(f"https://raw.githubusercontent.com/{v.REPO}/{v.PIN}/data/{region}/{path}",
              v.CACHE/v.PIN/region/path)
    f=pd.read_csv(io.BytesIO(raw),low_memory=False)
    required=["date","symbol","open","close","currency","price_basis","adjustment_factor","normalized_adj_close"]
    if any(k not in f.columns for k in required):
        raise ValueError(f"Incomplete adjusted OHLC {region}/{symbol}")
    if not (f["currency"]=="CNY").all() or not (f["price_basis"]=="raw_reconstructed_from_qlib").all():
        raise ValueError(f"Wrong currency/price source {region}/{symbol}")
    if not (f["symbol"].astype(str)==symbol).all():
        raise ValueError(f"Entity mismatched ticker {region}/{symbol}")
    f["date"]=pd.to_datetime(f["date"],errors="coerce")
    f["po"]=pd.to_numeric(f["open"],errors="coerce")*pd.to_numeric(f["adjustment_factor"],errors="coerce")
    f["pc"]=pd.to_numeric(f["normalized_adj_close"],errors="coerce")
    f=f.dropna(subset=["date","po","pc"])
    f=f[(f["po"]>0)&(f["pc"]>0)]
    return f[["date","po","pc"]].drop_duplicates("date").sort_values("date")

def combined(region_to_names,symbol):
    daily=[]
    for region in REGIONS:
        if symbol not in region_to_names[region]:
            continue
        d=get_daily(region,symbol)
        daily.append((region,d))
    if not daily or daily[0][0]!="csi300":
        raise AssertionError("Must preserve the original CSI 300 identification.")
    frames=[x[1] for x in daily]
    full=pd.concat(frames,ignore_index=True)
    disagreements=[]
    for date,g in full.groupby("date"):
        if len(g)<2:continue
        for col in ["po","pc"]:
            vals=g[col].to_numpy()
            ratio=vals.max()/vals.min()-1
            if ratio>0.001:
                disagreements.append({"date":str(date),"field":col,"ratio":float(ratio)})
    if disagreements:
        raise ValueError(f"Cross-source adjusted-price mismatch {symbol}: {disagreements[:2]}")
    # Only matching, audited Qlib symbol/currency/price-basis sources merged.
    dedup=full.sort_values("date").drop_duplicates("date")
    dedup["month"]=dedup["date"].dt.strftime("%Y-%m")
    monthly=dedup.groupby("month").agg(open=("po","first"),close=("pc","last"),days=("date","count"))
    monthly.loc[monthly.days<5,["open","close"]]=np.nan
    return monthly,{"regions":[x[0] for x in daily],
            "daily_rows":len(dedup),
            "overlap_duplicate_days":len(full)-len(dedup),
            "last_date":str(dedup["date"].max().date())}

def method_summary(out):
    return {name:{k:info[k] for k in ["months_complete","CAGR_PCT_ONLY_IF_NO_MISSING_MONTHS",
               "complete_case_CAGR_UNSAFE","missing_trade_instances","avg_turnover_one_way"]}
            for name,info in out["methods"].items()}

def main():
    names,n_total=v.region_symbols("csi300")
    sources={r:symbol_set(r) for r in REGIONS}
    index=v.load("csi300","INDEX",is_index=True)
    print("SAMPLE",len(names),"overlap csi500",len(set(names)&sources["csi500"]),
          "overlap csi1000",len(set(names)&sources["csi1000"]),flush=True)
    plain={}
    merged={}
    audits={}
    errors={}
    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        jobs={executor.submit(combined,sources,s):s for s in names}
        for job in cf.as_completed(jobs):
            sym=jobs[job]
            try:
                merged[sym],audits[sym]=job.result()
            except Exception as exc:
                errors[sym]=str(exc)
    # Parallel retrieval of original 100 for strictly matched comparison.
    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        jobs={executor.submit(v.load,"csi300",s):s for s in names}
        for job in cf.as_completed(jobs):
            sym=jobs[job]
            try:plain[sym]=job.result()
            except Exception as exc:errors[sym]=str(exc)
    # Matched 100 security set; never use survivorship pruning.
    both=sorted(set(plain)&set(merged))
    raw=v.evaluate("csi300_original", {s:plain[s] for s in both},index,n_total,errors)
    joined=v.evaluate("csi300_joined", {s:merged[s] for s in both},index,n_total,errors)
    report={"sampled":len(names),"matched":len(both),"error_names":errors,
      "supplemented_names":sum(1 for s in both if len(audits[s]["regions"])>1),
      "source_audits":audits,"original":raw,"joined":joined,
      "not_a_holdout":True,"no_alpha_certification":True,
      "critical":"Source folders from the same Qlib archive were joined only when symbol, price basis and CNY agreed and overlapping daily adjusted prices matched to 0.1%; no assumption that excluded delisted companies are present."}
    (OUTPUT/"china_bridge.json").write_text(json.dumps(report,indent=2))
    print("CHINA_BRIDGE_SUMMARY",json.dumps({"matched":len(both),"supplemented":report["supplemented_names"],
      "raw":method_summary(raw),"joined":method_summary(joined),"failed":errors},separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
