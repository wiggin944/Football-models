#!/usr/bin/env python3
"""Global Alpha v13B: EXPLORATORY five-region historical pilot. No trading."""
import concurrent.futures as cf
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

REPO="AndyLongest/HistoricalIndexPrices"
PIN="bdb5c01084b314a94edfad155547d9373d0d8191"
REGIONS=["sp500","eurostoxx50","nikkei225","hsi","csi300"]
SEED="globalalpha-v13-global-sample-20261115"
START="2022-01"
END="2025-06"
BPS_ONE_WAY=20
CACHE=Path("price_cache")
CACHE.mkdir(exist_ok=True)
MONTHS=pd.period_range("2015-01","2025-08",freq="M").strftime("%Y-%m").tolist()

def get(url, target, json_output=False):
    if target.exists():
        return json.loads(target.read_text()) if json_output else target.read_bytes()
    err=None
    for attempt in range(5):
        try:
            res=requests.get(url,timeout=70,headers={"User-Agent":"GlobalAlphaV13B/1.0","Accept":"application/vnd.github+json" if json_output else "*/*"})
            res.raise_for_status()
            if len(res.content)<50: raise ValueError("Short response for "+url)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(res.content)
            return res.json() if json_output else res.content
        except Exception as exc:
            err=exc
            time.sleep(min(2**attempt,8))
    raise RuntimeError(f"Could not fetch {url}: {err}")

def tree(sha):
    v=get(f"https://api.github.com/repos/{REPO}/git/trees/{sha}",CACHE/"trees"/(sha+".json"),True)
    if "tree" not in v: raise ValueError("Missing tree; API rate limit? "+str(v)[:200])
    return v["tree"]

def region_symbols(region):
    base=next(x["sha"] for x in tree(PIN) if x["path"]=="data")
    reg=next(x["sha"] for x in tree(base) if x["path"]==region)
    subtree=next(x["sha"] for x in tree(reg) if x["path"]=="constituents")
    names=sorted(x["path"] for x in tree(subtree) if x["type"]=="tree")
    return sorted(names,key=lambda s:hashlib.sha256((SEED+"|"+region+"|"+s).encode()).hexdigest())[:100],len(names)

def load(region, symbol, is_index=False):
    tail="index_prices_daily.csv" if is_index else f"constituents/{symbol}/prices_daily.csv"
    raw=get(f"https://raw.githubusercontent.com/{REPO}/{PIN}/data/{region}/{tail}",
            CACHE/PIN/region/tail)
    df=pd.read_csv(io.BytesIO(raw),low_memory=False)
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df=df.dropna(subset=["date"]).sort_values("date")
    opening=pd.to_numeric(df["open"],errors="coerce")
    closing=pd.to_numeric(df["close"],errors="coerce")
    if not is_index:
        adjusted=pd.to_numeric(df["normalized_adj_close"],errors="coerce")
        closing=adjusted.where(adjusted>0,closing)
        is_china=df["price_basis"].eq("raw_reconstructed_from_qlib")
        factor=pd.to_numeric(df["adjustment_factor"],errors="coerce")
        if (is_china & ((factor<=0)|factor.isna())).any():
            raise ValueError(f"Unadjustable China OHLC {region}/{symbol}")
        opening=opening*np.where(is_china,factor,1.0)
    df["po"]=opening
    df["pc"]=closing
    df=df[(df.po>0)&(df.pc>0)].copy()
    df["month"]=df["date"].dt.strftime("%Y-%m")
    monthly=df.groupby("month").agg(open=("po","first"),close=("pc","last"),days=("date","count"))
    monthly.loc[monthly.days<5,["open","close"]]=np.nan
    return monthly

def feat(stock,market,month):
    k=pd.Period(month,freq="M")
    mm=[(k-i).strftime("%Y-%m") for i in range(12)]
    try:
        a=stock.loc[mm,"close"].to_numpy(dtype=float)
        z=market.loc[mm,"close"].to_numpy(dtype=float)
    except KeyError:
        return None
    if not np.isfinite(a).all() or not np.isfinite(z).all() or min(a.min(),z.min())<=0:
        return None
    rets=a[:6]/a[1:7]-1
    return np.array([a[0]/a[1]-1,a[0]/a[3]-1,a[0]/a[6]-1,
       a[0]/a[11]-1,a[0]/a[1]-z[0]/z[1],a[0]/a[3]-z[0]/z[3],
       a[0]/max(a)-1,np.std(rets),z[0]/z[3]-1,0.0],float)

def holding(stock,month):
    m=pd.Period(month,freq="M")
    a=(m+1).strftime("%Y-%m")
    b=(m+2).strftime("%Y-%m")
    try:
        o=float(stock.loc[a,"open"])
        x=float(stock.loc[b,"open"])
    except KeyError:
        return None
    return x/o-1 if np.isfinite(o) and np.isfinite(x) and o>0 and x>0 else None

def evaluate(region, stocks, index, total_folders, load_errors):
    panels=[]
    for month in pd.period_range("2016-01","2025-06",freq="M").strftime("%Y-%m"):
        f={s:feat(v,index,month) for s,v in stocks.items()}
        f={s:x for s,x in f.items() if x is not None}
        if not f: continue
        mean_shock=float(np.mean([x[0] for x in f.values()]))
        for x in f.values():x[-1]=x[0]-mean_shock
        actual={s:holding(stocks[s],month) for s in f}
        panels.append({"month":month,"f":f,"actual":actual})
    series={x:[] for x in ["ridge","momentum","equal"]}
    last={x:{} for x in series}
    for panel in panels:
        month=panel["month"]
        if not START<=month<=END:continue
        train=[p for p in panels if pd.Period(p["month"])+2<=pd.Period(month)]
        X=[]
        y=[]
        for old in train:
            valid={s:v for s,v in old["actual"].items() if v is not None and np.isfinite(v) and v>-1}
            if len(valid)<5:continue
            avg=float(np.mean(list(valid.values())))
            for s,v in valid.items():
                X.append(old["f"][s])
                y.append(v-avg)
        symbols=sorted(panel["f"])
        if len(X)<150 or len(symbols)<5:continue
        model=make_pipeline(StandardScaler(),Ridge(alpha=80))
        model.fit(np.array(X),np.array(y))
        pred=dict(zip(symbols,model.predict(np.array([panel["f"][s] for s in symbols]))))
        n=max(3,math.ceil(0.30*len(symbols)))
        sets={"ridge":sorted(symbols,key=lambda s:pred[s],reverse=True)[:n],
              "momentum":sorted(symbols,key=lambda s:panel["f"][s][3],reverse=True)[:n],
              "equal":symbols}
        for method,chosen in sets.items():
            weights={s:1/len(chosen) for s in chosen}
            prior=last[method]
            turnover=(sum(abs(weights.get(s,0)-prior.get(s,0)) for s in set(weights)|set(prior))/2 if prior else 1.0)
            missing=[s for s in chosen if panel["actual"][s] is None]
            value=None if missing else float(np.mean([panel["actual"][s] for s in chosen]))-BPS_ONE_WAY/10000*turnover
            series[method].append({"month":month,"return":value,"missing":missing,
                     "turnover":round(turnover,4),"n":len(symbols)})
            last[method]=weights
    summary={}
    for name,rows in series.items():
        valid=[r["return"] for r in rows if r["return"] is not None]
        comp=len(valid)==len(rows) and len(rows)>=40
        wealth=1.
        peak=1.
        drawdown=0.
        for r in valid:
            wealth*=1+r
            peak=max(peak,wealth)
            drawdown=min(drawdown,wealth/peak-1)
        summary[name]={
            "months_requested":42,"months_evaluated":len(rows),
            "months_complete":len(valid),"missing_trade_months":[r["month"] for r in rows if r["missing"]],
            "missing_trade_instances":sum(len(r["missing"]) for r in rows),
            "avg_turnover_one_way":round(float(np.mean([r["turnover"] for r in rows])),3) if rows else None,
            "average_eligible_names":round(float(np.mean([r["n"] for r in rows])),1) if rows else None,
            "CAGR_PCT_ONLY_IF_NO_MISSING_MONTHS":round((wealth**(12/len(valid))-1)*100,3) if comp else None,
            "MAX_DRAWDOWN_PCT_ONLY_IF_NO_MISSING_MONTHS":round(100*drawdown,3) if comp else None,
            "complete_case_CAGR_UNSAFE":round((wealth**(12/len(valid))-1)*100,3) if valid and not comp else None,
            "not_valid_for_alpha":True}
    return {"region":region,"historical_security_folders":total_folders,"sampled":len(stocks)+len(load_errors),
            "fetched":len(stocks),"load_errors":load_errors,"methods":summary,"monthly":series,
            "CAVEAT":"Point-in-time folder slices can end before held-stock sale; do not silently drop missing exits; local currency, no FX, no full universe, no untouched holdout."}

def main():
    results={"version":"v13B","price_source_sha":PIN,"sample_seed":SEED,
             "evaluation":"Jan 2022 to Jun 2025, exploratory, prior-period repeatedly inspected",
             "regions":{},"status":"NO_ALPHA_CERTIFICATION"}
    for region in REGIONS:
        names,n_all=region_symbols(region)
        print("FETCH",region,len(names),"of",n_all,flush=True)
        index=load(region,"INDEX",is_index=True)
        stocks={}
        errors={}
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            jobs={pool.submit(load,region,s):s for s in names}
            for job in cf.as_completed(jobs):
                s=jobs[job]
                try:stocks[s]=job.result()
                except Exception as e:errors[s]=str(e)
        result=evaluate(region,stocks,index,n_all,errors)
        results["regions"][region]=result
        print("REGION_DONE",region,json.dumps(result["methods"],separators=(",",":")),flush=True)
        Path("results").mkdir(exist_ok=True)
        (Path("results")/f"{region}.json").write_text(json.dumps(result,indent=2))
    (Path("results")/"global_alpha_v13b_summary.json").write_text(json.dumps(results,indent=2))
    brief={k:{"sampled":v["sampled"],"fetched":v["fetched"],"load_errors":len(v["load_errors"]),"methods":v["methods"]} for k,v in results["regions"].items()}
    print("FINAL_RESULTS_JSON",json.dumps(brief,separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
