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
START="2025-08"
END="2026-09"
BPS_ONE_WAY=20
CACHE=Path("price_cache")
CACHE.mkdir(exist_ok=True)
MONTHS=pd.period_range("2015-01","2026-11",freq="M").strftime("%Y-%m").tolist()

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
    for month in pd.period_range("2016-01","2026-09",freq="M").strftime("%Y-%m"):
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
        comp=len(valid)==len(rows) and len(rows)>=14
        wealth=1.
        peak=1.
        drawdown=0.
        for r in valid:
            wealth*=1+r
            peak=max(peak,wealth)
            drawdown=min(drawdown,wealth/peak-1)
        summary[name]={
            "months_requested":14,"months_evaluated":len(rows),
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


# ---- Frozen v14 out-of-time experiment.  No trading. ----
import yfinance as yf
from threading import Lock

CUTOFF="2025-08-22"
HOLDOUT_START="2025-08"
HOLDOUT_END="2026-09"
FX_FEE_ONE_WAY=0.003  # provisional all-in retail FX fee per unit of trade
BENCH_ETF={"sp500":"SPY","eurostoxx50":"FEZ","nikkei225":"EWJ",
           "hsi":"EWH","csi300":"ASHR"}
STOCK_FX={"sp500":"GBPUSD=X","eurostoxx50":"GBPEUR=X","nikkei225":"GBPJPY=X",
          "hsi":"GBPHKD=X","csi300":"GBPCNY=X"}
REGIONAL_INDEX_YAHOO={"sp500":"^GSPC","eurostoxx50":"^STOXX50E",
               "nikkei225":"^N225","hsi":"^HSI","csi300":"000300.SS"}
YAHOO_START="2025-08-15"
YAHOO_END="2026-11-15"
_YF_LOCK=Lock()

def ticker_for(region,s):
    if region=="nikkei225": return s[2:]+".T"
    if region=="hsi": return s[2:]+".HK"
    if region=="csi300": return s[2:]+(".SS" if s.startswith("SH") else ".SZ")
    if region=="sp500": return s.replace(".","-")
    return s

def yahoo_daily(sym):
    """Unadjusted timestamps, adjusted OHLC (dividend/split-adjusted)."""
    for attempt in range(3):
        try:
            # Yahoo endpoints can throttle concurrent clients: cache independently.
            f=CACHE/"yahoo_v14"/(sym.replace("/","_").replace("^","index_").replace("=","_")+".csv")
            if f.exists():raw=pd.read_csv(f)
            else:
                with _YF_LOCK:
                    hist=yf.Ticker(sym).history(start=YAHOO_START,end=YAHOO_END,
                        auto_adjust=True,actions=False)
                if hist.empty:raise ValueError("No Yahoo adjusted prices")
                hist=hist.reset_index()
                col="Date" if "Date" in hist.columns else "Datetime"
                raw=hist[[col,"Open","Close"]].rename(columns={col:"date","Open":"po","Close":"pc"})
                f.parent.mkdir(parents=True,exist_ok=True)
                raw.to_csv(f,index=False)
            raw["date"]=pd.to_datetime(raw["date"]).dt.date
            raw["po"]=pd.to_numeric(raw["po"],errors="coerce")
            raw["pc"]=pd.to_numeric(raw["pc"],errors="coerce")
            raw=raw.dropna(subset=["date","po","pc"])
            raw=raw[(raw.po>0)&(raw.pc>0)].drop_duplicates("date").sort_values("date")
            if len(raw)<230:raise ValueError("Insufficient new-period observations")
            return raw
        except Exception as e:
            error=e
            time.sleep(0.6*(attempt+1))
    raise RuntimeError(str(error))

def new_monthly(df):
    t=df.copy()
    t["month"]=t["date"].astype(str).str[:7]
    x=t.groupby("month").agg(open=("po","first"),close=("pc","last"),days=("date","count"))
    x.loc[x.days<5,["open","close"]]=np.nan
    return x

def append_yahoo(old,region,sym,index=False):
    """Append future adjusted returns with an as-of-2025-08-22 archive anchor.
    Never alter pre-Aug-2025 historical prices."""
    ticker=REGIONAL_INDEX_YAHOO[region] if index else ticker_for(region,sym)
    daily=yahoo_daily(ticker)
    ref=daily[daily["date"]==pd.Timestamp(CUTOFF).date()]
    if ref.empty:raise ValueError("No exact closing-price anchor on 2025-08-22")
    if "2025-08" not in old.index or not np.isfinite(old.loc["2025-08","close"]):
        raise ValueError("Not active at the predeclared Aug-2025 cut-off")
    scale=float(old.loc["2025-08","close"])/float(ref["pc"].iloc[0])
    if not np.isfinite(scale) or not 0.05<scale<20:
        raise ValueError("Implausible cross-source scale")
    if not index:
        # Verify event-to-event returns at the anchor. Detect ticker reuse / wrong market proxies.
        cached=CACHE/PIN/region/"constituents"/sym/"prices_daily.csv"
        if cached.exists():
            df=pd.read_csv(cached,low_memory=False)
            df["d"]=pd.to_datetime(df["date"],errors="coerce").dt.date
            a=df[df["d"].isin([pd.Timestamp("2025-08-21").date(),pd.Timestamp(CUTOFF).date()])]
            b=daily[daily["date"].isin([pd.Timestamp("2025-08-21").date(),pd.Timestamp(CUTOFF).date()])]
            if len(a)==2 and len(b)==2:
                c="normalized_adj_close"
                z=a.sort_values("d")[c].astype(float).to_numpy()
                w=b.sort_values("date")["pc"].to_numpy(dtype=float)
                if abs((z[1]/z[0]-1)-(w[1]/w[0]-1))>0.035:
                    raise ValueError("Ticker identity/adjustment mismatch at original archive overlap")
    daily=daily.copy()
    daily["po"]*=scale
    daily["pc"]*=scale
    newer=new_monthly(daily)
    answer=old.copy()
    if "2025-08" not in newer.index or not np.isfinite(newer.loc["2025-08","close"]):
        raise ValueError("No complete August closing quote in new data")
    answer.loc["2025-08","close"]=newer.loc["2025-08","close"]
    for mon,row in newer.iterrows():
        if mon>"2025-08":answer.loc[mon,["open","close","days"]]=[row["open"],row["close"],row["days"]]
    return answer,scale

def raw_monthly_yahoo(sym):
    return new_monthly(yahoo_daily(sym))

def _open(d,period):
    key=str(period)
    if key not in d.index:return None
    z=float(d.loc[key,"open"])
    return z if np.isfinite(z) and z>0 else None

def convert_gbp(monthly,fx,method):
    """Convert 1-month local currency holding to GBP, with extra FX dealing charge.
    No FX fee on passive mark-to-market; FX fee proportional to traded notional."""
    rows=[]
    for r in monthly:
        k=pd.Period(r["month"],freq="M")
        f0=_open(fx,k+1);f1=_open(fx,k+2)
        loc=r["return"]
        good=loc is not None and f0 is not None and f1 is not None
        gbp=((1+loc)*(f0/f1)-1-FX_FEE_ONE_WAY*r["turnover"]) if good else None
        rows.append({"month":r["month"],"return_gbp":gbp,
                    "source_return":loc,"turnover":r["turnover"],
                    "missing_symbols":r["missing"]})
    return rows

def etf_benchmark_usd_in_gbp(months,etf,fx_usd):
    rows=[]
    for i,month in enumerate(months):
        k=pd.Period(month,freq="M")
        p0=_open(etf,k+1);p1=_open(etf,k+2)
        f0=_open(fx_usd,k+1);f1=_open(fx_usd,k+2)
        valid=all(v is not None for v in [p0,p1,f0,f1])
        # One initial investment, then buy-and-hold, 20bp security+30bp FX.
        r=((p1/p0)*(f0/f1)-1-(0.005 if i==0 else 0)) if valid else None
        rows.append({"month":month,"return_gbp":r})
    return rows

def calculate(arr,ret_key="return_gbp",n_expected=14):
    vals=[r.get(ret_key) for r in arr]
    complete=len(vals)==n_expected and all(x is not None and np.isfinite(x) and x>-1 for x in vals)
    if not complete:
        return {"complete":False,"n":len(vals),"observed":sum(x is not None for x in vals),
                "CAGR_GBP_PCT":None,"MAX_DD_GBP_PCT":None}
    growth=1.;peak=1.;maxdd=0.
    for r in vals:
        growth*=1+r
        peak=max(peak,growth)
        maxdd=min(maxdd,growth/peak-1)
    return {"complete":True,"n":len(vals),
            "CAGR_GBP_PCT":round(100*(growth**(12/len(vals))-1),3),
            "MAX_DD_GBP_PCT":round(100*maxdd,3),
            "TOTAL_RETURN_GBP_PCT":round(100*(growth-1),3),
            "MONTHLY_STD_PCT":round(100*float(np.std(vals,ddof=1)),3)}

def simple_bootstrap_paired(a,b,seed=29091,B=25000,block=3):
    if len(a)!=len(b) or not len(a):return None
    x=np.array(a)-np.array(b)
    rng=np.random.default_rng(seed)
    vals=[]
    for _ in range(B):
        inds=[]
        while len(inds)<len(x):
            s=int(rng.integers(0,len(x)))
            inds.extend(((s+j)%len(x) for j in range(block)))
        vals.append(1200*float(np.mean(x[np.array(inds[:len(x)])])))
    return {"mean_annualized_arithmetic_pp":round(1200*float(np.mean(x)),3),
            "interval_pp":[round(float(z),3) for z in np.quantile(vals,[.025,.975])],
            "caveat":"Descriptive block bootstrap, NOT independent validation or a CAGR confidence interval."}

def v14_main():
    global feat
    print("FROZEN_SETTINGS",json.dumps({"sample_seed":SEED,"start":START,"end":END,
           "macro_extra":"3 month price-index returns of all 5 regions, plus predeclared shock/drawdown/momentum interactions",
           "ridge_penalty":80,"selection":"top 30% long-only",
           "stock_bps":20,"fx_bps":30,
           "source_end":"2026-11-13","trade_cutoff":"end of each decision month"}),flush=True)
    indexes={}
    stocks={}
    errors={}
    history={}
    for reg in REGIONS:
        syms,n=region_symbols(reg)
        history[reg]={"symbol_folders":n,"predeclared_sample":len(syms)}
        idx=load(reg,"INDEX",is_index=True)
        try:
            indexes[reg],q=append_yahoo(idx,reg,"INDEX",index=True)
        except Exception as e:
            errors[reg+":INDEX"]=str(e)
            indexes[reg]=idx
        original={}
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            jobs={pool.submit(load,reg,s):s for s in syms}
            for job in cf.as_completed(jobs):
                s=jobs[job]
                try:
                    old=job.result()
                    if "2025-08" in old.index and pd.notna(old.loc["2025-08","close"]) and old.loc["2025-08","days"]>=5:
                        original[s]=old
                except Exception as e:
                    errors[reg+":"+s]="Archive "+str(e)
        print("AS_OF_CUTOFF",reg,len(original),"/",len(syms),flush=True)
        enhanced={}
        with cf.ThreadPoolExecutor(max_workers=5) as pool:
            jobs={pool.submit(append_yahoo,old,reg,s):s for s,old in original.items()}
            for job in cf.as_completed(jobs):
                s=jobs[job]
                try:enhanced[s],_=job.result()
                except Exception as e:errors[reg+":"+s]="Yahoo "+str(e)
        stocks[reg]=enhanced
        print("STITCHED",reg,len(enhanced),"/",len(original),"error_count",
              sum(x.startswith(reg+":") for x in errors),flush=True)
    # Frozen ancillary data. ETF comparators are US-listed research proxies,
    # not assertions of UK ISA investability.
    ext={}
    fx={}
    for code in sorted(set(STOCK_FX.values())|{"GBPUSD=X"}):
        try:fx[code]=raw_monthly_yahoo(code)
        except Exception as e:errors["FX:"+code]=str(e)
    for reg,code in BENCH_ETF.items():
        try:ext[reg]=raw_monthly_yahoo(code)
        except Exception as e:errors["ETF:"+reg]=str(e)
    orig_feat=feat
    results={"version":"v14_frozen_2025_2026_probe","source_sha":PIN,
             "cutoff":CUTOFF,"results_not_seen_before_code_freeze":True,
             "not_prospective":True,
             "research_only":True,"no_order_placement":True,
             "errors":errors,"counts":history,"regions":{}}
    for reg in REGIONS:
        if len(stocks[reg])<8:continue
        feat=orig_feat
        basic=evaluate(reg,stocks[reg],indexes[reg],history[reg]["symbol_folders"],{})
        def augmented(stock,market,mon):
            core=orig_feat(stock,market,mon)
            if core is None:return None
            k=pd.Period(mon,freq="M")
            z=[]
            for other in REGIONS:
                idx=indexes[other]
                a=k.strftime("%Y-%m");b=(k-3).strftime("%Y-%m")
                if a not in idx.index or b not in idx.index:return None
                p=float(idx.loc[a,"close"]);q=float(idx.loc[b,"close"])
                if not(np.isfinite(p) and np.isfinite(q) and p>0 and q>0):return None
                z.append(p/q-1)
            return np.concatenate([core[:-1],np.array(z,float),core[-1:]])
        feat=augmented
        macro=evaluate(reg,stocks[reg],indexes[reg],history[reg]["symbol_folders"],{})
        def augmented_interactions(stock,market,mon):
            # Raw market context is common to all companies in a decision month.
            # Interactions let the model LEARN how it changes a firm's shock response.
            expanded=augmented(stock,market,mon)
            if expanded is None:return None
            market_context=expanded[9:14]
            own_context=expanded[[4,6,2]]   # relative shock, drawdown, 6-month trend
            cross=np.outer(own_context,market_context).ravel()
            return np.concatenate([expanded[:-1],cross,expanded[-1:]])
        feat=augmented_interactions
        interacted=evaluate(reg,stocks[reg],indexes[reg],history[reg]["symbol_folders"],{})
        feat=orig_feat
        fx_series=fx.get(STOCK_FX[reg]);usd_fx=fx.get("GBPUSD=X")
        region={"asof_cutoff_names":len(stocks[reg]),"models":{}}
        for label,run in [("price",basic),("price_plus_global_market_context",macro),
                          ("price_plus_global_market_interactions",interacted)]:
            tmp={}
            for strategy,monthly in run["monthly"].items():
                gbp=convert_gbp(monthly,fx_series,strategy) if fx_series is not None else []
                tmp[strategy]={"local":run["methods"][strategy],
                     "gbp":calculate(gbp),"monthly_gbp":gbp}
            region["models"][label]=tmp
        months=[x["month"] for x in basic["monthly"]["ridge"]]
        if reg in ext and usd_fx is not None:
            bench=etf_benchmark_usd_in_gbp(months,ext[reg],usd_fx)
            region["research_etf_proxy"]={"ticker":BENCH_ETF[reg],
                 "summary":calculate(bench),"monthly_gbp":bench}
        a=region["models"]["price"]["ridge"]["monthly_gbp"]
        b=region["models"]["price"]["momentum"]["monthly_gbp"]
        if len(a)==len(b)==14 and all(x["return_gbp"] is not None for x in a+b):
            region["price_ridge_vs_momentum_bootstrap"]=simple_bootstrap_paired(
                [x["return_gbp"] for x in a],[x["return_gbp"] for x in b])
        results["regions"][reg]=region
        print("HOLDOUT_RESULT",reg,json.dumps({"names":len(stocks[reg]),
              "price":{k:v["gbp"] for k,v in region["models"]["price"].items()},
              "global_context":{k:v["gbp"] for k,v in region["models"]["price_plus_global_market_context"].items()},
              "global_interactions":{k:v["gbp"] for k,v in region["models"]["price_plus_global_market_interactions"].items()},
              "etf_proxy":region.get("research_etf_proxy",{}).get("summary")},separators=(",",":")),flush=True)
    feat=orig_feat
    Path("results").mkdir(exist_ok=True)
    Path("results/v14_frozen_holdout_results.json").write_text(json.dumps(results,indent=2))
    print("V14_COMPLETE",json.dumps({"evaluated":list(results["regions"]),
              "source_failures":len(errors),"research_only":True}),flush=True)

if __name__=="__main__":
    v14_main()
