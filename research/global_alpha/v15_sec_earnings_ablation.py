#!/usr/bin/env python3
"""Global Alpha v15: exploratory point-in-time SEC filing/event-context ablation.

EXPLORATORY: prior 2025-26 period has been inspected, so NOT a blind holdout.
No trading, no financial recommendations, not a global-fundamentals model.
"""
import concurrent.futures as cf
from datetime import date,timedelta
import io,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import v14_frozen_holdout as v

SOURCE_REPO="daggerok/fundamentals"
SEC_PIN="e775590c0daef2be0c1a7115a2a3fea8b358a5a7"
REVENUE_TAGS=[
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet"]
NET_TAGS=["NetIncomeLoss","ProfitLoss"]
WINDOW_START="2025-08"; WINDOW_END="2026-09"
MAX_FILE_LAG=150; AVAIL_LAG_DAYS=5

def sec_json(sym):
    path=f"data/fundamentals/{sym}.json"
    url=f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SEC_PIN}/{path}"
    raw=v.get(url,v.CACHE/SEC_PIN/path)
    j=json.loads(raw)
    if j.get("symbol")!=sym and j.get("ticker")!=sym:
        raise ValueError("SEC symbol/identity mismatch "+sym)
    if j.get("source")!="sec-companyfacts":
        raise ValueError("Unexpected SEC data source")
    return j

def quarterly(j,tags,allow_negative):
    facts=j.get("facts",{}).get("us-gaap",{})
    candidates=[]
    for rank,tag in enumerate(tags):
        for r in facts.get(tag,{}).get("units",{}).get("USD",[]):
            if r.get("form")!="10-Q":continue
            try:
                start=date.fromisoformat(r["start"]);end=date.fromisoformat(r["end"])
                filed=date.fromisoformat(r["filed"]);val=float(r["val"])
            except (KeyError,TypeError,ValueError):continue
            span=(end-start).days;lag=(filed-end).days
            if not(65<=span<=110 and 10<=lag<=MAX_FILE_LAG) or not np.isfinite(val):continue
            if not allow_negative and val<=0:continue
            candidates.append({"end":end,"start":start,"filed":filed,
                 "available":filed+timedelta(days=AVAIL_LAG_DAYS),
                 "val":val,"tag":tag,"rank":rank,"accn":r.get("accn","")})
    # Earliest actually filed record wins for a quarter; tag priority breaks only same-date ties.
    candidates.sort(key=lambda x:(x["end"],x["filed"],x["rank"],x["accn"]))
    output={}
    for r in candidates:
        output.setdefault(r["end"],r)
    return sorted(output.values(),key=lambda x:x["end"])

def add_growth(revenues):
    for i,q in enumerate(revenues):
        older=[p for p in revenues[:i] if 335<=(q["end"]-p["end"]).days<=395]
        if older:
            prior=min(older,key=lambda p:abs((q["end"]-p["end"]).days-365))
            q["yoy"]=q["val"]/prior["val"]-1
        else:q["yoy"]=None
        q["acceleration"]=None
        if q["yoy"] is not None:
            prev=[p for p in revenues[:i] if 60<=(q["end"]-p["end"]).days<=135 and p.get("yoy") is not None]
            if prev:
                p=min(prev,key=lambda z:abs((q["end"]-z["end"]).days-91))
                q["acceleration"]=q["yoy"]-p["yoy"]
    return revenues

def extract(j):
    return {
        "revenue":add_growth(quarterly(j,REVENUE_TAGS,False)),
        "income":quarterly(j,NET_TAGS,True),
        "cik":str(j.get("cik",""))
    }

def fundamental_features(facts,month,base):
    end=pd.Period(month,freq="M").end_time.date()
    qr=[q for q in facts["revenue"] if q["available"]<=end]
    qi=[q for q in facts["income"] if q["available"]<=end]
    rev=qr[-1] if qr else None
    yoy=rev.get("yoy") if rev else None
    acceleration=rev.get("acceleration") if rev else None
    matched=[q for q in qi if rev and q["end"]==rev["end"]]
    income=matched[0] if matched else None
    margin=(income["val"]/rev["val"]) if income and rev else None
    recent=any(q["available"].strftime("%Y-%m")==month for q in qr)
    age=min(12,max(0,(end-rev["available"]).days/30.44))/12 if rev else 1.0
    miss=[float(x is None or not math.isfinite(x)) for x in (yoy,acceleration,margin)]
    y=0.0 if miss[0] else float(np.clip(yoy,-2,5))
    a=0.0 if miss[1] else float(np.clip(acceleration,-3,3))
    m=0.0 if miss[2] else float(np.clip(margin,-3,3))
    return np.array([y,a,m,float(recent),*miss,age,
           y*base[4],a*base[4],a*base[6],float(recent)*base[4]],float)

def decorate(stock,index,month,sym,facts_by_symbol):
    base=v.PRICE_FEATURE_FUNCTION(stock,index,month)
    if base is None:return None
    extra=fundamental_features(facts_by_symbol.get(sym,EMPTY),month,base)
    # v.evaluate will overwrite LAST feature with contemporaneous cohort-relative shock.
    return np.concatenate([base[:-1],extra,base[-1:]])

EMPTY={"revenue":[],"income":[],"cik":""}

def run_with_features(stocks,index,fn):
    original=v.feat
    try:
        v.feat=fn
        return v.evaluate("sp500",stocks,index,709,{})
    finally:v.feat=original

def gbp_summary(run,fx):
    result={}
    for strategy,monthly in run["monthly"].items():
        x=v.convert_gbp(monthly,fx,strategy)
        result[strategy]={"summary":v.calculate(x),
                          "monthly_gbp":x,
                          "average_turnover":run["methods"][strategy]["avg_turnover_one_way"]}
    return result

def main():
    syms,all_hist=v.region_symbols("sp500")
    arch=v.load("sp500","INDEX",is_index=True)
    index,_=v.append_yahoo(arch,"sp500","INDEX",index=True)
    existing={}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(v.load,"sp500",s):s for s in syms}
        for job in cf.as_completed(jobs):
            sym=jobs[job]
            try:
                f=job.result()
                if "2025-08" in f.index and f.loc["2025-08","days"]>=5:
                    existing[sym]=f
            except Exception as exc:print("ARCHIVE_ERROR",sym,str(exc)[:60],flush=True)
    stocks={}
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        jobs={pool.submit(v.append_yahoo,f,"sp500",sym):sym for sym,f in existing.items()}
        for job in cf.as_completed(jobs):
            sym=jobs[job]
            try: stocks[sym],_=job.result()
            except Exception as exc:
                stocks[sym]=existing[sym] # do NOT drop absent future observations
                print("FUTURE_PRICE_MISSING",sym,str(exc)[:90],flush=True)
    print("EX_ANTE_UNIVERSE",len(stocks),"of",len(syms),flush=True)
    facts={};missing={}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(sec_json,s):s for s in stocks}
        for job in cf.as_completed(jobs):
            sym=jobs[job]
            try:facts[sym]=extract(job.result())
            except Exception as exc:missing[sym]=str(exc)[:150]
    print("SEC_FACTS",len(facts),"missing",len(missing),flush=True)
    v.PRICE_FEATURE_FUNCTION=v.feat
    price=run_with_features(stocks,index,v.PRICE_FEATURE_FUNCTION)
    def event_feat(stock,market,month):
        # v.evaluate only supplies the Pandas stock series, so identify without
        # looking at any future data using a stable in-memory identity lookup.
        s=id_to_symbol[id(stock)]
        return decorate(stock,market,month,s,facts)
    id_to_symbol={id(data):s for s,data in stocks.items()}
    contextual=run_with_features(stocks,index,event_feat)
    ordered=sorted(stocks)
    shift=7 if len(ordered)>7 else 1
    placebo={s:facts.get(ordered[(i+shift)%len(ordered)],EMPTY)
             for i,s in enumerate(ordered)}
    def placebo_feat(stock,market,month):
        return decorate(stock,market,month,id_to_symbol[id(stock)],placebo)
    negative_control=run_with_features(stocks,index,placebo_feat)
    fx=v.raw_monthly_yahoo("GBPUSD=X")
    etf=v.raw_monthly_yahoo("SPY")
    price_gbp=gbp_summary(price,fx)
    context_gbp=gbp_summary(contextual,fx)
    placebo_gbp=gbp_summary(negative_control,fx)
    months=[x["month"] for x in price["monthly"]["ridge"]]
    benchmark=v.etf_benchmark_usd_in_gbp(months,etf,fx)
    coverage={s:{"quarters_revenue":len(q["revenue"]),"quarters_income":len(q["income"]),
                 "rev_yoy_count":sum(x.get("yoy") is not None for x in q["revenue"]),
                 "acceleration_count":sum(x.get("acceleration") is not None for x in q["revenue"]),
                 "negative_income_quarters":sum(x["val"]<0 for x in q["income"])}
              for s,q in facts.items()}
    ctx=context_gbp["ridge"]["monthly_gbp"]
    def bootstrap(other):
        rs=other["ridge"]["monthly_gbp"]
        if len(ctx)!=14 or any(x["return_gbp"] is None for x in ctx+rs):return None
        return v.simple_bootstrap_paired([z["return_gbp"] for z in ctx],
                                         [z["return_gbp"] for z in rs])
    results={
        "version":"v15_US_filing_context_EXPLORATORY",
        "research_date":"2026-11-15","source_git_sha":SEC_PIN,
        "prior_period_already_inspected":True,
        "real_economic_alpha_proven":False,
        "universe":{"sample":len(syms),"as_of_cutoff":len(stocks),
                    "secs_available":len(facts),"secs_missing":missing},
        "coverage_by_symbol":coverage,
        "price":price_gbp,"fundamental_events":context_gbp,"shuffled_facts_placebo":placebo_gbp,
        "SPY_research_proxy":{"summary":v.calculate(benchmark),"monthly_gbp":benchmark},
        "fundamental_minus_price":bootstrap(price_gbp),
        "fundamental_minus_placebo":bootstrap(placebo_gbp),
        "limitations":[
            "SEC 10-Q dates +5 calendar days, NOT earlier earnings press-release timestamps",
            "Only US stocks; global universe remains untested for fundamentals",
            "Year-over-year revenue growth is a proxy, not analyst consensus surprise",
            "2025-26 already inspected; descriptive statistics only",
            "Instruments from frozen selected 2025-08 members; survivorship in history possible",
            "Missing/stale companyfacts encoded; source archive may still omit historical versions",
            "No portfolio cash allocation, monthly salary deposits, UK ISA restrictions or after-tax costs simulated",
            "Signed net income included; banks require sector-specific tags",
        ]}
    Path("results").mkdir(exist_ok=True)
    Path("results/v15_sec_earnings_ablation.json").write_text(json.dumps(results,indent=2))
    print("V15_SUMMARY",json.dumps({
        "sample":len(stocks),"sec_available":len(facts),
        "price":price_gbp["ridge"]["summary"],
        "context":context_gbp["ridge"]["summary"],
        "placebo":placebo_gbp["ridge"]["summary"],
        "momentum":price_gbp["momentum"]["summary"],
        "SPY":results["SPY_research_proxy"]["summary"],
        "paired_context_price":results["fundamental_minus_price"],
        "paired_context_placebo":results["fundamental_minus_placebo"]
    },separators=(",",":")),flush=True)

if __name__=="__main__":
    main()
