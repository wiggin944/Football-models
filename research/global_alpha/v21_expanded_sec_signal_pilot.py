#!/usr/bin/env python3
"""Global Alpha v21 — expanded US economic/industry-context diagnostic (NO ALPHA).

As-of fundamentals from original SEC filing vintage, labels from pinned historical
adjusted prices. Cohort dynamically limited by observed current-month prices.
2024–2025 is pre-inspected, with known delisting-data gaps: results exploratory.
No live orders, no ISA deployment and no claim of risk-adjusted global alpha.
"""
import concurrent.futures as cf
from collections import defaultdict
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import v14_frozen_holdout as price

SRC=Path("source/results/v20b_full_us_asfiled_ttm_peers.parquet")
OUT=Path("results")
T0=pd.Period("2022-01",freq="M")
T1=pd.Period("2025-06",freq="M")
E0=pd.Period("2024-01",freq="M")
E1=T1
ALPHA=80.0
PERCENT_SELECTED=.30
SEC_FIN=("ttm_revenue_yoy","ttm_margin","liabilities_assets",
         "cash_assets","ttm_cashflow_assets")
SEC_PEERS=tuple("sic2_peer_rel_"+k for k in SEC_FIN)
ALL_MODELS=("price_ridge","fund_ridge","fund_plus_peers",
            "fund_plus_shuffled_peers","momentum","equal_weight")
COST_ONE_WAY=.002

def finance_data(row,cols,with_interactions=False):
    vals=[float(row.get(k)) if row is not None and pd.notna(row.get(k)) else float("nan") for k in cols]
    bounds=[(-1,3),(-1,1),(0,3),(0,1),(-2,2)]
    value=np.array([0.0 if not np.isfinite(z) else float(np.clip(z,*bound))
                    for z,bound in zip(vals,bounds)],float)
    miss=np.array([not np.isfinite(z) for z in vals],float)
    return np.r_[value,miss]

def load_prices():
    symbols=sorted(x["path"] for x in price.tree(
        next(x["sha"] for x in price.tree(
            next(x["sha"] for x in price.tree(price.PIN) if x["path"]=="data"))
            if x["path"]=="sp500") and []]) if False else None
    # Source names from the pinned full historical S&P constituent tree.
    root=price.tree(price.PIN)
    data=next(x["sha"] for x in root if x["path"]=="data")
    sp=next(x["sha"] for x in price.tree(data) if x["path"]=="sp500")
    folder=next(x["sha"] for x in price.tree(sp) if x["path"]=="constituents")
    symbols=sorted(x["path"] for x in price.tree(folder) if x["type"]=="tree")
    if len(symbols)!=709:raise RuntimeError("Archive population changed from frozen v20")
    index=price.load("sp500","INDEX",is_index=True)
    stocks={};errors={}
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        tasks={pool.submit(price.load,"sp500",sym):sym for sym in symbols}
        for j,task in enumerate(cf.as_completed(tasks),1):
            sym=tasks[task]
            try:stocks[sym]=task.result()
            except Exception as err:errors[sym]=str(err)[:90]
            if j%100==0:print("V21_PRICES_LOADED",j,"of",len(symbols),flush=True)
    if errors:raise RuntimeError("Missing historical stock files: "+json.dumps(errors)[:350])
    return stocks,index

def panels(stocks,index,facts):
    output=[]
    for period in pd.period_range(T0,T1,freq="M"):
        month=str(period)
        rr=[]
        for symbol,series in stocks.items():
            px=price.feat(series,index,month)
            if px is None:continue
            holding=price.holding(series,month)
            rr.append({"symbol":symbol,"price":px,
                       "realized":holding,
                       "fund_row":facts.get((symbol,month))})
        if len(rr)<100:raise RuntimeError("Unexpected small population "+month)
        # Current-month cohort-relative price movement known at decision cutoff.
        shock=float(np.mean([r["price"][0] for r in rr]))
        for r in rr:r["price"][-1]=r["price"][0]-shock
        usable=[r["realized"] for r in rr if r["realized"] is not None
           and np.isfinite(r["realized"]) and r["realized"]>-1]
        m=float(np.mean(usable)) if usable else None
        for r in rr:
            r["label"]=(r["realized"]-m if m is not None and r["realized"] is not None
                   and np.isfinite(r["realized"]) and r["realized"]>-1 else None)
            r["fund"]=finance_data(r["fund_row"],SEC_FIN)
            r["peer"]=finance_data(r["fund_row"],SEC_PEERS)
        # Negative control: rotate PEER features only among the same SIC2 when possible.
        bysic=defaultdict(list)
        for r in rr:
            sig=r["fund_row"].get("sic2_pit") if r["fund_row"] else None
            if pd.notna(sig):bysic[str(sig)].append(r)
        for sector,same in bysic.items():
            same.sort(key=lambda r:r["symbol"])
            if len(same)<3:continue
            for i,r in enumerate(same):
                r["fake_peer"]=same[(i+2)%len(same)]["peer"].copy()
        for r in rr:
            if "fake_peer" not in r:
                # No false surrogate for a company with no same-sector comparator.
                r["fake_peer"]=np.r_[np.zeros(5),np.ones(5)]
        output.append({"date":period,"rows":rr})
        if period.month==12 or period==T1:
            print("V21_PANEL_PREPARED",month,len(rr),"price_rows",len(usable),flush=True)
    return output

def matrix(r,kind):
    px=r["price"]
    if kind=="price_ridge":return px
    if kind=="fund_ridge":return np.r_[px,r["fund"]]
    if kind=="fund_plus_peers":return np.r_[px,r["fund"],r["peer"]]
    if kind=="fund_plus_shuffled_peers":return np.r_[px,r["fund"],r["fake_peer"]]
    raise KeyError(kind)

def holdings(rows,scores,prior,mode):
    if mode=="equal_weight":chosen=rows
    else:
        n=max(3,int(math.ceil(len(rows)*PERCENT_SELECTED)))
        chosen=sorted(rows,key=lambda z:(-scores[z["symbol"]],z["symbol"]))[:n]
    weights={x["symbol"]:1/len(chosen) for x in chosen}
    turn=(.5*sum(abs(weights.get(k,0)-prior.get(k,0))
          for k in set(weights)|set(prior)) if prior else 1.0)
    missing=[x["symbol"] for x in chosen if x["realized"] is None
          or not np.isfinite(x["realized"]) or x["realized"]<=-1]
    ret=(None if missing else float(np.mean([x["realized"] for x in chosen]))-COST_ONE_WAY*turn)
    return {"ret":ret,"missing":missing,"turnover":turn,
            "eligible":len(rows),"n_selected":len(chosen)},weights

def rank_ic(rows,scores):
    obs=[r for r in rows if r["label"] is not None and np.isfinite(r["label"])]
    if len(obs)<30:return None
    df=pd.DataFrame({"score":[scores[r["symbol"]] for r in obs],
                     "return":[r["label"] for r in obs]})
    return float(df.score.corr(df["return"],method="spearman"))

def backtest(panel):
    records={k:[] for k in ALL_MODELS}
    prev={k:{} for k in ALL_MODELS}
    quality=[]
    for item in panel:
        t=item["date"]
        if not E0<=t<=E1:continue
        # All training labels must have fully matured before decision t.
        tr=[r for old in panel if old["date"]+2<=t
            for r in old["rows"] if r["label"] is not None]
        if len(tr)<2000:raise RuntimeError("Insufficient predecision training")
        scored={}
        for model in ALL_MODELS[:4]:
            XX=np.stack([matrix(row,model) for row in tr])
            yy=np.array([row["label"] for row in tr],float)
            if not np.isfinite(XX).all():raise AssertionError("Nonfinite financial features")
            pipe=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA))
            pipe.fit(XX,yy)
            preds=pipe.predict(np.stack([matrix(r,model) for r in item["rows"]]))
            scored[model]={r["symbol"]:float(x) for r,x in zip(item["rows"],preds)}
        scored["momentum"]={r["symbol"]:float(r["price"][3]) for r in item["rows"]}
        scored["equal_weight"]={r["symbol"]:0.0 for r in item["rows"]}
        for method in ALL_MODELS:
            record,weights=holdings(item["rows"],scored[method],prev[method],method)
            record.update({"month":str(t),"spearman_ic_observed":rank_ic(item["rows"],scored[method])})
            prev[method]=weights
            records[method].append(record)
        quality.append({"month":str(t),"training_observed_rows":len(tr),
            "decision_cohort":len(item["rows"]),
            "missing_future_price_stocks":sum(r["realized"] is None for r in item["rows"])})
        print("V21_WALK_FORWARD",str(t),"train",len(tr),"decisions",len(item["rows"]),flush=True)
    return records,quality

def calc(rows):
    vals=[r["ret"] for r in rows]
    if len(rows)!=18 or any(x is None or x<=-1 or not np.isfinite(x) for x in vals):
        return {"complete":False,"n":len(rows),
            "n_missing_months":sum(x is None for x in vals),
            "cagr_usd_pct":None,"max_drawdown_pct":None,
            "one_way_turnover_month_avg":round(float(np.mean([r["turnover"] for r in rows])),3)}
    cap=1.;peak=1.;dd=0
    for r in vals:
        cap*=1+r;peak=max(peak,cap);dd=min(dd,cap/peak-1)
    return {"complete":True,"n":len(vals),
        "cagr_usd_pct":round((cap**(12/len(vals))-1)*100,3),
        "max_drawdown_pct":round(dd*100,3),
        "one_way_turnover_month_avg":round(float(np.mean([r["turnover"] for r in rows])),3),
        "avg_rank_ic_observed":round(float(np.mean([r["spearman_ic_observed"] for r in rows
            if r["spearman_ic_observed"] is not None])),4)}

def main():
    if not SRC.exists():raise RuntimeError("Missing verified v20B source")
    data=pd.read_parquet(SRC)
    if data.duplicated(["ticker","month"]).any():raise AssertionError("SEC duplicated ticker-month")
    for c in ("revenue_available","earnings_available"):
        future=(pd.to_datetime(data[c],errors="coerce")>
                pd.to_datetime(data.decision_date))
        if future.any():raise AssertionError("Future SEC filing leaked into "+c)
    facts={(row.ticker,row.month):row._asdict() for row in data.itertuples(index=False)}
    print("V21_SECRECORDS",len(data),"mapped",data.ticker.nunique(),flush=True)
    stocks,index=load_prices()
    panel=panels(stocks,index,facts)
    results,quality=backtest(panel)
    OUT.mkdir(exist_ok=True)
    report={
       "version":"v21_us_exploratory_sector_ttm_ablation",
       "evaluated":"January 2024–June 2025, 18 monthly decision dates",
       "return_units":"USD; nominal costs of 20bps times one-way turnover; FX not modelled",
       "not_untouched":True,"alpha_verified":False,
       "price_archive_commit":price.PIN,
       "train_period":"from Jan 2022 with next-month-open-to-month-after-open labels; maturity by t-2",
       "features":"SEC original-filing Q4/TTM economics and SIC2 peer median differential",
       "fair_comparison":"Identical US price-eligible rows, labels, execution and costs across methods",
       "methods":{method:{"metrics":calc(rows),"monthly":rows} for method,rows in results.items()},
       "cohort_quality":quality,
       "limitations":[
           "Historical index archive contains ticker spans and is NOT a proven PIT index-membership master.",
           "Stock exit prices missing for delisted/restructured securities; fail-closed returns rather than dropping trades.",
           "As-filed SEC data linked via modern ticker/CIK snapshot, not verified historical identity chain.",
           "Price-only and expanded models are both Ridge, with chosen alpha and selection cutoff not independently tuned.",
           "No valuation, analyst estimates, actual competitor metadata, news or earnings publication events.",
           "SIC2 peers are broad approximations of real competitors.",
           "This 2024–25 data has previously been examined, so no independent alpha inference.",
           "Missing SEC data retained through missingness flags, not outcome-informed deletion.",
           "Trading costs are illustrative. Corporate actions and share class mappings remain unvalidated.",
           "Research only; no real money, broker orders, ISA integration or automation of deposits."
       ]}
    (OUT/"v21_exploratory_ablation.json").write_text(json.dumps(report,indent=2))
    print("V21_RESULTS",json.dumps({
       "months":len(quality),
       "eligible_range":[min(q["decision_cohort"] for q in quality),max(q["decision_cohort"] for q in quality)],
       "metrics":{k:calc(v) for k,v in results.items()}},separators=(",",":")),flush=True)

if __name__=="__main__":main()
