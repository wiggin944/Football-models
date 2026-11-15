#!/usr/bin/env python3
"""Global Alpha v16B: real-price global pooled nonlinear peer-anomaly ablation.

Every feature uses data available by the decision month's end; buys at following
month's opening price. This is exploratory because 2025-26 is inspected history.
This is NOT evidence of alpha, investment advice, or a broker connection.
"""
import concurrent.futures as cf
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import v14_frozen_holdout as v

REGIONS=["sp500","eurostoxx50","nikkei225","hsi"]
REG_INDEX={r:i for i,r in enumerate(REGIONS)}
DSTART=pd.Period("2017-01",freq="M")
DEND=pd.Period("2026-09",freq="M")
TEST_START=pd.Period("2025-08",freq="M")
TEST_END=pd.Period("2026-09",freq="M")
LOOKBACK_MONTHS=25
PEER_COUNT=5
SEED=23699
PARAMS={"max_iter":100,"max_leaf_nodes":10,"learning_rate":0.05,
        "min_samples_leaf":50,"l2_regularization":5.0,
        "early_stopping":False,"random_state":SEED}

def month_price(series,period):
    k=str(period)
    if k not in series.index:return None
    a=float(series.loc[k,"close"])
    return a if np.isfinite(a) and a>0 else None

def history(series,t):
    # t-25 .. t-1, EXCLUDING current-month return when matching peers.
    a=[month_price(series,t-i) for i in range(LOOKBACK_MONTHS,0,-1)]
    if any(z is None for z in a):return None
    return np.diff(np.log(np.array(a,float))) # 24 monthly returns

def build_panel(reg,assets,idx,indexes,t):
    features={}
    histories={}
    for sym,series in assets.items():
        x=v.feat(series,idx,str(t))
        if x is None:continue
        h=history(series,t)
        if h is None:continue
        features[sym]=x
        histories[sym]=h
    if len(features)<6:return None
    names=sorted(features)
    # Predict the effect of the last one/three-month moves relative to peers.
    current=np.array([features[sym] for sym in names],float)
    core=np.array([features[sym] for sym in names],float)
    core[:,-1]=core[:,0]-np.mean(core[:,0])
    hist=np.stack([histories[s] for s in names])
    correlations=np.corrcoef(hist)
    correlations=np.nan_to_num(correlations,nan=-1.0,posinf=-1.0,neginf=-1.0)
    np.fill_diagonal(correlations,-np.inf)
    peer_features={}
    for i,sym in enumerate(names):
        near=np.argsort(correlations[i])[::-1][:PEER_COUNT]
        p=core[near]
        rel=float(np.mean(correlations[i][near]))
        # Actual peers are dynamic price-correlation neighbours, not claims
        # about economic competitors or the company's actual supply chain.
        peer_features[sym]=np.array([
           core[i,0]-np.mean(p[:,0]), # current idiosyncratic one-month move
           core[i,1]-np.mean(p[:,1]), # three-month peer gap
           core[i,2]-np.mean(p[:,2]), # six-month peer gap
           float(np.mean(p[:,0])),  # peer short-term direction
           float(np.mean(p[:,1])),  # peer medium-term direction
           float(np.std(p[:,0])),   # peer dispersion
           float(np.mean(p[:,0]<0)), # fraction peers negative
           rel # past-return correlation
        ],float)
    reg_onehot=np.eye(len(REGIONS),dtype=float)[REG_INDEX[reg]]
    gl=[]
    for j in REGIONS:
        a=month_price(indexes[j],t)
        b=month_price(indexes[j],t-3)
        if a is None or b is None:
            return None
        gl.append(a/b-1)
    macro=np.array(gl,float)
    panel=[]
    actual={}
    for i,s in enumerate(names):
        peer=peer_features[s]
        base=np.concatenate([core[i],reg_onehot,macro])
        future=v.holding(assets[s],str(t))
        actual[s]=future
        panel.append({"region":reg,"month":str(t),"symbol":s,"base":base,
           "peer":peer,"ret":future})
    # Fixed within-region permutation: real distribution of peer features,
    # deliberately mismatched to stock identity.
    ordered=sorted(panel,key=lambda x:x["symbol"])
    shift=7%len(ordered)
    for j,item in enumerate(ordered):
        item["placebo_peer"]=ordered[(j+shift)%len(ordered)]["peer"].copy()
    valid=[r["ret"] for r in panel if r["ret"] is not None and np.isfinite(r["ret"]) and r["ret"]>-1]
    if len(valid)>=4:
        mu=float(np.mean(valid))
        for row in panel:
            row["y"]=row["ret"]-mu if row["ret"] is not None and np.isfinite(row["ret"]) and row["ret"]>-1 else None
    else:
        for row in panel:row["y"]=None
    return panel

def load_region(reg):
    syms,total=v.region_symbols(reg)
    base=v.load(reg,"INDEX",is_index=True)
    index,_=v.append_yahoo(base,reg,"INDEX",index=True)
    old={}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(v.load,reg,s):s for s in syms}
        for job in cf.as_completed(jobs):
            s=jobs[job]
            try:
                hist=job.result()
                if "2025-08" in hist.index and hist.loc["2025-08","days"]>=5:
                    old[s]=hist
            except Exception:pass
    stocks={}
    errors={}
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        jobs={pool.submit(v.append_yahoo,series,reg,s):s for s,series in old.items()}
        for job in cf.as_completed(jobs):
            s=jobs[job]
            try:stocks[s],_=job.result()
            except Exception as exc:
                stocks[s]=old[s]
                errors[s]=str(exc)[:100]
    print("REGION_LOADED",reg,"cohort",len(stocks),"of archive",total,
          "continuation_errors",len(errors),flush=True)
    return {"index":index,"stocks":stocks,"errors":errors,
            "historical_folders":total,"asof_eligible":len(stocks)}

def select_return(rows,predictions,previous):
    k=max(3,math.ceil(len(rows)*0.30))
    chosen=sorted(rows,key=lambda r:predictions[r["symbol"]],reverse=True)[:k]
    weights={r["symbol"]:1/k for r in chosen}
    if not previous:turnover=1.0
    else:turnover=0.5*sum(abs(weights.get(s,0)-previous.get(s,0)) for s in set(weights)|set(previous))
    absent=[r["symbol"] for r in chosen if r["ret"] is None or not np.isfinite(r["ret"])]
    ret=(None if absent else float(np.mean([r["ret"] for r in chosen]))-v.BPS_ONE_WAY/10000*turnover)
    return ret,weights,turnover,absent

def run():
    datasets={r:load_region(r) for r in REGIONS}
    stocks={r:z["stocks"] for r,z in datasets.items()}
    indexes={r:z["index"] for r,z in datasets.items()}
    panels=[]
    counters=defaultdict(int)
    for t in pd.period_range(DSTART,DEND,freq="M"):
        for r in REGIONS:
            pane=build_panel(r,stocks[r],indexes[r],t)
            if pane:
                panels.append({"month":t,"region":r,"rows":pane})
                counters[r]+=len(pane)
        if t.month==12:print("PREPARED_YEAR",t.year,flush=True)
    def keys(kind):
        if kind=="price":return lambda row:row["base"]
        if kind=="peer":return lambda row:np.r_[row["base"],row["peer"]]
        if kind=="placebo":return lambda row:np.r_[row["base"],row["placebo_peer"]]
        raise ValueError(kind)
    methods=("gbdt_price","gbdt_peer","gbdt_placebo","momentum")
    results={r:{kind:[] for kind in methods} for r in REGIONS}
    prev={r:{kind:{} for kind in methods} for r in REGIONS}
    counts=[]
    for t in pd.period_range(TEST_START,TEST_END,freq="M"):
        train=[row for panel in panels if panel["month"]+2<=t
               for row in panel["rows"] if row["y"] is not None]
        if len(train)<200:raise RuntimeError("Insufficient training, fail closed")
        selected=[panel for panel in panels if panel["month"]==t]
        if len(selected)!=len(REGIONS):
            raise RuntimeError("Missing contemporaneous market or price panel "+str(t))
        models={}
        for variant in ("price","peer","placebo"):
            f=keys(variant)
            XX=np.stack([f(row) for row in train])
            yy=np.array([row["y"] for row in train],float)
            model=HistGradientBoostingRegressor(**PARAMS)
            model.fit(XX,yy)
            models[variant]=model
        counts.append({"month":str(t),"training_rows":len(train)})
        for panel in selected:
            reg=panel["region"];rs=panel["rows"]
            scored={}
            for variant,m in models.items():
                val=m.predict(np.stack([keys(variant)(row) for row in rs]))
                scored["gbdt_"+variant]={row["symbol"]:float(x) for row,x in zip(rs,val)}
            scored["momentum"]={row["symbol"]:float(row["base"][3]) for row in rs}
            for method in methods:
                r,w,turn,absent=select_return(rs,scored[method],prev[reg][method])
                results[reg][method].append({"month":str(t),"return":r,
                    "turnover":float(turn),"missing":absent,"n":len(rs)})
                prev[reg][method]=w
        print("TEST_MONTH",str(t),"train_rows",len(train),
              "market_count",len(selected),flush=True)
    fx={r:v.raw_monthly_yahoo(v.STOCK_FX[r]) for r in REGIONS}
    etfs={r:v.raw_monthly_yahoo(v.BENCH_ETF[r]) for r in REGIONS}
    usd=v.raw_monthly_yahoo("GBPUSD=X")
    output={"version":"v16B","date":"2026-11-15",
     "universe":"As-of-August-2025 symbols sampled from retrospective index folders; NOT survivorship-free",
     "test_period":"August 2025 to September 2026; already examined; exploratory not untouched holdout",
     "data_source_sha":v.PIN,"alpha_proven":False,
     "features":"Basic price/context plus four market index regimes, and rolling 24m top-five price-correlation peers (not sector/competitor fundamental metadata)",
     "policy":"Pooled GBDT one-month region-relative labels, 30% selected per region, 20bp stock and 30bp FX one-way, no short/leverage/cash",
     "fixed_model":PARAMS,"train_counts":counts,
     "companies_asof":{r:datasets[r]["asof_eligible"] for r in REGIONS},
     "source_errors":{r:datasets[r]["errors"] for r in REGIONS},
     "regions":{}}
    summary={}
    for reg in REGIONS:
        m={}
        for kind,monthly in results[reg].items():
            trans=v.convert_gbp(monthly,fx[reg],kind)
            m[kind]={"summary":v.calculate(trans),
                     "mean_oneway_turnover":float(np.mean([x["turnover"] for x in monthly])),
                     "monthly_gbp":trans}
        dates=[x["month"] for x in results[reg]["gbdt_peer"]]
        bench=v.etf_benchmark_usd_in_gbp(dates,etfs[reg],usd)
        benchmark=v.calculate(bench)
        def pair(a,b):
            x=m[a]["monthly_gbp"];y=m[b]["monthly_gbp"]
            if len(x)!=14 or len(y)!=14 or any(i["return_gbp"] is None for i in x+y):return None
            return v.simple_bootstrap_paired([i["return_gbp"] for i in x],
                      [i["return_gbp"] for i in y],B=12000)
        output["regions"][reg]={"methods":m,"ETF":{"ticker":v.BENCH_ETF[reg],
                    "summary":benchmark,"monthly_gbp":bench},
                    "peer_minus_base":pair("gbdt_peer","gbdt_price"),
                    "peer_minus_placebo":pair("gbdt_peer","gbdt_placebo")}
        summary[reg]={"n":datasets[reg]["asof_eligible"],
            "methods":{k:d["summary"] for k,d in m.items()},
            "etf":benchmark,
            "peer_minus_base":output["regions"][reg]["peer_minus_base"]}
        print("REGION_SUMMARY",reg,json.dumps(summary[reg],separators=(",",":")),flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/v16b_global_peer_ml.json").write_text(json.dumps(output,indent=2))
    print("GLOBAL_PEER_RESULTS",json.dumps(summary,separators=(",",":")),flush=True)

if __name__=="__main__":
    run()
