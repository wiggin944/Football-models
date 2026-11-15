#!/usr/bin/env python3
"""Global Alpha v18 — chronologically refitted, SEC-as-filed US fundamentals ablation.

PRE-INSPECTED evaluation (Aug 2025–Sep 2026); descriptive ONLY, not an
untouched holdout. Monthly long-only research portfolios, NO trade execution.
All stock memberships and returns retain unavailable future exit prices as
explicit missing returns. Training uses 2022+ months with only matured labels.
"""
import json, math
from pathlib import Path
import concurrent.futures as cf
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import v14_frozen_holdout as v

RAW_FEATURE_FILE=Path("source/results/v17d_asof_fundamental_features.parquet")
OUT=Path("results")
START=pd.Period("2022-01",freq="M")
END=pd.Period("2026-09",freq="M")
TEST_START=pd.Period("2025-08",freq="M")
TEST_END=END
ALPHA=80.0
N_SELECTED_FRACTION=0.30
MODEL_NAMES=("price","filing_fundamentals","filing_placebo","momentum","equal")
FIELDS=("fund_revenue_yoy","fund_net_margin","fund_leverage","fund_cash_assets")

def quality_gate(raw):
    d=raw.copy()
    for name in ("decision_date","latest_revenue_release","latest_income_release","latest_revenue_period"):
        d[name]=pd.to_datetime(d[name],errors="coerce")
    for name in ("revenue_yoy","net_margin","leverage_liabilities_assets","cash_assets"):
        d[name]=pd.to_numeric(d[name],errors="coerce")
    for name in [c for c in d if c.endswith("_lag_days")]:
        d[name]=pd.to_numeric(d[name],errors="coerce")
    future={}
    for k in ("latest_revenue_release","latest_income_release","latest_revenue_period"):
        future[k]=int((d[k].notna() & (d[k]>d["decision_date"])).sum())
    if sum(future.values())>0:raise AssertionError(f"SEC look-ahead rows: {future}")
    fresh_rev=d.sales_lag_days.between(0,180)
    fresh_earn=d.earnings_lag_days.between(0,180)
    good_margin=d.net_margin.abs().le(1)
    fresh_balance=d.assets_lag_days.between(0,365)&d.liabilities_lag_days.between(0,365)
    fresh_cash=d.assets_lag_days.between(0,365)&d.cash_lag_days.between(0,365)
    d["fund_revenue_yoy"]=d.revenue_yoy.where(fresh_rev)
    d["fund_net_margin"]=d.net_margin.where(fresh_rev&fresh_earn&good_margin)
    d["fund_leverage"]=d.leverage_liabilities_assets.where(fresh_balance)
    d["fund_cash_assets"]=d.cash_assets.where(fresh_cash)
    d["fund_context_usable"]=d.fund_revenue_yoy.notna()&d.fund_net_margin.notna()
    assert not d.duplicated(["ticker","month"]).any()
    audit=d[d.month.between(str(TEST_START),str(TEST_END))]
    summary={"rows":len(d),"companies":int(d.ticker.nunique()),"future_release_rows":future,
             "test_coverage":{k:round(float(audit[k].notna().mean()),4) for k in FIELDS}}
    if len(d)!=4617 or int(d.ticker.nunique())!=81:
        raise AssertionError(f"Unexpected SEC archive snapshot {summary}")
    print("SEC_QUALITY_GATE",json.dumps(summary),flush=True)
    return d,summary

def fundamental_vec(rec,price):
    if rec is None: vals=[None]*4
    else: vals=[rec.get(k) for k in FIELDS]
    vals=[float(x) if x is not None and pd.notna(x) else np.nan for x in vals]
    missing=np.array([not np.isfinite(x) for x in vals],float)
    # Bounds are predeclared stability safeguards, not tuned to evaluation returns.
    limits=((-1,3),(-1,1),(0,3),(0,1))
    numeric=np.array([0.0 if not np.isfinite(x) else float(np.clip(x,*bounds))
                     for x,bounds in zip(vals,limits)],float)
    # Company's short-term relative move and drawdown can interact with fundamentals.
    interactions=np.array([numeric[0]*price[4],numeric[1]*price[4],
                           numeric[2]*price[6],numeric[3]*price[6]],float)
    return np.r_[numeric,missing,interactions]

def load_market():
    syms,folders=v.region_symbols("sp500")
    index,_=v.append_yahoo(v.load("sp500","INDEX",is_index=True),"sp500","INDEX",index=True)
    historic={}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        jobs={pool.submit(v.load,"sp500",sym):sym for sym in syms}
        for fut in cf.as_completed(jobs):
            sym=jobs[fut]
            try:
                x=fut.result()
                if "2025-08" in x.index and pd.notna(x.loc["2025-08","close"]) and x.loc["2025-08","days"]>=5:
                    historic[sym]=x
            except Exception as e:print("ARCHIVE_ERROR",sym,type(e).__name__,flush=True)
    stocks={};failures={}
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        jobs={pool.submit(v.append_yahoo,x,"sp500",sym):sym for sym,x in historic.items()}
        for fut in cf.as_completed(jobs):
            sym=jobs[fut]
            try:stocks[sym],_=fut.result()
            except Exception as e:
                stocks[sym]=historic[sym]  # Do not drop historical eligible stock on missing forward prices.
                failures[sym]=str(e)[:140]
    if not 50<=len(stocks)<=100:raise RuntimeError("Surprising as-of population: "+str(len(stocks)))
    print("ASOF_US_UNIVERSE",len(stocks),"sampled",len(syms),"missing_future",failures,flush=True)
    return stocks,index,failures

def make_panel(stock_data,index,fund_by_pair):
    panels=[]
    names=sorted(stock_data)
    for t in pd.period_range(START,END,freq="M"):
        s=str(t); rows=[]
        for sym in names:
            price=v.feat(stock_data[sym],index,s)
            if price is None:continue
            ret=v.holding(stock_data[sym],s)
            rows.append({"ticker":sym,"price":price,"ret":ret})
        if len(rows)<5:continue
        # Current-month population shock uses only observed current price moves.
        shock=float(np.mean([r["price"][0] for r in rows]))
        for row in rows:row["price"][-1]=row["price"][0]-shock
        rows.sort(key=lambda a:a["ticker"])
        names_current=[r["ticker"] for r in rows]
        # A fixed, predeclared cross-company permutation for the negative control.
        rotated=names_current[7%len(rows):]+names_current[:7%len(rows)]
        mapping=dict(zip(names_current,rotated))
        valid=[r["ret"] for r in rows if r["ret"] is not None and np.isfinite(r["ret"]) and r["ret"]>-1]
        mu=float(np.mean(valid)) if valid else None
        for row in rows:
            rec=fund_by_pair.get((row["ticker"],s))
            neg=fund_by_pair.get((mapping[row["ticker"]],s))
            row["fundamental"]=fundamental_vec(rec,row["price"])
            row["placebo"]=fundamental_vec(neg,row["price"])
            row["label"]=(row["ret"]-mu if mu is not None and row["ret"] is not None
                          and np.isfinite(row["ret"]) and row["ret"]>-1 else None)
        panels.append({"month":t,"rows":rows})
    return panels

def get_x(row,kind):
    if kind=="price":return row["price"]
    if kind=="filing_fundamentals":return np.r_[row["price"],row["fundamental"]]
    if kind=="filing_placebo":return np.r_[row["price"],row["placebo"]]
    raise ValueError(kind)

def model_train(rows,kind):
    xx=np.stack([get_x(z,kind) for z in rows])
    yy=np.array([z["label"] for z in rows],float)
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():raise AssertionError("Nonfinite training observations")
    model=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA))
    model.fit(xx,yy)
    return model

def select(rows,scores,prior):
    n=max(3,int(math.ceil(N_SELECTED_FRACTION*len(rows))))
    chosen=sorted(rows,key=lambda r:(-scores[r["ticker"]],r["ticker"]))[:n]
    now={r["ticker"]:1/n for r in chosen}
    turnover=(0.5*sum(abs(now.get(s,0)-prior.get(s,0)) for s in set(now)|set(prior))
              if prior else 1.0)
    missing=[r["ticker"] for r in chosen if r["ret"] is None or
             not np.isfinite(r["ret"]) or r["ret"]<=-1]
    gross=None if missing else float(np.mean([r["ret"] for r in chosen]))
    net=None if gross is None else gross-v.BPS_ONE_WAY/10000*turnover
    return {"return":net,"missing":missing,"turnover":turnover,
            "n_selected":len(chosen),"eligible":len(rows)},now

def analyze(panels):
    logs={kind:[] for kind in MODEL_NAMES}
    prev={kind:{} for kind in MODEL_NAMES}
    train_n=[]
    for p in panels:
        if p["month"]<TEST_START or p["month"]>TEST_END:continue
        # Labels from decision month k refer to next-month open through k+2 open.
        # A return is only used for training if k+2 <= current decision month.
        training=[r for old in panels if old["month"]+2<=p["month"]
                  for r in old["rows"] if r["label"] is not None]
        if len(training)<300:raise RuntimeError("No sufficient matured training")
        train_n.append(len(training))
        current=p["rows"]
        preds={}
        for kind in ("price","filing_fundamentals","filing_placebo"):
            est=model_train(training,kind)
            scores=est.predict(np.stack([get_x(r,kind) for r in current]))
            preds[kind]={r["ticker"]:float(s) for r,s in zip(current,scores)}
        preds["momentum"]={r["ticker"]:float(r["price"][3]) for r in current}
        preds["equal"]={r["ticker"]:0.0 for r in current}
        for kind in MODEL_NAMES:
            if kind=="equal":
                scores=preds[kind]
                # Equal-weight full eligible universe; not top 30%.
                tmp=current; prior=prev[kind]
                now={r["ticker"]:1/len(tmp) for r in tmp}
                turnover=(0.5*sum(abs(now.get(s,0)-prior.get(s,0)) for s in set(now)|set(prior))
                          if prior else 1.0)
                missing=[r["ticker"] for r in tmp if r["ret"] is None]
                net=(None if missing else float(np.mean([r["ret"] for r in tmp]))-v.BPS_ONE_WAY/10000*turnover)
                res={"return":net,"missing":missing,"turnover":turnover,
                    "n_selected":len(tmp),"eligible":len(tmp)}
            else:
                res,now=select(current,preds[kind],prev[kind])
            logs[kind].append({"month":str(p["month"]),**res})
            prev[kind]=now
        print("MODEL_MONTH",str(p["month"]),"stocks",len(current),"train",len(training),flush=True)
    return logs,train_n

def summarize(logs):
    fx=v.raw_monthly_yahoo("GBPUSD=X")
    usd_etf=v.raw_monthly_yahoo("SPY")
    out={}
    for kind,rows in logs.items():
        gbp=v.convert_gbp(rows,fx,kind)
        out[kind]={"summary":v.calculate(gbp,n_expected=14),
                   "avg_one_way_turnover":round(float(np.mean([z["turnover"] for z in rows])),4),
                   "months_missing":{z["month"]:z["missing"] for z in rows if z["missing"]},
                   "monthly_gbp":gbp}
    months=[r["month"] for r in logs["price"]]
    etf=v.etf_benchmark_usd_in_gbp(months,usd_etf,fx)
    out["SPY_benchmark"]={"summary":v.calculate(etf,n_expected=14),"monthly_gbp":etf}
    return out

def paired(out,a,b):
    x=out[a]["monthly_gbp"];y=out[b]["monthly_gbp"]
    if len(x)!=len(y) or len(x)!=14 or any(r["return_gbp"] is None for r in x+y):return None
    return v.simple_bootstrap_paired(
       [z["return_gbp"] for z in x],[z["return_gbp"] for z in y],
       B=15000,seed=71713)

def selftest():
    x=np.array([.01,.02,.03,.04,.05,.06,.07,.08,.09,.1])
    f=fundamental_vec(None,x)
    assert len(f)==12 and all(f[4:8]==1.0) and all(f[:4]==0)
    n=fundamental_vec({"fund_revenue_yoy":.2,"fund_net_margin":-.1,
           "fund_leverage":.6,"fund_cash_assets":.15},x)
    assert len(n)==12 and not n[4:8].any()
    test=[{"ticker":"A","ret":.1},{"ticker":"B","ret":None},
          {"ticker":"C","ret":-.03},{"ticker":"D","ret":.04},
          {"ticker":"E","ret":.01},{"ticker":"F","ret":.02}]
    result,_=select(test,{k["ticker"]:(1 if k["ticker"]=="B" else 0) for k in test},{})
    assert result["return"] is None and result["missing"]==["B"]
    print("V18_INTERNAL_VALIDATION_PASS",flush=True)

def main():
    selftest()
    if not RAW_FEATURE_FILE.exists():raise RuntimeError("Missing exact v17D as-filed artifact")
    raw=pd.read_parquet(RAW_FEATURE_FILE)
    d,qa=quality_gate(raw)
    lookup={(str(row.ticker),str(row.month)):{k:getattr(row,k) for k in FIELDS}
            for row in d.itertuples(index=False)}
    stocks,index,errors=load_market()
    panels=make_panel(stocks,index,lookup)
    logs,train_n=analyze(panels)
    OUT.mkdir(exist_ok=True,parents=True)
    # CHECKPOINT actual local return series before FX download so we never lose model fitting.
    (OUT/"v18_local_portfolios_checkpoint.json").write_text(json.dumps(logs,indent=2))
    out=summarize(logs)
    result={"version":"v18_filing_earnings_ablation",
       "research_only":True,"trades_executed":False,"alpha_verified":False,
       "fundamental_source":"Original SEC accession as-filed DuckDB mirror + 5 calendar-day publication lag",
       "fundamental_archive_artifact_run":43761156057,
       "test_period":"2025-08 to 2026-09 decision month; 14 subsequent GBP return months",
       "prior_test_history_seen":True,"SEC_coverage":qa,
       "universe_asof_2025_08":len(stocks),"missing_future_price_continuations":errors,
       "training_rows_min":min(train_n),"training_rows_max":max(train_n),
       "methods":out,"fund_vs_price":paired(out,"filing_fundamentals","price"),
       "fund_vs_placebo":paired(out,"filing_fundamentals","filing_placebo"),
       "limitations":[
         "Sample is a deterministic subset of today's retrospective historical index archive.",
         "2026 identifier directory is not a true historical security ID master.",
         "US only for as-filed fundamentals; user objective remains globally diversified.",
         "Sector economic comparability of reported financial concepts is not yet verified.",
         "No analyst consensus surprises, full valuation, sector competition or news.",
         "Non-independently inspected 2025-26 period cannot certify alpha.",
         "Fixed brokerage and FX costs are illustrative. No execution prices or slippage audit.",
         "No dividends/corporate actions audit outside source-adjusted OHLC; UK ISA eligible stock filtering absent.",
         "No shorting, leverage, cash optimisation, monthly-contribution cashflows or broker orders."
       ]}
    (OUT/"v18_sec_fundamental_ablation.json").write_text(json.dumps(result,indent=2))
    print("V18_RESULTS",json.dumps({
       "sample":len(stocks),"training":[min(train_n),max(train_n)],
       "CAGR":{k:v["summary"].get("CAGR_GBP_PCT") for k,v in out.items()},
       "max_drawdown":{k:v["summary"].get("MAX_DD_GBP_PCT") for k,v in out.items()},
       "fund_vs_price":result["fund_vs_price"],
       "fund_vs_placebo":result["fund_vs_placebo"]
    },separators=(",",":")),flush=True)

if __name__=="__main__":main()
