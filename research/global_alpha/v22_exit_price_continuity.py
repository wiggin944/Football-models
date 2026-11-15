#!/usr/bin/env python3
"""Global Alpha v22: audit 13 observed price-history gaps after S&P index departures.

Never assume that disappearance from index constituent archive means delisting.
Independent Yahoo adjusted OHLC is allowed to fill a missing *realised return*
ONLY when multiple original-vs-Yahoo daily returns agree at an overlapping anchor.
This is RESEARCH-ONLY and a post-inspection data repair, NOT a holdout or alpha.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import v14_frozen_holdout as price

OUT=Path("results")
OUT.mkdir(parents=True,exist_ok=True)
# Symbols selected because a previously inspected fixed-universe v21 run
# lacked next-month open prices for one or more strategies. This is data QA,
# not a new, outcome-selected trading rule.
TARGETS={
 "AAL":["2024-08","2024-09"],
 "BBWI":["2024-08","2024-09"],
 "BIO":["2024-08","2024-09"],
 "BWA":["2025-02","2025-03"],
 "CE":["2025-02","2025-03"],
 "ETSY":["2024-08","2024-09"],
 "FMC":["2025-02","2025-03"],
 "ILMN":["2024-05","2024-06"],
 "QRVO":["2024-11","2024-12"],
 "RHI":["2024-05","2024-06"],
 "TFX":["2025-02","2025-03"],
 "VFC":["2024-02","2024-03"],
 "WHR":["2024-02","2024-03"],
}

def archive(sym):
    url=f"https://raw.githubusercontent.com/{price.REPO}/{price.PIN}/data/sp500/constituents/{sym}/prices_daily.csv"
    path=price.CACHE/price.PIN/"sp500"/"constituents"/sym/"prices_daily.csv"
    content=price.get(url,path)
    from io import BytesIO
    df=pd.read_csv(BytesIO(content),low_memory=False)
    df["date"]=pd.to_datetime(df["date"],errors="coerce").dt.date
    df["close_final"]=pd.to_numeric(df["normalized_adj_close"],errors="coerce")
    close=pd.to_numeric(df["close"],errors="coerce")
    df["close_final"]=df["close_final"].where(df["close_final"]>0,close)
    return df[["date","close_final"]].dropna().drop_duplicates("date").set_index("date").sort_index()

def yahoo(sym):
    h=yf.Ticker(sym).history(start="2024-01-01",end="2025-09-05",
                             auto_adjust=True,actions=False)
    if h.empty:raise ValueError("No Yahoo adjusted prices in requested interval")
    h=h.reset_index()
    col="Date" if "Date" in h.columns else "Datetime"
    h["date"]=pd.to_datetime(h[col].astype(str).str.slice(0,10),errors="raise").dt.date
    h["close_final"]=pd.to_numeric(h["Close"],errors="coerce")
    h["open_final"]=pd.to_numeric(h["Open"],errors="coerce")
    h=h[(h.close_final>0)&(h.open_final>0)].dropna(subset=["date","close_final","open_final"])
    return h[["date","close_final","open_final"]].drop_duplicates("date").set_index("date").sort_index()

def verify_overlap(source,extra):
    shared=source.index.intersection(extra.index).sort_values()
    # Use last 25 overlapping daily dates near original constituent disappearance,
    # comparing daily percentage changes (invariant to future corporate-action
    # multiplicative adjustment on both historical series).
    shared=shared[-25:]
    if len(shared)<10:raise ValueError("Insufficient source-identification overlap")
    a=source.loc[shared,"close_final"].astype(float).to_numpy()
    b=extra.loc[shared,"close_final"].astype(float).to_numpy()
    if any(x<=0 for x in np.r_[a,b]):raise ValueError("Nonpositive adjusted close in anchor")
    deltas=np.abs(np.diff(np.log(a))-np.diff(np.log(b)))
    score={"overlap_days":len(shared),"overlap_first":str(shared[0]),
       "overlap_last":str(shared[-1]),
       "median_abs_daily_log_return_disagreement":float(np.median(deltas)),
       "max_abs_daily_log_return_disagreement":float(np.max(deltas)),
       "95pct_abs_daily_log_return_disagreement":float(np.quantile(deltas,.95))}
    # Threshold chosen BEFORE inspecting return outcomes. Any mismatch stays
    # unresolved to prevent ticker reuse/price-adjustment/corporate-action mistakes.
    if score["median_abs_daily_log_return_disagreement"]>.0025 or score["max_abs_daily_log_return_disagreement"]>.025:
        raise ValueError("Adjusted-price continuity failed "+str(score))
    return score

def open_month(h,period):
    month=str(period)
    a=h.reset_index()
    a["month"]=a["date"].astype(str).str[:7]
    m=a[a.month==month]
    return float(m.open_final.iloc[0]) if len(m) else None

def repair_ret(h,t):
    mon=pd.Period(t,freq="M")
    a=open_month(h,mon+1);b=open_month(h,mon+2)
    if a is None or b is None:return None
    r=b/a-1
    return float(r) if -1<r<5 and np.isfinite(r) else None

def main():
    audit={"version":"v22_index_exit_continuity_audit",
       "price_archive_commit":price.PIN,
       "purpose":"Resolve missing next-month OPEN returns, not generate price-only alpha",
       "selection":"13 securities observed missing in pre-inspected v21 model portfolios",
       "assumed":False,"symbols":{},"resolved_return_lookup":{},
       "warnings":[
          "External Yahoo adjustment history can change retrospectively, but overlapping daily returns must agree.",
          "A corporate merger or ticker reassignment requires manual handling; rejected symbols are NOT dropped.",
          "The 2024–25 research period was seen before this repair. No independent alpha inference.",
          "Trade month opening is proxied by first reported exchange day of each calendar month.",
          "No proof yet that symbols were investable under UK ISA restrictions.",
          "This repairs realised-return labels only, not financial valuation denominators."
       ]}
    for sym,times in TARGETS.items():
        rec={"missing_decision_months":times,"source":"archive vs Yahoo adjusted OHLC"}
        try:
            base=archive(sym)
            other=yahoo(sym)
            rec["archive_last"]=str(base.index.max())
            rec["yahoo_last"]=str(other.index.max())
            rec["anchor"]=verify_overlap(base,other)
            rec["repaired"]={}
            for t in times:
                r=repair_ret(other,t)
                rec["repaired"][t]=r
                if r is not None:
                    audit["resolved_return_lookup"][sym+"|"+t]=r
            rec["verified"]=all(x is not None for x in rec["repaired"].values())
        except Exception as e:
            rec["verified"]=False
            rec["failure"]=type(e).__name__+": "+str(e)[:350]
        audit["symbols"][sym]=rec
        print("V22_SYMBOL",sym,json.dumps(rec,separators=(",",":")),flush=True)
    resolved=[sym for sym,r in audit["symbols"].items() if r["verified"]]
    audit["total_symbols"]=len(TARGETS)
    audit["verified_symbols"]=len(resolved)
    audit["missing_decision_pairs"]=sum(len(z) for z in TARGETS.values())
    audit["repaired_decision_pairs"]=len(audit["resolved_return_lookup"])
    (OUT/"v22_exit_price_continuity_audit.json").write_text(json.dumps(audit,indent=2))
    print("V22_CONTINUITY_SUMMARY",json.dumps({
       "targets":audit["total_symbols"],"verified":audit["verified_symbols"],
       "total_missing_pairs":audit["missing_decision_pairs"],
       "repaired_pairs":audit["repaired_decision_pairs"]}),flush=True)

if __name__=="__main__":main()
