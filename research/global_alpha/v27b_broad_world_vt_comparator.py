#!/usr/bin/env python3
"""Global Alpha v27B: independent broad global ETF market comparator.

Compare frozen quality/value/momentum research return history against
dividend-adjusted market price of Vanguard Total World Stock ETF (VT).
Strategy fixed BEFORE reading VT returns. Since VT 2008 inception,
evaluate full-calendar-month overlap Jan 2009 through Dec 2025 in USD.
Do not infer an investable stock strategy or UK ISA eligibility.
"""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from v27_simple_global_factor_backtest import (
  metric,interval,rolling_realised_winshare,DATA,REGION_WEIGHTS,
  load_all,series_from_sources,sleeve_returns,rebalance_monthly,
  COST_ONE_WAY,SEED)

OUT=Path("results");OUT.mkdir(exist_ok=True)
SOURCE=OUT/"v27_monthly_factor_returns.csv"
SOURCE_REPORT=OUT/"v27_global_factor_research.json"
START="2009-01";END="2025-12"

def fetch_vt():
    bars=yf.Ticker("VT").history(start="2008-06-01",end="2026-11-01",
       interval="1d",auto_adjust=True,actions=False,raise_errors=True)
    if bars.empty or len(bars)<3000:
        raise RuntimeError("Missing fully adjusted ETF history")
    if bars.index.tz is None or bars.index.has_duplicates:
        raise RuntimeError("VT timestamps must carry an exchange timezone")
    if not bars.index.is_monotonic_increasing:
        raise RuntimeError("VT historical bars out of time order")
    date=[str(k)[:10] for k in bars.index]
    local_month=pd.PeriodIndex(pd.to_datetime(date).to_period("M"),freq="M")
    close=pd.Series(bars["Close"].to_numpy(float),index=local_month)
    monthly=close.groupby(level=0).last()
    if monthly.le(0).any() or monthly.isna().any():
        raise RuntimeError("Bad split/dividend adjusted VT prices")
    returns=monthly.pct_change(fill_method=None).dropna()
    return returns,{"source":"Yahoo adjusted daily VT prices","kind":"split/dividend adjusted",
       "first":str(returns.index.min()),"last":str(returns.index.max()),
       "daily_bars":len(bars),"market_ticker":"VT",
       "index":"FTSE Global All Cap Index; not MSCI ACWI IMI"}

def submetric(strategy,vt,first,last):
    a=strategy.loc[first:last]
    b=vt.loc[first:last]
    if len(a)!=len(b) or len(a)<50 or not a.index.equals(b.index):
        raise RuntimeError("Mismatch against VT on fixed months")
    return {"month_count":len(a),"factor":metric(a),
      "vt":metric(b),"annual_arithmetic_difference_pp":round(float((a-b).mean()*1200),3),
      "rolling_36m_win_fraction":rolling_realised_winshare(a,b),
      "difference_interval":interval(a,b)}

def main():
    if not SOURCE.exists() or not SOURCE_REPORT.exists():
        raise RuntimeError("Frozen historical multi-factor research artifact missing")
    original=json.loads(SOURCE_REPORT.read_text())
    if (original["region_weights"]!={"us":.55,"developed_ex_us":.30,"emerging":.15}
       or original["factor_weights"]!={"value":1/3,"quality":1/3,"momentum":1/3}):
        raise RuntimeError("Factor allocations changed after first performance reveal")
    raw=pd.read_csv(SOURCE)
    factor=pd.Series(raw.multifactor.to_numpy(float),
                     index=pd.PeriodIndex(raw.month,freq="M"))
    market=pd.Series(raw.market.to_numpy(float),
                     index=pd.PeriodIndex(raw.month,freq="M"))
    vt,source=fetch_vt()
    idx=pd.period_range(START,END,freq="M")
    if idx.difference(factor.index).size or idx.difference(vt.index).size:
        raise RuntimeError("A historical return month missing")
    factor=factor.loc[idx]
    vt=vt.loc[idx]
    market=market.loc[idx]
    periods={"since_VT_start":"2009-01:2025-12",
             "older_VT_era":"2009-01:2015-12",
             "modern_era":"2016-01:2025-12"}
    stats={name:submetric(factor,vt,*s.split(":")) for name,s in periods.items()}
    market_vs_vt=submetric(market,vt,START,END)
    stress={}
    if original["time_start"]!="1995-01":raise RuntimeError("Data vintage mismatch")
    frames,_=load_all()  # Reuse saved verified original ZIPs; NO new source substitution
    reg,months=series_from_sources(frames)
    sleeve,w=sleeve_returns(reg,months,"multifactor")
    for annual in (0.,.005,.01,.015,.02):
        trial=rebalance_monthly(sleeve,w,COST_ONE_WAY,annual)
        x=pd.Series(trial.net_return.to_numpy(float),
            index=pd.PeriodIndex(trial.index,freq="M")).loc[idx]
        stress[str(annual)]={"factor_cagr_pct":metric(x)["cagr_pct"],
             "VT_CAGR_pct":metric(vt)["cagr_pct"],
             "arithmetic_annual_advantage_pp":round(float((x-vt).mean()*1200),3)}
    out={"version":"v27B_world_etf_vt_comparison",
      "benchmark":"Vanguard Total World Stock ETF (VT), dividend-adjusted monthly closing USD price",
      "ETF_inception":"2008-06-24","ETF_index":"FTSE Global All Cap",
      "data_vintage":"Yahoo at script runtime; different market close and academic return boundaries",
      "risk_status":"Research only; never claim investable reconstructed factor portfolios",
      "factor_frozen_before_vt_inspection":True,"source":source,
      "methods":original["region_weights"],"periods":periods,
      "results":stats,"synthetic_market_vs_vt":market_vs_vt,
      "implementation_annual_haircut_stress":stress,
      "limits":[
        "A pre-specified academic sleeve blend is not a stock-by-stock investable portfolio.",
        "VT is a real US-domiciled market-cap-weighted ETF with embedded expense ratio, not necessarily UK ISA-eligible.",
        "Academic monthly regional return series and VT close-to-close returns are not identical investable executions.",
        "Factor costs are sensitivity assumptions; underlying constituent turnover, market impact and FX costs not measured.",
        "Historical portfolio reconstructions revise old records and cannot be treated as prospective live performance.",
        "No GBP returns, salary contributions, reliable long-term low-risk alpha or broker trades demonstrated.",
        "US, developed ex-US and emerging academic regions are broad, but not every world listing/country."],
      "alpha_confirmed":False,"live_orders":0}
    (OUT/"v27b_vt_benchmark_audit.json").write_text(json.dumps(out,indent=2))
    pd.DataFrame({"month":[str(x) for x in idx],
      "factor_net":factor.to_numpy(),"VT_adjusted":vt.to_numpy(),
      "fixed_region_market":market.to_numpy()}).to_csv(OUT/"v27b_VT_matched_monthly.csv",index=False)
    print("V27B_VT_COMPARISON",json.dumps({
      "results":{k:{"factor_cagr":v["factor"]["cagr_pct"],
                     "VT_cagr":v["vt"]["cagr_pct"],
                     "factor_maxDD":v["factor"]["max_drawdown_pct"],
                     "VT_maxDD":v["vt"]["max_drawdown_pct"],
                     "factor_rolling_win":v["rolling_36m_win_fraction"],
                     "arithmetic_advantage":v["annual_arithmetic_difference_pp"]}
                  for k,v in stats.items()},
      "stress":stress},separators=(",",":")),flush=True)
if __name__=="__main__":main()
