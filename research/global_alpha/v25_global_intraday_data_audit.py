#!/usr/bin/env python3
"""Global Alpha v25: real recent 5-minute OHLC coverage diagnostic, not live quotes.

Tests 11 geographically diverse symbols (illustrations only, NOT a universe
whitelist). Yahoo's 5-minute bars are not a licensed global realtime bid/ask
feed and cannot be used to simulate executable spread or fills. No orders.
"""
import concurrent.futures as cf
import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# Coverage probes, not the stock discovery universe. A real exchange registry
# must enumerate active + delisted all instruments for all countries.
SAMPLE=(
 ("US","AAPL","XNAS"), ("GB","HSBA.L","XLON"), ("DE","SAP.DE","XETR"),
 ("JP","7203.T","XTKS"), ("HK","0700.HK","XHKG"),
 ("IN","RELIANCE.NS","XNSE"), ("KR","005930.KS","XKRX"),
 ("AU","BHP.AX","XASX"), ("BR","PETR4.SA","BVMF"),
 ("CA","SHOP.TO","XTSE"), ("ZA","NPN.JO","XJSE")
)

def probe(entry):
    country,symbol,mic=entry
    state={"country":country,"symbol":symbol,"expected_mic":mic,
           "data_kind":"5min historical OHLC bars, unadjusted",
           "licensed_top_of_book":False,
           "broker_order_allowed":False}
    try:
        f=yf.Ticker(symbol).history(period="5d",interval="5m",
              auto_adjust=False,actions=False,prepost=False,raise_errors=True)
        if f.empty:raise ValueError("Empty intraday history")
        f=f.dropna(subset=["Open","High","Low","Close"])
        dt=pd.DatetimeIndex(f.index)
        if dt.tz is None:raise ValueError("No source timezone on bars")
        ts=dt.tz_convert("UTC")
        now=datetime.now(timezone.utc)
        if not ts.is_monotonic_increasing or ts.has_duplicates:
            raise ValueError("Nonmonotonic or duplicate market timestamps")
        if (ts.to_pydatetime() > now+timedelta(minutes=15)).any():
            raise ValueError("Future-dated exchange bars")
        prices=f[["Open","High","Low","Close"]].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or (prices<=0).any():
            raise ValueError("Nonfinite or nonpositive OHLC data")
        if (prices[:,1]+1e-7<np.max(prices[:,[0,3]],axis=1)).any() or (
           prices[:,2]-1e-7>np.min(prices[:,[0,3]],axis=1)).any():
            raise ValueError("OHLC contradictions")
        sessions=len(set(dt.date))
        state.update({"valid":True,"bar_count":len(f),"sessions":sessions,
              "exchange_tz":str(dt.tz),"first_utc":ts[0].isoformat(),
              "last_utc":ts[-1].isoformat(),
              "first_usable_signal_utc":(ts[0]+pd.Timedelta(minutes=5)).isoformat(),
              "last_close_local":round(float(f.Close.iloc[-1]),5),
              "last_quote_age_hours":round((now-ts[-1].to_pydatetime()).total_seconds()/3600,1),
              "source_has_actual_bid_ask":False,
              "source_has_trades_volume":bool("Volume" in f.columns),
              "caveat":"Do NOT equate historical OHLC to executable bid/ask or live venue state."})
        if len(f)<20 or sessions<2:raise ValueError("Too little recent intraday history")
    except Exception as ex:
        state.update({"valid":False,"error":type(ex).__name__+": "+str(ex)[:220]})
    return state

def main():
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        out=list(pool.map(probe,SAMPLE))
    success=sum(int(x.get("valid",False)) for x in out)
    report={"version":"v25_realtime_data_availability_probe",
      "research_only":True,"tested_asof_utc":datetime.now(timezone.utc).isoformat(),
      "markets_in_probe":len(SAMPLE),"usable_research_5min_bars":success,
      "observations":out,
      "not_a_global_universe":True,
      "not_live_executable_quotes":True,
      "next_step":"Licensed exchange-level streaming quote+trade data and historical delisted listing master with entitlement checks; no broker connected.",
      "trading_recommendation":"NO LIVE TRADING"}
    Path("results").mkdir(parents=True,exist_ok=True)
    Path("results/v25_intraday_5m_global_coverage.json").write_text(json.dumps(report,indent=2))
    for rec in out:
        print("V25_INTRADAY_REGION",json.dumps({k:rec.get(k) for k in ("country","symbol","valid","bar_count","sessions","exchange_tz","error")},separators=(",",":")),flush=True)
    print("V25_INTRADAY_PROBE_SUMMARY",json.dumps({"markets":len(out),"valid_recent_5m":success,
       "all_markets_valid":success==len(out),"broker_orders":0}),flush=True)
    if success<5:raise SystemExit("Too few regions returned usable 5-minute bars. FAIL CLOSED.")

if __name__=="__main__":main()
