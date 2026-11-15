"""Source-only check: does free Yahoo historical adjusted data work in GitHub Actions?"""
import json, sys
from pathlib import Path
import yfinance as yf
syms=["SPY","EWJ","FEZ","EWH","ASHR","AAPL","1332.T","0001.HK","ABI.BR","600000.SS","GBPUSD=X","GBPJPY=X","GBPHKD=X","GBPCNY=X","GBPEUR=X"]
results={}
for sym in syms:
    try:
        data=yf.Ticker(sym).history(start="2025-08-15",end="2026-11-15",auto_adjust=True,actions=False)
        if len(data):
            results[sym]={"rows":len(data),"first":str(data.index.min().date()),"last":str(data.index.max().date()),"last_close":round(float(data.Close.iloc[-1]),4)}
        else:results[sym]={"rows":0,"error":"empty"}
    except Exception as exc:
        results[sym]={"rows":0,"error":str(exc)[:180]}
    print("PROBE",sym,json.dumps(results[sym],separators=(",",":")),flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/v14_source_probe.json").write_text(json.dumps(results,indent=2))
if sum(1 for x in results.values() if x["rows"]>20)<10:
    raise SystemExit("Source probe inadequate: fewer than 10 liquid symbol series")
