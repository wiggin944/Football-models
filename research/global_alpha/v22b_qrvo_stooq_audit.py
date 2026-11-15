#!/usr/bin/env python3
"""QRVO post-index exit source repair: Stooq full daily OHLC, strict archive overlap.

If Stooq differs in split adjusted DAILY RETURNS from historical archive,
data must remain missing (do not impute).
"""
from io import StringIO
import json
from pathlib import Path
import pandas as pd,requests,numpy as np
import v22_exit_price_continuity as prior
from v22_exit_price_continuity import verify_overlap,archive,open_month,repair_ret

p=Path("results");p.mkdir(exist_ok=True)
url="https://stooq.com/q/d/l/?s=qrvo.us&i=d&d1=20240101&d2=20250905"
r=requests.get(url,timeout=35,headers={"User-Agent":"Mozilla/5.0 (compatible; GlobalAlphaResearch/0.22)"})
report={"source":"Stooq QRVO.US independent daily OHLC", "http_status":r.status_code}
try:
    r.raise_for_status()
    z=pd.read_csv(StringIO(r.text))
    if not set(["Date","Open","Close"]).issubset(z.columns):raise ValueError("Missing OHLC fields "+repr(list(z.columns)))
    z["date"]=pd.to_datetime(z["Date"],errors="raise").dt.date
    z["close_final"]=pd.to_numeric(z["Close"],errors="raise")
    z["open_final"]=pd.to_numeric(z["Open"],errors="raise")
    z=z[["date","close_final","open_final"]].sort_values("date").drop_duplicates("date").set_index("date")
    anchor=verify_overlap(archive("QRVO"),z)
    report["anchor"]=anchor
    report["dates"]={"first":str(z.index.min()),"last":str(z.index.max())}
    report["repaired"]={m:repair_ret(z,m) for m in ("2024-11","2024-12")}
    report["verified"]=all(x is not None for x in report["repaired"].values())
except Exception as ex:
    report["verified"]=False
    report["error"]=type(ex).__name__+": "+str(ex)[:600]
(p/"v22b_qrvo_source_fallback.json").write_text(json.dumps(report,indent=2))
print("V22B_QRVO_AUDIT",json.dumps(report,separators=(",",":")),flush=True)
