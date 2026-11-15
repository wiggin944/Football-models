#!/usr/bin/env python3
"""Read-only point-in-time SEC quarterly ZIP probe.

Makes two bounded HTTP GET requests. Streamed Range reads deliberately avoid
downloading large SEC archives if the endpoint does not honour byte ranges.
This is a source-connectivity check, NOT a financial backtest.
"""
import json, requests
from pathlib import Path
UA="GlobalAlpha-Personal-Research (contact: https://github.com/wiggin944/Football-models)"
ROOT="https://www.sec.gov/files/dera/data/financial-statement-data-sets/"
periods=("2009q3","2025q3")
results={}
for period in periods:
    url=ROOT+period+".zip"
    try:
        with requests.get(url,headers={"User-Agent":UA,"Accept":"application/zip",
                "Range":"bytes=0-2047"},timeout=(12,20),stream=True) as r:
            first=next(r.iter_content(2048),b"")
            results[period]={"status":r.status_code,"content_type":r.headers.get("Content-Type"),
                 "content_range":r.headers.get("Content-Range"),
                 "content_length":r.headers.get("Content-Length"),
                 "zip_magic":first.startswith(b"PK"),"first_bytes":len(first),
                 "response_prefix":repr(first[:90]) if not first.startswith(b"PK") else None}
    except Exception as e:
        results[period]={"error":str(e)[:200]}
    print("SEC_QUARTER_PROBE",period,json.dumps(results[period],separators=(",",":")),flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/v17_sec_quarter_access.json").write_text(json.dumps(results,indent=2))
print("ZIP_GATE",json.dumps({"accessible_count":sum(int(x.get("zip_magic",False)) for x in results.values()),"total":len(results)}),flush=True)
