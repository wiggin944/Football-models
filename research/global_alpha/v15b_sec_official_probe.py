"""Minimal read-only official SEC companyfacts access probe; no mass downloads."""
import json,requests
from pathlib import Path
headers={
 "User-Agent":"GlobalAlphaAcademicResearch/0.1 (https://github.com/wiggin944/Football-models)",
 "Accept":"application/json",
 "Accept-Encoding":"gzip, deflate"
}
out={}
for name,url in [
 ("ticker_map","https://www.sec.gov/files/company_tickers.json"),
 ("apple_facts","https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"),
]:
 try:
  res=requests.get(url,headers=headers,timeout=25)
  out[name]={"status":res.status_code,"nbytes":len(res.content),
             "content_type":res.headers.get("content-type","")}
  if res.ok:
   j=res.json()
   if name=="ticker_map":
    found=[{"ticker":v["ticker"],"cik":v["cik_str"]}
           for v in j.values() if v.get("ticker") in ("ECL","AIG","GEHC","AAPL")]
    out[name]["lookup"]=found
   else:out[name]["cik"]=j.get("cik")
  else:out[name]["excerpt"]=res.text[:100]
 except Exception as exc:
  out[name]={"error":type(exc).__name__+":"+str(exc)[:80]}
 print("SEC_PROBE",name,json.dumps(out[name]),flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/v15b_official_sec_source_probe.json").write_text(json.dumps(out,indent=2))
