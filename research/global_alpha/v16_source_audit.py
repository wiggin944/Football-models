#!/usr/bin/env python3
"""Global Alpha v16 public research-data availability, provenance and schema gate.

Read-only. Does not use any 2026 fields as historical forecasts.
"""
import json, io, requests
from pathlib import Path
import pandas as pd
ROOT="https://huggingface.co"
REPO="zalizedata/us-public-company-financials-dataset"
s=requests.Session()
s.headers.update({"User-Agent":"GlobalAlphaPersonalAcademicResearch/0.16 (https://github.com/wiggin944/Football-models)"})
out={"research_only":True,"source":REPO,"datasets":{}}
for kind in ("companies","financials"):
    listing=f"{ROOT}/api/datasets/{REPO}/tree/main/data/{kind}"
    resp=s.get(listing,timeout=50)
    if not resp.ok:
        out["datasets"][kind]={"error":"API listing status "+str(resp.status_code),"body":resp.text[:140]}
        continue
    filenames=[x["path"] for x in resp.json() if x.get("path","").endswith(".parquet")]
    if not filenames:
        out["datasets"][kind]={"error":"No Parquet file","listing":resp.json()[:5]}
        continue
    if len(filenames)!=1:raise ValueError(f"Expected exactly one {kind} shard; got {len(filenames)}")
    path=filenames[0]
    raw=s.get(f"{ROOT}/datasets/{REPO}/resolve/main/{path}",timeout=150)
    if not raw.ok:
        out["datasets"][kind]={"error":"Parquet status "+str(raw.status_code)}
        continue
    df=pd.read_parquet(io.BytesIO(raw.content))
    info={"path":path,"download_bytes":len(raw.content),"rows":len(df),
       "columns":[{"name":str(c),"dtype":str(df[c].dtype),"null_frac":round(float(df[c].isna().mean()),3)} for c in df.columns]}
    important=[c for c in df.columns if any(k in c.lower() for k in
      ("date","filed","accept","known","valid","as_of","cik","ticker","revenue",
       "income","asset","liab","cash","shares","sector","sic","source","restat"))]
    samples={}
    for c in important:
        col=df[c]
        vals=col.dropna()
        if len(vals)==0:continue
        if any(k in c.lower() for k in ("date","filed","accept","known","valid","as_of","period")):
            try:
                parsed=pd.to_datetime(vals,errors="coerce",utc=True)
                samples[c]={"non_null":len(vals),"earliest":str(parsed.min()),"latest":str(parsed.max())}
            except Exception as e:samples[c]={"error":str(e)[:70]}
        else:
            samples[c]={"non_null":len(vals),"examples":[str(x)[:90] for x in vals.head(3).to_list()]}
    info["interesting_columns"]=samples
    if kind=="companies":
        for col in ("ticker","cik"):
            if col in df:info[col+"_unique"]=int(df[col].nunique())
        for sym in ("AAPL","ABBV","ADBE","ECL","AIG"):
            rows=df[df["ticker"].eq(sym)] if "ticker" in df else df.iloc[:0]
            info.setdefault("example_identifiers",{})[sym]=rows[["ticker","cik"]].to_dict("records")[:2] if len(rows) else []
    else:
        for col in ("ticker","cik","company_id"):
            if col in df:info[col+"_unique"]=int(df[col].nunique())
        info["historical_asof_fields_present"]=any("file" in c.lower() or "accepted" in c.lower() or "available" in c.lower() for c in df.columns)
        info["sample_rows"]=df.head(2).astype(str).to_dict("records")
    out["datasets"][kind]=info
    print("PROBE_SUCCESS",kind,"rows",len(df),"columns",list(df.columns),flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/v16_data_source_audit.json").write_text(json.dumps(out,indent=2,default=str))
print("V16_SOURCE_GATE",json.dumps({"available":list(out["datasets"]),
 "financial_rows":out["datasets"].get("financials",{}).get("rows"),
 "point_in_time_date_available":out["datasets"].get("financials",{}).get("historical_asof_fields_present")}),flush=True)
