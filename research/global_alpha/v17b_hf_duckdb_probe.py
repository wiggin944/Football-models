#!/usr/bin/env python3
"""Bounded 5.28GB remote DuckDB mirror audit; no full download, no trading."""
import json,os,time,requests,duckdb
from pathlib import Path
URI="https://huggingface.co/datasets/erlenbusch/sec-edgar/resolve/main/sec_edgar.duckdb"
OUT={"uri":URI,"probe_date":"2026-11-15","research_only":True}
h={"User-Agent":"GlobalAlphaResearch/0.17"}
try:
    r=requests.head(URI,headers=h,allow_redirects=True,timeout=25)
    OUT["head"]={"status":r.status_code,"size":r.headers.get("content-length"),
        "accept_ranges":r.headers.get("accept-ranges"),"content_type":r.headers.get("content-type"),
        "hostname":requests.utils.urlparse(r.url).hostname}
    print("HEAD",json.dumps(OUT["head"]),flush=True)
except Exception as e:OUT["head"]={"error":str(e)[:180]}
try:
    with requests.get(URI,headers={**h,"Range":"bytes=0-1023"},stream=True,timeout=25) as res:
        data=next(res.iter_content(chunk_size=1024),b"")
        OUT["range"]={"status":res.status_code,"range":res.headers.get("content-range"),
            "bytes":len(data),"type":res.headers.get("content-type")}
    print("RANGE",json.dumps(OUT["range"]),flush=True)
except Exception as e:OUT["range"]={"error":str(e)[:180]}
try:
    cn=duckdb.connect(database=":memory:")
    cn.execute("INSTALL httpfs")
    cn.execute("LOAD httpfs")
    print("HTTPFS_LOADED",flush=True)
    cn.execute("SET http_timeout=50000")
    cn.execute("ATTACH ? AS sec (READ_ONLY)",[URI])
    print("REMOTE_ATTACH_OK",flush=True)
    tables=cn.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog='sec' ORDER BY table_name").fetchall()
    OUT["tables"]=[x[0] for x in tables]
    for table in ("submissions","numbers"):
        if table not in OUT["tables"]:raise ValueError(f"Missing {table}")
        columns=cn.execute(f"DESCRIBE sec.main.{table}").fetchall()
        OUT[table+"_schema"]=[{"column":str(row[0]),"type":str(row[1])} for row in columns]
        print("SCHEMA",table,json.dumps(OUT[table+"_schema"][:35]),flush=True)
    names={x["column"] for x in OUT["submissions_schema"]}
    if not ("filed" in names and ("adsh" in names or "accession" in names) and "cik" in names):
        raise ValueError("Point-in-time submission join keys missing")
    cols=[x["column"] for x in OUT["submissions_schema"]]
    query="SELECT "+", ".join('"'+c+'"' for c in cols if c in
        ("adsh","cik","filed","form","name","period","sic"))+" FROM sec.main.submissions LIMIT 3"
    OUT["sample"]=cn.execute(query).fetchdf().astype(str).to_dict("records")
    print("PIT_SAMPLE",json.dumps(OUT["sample"]),flush=True)
    OUT["status"]="SCHEMA_PIT_PASS"
except Exception as e:
    OUT["status"]="FAILED_REMOTE_QUERY"
    OUT["error"]=str(e)[:1200]
    print("MIRROR_QUERY_ERROR",str(e)[:900],flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/v17b_huggingface_asfiled_probe.json").write_text(json.dumps(OUT,indent=2))
print("V17B_GATE",OUT["status"],flush=True)
if OUT["status"]!="SCHEMA_PIT_PASS":
    raise SystemExit(4)
