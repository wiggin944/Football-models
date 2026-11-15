#!/usr/bin/env python3
"""SEC filing-level availability and company-identity audit. No trading or alpha claims."""
import io,json,os
from pathlib import Path
import requests,duckdb,pandas as pd
import v14_frozen_holdout as v
HF_META="zalizedata/us-public-company-financials-dataset"
HF_FIN="https://huggingface.co/datasets/erlenbusch/sec-edgar/resolve/main/sec_edgar.duckdb"
def map_companies():
    q=requests.get(f"https://huggingface.co/api/datasets/{HF_META}/tree/main/data/companies",timeout=35)
    q.raise_for_status()
    files=[x["path"] for x in q.json() if x.get("path","").endswith(".parquet")]
    assert len(files)==1,files
    r=requests.get(f"https://huggingface.co/datasets/{HF_META}/resolve/main/{files[0]}",timeout=100)
    r.raise_for_status()
    df=pd.read_parquet(io.BytesIO(r.content))
    df["ticker"]=df["ticker"].astype("string").str.upper().str.strip()
    df["cik_norm"]=df["cik"].apply(lambda x:str(int(x)).zfill(10) if pd.notna(x) else None)
    return df
def main():
    df=map_companies()
    selected,_=v.region_symbols("sp500")
    found=df[df.ticker.isin(selected)].copy()
    conflict=found.groupby("ticker").cik_norm.nunique()
    bad=conflict[conflict>1].index.tolist()
    unique=found[~found.ticker.isin(bad)].drop_duplicates("ticker")
    ids=sorted(set(unique.cik_norm.dropna()))
    result={"version":"v17C","research_only":True,
      "ticker_map_is_2026_snapshot":True,"not_historical_ticker_master":True,
      "sampled_historical_folders":len(selected),"mapped_symbols":len(unique),
      "unmapped":sorted(set(selected)-set(unique.ticker)),"conflict_tickers":bad,
      "source":"https://huggingface.co/datasets/erlenbusch/sec-edgar"}
    print("CIK_COVERAGE",json.dumps({"sample":len(selected),"mapped":len(unique),
          "unmapped":len(result["unmapped"]),"conflicts":bad}),flush=True)
    cn=duckdb.connect(":memory:")
    cn.execute("INSTALL httpfs");cn.execute("LOAD httpfs")
    cn.execute("SET http_timeout=70000")
    cn.execute("SET memory_limit='4GB'")
    cn.execute("ATTACH ? AS sec (READ_ONLY)",[HF_FIN])
    marks=",".join("?" for _ in ids)
    sql=f"""SELECT cik,COUNT(DISTINCT adsh) filings,
      COUNT(DISTINCT CASE WHEN accepted IS NOT NULL THEN adsh END) timestamped,
      MIN(filed) earliest,MAX(filed) latest
      FROM sec.main.submissions
      WHERE cik IN ({marks}) AND form IN ('10-Q','10-K')
      AND filed<=DATE '2026-09-30' GROUP BY cik"""
    counts=cn.execute(sql,ids).fetchdf()
    result["matched_ciks"]=len(counts)
    result["unique_ciks"]=len(ids)
    result["filing_count"]=int(counts.filings.sum())
    result["acceptance_timestamps"]=int(counts.timestamped.sum())
    result["unmatched_ciks"]=sorted(set(ids)-set(counts.cik))
    result["coverage_rows"]=counts.head(12).astype(str).to_dict("records")
    print("SEC_FILING_COVERAGE",json.dumps({"mapped":len(ids),
       "with_reports":len(counts),"n_filings":result["filing_count"],
       "acceptance_timestamps":result["acceptance_timestamps"]}),flush=True)
    # Predeclared numeric audit CIKs, regardless of whether they entered sampled universe.
    check=df[df.ticker.isin(("AAPL","ABBV","BAC","JPM"))].drop_duplicates("ticker")
    audit_ciks=sorted(set(check.cik_norm.dropna()))
    tags=("Revenues","RevenueFromContractWithCustomerExcludingAssessedTax",
          "SalesRevenueNet","NetIncomeLoss","Assets","Liabilities")
    print("NUMERIC_JOIN_START",json.dumps({"ciks":audit_ciks,"tags":tags}),flush=True)
    sql=f"""SELECT s.cik,s.adsh,s.form,s.filed,s.accepted,
       n.tag,n.ddate,n.qtrs,n.uom,n.value,n.segments
       FROM sec.main.numbers n
       JOIN sec.main.submissions s ON n.adsh=s.adsh
       WHERE s.cik IN ({','.join('?' for _ in audit_ciks)})
       AND s.filed BETWEEN DATE '2024-01-01' AND DATE '2025-09-30'
       AND s.form IN ('10-Q','10-K')
       AND n.tag IN ({','.join('?' for _ in tags)})
       AND n.uom='USD'
       ORDER BY s.filed,n.tag,n.ddate
       LIMIT 100"""
    rows=cn.execute(sql,audit_ciks+list(tags)).fetchdf()
    result["numeric_row_count"]=len(rows)
    result["numeric_sample"]=rows.head(25).astype(str).to_dict("records")
    result["numeric_tags"]=sorted(rows.tag.unique().tolist()) if len(rows) else []
    print("SEC_NUMERIC_AUDIT",json.dumps({"rows":len(rows),
      "tags":result["numeric_tags"]}),flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/v17c_us_filing_coverage.json").write_text(json.dumps(result,indent=2))
    if len(counts)<max(8,len(ids)*0.6) or len(rows)<10:
       raise SystemExit("Source coverage or numeric join insufficient")
    print("V17C_PIT_COVERAGE_PASS",flush=True)
if __name__=="__main__":main()
