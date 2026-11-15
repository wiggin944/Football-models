#!/usr/bin/env python3
"""Global Alpha v19: preserve SEC original filing rows for PIT TTM development.

This creates a pinned research-only raw filing extract and a company identity audit,
not a backtest and not a legal historical-security master. Non-US still missing.
"""
from pathlib import Path
import hashlib,json
import pandas as pd
import v14_frozen_holdout as price
import v17d_asof_fundamentals as sec

OUT=Path("results")
OUT.mkdir(parents=True,exist_ok=True)
sample,_=price.region_symbols("sp500")
directory=sec.mapping()
matches=directory[directory.ticker.isin(sample)].copy()
conflicts=matches.groupby("ticker").cik_norm.nunique()
bad=set(conflicts[conflicts>1].index)
matches=matches[~matches.ticker.isin(bad)].drop_duplicates("ticker")
ciks=sorted(set(matches.cik_norm.dropna()))
print("V19_RAW_PULL",len(ciks),"CIKs",len(sample),"original historical stock folders",flush=True)
raw=sec.filings(ciks)
if not len(raw) or len(raw)<20000:raise RuntimeError("Raw as-filed SEC extract unexpectedly small")
assert raw.cik.nunique()==len(ciks)
assert raw.adsh.notna().all() and raw.filed.notna().all()
assert raw.tag.notna().all() and raw.value.notna().all()
assert raw.qtrs.isin([0,1,2,3,4]).all()
assert raw.form.isin(["10-K","10-Q"]).all()
assert raw.accepted.notna().all()
files={"filing_extract":"v19_original_sec_facts.parquet",
       "identity_snapshot":"v19_2026_ticker_cik_map.parquet"}
raw.to_parquet(OUT/files["filing_extract"],index=False,compression="zstd")
cols=["ticker","cik_norm","company_name","sic","sic_description"]
cols=[c for c in cols if c in matches.columns]
matches[cols].to_parquet(OUT/files["identity_snapshot"],index=False,compression="zstd")
sizes={k:(OUT/v).stat().st_size for k,v in files.items()}
sha={k:hashlib.sha256((OUT/v).read_bytes()).hexdigest() for k,v in files.items()}
audit={
 "version":"v19_raw_with_YTD_quarters_0_1_2_3_4","record_count":len(raw),
 "companies":int(raw.cik.nunique()),"unique_accessions":int(raw.adsh.nunique()),
 "accepted_timestamps":int(raw.accepted.notna().sum()),
 "reported_groups":raw.qtrs.value_counts().to_dict(),
 "rows_by_tag":raw.tag.value_counts().head(20).to_dict(),
 "source_database":"https://huggingface.co/datasets/erlenbusch/sec-edgar",
 "source_code":"v17d SEC original-filing extract; via read-only signed-CDN and local range server",
 "source_pit":"filing accession + filed/accepted time; DO NOT collapse by economic period until as-of gate",
 "snapshot_map_source":"zalizedata/us-public-company-financials-dataset",
 "snapshot_map_not_historical":True,
 "files":files,"file_bytes":sizes,"sha256":sha,
 "missing_identifiers":sorted(set(sample)-set(matches.ticker)),
 "multi_cik_collisions":sorted(bad),
 "model_trade_status":"NO TRADING / NO ALPHA CLAIM"
}
(OUT/"v19_raw_extract_audit.json").write_text(json.dumps(audit,indent=2,default=str))
print("V19_SEC_EXTRACT_PASS",json.dumps({k:audit[k] for k in ("record_count","companies","unique_accessions","reported_groups","file_bytes")},default=str),flush=True)
