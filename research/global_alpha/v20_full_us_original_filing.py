#!/usr/bin/env python3
"""Global Alpha v20 / expanded historical US population and original-filing source.

Research data only: not full survivorship-free security master, no returns, no orders.
Reuses independently verified SEC filing-vintage extraction and archives all mappings.
"""
import hashlib,json,re
from pathlib import Path
import pandas as pd,requests
import v14_frozen_holdout as price
import v17d_asof_fundamentals as sec

PIN=price.PIN
BASE=Path("results");BASE.mkdir(exist_ok=True,parents=True)
def full_symbols():
    root=price.tree(PIN)
    data=next(x["sha"] for x in root if x["path"]=="data")
    sp=next(x["sha"] for x in price.tree(data) if x["path"]=="sp500")
    folder=next(x["sha"] for x in price.tree(sp) if x["path"]=="constituents")
    return sorted(x["path"] for x in price.tree(folder) if x["type"]=="tree")

def archive_spans():
    url=f"https://raw.githubusercontent.com/{price.REPO}/{PIN}/data/STOCK_COMPANY_MAP.md"
    r=requests.get(url,timeout=60);r.raise_for_status()
    content=r.text
    if "## S&P 500" not in content:raise RuntimeError("No expected S&P 500 section")
    lines=content.split("## S&P 500",1)[1].splitlines()
    out={}
    for line in lines:
        if not line.startswith("| `"):continue
        vals=[z.strip().strip("`") for z in line.split("|")[1:-1]]
        if len(vals)<6:continue
        sym,lookup,name,first,last,confidence=vals[:6]
        if not(re.fullmatch(r"\d{4}-\d{2}-\d{2}",first) and re.fullmatch(r"\d{4}-\d{2}-\d{2}",last)):continue
        if sym in out:
            old=out[sym]
            if first<old["price_first"]:old["price_first"]=first
            if last>old["price_last"]:old["price_last"]=last
            continue
        out[sym]={"ticker":sym,"provider_lookup":lookup,"archive_company_name":name,
            "price_first":first,"price_last":last,"metadata_confidence":confidence}
    return out

def main():
    symbols=full_symbols()
    spans=archive_spans()
    if not 650<=len(symbols)<=850:raise RuntimeError("Unexpected full population "+str(len(symbols)))
    if not set(symbols).issubset(spans):raise RuntimeError("Stock price-span metadata missing folders "+str(set(symbols)-set(spans)))
    directory=sec.mapping()
    directory=directory[directory.ticker.isin(symbols)].copy()
    conflicts=directory.groupby("ticker").cik_norm.nunique(dropna=True)
    collisions=sorted(conflicts[conflicts>1].index)
    directory=directory[~directory.ticker.isin(collisions)].drop_duplicates("ticker")
    if directory.ticker.duplicated().any():raise AssertionError("Duplicate stock identifiers")
    lookup=directory.set_index("ticker")["cik_norm"].to_dict()
    full=pd.DataFrame([{**spans[sym],"cik_norm":lookup.get(sym)}
        for sym in symbols])
    full["active_at_2025_08_22_archive"]=(full.price_first.le("2025-08-22")&
                                          full.price_last.ge("2025-08-22"))
    ci=sorted(set(full.cik_norm.dropna()))
    matched=full[full.cik_norm.notna()]
    active=full[full.active_at_2025_08_22_archive]
    print("V20_FULL_POPULATION",json.dumps({
        "folders":len(full),"mapped_tickers":len(matched),"distinct_cik":len(ci),
        "active_archive_2025_08":len(active),
        "active_with_cik":int(active.cik_norm.notna().sum()),
        "former_unmapped":int((~full.active_at_2025_08_22_archive & full.cik_norm.isna()).sum()),
        "multiple_cik_conflict":len(collisions)},separators=(",",":")),flush=True)
    if len(ci)<300:raise RuntimeError("Insufficient SEC CIK coverage for full universe")
    raw=sec.filings(ci)
    if len(raw)<150000 or raw.cik.nunique()<300:raise RuntimeError("Too few usable original SEC records")
    assert raw.adsh.notna().all() and raw.filed.notna().all() and raw.accepted.notna().all()
    assert raw.qtrs.isin([0,1,2,3,4]).all()
    raw.to_parquet(BASE/"v20_original_sec_facts.parquet",index=False,compression="zstd")
    full.to_parquet(BASE/"v20_us_security_mapping.parquet",index=False,compression="zstd")
    files=["v20_original_sec_facts.parquet","v20_us_security_mapping.parquet"]
    audit={
      "version":"v20_full_historical_us_original_filing",
      "price_archive_commit":PIN,
      "historical_security_folder_count":len(symbols),
      "mapped_tickers":len(matched),
      "distinct_mapped_sec_cik":len(ci),
      "asof_2025_08_archive_price_span":len(active),
      "asof_2025_08_cik_mapped":int(active.cik_norm.notna().sum()),
      "unmapped_tickers":sorted(set(symbols)-set(matched.ticker)),
      "multi_cik_tickers_rejected":collisions,
      "original_numeric_facts":len(raw),
      "ciks_with_original_facts":int(raw.cik.nunique()),
      "distinct_accession_numbers":int(raw.adsh.nunique()),
      "acceptance_timestamps":int(raw.accepted.notna().sum()),
      "qtrs_counts":{str(k):int(v) for k,v in raw.qtrs.value_counts().items()},
      "artefacts":{fn:{"sha256":hashlib.sha256((BASE/fn).read_bytes()).hexdigest(),
                        "size":(BASE/fn).stat().st_size} for fn in files},
      "warning":"SEC CIK map from 2026 contemporary snapshot. Historical prices may contain corporate actions and index-absence gaps. price_first/price_last are archive coverage span, not historical index membership or guaranteed tradability.",
      "research_only":True,"global_alpha_proven":False,"cash_orders_executed":0
    }
    (BASE/"v20_full_us_source_audit.json").write_text(json.dumps(audit,indent=2,default=str))
    print("V20_FULL_SOURCE_COMPLETE",json.dumps({
      "folders":audit["historical_security_folder_count"],
      "mapped_ciks":audit["distinct_mapped_sec_cik"],
      "asof_2025_08_archive":audit["asof_2025_08_archive_price_span"],
      "asof_2025_08_mapped":audit["asof_2025_08_cik_mapped"],
      "numeric_facts":audit["original_numeric_facts"],
      "accessions":audit["distinct_accession_numbers"]},separators=(",",":")),flush=True)
if __name__=="__main__":main()
