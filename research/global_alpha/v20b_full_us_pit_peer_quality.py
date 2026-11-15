#!/usr/bin/env python3
"""Global Alpha v20B: full US SEC as-filed TTM & same-industry peer coverage.

Historical S&P folder spans are used ONLY as coarse coverage filters (not as proof
of historical index membership, tradability, or independent security identity).
No portfolio return testing, security recommendation or order execution.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import v19_pit_ttm_peer_features as prev

SOURCE=Path("source/results/v20_original_sec_facts.parquet")
MAPPING=Path("source/results/v20_us_security_mapping.parquet")
DEST=Path("results")
START="2022-01"
END="2026-09"
ARCHIVE_CUTOFF="2025-08-22"
TEST_START="2025-08"
TEST_END=END
METRICS=["ttm_revenue_yoy","ttm_margin","liabilities_assets","cash_assets","ttm_cashflow_assets"]

def observed_price_span(ticker_record,month):
    # Month-level approximation to archive price availability; no day-level gaps audited.
    first=str(ticker_record.price_first)[:7]
    last=str(ticker_record.price_last)[:7]
    if month<=ARCHIVE_CUTOFF[:7]:
        return first<=month<=last
    return bool(ticker_record.active_at_2025_08_22_archive)

def eligible_raw_data():
    if not SOURCE.exists() or not MAPPING.exists():
        raise RuntimeError("Missing pinned original SEC archive; no source substitution.")
    frame=pd.read_parquet(SOURCE)
    ident=pd.read_parquet(MAPPING)
    if frame.empty or ident.empty:raise RuntimeError("Empty SEC full archive")
    if ident.ticker.duplicated().any():raise AssertionError("Duplicated security ticker mapping")
    if frame.adsh.isna().any() or frame.accepted.isna().any():
        raise AssertionError("Missing accession or accepted timestamps")
    cleaned=prev.clean_frame(frame)
    by_cik=defaultdict(list)
    for row in cleaned:by_cik[row["cik"]].append(row)
    return frame,ident,by_cik

def monthly_data(ident,by_cik):
    rows=[]
    months=pd.period_range(START,END,freq="M")
    for i,issuer in enumerate(ident.itertuples(index=False)):
        cid=getattr(issuer,"cik_norm")
        if pd.isna(cid):continue
        issuer_records=by_cik.get(str(cid),[])
        for month in months:
            period=str(month)
            # Explicit conservative weekday approximation; exchange calendar a later gate.
            date=pd.bdate_range(month.start_time,month.end_time)[-1].date()
            state=prev.company_asof(issuer_records,date)
            rows.append({"ticker":str(issuer.ticker),"cik":str(cid),
               "month":period,"decision_date":str(date),
               "archive_price_span":observed_price_span(issuer,period),
               "active_at_2025_08_22_archive":bool(issuer.active_at_2025_08_22_archive),
               **state})
        if (i+1)%75==0:print("V20B_ISSUERS_PROCESSED",i+1,flush=True)
    df=pd.DataFrame(rows)
    if df.duplicated(["ticker","month"]).any():raise AssertionError("Duplicate security month")
    return df

def cross_sectional_peers(df):
    """Exclude self from median; only contemporaneously observed price-span peers.
    Comparisons are SIC2 proxies, NOT confirmed direct industry competitors.
    """
    for metric in METRICS:
        df["sic2_peer_rel_"+metric]=np.nan
        df["sic2_peer_n_"+metric]=0
    eligible=df[(df.archive_price_span)&(df.sic2_pit.notna())]
    for (month,sic),group in eligible.groupby(["month","sic2_pit"],sort=False):
        for metric in METRICS:
            nonmissing=group[metric].dropna()
            count=len(nonmissing)
            if count<=1:continue
            values=nonmissing.to_numpy(float)
            inds=nonmissing.index.to_numpy()
            # Leave one out; this is intentionally exact, not a group average.
            for pos,idx in enumerate(inds):
                others=np.concatenate((values[:pos],values[pos+1:]))
                n=len(others)
                df.at[idx,"sic2_peer_n_"+metric]=n
                if n>=3:
                    df.at[idx,"sic2_peer_rel_"+metric]=float(values[pos]-np.median(others))
    return df

def audit(df,raw,ident):
    period=df[df.month.between(TEST_START,TEST_END)&df.active_at_2025_08_22_archive].copy()
    # Only compare current archived universe; old firms cannot be used as future peers.
    coverage={key:round(float(period[key].notna().mean()),4) for key in METRICS}
    peer_coverage={"sic2_peer_rel_"+key:round(float(period["sic2_peer_rel_"+key].notna().mean()),4)
                   for key in METRICS}
    n_groups=period.groupby("month").sic2_pit.nunique()
    n_eligible=df[df.archive_price_span].groupby("month").ticker.nunique()
    report={
       "version":"v20B_full_2025_cohort_original_SEC_ttm_peers",
       "original_source_rows":len(raw),
       "original_accessions":int(raw.adsh.nunique()),
       "historical_security_folders":int(len(ident)),
       "historical_tickers_mapped_to_cik":int(ident.cik_norm.notna().sum()),
       "company_month_rows":len(df),
       "asof_2025_08_22_archived_symbols":int(ident.active_at_2025_08_22_archive.sum()),
       "asof_2025_08_22_mapped_symbols":int((ident.active_at_2025_08_22_archive&ident.cik_norm.notna()).sum()),
       "months_2025_08_to_2026_09":len(period.month.unique()),
       "monthly_2025_26_eligible_rows":len(period),
       "contemporaneous_sic2_groups_median":float(n_groups.median()),
       "earliest_2022_archive_month_names":int(n_eligible.get("2022-01",0)),
       "latest_2025_08_archive_month_names":int(n_eligible.get("2025-08",0)),
       "ttm_coverage_2025_26":coverage,
       "sic2_peer_coverage_2025_26":peer_coverage,
       "prior_81_company_peer_coverage_v19":{"ttm_revenue_yoy":0.4735,
                    "ttm_margin":0.4735,"liabilities_assets":0.1852,
                    "cash_assets":0.642,"ttm_cashflow_assets":0.3377},
       "release_check":"SEC filed plus 5 calendar days, from original accession. Synthetic amendment tests in v19.",
       "limitations":[
           "The modern 2026 ticker-to-CIK directory is NOT an authentic historical identifier master.",
           "Source archive price-first/last bounds are coarse observed coverage, not full daily delisting or index-membership history.",
           "For 2025-09 onward, tickers observed on 2025-08-22 are carried as a research cohort WITHOUT verifying future trading.",
           "SIC2 peers are approximate category neighbours, not proven same-business competitors.",
           "Model does not yet have market cap or fully-adjusted share-count valuations.",
           "Data source licensed for personal/non-commercial research where applicable.",
           "US-only fundamentals; full globally diversified investment system remains unbuilt.",
           "No model backtest, independently validated alpha or trading/broker connection."
       ],
       "alpha_proven":False,"live_orders":0
    }
    for col in ("revenue_available","earnings_available"):
        seen=pd.to_datetime(df[col],errors="coerce")
        when=pd.to_datetime(df.decision_date)
        if bool(((seen>when)&seen.notna()).any()):
            raise AssertionError("FUTURE INFORMATION in "+col)
    if len(period.month.unique())!=14:raise AssertionError("Missing evaluation month in as-of cohort")
    return report

def main():
    prev.tests()
    raw,ident,by_cik=eligible_raw_data()
    print("V20B_SOURCE",json.dumps({"folders":len(ident),
        "mapped":int(ident.cik_norm.notna().sum()),"facts":len(raw),"cik":raw.cik.nunique()}),flush=True)
    df=monthly_data(ident,by_cik)
    df=cross_sectional_peers(df)
    report=audit(df,raw,ident)
    DEST.mkdir(exist_ok=True,parents=True)
    df.to_parquet(DEST/"v20b_full_us_asfiled_ttm_peers.parquet",index=False,compression="zstd")
    (DEST/"v20b_us_peer_coverage_audit.json").write_text(json.dumps(report,indent=2))
    print("V20B_COMPLETED",json.dumps({
        "historical_folder_count":report["historical_security_folders"],
        "mapped":report["historical_tickers_mapped_to_cik"],
        "active_at_2025_08_22_mapped":report["asof_2025_08_22_mapped_symbols"],
        "company_month_rows":report["company_month_rows"],
        "TTM":report["ttm_coverage_2025_26"],
        "peers":report["sic2_peer_coverage_2025_26"]},separators=(",",":")),flush=True)

if __name__=="__main__":main()
