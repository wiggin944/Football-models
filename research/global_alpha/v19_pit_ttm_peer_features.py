#!/usr/bin/env python3
"""Global Alpha v19: as-filed quarterly reconstruction, TTM and contemporaneous SIC peers.

DATA QUALITY ONLY. No return optimisation, trading orders, or alpha assertion.
Preserves SEC accession filing vintages and refuses to mix incompatible concept tags.
"""
import argparse,collections,datetime,json,math
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE=Path("source/results/v19_original_sec_facts.parquet")
IDS=Path("source/results/v19_2026_ticker_cik_map.parquet")
OUT=Path("results")
FILING_BUFFER_DAYS=5
CONCEPTS={
 "revenue":["RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet","Revenues","RevenueFromContractWithCustomer",
            "RevenuesNetOfInterestExpense","SalesRevenueGoodsNet","SalesRevenueServicesNet"],
 "earnings":["NetIncomeLoss","ProfitLoss"],
 "opcash":["NetCashProvidedByUsedInOperatingActivities"],
 "assets":["Assets"],
 "liabilities":["Liabilities"],
 "cash":["CashAndCashEquivalentsAtCarryingValue",
         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
TAG_KIND={tag:(kind,i) for kind,ts in CONCEPTS.items() for i,tag in enumerate(ts)}
FLOW_KINDS={"revenue","earnings","opcash"}
BALANCE_KINDS=set(CONCEPTS)-FLOW_KIN
GAP_MIN,GAP_MAX=52,142
MAX_STALENESS_DAYS=210
MIN_PEERS=3

def as_date(x):
    return pd.Timestamp(x).date()

def clean_frame(f):
    rows=[]
    for r in f.itertuples(index=False):
        if r.tag not in TAG_KIND:continue
        kind,priority=TAG_KIND[r.tag]
        qtrs=int(r.qtrs)
        if kind in FLOW_KIN and qtrs not in (1,4):continue
        if kind in BALANCE_KINDS and qtrs!=0:continue
        if r.uom!="USD" or not np.isfinite(float(r.value)):continue
        filed=as_date(r.filed)
        ddate=as_date(r.ddate)
        accepted=pd.Timestamp(r.accepted)
        rows.append({"cik":str(r.cik).zfill(10),"adsh":str(r.adsh),
            "filed":filed,"accepted":accepted,
            "available":filed+datetime.timedelta(days=FILING_BUFFER_DAYS),
            "end":ddate,"kind":kind,"tag":str(r.tag),"priority":priority,
            "qtrs":qtrs,"value":float(r.value),"sic":str(r.sic) if pd.notna(r.sic) else None})
    return rows

def latest(records):
    return max(records,key=lambda r:(r["available"],r["accepted"],-r["priority"],r["adsh"]))

def by_period(known,tag,qtr):
    periods=collections.defaultdict(list)
    for r in known:
        if r["tag"]==tag and r["qtrs"]==qtr:
            periods[r["end"]].append(r)
    return {end:latest(versions) for end,versions in periods.items()}

def quarter_gap_is_valid(older,newer):
    d=(newer-older).days
    return GAP_MIN<=d<=GAP_MAX

def get_quarters(known,tag):
    raw_q=by_period(known,tag,1)
    annuals=by_period(known,tag,4)
    chosen=dict(raw_q)
    derived={}
    for yearend,fy in annuals.items():
        if yearend in raw_q:continue
        # FY fourth quarter equals known FY total minus the THREE quarters
        # originally available by that FY report's availability date.
        earlier=[(end,r) for end,r in raw_q.items()
                 if 40<(yearend-end).days<355 and r["available"]<=fy["available"]]
        if len(earlier)<3:continue
        earlier.sort(key=lambda z:z[0],reverse=True)
        recent=earlier[:3]
        q3,q2,q1=recent[0],recent[1],recent[2]
        if not all(quarter_gap_is_valid(*pair) for pair in [
              (q1[0],q2[0]),(q2[0],q3[0]),(q3[0],yearend)]):
            continue
        value=fy["value"]-sum(r["value"] for _,r in recent)
        q4={**fy,"value":value,"qtrs":1,"synthetic_q4":True,
            "source_accessions":[fy["adsh"]]+[r["adsh"] for _,r in reversed(recent)]}
        chosen[yearend]=q4
        derived[yearend]=q4
    return chosen,derived

def ttms(known,kind,asof):
    candidates=[]
    for tag in CONCEPTS[kind]:
        quarter_map,derived=get_quarters(known,tag)
        dates=sorted(quarter_map,reverse=True)
        if len(dates)<4:continue
        # Only use latest four quarters, with a valid sequence of economic periods.
        four=dates[:4]
        if any(not quarter_gap_is_valid(four[j+1],four[j]) for j in range(3)):continue
        if (asof-four[0]).days>MAX_STALENESS_DAYS:continue
        records=[quarter_map[d] for d in four]
        if any(r["available"]>asof for r in records):raise AssertionError("Future release")
        current=sum(r["value"] for r in records)
        # Compare against same fiscal quarter TTM 4 earlier quarters:
        # eight consecutive quarterly points required for reliable year-over-year.
        yoy=None
        if len(dates)>=8:
            eight=dates[:8]
            if all(quarter_gap_is_valid(eight[j+1],eight[j]) for j in range(7)):
                prev=sum(quarter_map[d]["value"] for d in dates[4:8])
                if prev>0 and kind=="revenue":yoy=current/prev-1
        candidates.append({"kind":kind,"tag":tag,"priority":TAG_KIND[tag][1],
            "end":four[0],"value":current,"yoy":yoy,
            "available":max(r["available"] for r in records),
            "quarter_ends":[str(x) for x in four],
            "derived_q4_count":sum(r.get("synthetic_q4",False) for r in records)})
    if not candidates:return None
    # More recent quarterly data first, then preferred concept tag.
    return max(candidates,key=lambda r:(r["end"],-r["priority"]))

def latest_balance(known,kind,asof):
    rs=[r for r in known if r["kind"]==kind and r["qtrs"]==0]
    if not rs:return None
    periods=collections.defaultdict(list)
    for r in rs:periods[r["end"]].append(r)
    top=max((latest(sub) for sub in periods.values()),
            key=lambda r:(r["end"],-r["priority"]))
    if (asof-top["end"]).days>400:return None
    return top

def company_asof(records,asof):
    known=[r for r in records if r["available"]<=asof and r["end"]<=asof]
    known.sort(key=lambda x:(x["filed"],x["accepted"]))
    sector=next((r["sic"] for r in reversed(known)
                 if r["sic"] and r["sic"].isdigit() and len(r["sic"])>=3),None)
    r=ttms(known,"revenue",asof)
    e=ttms(known,"earnings",asof)
    cf=ttms(known,"opcash",asof)
    a=latest_balance(known,"assets",asof)
    l=latest_balance(known,"liabilities",asof)
    cash=latest_balance(known,"cash",asof)
    def aligned(x,y,days=21):
        return bool(x and y and abs((x["end"]-y["end"]).days)<=days)
    margin=e["value"]/r["value"] if aligned(e,r) and r["value"]>0 else None
    leverage=l["value"]/a["value"] if aligned(l,a) and a["value"]>0 else None
    cash_ratio=cash["value"]/a["value"] if aligned(cash,a) and a["value"]>0 else None
    cashflow_asset=cf["value"]/a["value"] if aligned(cf,a,days=145) and a["value"]>0 else None
    # Do not claim generic leverage is economically comparable for banks/insurers.
    is_financial=bool(sector and 60<=int(sector[:2])<=67)
    if is_financial:
        leverage=None
        cashflow_asset=None
    out={
      "sic_pit":sector,"sic2_pit":sector[:2] if sector else None,
      "finance_sector":int(is_financial),
      "ttm_revenue":r["value"] if r and r["value"]>0 else None,
      "ttm_revenue_yoy":r["yoy"] if r else None,
      "ttm_earnings":e["value"] if e else None,
      "ttm_opcash":cf["value"] if cf else None,
      "ttm_margin":margin,
      "liabilities_assets":leverage,
      "cash_assets":cash_ratio,
      "ttm_cashflow_assets":cashflow_asset,
      "revenue_derived_q4":r["derived_q4_count"] if r else 0,
      "earnings_derived_q4":e["derived_q4_count"] if e else 0,
      "opcash_derived_q4":cf["derived_q4_count"] if cf else 0,
      "revenue_end":str(r["end"]) if r else None,
      "revenue_available":str(r["available"]) if r else None,
      "earnings_end":str(e["end"]) if e else None,
      "earnings_available":str(e["available"]) if e else None,
    }
    if r and r["available"]>asof:raise AssertionError("Revenue leaked after decision")
    if e and e["available"]>asof:raise AssertionError("Earnings leaked after decision")
    return out

def build_sector_peers(df):
    # Within each month, SIC2 groups, leave-one-company-out.
    # At least three OTHER peers with available *comparable* metrics required.
    audit={}
    for metric in ["ttm_revenue_yoy","ttm_margin","liabilities_assets",
                   "cash_assets","ttm_cashflow_assets"]:
        vals=[];counts=[]
        for period,g in df.groupby("month",sort=False):
            for _,firm in g.iterrows():
                if pd.isna(firm["sic2_pit"]) or pd.isna(firm[metric]):
                    vals.append((firm.name,np.nan,0));continue
                peers=g[(g["sic2_pit"]==firm["sic2_pit"])&
                        (g["ticker"]!=firm["ticker"])&(g[metric].notna())]
                n=len(peers)
                diff=float(firm[metric]-peers[metric].median()) if n>=MIN_PEERS else np.nan
                vals.append((firm.name,diff,n))
        peer=pd.DataFrame(vals,columns=["index","diff","n"]).set_index("index")
        df["sector_peer_diff_"+metric]=peer["diff"].reindex(df.index)
        df["sector_peer_count_"+metric]=peer["n"].reindex(df.index).fillna(0).astype(int)
        audit[metric]={"n_with_three_peers":int((df["sector_peer_count_"+metric]>=MIN_PEERS).sum()),
                       "median_peers":float(df["sector_peer_count_"+metric].median())}
    return df,audit

def tests():
    D=datetime.date
    def mk(tag,ddate,val,filed,adsh,qtrs=1,sic="3571"):
        kind,rank=TAG_KIND[tag]
        return {"cik":"0000000001","adsh":adsh,"filed":filed,
          "accepted":pd.Timestamp(filed),"available":filed+datetime.timedelta(days=5),
          "end":ddate,"kind":kind,"tag":tag,"priority":rank,
          "qtrs":qtrs,"value":val,"sic":sic}
    tag=CONCEPTS["revenue"][0]
    r=[
        mk(tag,D(2023,3,31),20,D(2023,5,1),"Q1"),
        mk(tag,D(2023,6,30),25,D(2023,8,1),"Q2"),
        mk(tag,D(2023,9,30),30,D(2023,11,1),"Q3"),
        mk(tag,D(2023,12,31),100,D(2024,2,15),"FY",qtrs=4)
    ]
    assert ttms([x for x in r if x["available"]<=D(2024,2,18)],"revenue",D(2024,2,18)) is None
    t=ttms([x for x in r if x["available"]<=D(2024,2,25)],"revenue",D(2024,2,25))
    assert t is not None and t["derived_q4_count"]==1 and t["value"]==100
    q4=get_quarters(r,tag)[0][D(2023,12,31)]
    assert abs(q4["value"]-25)<1e-10
    # Future restatement of a past Q2 MUST NOT enter an earlier as-of decision.
    newer=mk(tag,D(2023,6,30),99,D(2025,4,1),"Q2_REVISED")
    earlier=[x for x in r+[newer] if x["available"]<=D(2024,2,25)]
    assert ttms(earlier,"revenue",D(2024,2,25))["value"]==100
    # Negative profits remain negative, not silently discarded.
    loss=mk(CONCEPTS["earnings"][0],D(2023,3,31),-40,D(2023,5,1),"NEG")
    assert get_quarters([loss],loss["tag"])[0][D(2023,3,31)]["value"]==-40
    # SIC from future filing cannot be known before its publication.
    other=mk(tag,D(2024,3,31),3,D(2024,5,1),"NEXT",sic="6021")
    assert company_asof(r+[other],D(2024,3,1))["sic2_pit"]=="35"
    print("V19_QUARTER_TTM_PIT_TESTS_PASS",flush=True)

def main():
    tests()
    if not SOURCE.exists() or not IDS.exists():raise RuntimeError("Pinned v19 raw source artifacts absent")
    raw=pd.read_parquet(SOURCE)
    identities=pd.read_parquet(IDS)
    cleaned=clean_frame(raw)
    by_cik=collections.defaultdict(list)
    for r in cleaned:by_cik[r["cik"]].append(r)
    results=[]
    for i,row in identities.iterrows():
        ticker=str(row.ticker)
        cik=str(row.cik_norm)
        issuer=by_cik[cik]
        for month in pd.period_range("2022-01","2026-09",freq="M"):
            asof=pd.bdate_range(month.start_time,month.end_time)[-1].date()
            result=company_asof(issuer,asof)
            results.append({"ticker":ticker,"cik":cik,"month":str(month),
                "decision_date":str(asof),**result})
        if (i+1)%20==0:print("PIT_COMPANIES_PROCESSED",i+1,flush=True)
    df=pd.DataFrame(results)
    if len(df)!=4617 or df.duplicated(["ticker","month"]).any():
        raise AssertionError("Unexpected panel size or duplicate month-issuer")
    df,peer_audit=build_sector_peers(df)
    # Gate the as-of dates before saving.
    for k in ("revenue_available","earnings_available"):
        d=pd.to_datetime(df[k],errors="coerce")
        asof=pd.to_datetime(df["decision_date"])
        if bool(((d>asof)&d.notna()).any()):raise AssertionError("Future release leaked in "+k)
    test=df[df.month.between("2025-08","2026-09")]
    cols=["ttm_revenue","ttm_revenue_yoy","ttm_earnings","ttm_opcash",
          "ttm_margin","ttm_cashflow_assets","liabilities_assets","cash_assets"]
    coverage={k:round(float(test[k].notna().mean()),4) for k in cols}
    peer_coverage={k:round(float(test["sector_peer_diff_"+k].notna().mean()),4)
           for k in ["ttm_revenue_yoy","ttm_margin","liabilities_assets","cash_assets","ttm_cashflow_assets"]}
    report={
      "version":"v19_pit_annual_to_quarter_ttm",
      "row_count":len(df),"companies":int(df.ticker.nunique()),
      "unique_pit_sic2":int(df.sic2_pit.nunique()),
      "fundamental_coverage_2025_26":coverage,
      "peer_relative_coverage_2025_26":peer_coverage,
      "sector_peer_quality":peer_audit,
      "records_with_synthetic_Q4_revenue":int((test.revenue_derived_q4>0).sum()),
      "records_with_synthetic_Q4_earnings":int((test.earnings_derived_q4>0).sum()),
      "financial_sector_rows":int(test.finance_sector.sum()),
      "leakage_tests":"PASS synthetic future amendments, FY Q4 sum, signed losses, future SIC",
      "alpha_proven":False,"portfolio_returns_not_tested":True,
      "limitations":[
        "Historical ticker/CIK mapping from 2026 snapshot, not a full historically changing security master",
        "SIC 2-digit group with only the sampled universe; peer availability is sparse and not full economic competitor mapping",
        "Q4=FY annual minus precisely matching same-tag three preceding quarters known by FY filing; missing otherwise",
        "Quarterly TTM requires four consecutive ~quarterly period ends of one XBRL tag",
        "Filing availability buffered by five days; exchange trading calendars not fully modelled",
        "Restated annual and quarterly facts can be inconsistent despite as-of checks; no claim of exhaustive corrections",
        "Financial-sector leverage and OCF ratio excluded pending separate accounting semantics",
        "Not a globally representative valuation or earnings dataset; non-US filings remain unintegrated",
        "No full corporate-action-adjusted investable universe, independent holdout, salary allocator or live orders"
      ]}
    OUT.mkdir(parents=True,exist_ok=True)
    df.to_parquet(OUT/"v19_pit_ttm_sector_peer_features.parquet",index=False,compression="zstd")
    (OUT/"v19_ttm_quality_report.json").write_text(json.dumps(report,indent=2))
    print("V19_TTM_COVERAGE",json.dumps(coverage),flush=True)
    print("V19_PEER_COVERAGE",json.dumps(peer_coverage),flush=True)
    print("V19_TTM_FEATURE_STORE_PASS",json.dumps({"rows":len(df),"companies":df.ticker.nunique(),"industries":df.sic2_pit.nunique()}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--test-only",action="store_true");a=ap.parse_args()
    if a.test_only:tests()
    else:main()
