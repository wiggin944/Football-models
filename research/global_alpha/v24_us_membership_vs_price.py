#!/usr/bin/env python3
"""v24: audit difference between archived price coverage and date-correct S&P 500 membership.
US-only DIAGNOSTIC. It does not define Global Alpha's worldwide equity universe.
"""
import csv,io,datetime,json
from pathlib import Path
import pandas as pd
import requests
import v21_expanded_sec_signal_pilot as prior
import v14_frozen_holdout as price

URL="https://raw.githubusercontent.com/thuningxu/sp500nq100/1cc1de2770874293597a418b41b079e5337b260c/sp500_components_history.csv"
OUT=Path("results");OUT.mkdir(exist_ok=True)
def clean(s):
    return s.strip().upper().replace(".","-").replace("/","-")
def snap():
    r=requests.get(URL,timeout=(12,90),headers={"User-Agent":"GlobalAlphaUSUniverseAudit/0.24"})
    r.raise_for_status()
    rows=[]
    raw=csv.DictReader(io.StringIO(r.content.decode("utf-8-sig")))
    if "date" not in raw.fieldnames or "tickers" not in raw.fieldnames:
        raise RuntimeError("Unexpected historical membership schema "+str(raw.fieldnames))
    for d in raw:
        t=d.get("date"); ss=d.get("tickers")
        if not(t and ss):continue
        symbols={clean(x) for x in ss.replace(";",",").split(",") if x.strip()}
        dt=pd.to_datetime(t).date()
        if not 450<=len(symbols)<=520:
            raise RuntimeError(f"Invalid snapshot {dt} constituents={len(symbols)}")
        rows.append((dt,symbols))
    rows.sort(key=lambda p:p[0])
    if not rows or rows[0][0]>datetime.date(2024,1,1) or rows[-1][0]<datetime.date(2025,6,1):
        raise RuntimeError("Membership source does not span predeclared audit")
    return rows
def point_in_time(rows,date):
    x=[v for d,v in rows if d<=date]
    if not x:raise RuntimeError("Missing prior snapshot")
    return x[-1]

def main():
    events=snap()
    print("MEMBERSHIP_SOURCE_VERIFIED",json.dumps({
        "effective_changes":len(events),"first":str(events[0][0]),
        "last":str(events[-1][0]),"source":URL}),flush=True)
    stocks,idx=prior.load_prices()
    comparisons=[]
    for month in pd.period_range("2024-01","2025-06",freq="M"):
        clock=pd.bdate_range(month.start_time,month.end_time)[-1].date()
        members=point_in_time(events,clock)
        available={clean(sym) for sym,series in stocks.items()
                   if price.feat(series,idx,str(month)) is not None}
        both=members&available
        archive_nonmembers=sorted(available-members)
        missing_member_prices=sorted(members-available)
        report={"decision_month":str(month),"cutoff":str(clock),
                "members_at_cutoff":len(members),
                "eligible_by_archived_price":len(available),
                "genuine_members_with_eligible_prices":len(both),
                "archive_price_eligible_but_not_members":len(archive_nonmembers),
                "index_members_missing_complete_price_context":len(missing_member_prices),
                "false_eligible_example":archive_nonmembers[:25],
                "unobserved_member_example":missing_member_prices[:25],
                "false_eligible_symbols":archive_nonmembers,
                "missing_member_symbols":missing_member_prices}
        comparisons.append(report)
        print("MONTH_MEMBERSHIP_AUDIT",json.dumps({k:report[k] for k in
            ("decision_month","members_at_cutoff","eligible_by_archived_price",
             "genuine_members_with_eligible_prices",
             "archive_price_eligible_but_not_members",
             "index_members_missing_complete_price_context")},separators=(",",":")),flush=True)
    n=len(comparisons)
    summary={
       "version":"v24_US_membership_only_not_global_scope",
       "scope":"Independent S&P membership audit only; worldwide all-stock research objective unchanged",
       "price_archive_commit":price.PIN,
       "membership_url":URL,
       "months":n,
       "mean_archived_price_eligible":round(sum(x["eligible_by_archived_price"] for x in comparisons)/n,1),
       "mean_true_members_with_price_features":round(sum(x["genuine_members_with_eligible_prices"] for x in comparisons)/n,1),
       "mean_archive_nonmembers_in_past_cohort":round(sum(x["archive_price_eligible_but_not_members"] for x in comparisons)/n,1),
       "mean_current_members_missing_price_context":round(sum(x["index_members_missing_complete_price_context"] for x in comparisons)/n,1),
       "month_by_month":comparisons,
       "portfolio_return_recalculation_not_performed":True,
       "limitation":"Membership reconstructed from public historical tables; separate historical issuer ID, split/merger, holiday and true security listing verification remains required.",
       "prior_v23_returns_are_not_membership_corrected":True,
       "live_orders":0}
    (OUT/"v24_us_membership_vs_price_coverage.json").write_text(json.dumps(summary,indent=2))
    print("V24_MEMBERSHIP_RESULTS",json.dumps({k:summary[k] for k in [
        "months","mean_archived_price_eligible","mean_true_members_with_price_features",
        "mean_archive_nonmembers_in_past_cohort","mean_current_members_missing_price_context"]}),flush=True)
if __name__=="__main__":
    main()
