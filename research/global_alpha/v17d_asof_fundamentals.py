#!/usr/bin/env python3
"""Global Alpha v17D: original SEC filing vintages -> monthly PIT numeric features.

Read-only exploratory data pipeline. Market price history and company identities
are not yet survivorship-free. No investment strategy or live orders.
"""
import argparse,collections,datetime,io,json,math
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from threading import Thread
from pathlib import Path
import numpy as np,pandas as pd,requests,duckdb
import v14_frozen_holdout as v

HF_FIN="https://huggingface.co/datasets/erlenbusch/sec-edgar/resolve/main/sec_edgar.duckdb"
HF_MAP="zalizedata/us-public-company-financials-dataset"
DEST=Path("results")
TAG_GROUPS={
 "sales":["RevenueFromContractWithCustomerExcludingAssessedTax",
          "RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueNet",
          "Revenues","RevenueFromContractWithCustomer", "RevenuesNetOfInterestExpense",
          "SalesRevenueGoodsNet","SalesRevenueServicesNet"],
 "earnings":["NetIncomeLoss","ProfitLoss"],
 "assets":["Assets"],
 "liabilities":["Liabilities"],
 "cash":["CashAndCashEquivalentsAtCarryingValue","CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
 "operating_cashflow":["NetCashProvidedByUsedInOperatingActivities"],
 "shares":["EntityCommonStockSharesOutstanding","CommonStockSharesOutstanding"],
}
TAGS={tag:(group,i) for group,ts in TAG_GROUPS.items() for i,tag in enumerate(ts)}
def mapping():
    api=f"https://huggingface.co/api/datasets/{HF_MAP}/tree/main/data/companies"
    r=requests.get(api,timeout=50);r.raise_for_status()
    names=[x["path"] for x in r.json() if x.get("path","").endswith(".parquet")]
    if len(names)!=1:raise RuntimeError("Unexpected mapping shards: "+str(names))
    x=requests.get(f"https://huggingface.co/datasets/{HF_MAP}/resolve/main/{names[0]}",timeout=80)
    x.raise_for_status()
    d=pd.read_parquet(io.BytesIO(x.content))
    d["ticker"]=d.ticker.astype("string").str.upper().str.strip()
    d["cik_norm"]=d.cik.apply(lambda c:str(int(c)).zfill(10) if pd.notna(c) else None)
    return d
def filings(ciks):
    db=duckdb.connect(":memory:")
    db.execute("INSTALL httpfs");db.execute("LOAD httpfs")
    db.execute("SET http_timeout=90000")
    db.execute("SET memory_limit='5GB'")
    # Hugging Face resolver can rate-limit thousands of DuckDB range GETs.
    # Resolve ONCE to the provider-authorised signed CDN URL; DuckDB reads
    # byte ranges from that URL without repeatedly hitting the HF resolver.
    redirect=requests.head(HF_FIN,allow_redirects=True,timeout=45)
    redirect.raise_for_status()
    direct=redirect.url
    host=requests.utils.urlparse(direct).hostname or ""
    if not host.endswith(".hf.co") or host=="huggingface.co":
        raise RuntimeError("Expected an authorised Hugging Face CDN redirect, got "+host)
    size=int(redirect.headers["Content-Length"])
    print("SIGNED_CDN_SOURCE",host,"bytes",size,flush=True)
    # DuckDB 1.x probes for a sibling .wal recovery file even in READ_ONLY mode.
    # A signed CDN URL valid ONLY for the database cannot fetch that sibling,
    # producing HTTP 403 (see DuckDB PR #28498). A loopback read-only adapter
    # returns 404 for the non-existent WAL and forwards ONLY bounded byte-range
    # GETs to the authorised file URL. Never exposes the signed URL in output.
    class ReadOnlyRange(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def log_message(self,*args):pass
        def do_HEAD(self):
            if self.path!="/sec_edgar.duckdb":
                self.send_error(404);return
            self.send_response(200)
            self.send_header("Content-Length",str(size))
            self.send_header("Accept-Ranges","bytes")
            self.send_header("Content-Type","application/octet-stream")
            self.end_headers()
        def do_GET(self):
            if self.path!="/sec_edgar.duckdb":
                self.send_error(404);return
            value=self.headers.get("Range","")
            if not value.startswith("bytes="):
                self.send_error(416,"Range requests required");return
            try:
                a,b=value[6:].split("-",1)
                start,end=int(a),int(b)
                if start<0 or end<start or end-start>8*1024*1024:
                    self.send_error(416,"Invalid or oversized range");return
                with requests.get(direct,headers={"Range":value},timeout=(15,90),stream=True) as upstream:
                    if upstream.status_code!=206:
                        self.send_error(502,"Upstream denied bounded range");return
                    self.send_response(206)
                    for k in ("Content-Type","Content-Length","Content-Range","Accept-Ranges","ETag"):
                        if upstream.headers.get(k):self.send_header(k,upstream.headers[k])
                    self.end_headers()
                    for buf in upstream.iter_content(1024*128):
                        if buf:self.wfile.write(buf)
            except (ValueError,TypeError):
                self.send_error(416,"Invalid range")
            except (ConnectionError,BrokenPipeError):
                pass
            except Exception as error:
                print("RANGE_PROXY_IO_ERROR",type(error).__name__,str(error)[:120],flush=True)
    proxy=ThreadingHTTPServer(("127.0.0.1",0),ReadOnlyRange)
    thread=Thread(target=proxy.serve_forever,daemon=True);thread.start()
    print("LOOPBACK_READ_ONLY_RANGE_READY",proxy.server_port,flush=True)
    db.execute("SET threads=4")
    try:
        db.execute("ATTACH ? AS filings (READ_ONLY)",
                   [f"http://127.0.0.1:{proxy.server_port}/sec_edgar.duckdb"])
        params=list(ciks)+list(TAGS)
        sql=f"""SELECT s.cik,s.adsh,s.filed,s.accepted,s.form,s.sic,
             n.tag,n.ddate,n.qtrs,n.uom,n.value
           FROM filings.main.numbers n JOIN filings.main.submissions s ON s.adsh=n.adsh
           WHERE s.cik IN ({','.join('?' for _ in ciks)})
             AND s.form IN ('10-Q','10-K')
             AND s.filed BETWEEN DATE '2015-01-01' AND DATE '2026-09-30'
             AND n.tag IN ({','.join('?' for _ in TAGS)})
             AND n.uom IN ('USD','shares')
             AND n.segments IS NULL
             AND n.ddate BETWEEN DATE '2014-01-01' AND DATE '2026-09-30'
             AND n.qtrs IN (0,1,2,3,4)"""
        df=db.execute(sql,params).fetchdf()
        return df
    finally:
        db.close()
        proxy.shutdown()
        proxy.server_close()
def availability(filed):
    """Use a conservative five-day filing delay, no guessing press release time."""
    return filed+datetime.timedelta(days=5)
def clean_rows(data):
    d=data.copy()
    for k in ("filed","ddate"):
        d[k]=pd.to_datetime(d[k],errors="coerce").dt.date
    d["value"]=pd.to_numeric(d.value,errors="coerce")
    d=d.dropna(subset=["filed","ddate","value","cik","tag"])
    d["available"]=d.filed.apply(availability)
    d["group"]=d.tag.map(lambda k:TAGS.get(k,("unknown",99))[0])
    d["priority"]=d.tag.map(lambda k:TAGS.get(k,("unknown",99))[1])
    d["qtr"]=d.qtrs.apply(lambda x:int(x) if pd.notna(x) else -1)
    d=d[((d.qtr==0)&(d.group.isin(["assets","liabilities","cash","shares"])))
         | ((d.qtr==1)&(d.group.isin(["sales","earnings","operating_cashflow"])))
         | ((d.qtr==4)&(d.group.isin(["sales","earnings","operating_cashflow"])))]
    return d
def asof_one(issuer,group,asof,qtrs):
    """Latest statement economic period, then latest known original filing vintage.
    Prefer highest-priority tag ONLY among equivalent period+vintage records.
    """
    xs=[r for r in issuer if r["group"]==group and r["qtr"]==qtrs
        and r["available"]<=asof and r["ddate"]<=asof]
    if not xs:return None
    xs.sort(key=lambda r:(r["ddate"],r["available"],-r["priority"],r["adsh"]),reverse=True)
    return xs[0]
def quarter_yoy(issuer,record,asof):
    if record is None:return None
    # Use a prior quarter of the SAME concept group; date range allows 53-week FY.
    prior=[r for r in issuer if r["group"]==record["group"] and r["qtr"]==1
       and r["available"]<=asof and 330<=(record["ddate"]-r["ddate"]).days<=400]
    if not prior:return None
    prior.sort(key=lambda r:(-abs((record["ddate"]-r["ddate"]).days-365),
        r["available"],-r["priority"]),reverse=True)
    val=prior[0]["value"]
    return record["value"]/val-1 if val>0 else None
def month_features(rows,asof):
    top={k:asof_one(rows,k,asof,1 if k in ("sales","earnings","operating_cashflow") else 0)
        for k in ("sales","earnings","assets","liabilities","cash","operating_cashflow","shares")}
    out={}
    for k,r in top.items():
        out[k+"_available"]=int(r is not None)
        out[k+"_lag_days"]=(asof-r["available"]).days if r else None
    rev=top["sales"];inc=top["earnings"];assets=top["assets"];liabs=top["liabilities"]
    out["revenue_yoy"]=quarter_yoy(rows,rev,asof)
    # A different period's profit must not be divided by current-quarter revenue.
    out["net_margin"]=(inc["value"]/rev["value"] if rev and inc
      and abs((inc["ddate"]-rev["ddate"]).days)<12 and rev["value"]>0 else None)
    out["leverage_liabilities_assets"]=(liabs["value"]/assets["value"] if assets and liabs
      and abs((assets["ddate"]-liabs["ddate"]).days)<12 and assets["value"]>0 else None)
    out["cash_assets"]=(top["cash"]["value"]/assets["value"] if assets and top["cash"]
      and abs((assets["ddate"]-top["cash"]["ddate"]).days)<12 and assets["value"]>0 else None)
    out["net_income_negative"]=int(inc["value"]<0) if inc is not None else None
    out["latest_revenue_period"]=str(rev["ddate"]) if rev else None
    out["latest_revenue_release"]=str(rev["available"]) if rev else None
    out["latest_income_release"]=str(inc["available"]) if inc else None
    return out
def testcase():
    t=datetime.date
    rows=[{"group":"earnings","qtr":1,"ddate":t(2023,3,31),"filed":t(2023,5,10),
           "available":availability(t(2023,5,10)),"value":-2.0,
           "priority":0,"adsh":"original"},
          {"group":"earnings","qtr":1,"ddate":t(2023,3,31),"filed":t(2024,1,10),
           "available":availability(t(2024,1,10)),"value":-5.0,
           "priority":0,"adsh":"amended"}]
    assert asof_one(rows,"earnings",t(2023,5,12),1) is None
    assert asof_one(rows,"earnings",t(2023,5,20),1)["value"]==-2.0
    assert asof_one(rows,"earnings",t(2024,1,20),1)["value"]==-5.0
    assert month_features(rows,t(2023,6,30))["net_income_negative"]==1
    print("SYNTHETIC_PIT_TEST_PASS",flush=True)
def main():
    testcase()
    m=mapping()
    names,_=v.region_symbols("sp500")
    tmp=m[m.ticker.isin(names)].copy()
    conflicts=tmp.groupby("ticker").cik_norm.nunique()
    tmp=tmp[~tmp.ticker.isin(conflicts[conflicts>1].index)].drop_duplicates("ticker")
    ciks=sorted(set(tmp.cik_norm.dropna()))
    print("PIT_PULL",len(ciks),"CIKs from",len(names),"sampled historical security folders",flush=True)
    facts=filings(ciks)
    print("PIT_NUMERIC_ROWS",len(facts),"distinct_issuers",facts.cik.nunique(),flush=True)
    d=clean_rows(facts)
    print("PIT_CLEAN_ROWS",len(d),"concepts",d.group.value_counts().to_dict(),flush=True)
    groups={k:g.to_dict("records") for k,g in d.groupby("cik")}
    monthly=[]
    for _,row in tmp.iterrows():
        cik=row.cik_norm
        issuer=groups.get(cik,[])
        for month in pd.period_range("2022-01","2026-09",freq="M"):
            # For the exploratory monthly feature audit, use the final weekday
            # before calendar month-end; availability uses an extra 5-day buffer.
            date=(pd.bdate_range(month.start_time,month.end_time)).max().date()
            feat=month_features(issuer,date)
            monthly.append({"ticker":str(row.ticker),"cik":cik,"month":str(month),
                            "decision_date":str(date),**feat})
    frame=pd.DataFrame(monthly)
    DEST.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(DEST/"v17d_asof_fundamental_features.parquet",index=False)
    metrics=["revenue_yoy","net_margin","leverage_liabilities_assets","cash_assets"]
    period=frame[frame.month.between("2025-08","2026-09")]
    coverage={k:round(float(period[k].notna().mean()),3) for k in metrics}
    report={"version":"v17D","study":"SEC original accession as-of features",
     "selected_ticker_count":len(names),"mapped_ciks":len(ciks),
     "matched_numeric_ciks":d.cik.nunique(),"raw_numeric_rows":len(facts),
     "usable_statement_rows":len(d),"features":metrics,
     "period_2025_26_coverage":coverage,"source":"public-domain SEC mirror",
     "missing_2026_ticker_map":list(sorted(set(names)-set(tmp.ticker))),
     "limitations":["Ticker CIK mapping comes from a 2026 snapshot, not historical security master.",
       "Business-sector concepts are not comparable across banks, REITs and other industries.",
       "Simple as-of uses filed date +5 calendar days and weekday month-end; actual exchange holidays require exchange-specific calendars.",
       "All 2025-26 model results are already explored; this is data QC, not untouched alpha test.",
       "No data on non-US filings; global research remains open.",
       "No live trades and no investment recommendation."]}
    (DEST/"v17d_asof_feature_report.json").write_text(json.dumps(report,indent=2))
    print("V17D_COVERAGE",json.dumps(coverage),flush=True)
    print("V17D_PIT_STORE_COMPLETE",json.dumps({"companies":len(ciks),"months":frame.month.nunique(),"rows":len(frame)}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--synthetic-test",action="store_true");x=p.parse_args()
    if x.synthetic_test:testcase()
    else:main()
