#!/usr/bin/env python3
"""Global Alpha v20C: EU ESEF filing publication-time coverage, research-only.

Four official year-index downloads, no scraping or individual filing downloads.
A published timestamp is eligible for PIT only if present; processed != released.
"""
import json,collections,datetime,requests
from pathlib import Path

URL="https://filings.xbrl.org/indexes/filings-{}.json"
YEARS=(2022,2023,2024,2025)
OUT=Path("results");OUT.mkdir(parents=True,exist_ok=True)
HEADERS={"User-Agent":"GlobalAlphaAcademicResearch/0.20C (https://github.com/wiggin944/Football-models)",
  "Accept":"application/json"}
EUCOUNTRIES=("FR","DE","NL","BE","IT","ES","FI","IE","AT","LU","PT")

def year(year):
    r=requests.get(URL.format(year),headers=HEADERS,timeout=(15,120))
    r.raise_for_status()
    j=r.json()
    records=j.get("data",[])
    if not records:raise ValueError("Empty ESEF index "+str(year))
    out={"year":year,"count":len(records),"reported_count":j.get("meta",{}).get("count"),
       "last_modified":r.headers.get("Last-Modified"),
       "date_fields":sorted(records[0].get("attributes",{})),
       "countries":{},"published_nonnull":0,"rejected_future_published":0}
    stats=collections.defaultdict(lambda:{"filings":0,"published":0,"unique_entities":set(),"examples":[]})
    for row in records:
        attr=row.get("attributes",{})
        country=attr.get("country") or "UNKNOWN"
        pub=attr.get("published")
        period_end=attr.get("period_end")
        entity=row.get("relationships",{}).get("entity",{}).get("data",{})
        entity_id=(entity.get("id") if isinstance(entity,dict) else None)
        stats[country]["filings"]+=1
        if entity_id:stats[country]["unique_entities"].add(str(entity_id))
        if pub:
            out["published_nonnull"]+=1
            stats[country]["published"]+=1
            try:
                if period_end and str(pub)[:10]<str(period_end)[:10]:
                    out["rejected_future_published"]+=1
            except Exception:pass
        if len(stats[country]["examples"])<2:
            stats[country]["examples"].append({"filing_id":str(row.get("id",""))[:120],
                "period_end":period_end,"published":pub,"entity_id":entity_id})
    for c,s in stats.items():
        out["countries"][c]={"filings":s["filings"],"published":s["published"],
            "publication_coverage":round(s["published"]/max(1,s["filings"]),3),
            "entities":len(s["unique_entities"]),
            "examples":s["examples"]}
    print("EU_ESEF_INDEX_YEAR",json.dumps({"year":year,"rows":len(records),
       "published":out["published_nonnull"],"published_fraction":round(out["published_nonnull"]/len(records),3),
       "eurostoxx_countries":sum(out["countries"].get(c,{}).get("filings",0) for c in EUCOUNTRIES)}),flush=True)
    return out

def main():
    result={"version":"v20C_EU_ESEF_publication_audit",
       "source":"https://filings.xbrl.org/docs/api#index-files",
       "only_original_published_if_present":True,
       "years":{}}
    for y in YEARS:
        try:result["years"][str(y)]=year(y)
        except Exception as ex:
            result["years"][str(y)]={"failure":str(ex)[:240]}
            print("EU_INDEX_FETCH_FAILED",y,str(ex)[:220],flush=True)
    collected=sum(1 for x in result["years"].values() if "count" in x)
    eu_tot=sum(sum(yr.get("countries",{}).get(c,{}).get("filings",0) for c in EUCOUNTRIES)
       for yr in result["years"].values())
    eu_pub=sum(sum(yr.get("countries",{}).get(c,{}).get("published",0) for c in EUCOUNTRIES)
       for yr in result["years"].values())
    result["eurostoxx_region_aggregate"]={"filings":eu_tot,
       "published":eu_pub,"availability_fraction":round(eu_pub/max(1,eu_tot),3)}
    result["no_assumed_filings_for_missing_published"]=True
    result["limitations"]=[
       "Published dates not uniformly supplied by country; source semantics vary.",
       "2026 ESEF archive indexes are backfilled; snapshots today do not establish historic API availability.",
       "Publication-time alone not enough; financial facts and ESEF entity-to-security LEI must be mapped with PIT identities.",
       "Mostly annual ESEF reports, so quarterly price interpretation still needs separate country sources.",
       "No investment returns, currency conversions, or alpha tested."]
    (OUT/"v20c_eu_filings_publication_gate.json").write_text(json.dumps(result,indent=2))
    print("EU_ESEF_PIT_GATE",json.dumps({"downloaded_years":collected,
       "target_countries_records":eu_tot,"published":eu_pub,
       "coverage":result["eurostoxx_region_aggregate"]["availability_fraction"]}),flush=True)
    if collected<2 or eu_tot<100:raise RuntimeError("Insufficient historic European ESEF index records")

if __name__=="__main__":main()
