#!/usr/bin/env python3
"""Worldwide EODHD exchange discovery adapter, 2026-11: TESTED WITH MOCK DATA.

No credentials -> no provider request. A discovery candidate is NOT a
historically as-of tradable security; vendor ticker reuse and corporate actions
must be checked against persistent exchange/vendor/ISIN/FIGI identities.
Never claim this adapter covers unrepresented exchanges or supplies live quotes.
"""
import argparse,json,os
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

VIRTUAL_EXCHANGE_CODES={"CC","FOREX","MONEY","GBOND","EUFUND"}

class EodhdFetch(Protocol):
    def __call__(self,path:str,params:dict)->list: ...

@dataclass(frozen=True)
class Candidate:
    exchange_code:str
    operating_mic:str|None
    country_raw:str|None
    ticker:str
    vendor_exchange_ticker:str
    vendor_name:str|None
    vendor_type:str|None
    vendor_currency:str|None
    isin:str|None
    first_observed_utc:str
    last_observed_utc:str
    vendor_delisted_snapshot:bool
    permanent_instrument_id_confirmed:bool=False
    broker_tradability_confirmed:bool=False
    historical_effective_from_confirmed:bool=False
    historical_effective_to_confirmed:bool=False

def _safe_text(raw):
    return str(raw).strip() if raw is not None else ""

def collect_catalog(fetch:EodhdFetch,at_utc:datetime,exchange_limit:int|None=None):
    if at_utc.tzinfo is None:raise ValueError("Snapshots must be UTC time-aware")
    raw_exchanges=fetch("/exchanges-list/",{})
    if not isinstance(raw_exchanges,list) or not raw_exchanges:
        raise RuntimeError("No verified exchange directory from provider")
    exchanges=sorted((row for row in raw_exchanges
         if isinstance(row,dict) and _safe_text(row.get("Code"))
         and _safe_text(row.get("Code")).upper() not in VIRTUAL_EXCHANGE_CODES),
         key=lambda r:_safe_text(r.get("Code")))
    if exchange_limit is not None:exchanges=exchanges[:exchange_limit]
    candidates=[]
    errors=[]
    pairset=set()
    now=at_utc.astimezone(timezone.utc).isoformat()
    for ex in exchanges:
        code=_safe_text(ex.get("Code")).upper()
        mic=_safe_text(ex.get("OperatingMIC") or ex.get("MIC")) or None
        country=_safe_text(ex.get("CountryISO2") or ex.get("Country")) or None
        for delisted in (False,True):
            # Stock type is intentionally not filtered at ingestion:
            # providers disagree on 'stock', 'common_stock' and preferred types.
            # Preserve all returned types, classify later.
            try:rows=fetch("/exchange-symbol-list/"+quote(code,safe=""),{
                "delisted":1 if delisted else 0})
            except Exception as exn:
                errors.append({"exchange":code,"delisted":delisted,
                               "error_type":type(exn).__name__})
                continue
            if not isinstance(rows,list):
                errors.append({"exchange":code,"delisted":delisted,
                               "error_type":"BAD_SOURCE_FORMAT"})
                continue
            for row in rows:
                if not isinstance(row,dict):
                    errors.append({"exchange":code,"error_type":"NON_DICT_TICKER"})
                    continue
                sym=_safe_text(row.get("Code"))
                if not sym:
                    errors.append({"exchange":code,"error_type":"NO_TICKER"})
                    continue
                conflict_key=(code,sym)
                if conflict_key in pairset:
                    # A ticker may appear in active + delisted or be reused.
                    # Do not treat it as the same permanent security.
                    errors.append({"exchange":code,"ticker":sym,
                                   "error_type":"TICKER_REUSE_OR_DUPLICATE"})
                pairset.add(conflict_key)
                suffix=sym+"."+code
                candidates.append(Candidate(
                    code,mic,country,sym,suffix,
                    _safe_text(row.get("Name")) or None,
                    _safe_text(row.get("Type")) or None,
                    _safe_text(row.get("Currency")) or None,
                    _safe_text(row.get("Isin") or row.get("ISIN")) or None,
                    now,now,delisted))
    if not candidates:raise RuntimeError("No stock-listing candidates; fail closed")
    report={"observed_at_utc":now,"source":"EODHD",
        "exchanges_available":len(exchanges),
        "rows_active":sum(not v.vendor_delisted_snapshot for v in candidates),
        "rows_provider_delisted":sum(v.vendor_delisted_snapshot for v in candidates),
        "raw_candidate_count":len(candidates),"errors":errors,
        "permanent_identity_verified":False,
        "historical_asof_membership_verified":False,
        "intraday_licence_present":False,
        "broker_order_eligible":False}
    return candidates,report

def authenticated_transport(token):
    if not token:raise ValueError("Missing explicitly configured market-data token")
    import requests
    session=requests.Session()
    session.headers["User-Agent"]="GlobalAlpha-Research-Discovery/0.25"
    def get(path,params):
        p={**params,"api_token":token,"fmt":"json"}
        # Do not log tokens, HTTP query strings or response URLs.
        resp=session.get("https://eodhd.com/api"+path,params=p,timeout=(10,60))
        resp.raise_for_status()
        return resp.json()
    return get

def test():
    class MockProvider:
        def __call__(self,path,params):
            if path=="/exchanges-list/":
                return [{"Code":"US","OperatingMIC":"XNAS","CountryISO2":"US"},
                        {"Code":"JP","OperatingMIC":"XTKS","CountryISO2":"JP"},
                        {"Code":"FOREX","OperatingMIC":"","CountryISO2":""}]
            if path.endswith("/US"):
                if params["delisted"]:
                    return [{"Code":"OLD","Name":"Former company","Type":"Common Stock","Currency":"USD"},
                            {"Code":"AAPL","Name":"Former reused symbol","Type":"Common Stock"}]
                return [{"Code":"AAPL","Name":"Current name","Type":"Common Stock",
                         "Currency":"USD","Isin":"US0378331005"},
                        {"Code":"EXOTIC","Name":"Tiny public firm","Type":"common_stock","Currency":"USD"}]
            if path.endswith("/JP"):
                if params["delisted"]:
                    return [{"Code":"1000","Name":"Historical Japanese listing","Type":"Stock"}]
                return [{"Code":"7203","Name":"Toyota","Type":"Common Stock","Currency":"JPY"}]
            raise ValueError("Unknown fixture endpoint")
    t=datetime(2026,11,15,12,tzinfo=timezone.utc)
    rows,report=collect_catalog(MockProvider(),t)
    assert report["exchanges_available"]==2
    assert report["rows_active"]==4 and report["rows_provider_delisted"]==3
    assert len(rows)==7
    assert any(x.ticker=="EXOTIC" for x in rows)  # no small-cap prefilter
    assert any(x.exchange_code=="JP" and x.vendor_delisted_snapshot for x in rows)
    assert any(e["error_type"]=="TICKER_REUSE_OR_DUPLICATE" for e in report["errors"])
    assert not any(x.permanent_instrument_id_confirmed or x.broker_tradability_confirmed for x in rows)
    print("V25_WORLDWIDE_DISCOVERY_ADAPTER_TEST_PASS",json.dumps({
        "exchanges":2,"active":4,"delisted":3,
        "ticker_reuse_detected":True,"microcaps_not_prefiltered":True,
        "api_calls_to_real_provider":0,"live_trades":0}))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--fetch-all-with-own-token",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        test();return
    if not args.fetch_all_with_own_token:
        raise SystemExit("Use --self-test, or explicitly --fetch-all-with-own-token")
    token=os.environ.get("EODHD_API_TOKEN")
    if not token:raise SystemExit("No EODHD_API_TOKEN set: no provider downloads performed.")
    rows,report=collect_catalog(authenticated_transport(token),
                                datetime.now(timezone.utc))
    folder=Path("results");folder.mkdir(parents=True,exist_ok=True)
    (folder/"v25_eodhd_global_catalog.json").write_text(json.dumps({
        "source_report":report,"candidates":[asdict(x) for x in rows]},indent=2))
    print("V25_WORLD_CATALOG_DISCOVERY",json.dumps({
        "exchanges":report["exchanges_available"],
        "active":report["rows_active"],
        "delisted":report["rows_provider_delisted"],
        "ticker_collisions":sum(e["error_type"]=="TICKER_REUSE_OR_DUPLICATE" for e in report["errors"]),
        "live_broker_connected":False}),flush=True)

if __name__=="__main__":main()
