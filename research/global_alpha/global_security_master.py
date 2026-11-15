#!/usr/bin/env python3
"""Global Alpha global-universe primitives: no country/index whitelist.

Models exchange listings and issuer identity separately. Feed coverage and order
eligibility are explicit, audited per listing and trading timestamp. Research only.
"""
from dataclasses import dataclass
from datetime import date,datetime,timezone
from typing import Optional,Iterable
import csv
import io

@dataclass(frozen=True)
class Listing:
    security_id:str            # persistent vendor security identity, NOT ticker
    issuer_id:str              # consolidated issuer identity, NOT ticker
    ticker:str
    mic:str                    # ISO 10383 exchange MIC
    country:str                # ISO 3166-1 alpha-2
    currency:str               # ISO 4217
    security_type:str          # ordinary_share, preferred, ADR, etc.
    effective_from:date
    effective_to:Optional[date]  # None = still listing; dated = terminal listing
    broker_contract_id:Optional[str]=None
    exchange_time_zone:Optional[str]=None
    isin:Optional[str]=None
    figi:Optional[str]=None
    legal_restricted:bool=False

    def __post_init__(self):
        if not self.security_id or not self.issuer_id or not self.ticker:
            raise ValueError("Stable security, issuer identity and listing ticker required")
        if not(len(self.mic)==4 and len(self.country)==2 and len(self.currency)==3):
            raise ValueError("Nonstandard exchange/country/currency code")
        if self.effective_to and self.effective_to<self.effective_from:
            raise ValueError("Listing cannot end before it starts")

    def existed(self,at:date)->bool:
        return self.effective_from<=at and (self.effective_to is None or at<=self.effective_to)

@dataclass(frozen=True)
class VenueObservation:
    security_id:str
    received_at_utc:datetime
    quoted_bid:float
    quoted_ask:float
    daily_dollar_volume:float
    exchange_open:bool
    auction_or_halted:bool
    source_is_live:bool

def discover_at(listings:Iterable[Listing],at:date):
    """ALL valid listings, including those NOT tradable. Do not filter by country or index."""
    candidates={}
    for l in listings:
        if not l.existed(at):continue
        if l.security_id in candidates:raise ValueError("Overlapping duplicate security identity")
        candidates[l.security_id]=l
    return candidates

def can_route_order(listing:Listing,quote:VenueObservation,now:datetime,
                    max_stale_seconds:int=15,min_daily_dollar_volume:float=100000.0):
    """A deliberately strict readiness gate; not a recommended investment threshold."""
    blockers=[]
    if listing.security_id!=quote.security_id:blockers.append("MISMATCHED_INSTRUMENT")
    if not listing.existed(now.date()):blockers.append("NOT_LISTED_ON_DATE")
    if listing.legal_restricted:blockers.append("LEGAL_RESTRICTION")
    if not listing.broker_contract_id:blockers.append("NO_BROKER_CONTRACT")
    if not listing.exchange_time_zone:blockers.append("UNKNOWN_MARKET_CALENDAR")
    if not quote.source_is_live:blockers.append("DELAYED_OR_UNLICENSED_FEED")
    if not quote.exchange_open or quote.auction_or_halted:blockers.append("EXCHANGE_NOT_CONTINUOUSLY_OPEN")
    if not (quote.quoted_bid>0 and quote.quoted_ask>=quote.quoted_bid):
        blockers.append("INVALID_BID_ASK")
    if quote.daily_dollar_volume<min_daily_dollar_volume:
        blockers.append("LIQUIDITY_INSUFFICIENT_FOR_TEST")
    timestamp=quote.received_at_utc
    if timestamp.tzinfo is None: blockers.append("NAIVE_TIMESTAMP")
    else:
        seconds=(now.astimezone(timezone.utc)-timestamp.astimezone(timezone.utc)).total_seconds()
        if seconds<0 or seconds>max_stale_seconds:blockers.append("STALE_OR_FUTURE_QUOTE")
    return (not blockers,blockers)

def tests():
    from datetime import timedelta
    now=datetime(2026,11,15,12,tzinfo=timezone.utc)
    listings=[
        Listing("US1234","ISSUER_A","ABC","XNAS","US","USD","ordinary_share",date(2014,1,1),None,
                "IBKR-US1234","America/New_York"),
        Listing("JP1234","ISSUER_B","1234","XTKS","JP","JPY","ordinary_share",date(2010,1,1),None,
                "IBKR-JP1234","Asia/Tokyo"),
        Listing("IN1234","ISSUER_C","FOO","XNSE","IN","INR","ordinary_share",date(2020,1,1),None,
                None,"Asia/Kolkata"),
        Listing("ZA1234","ISSUER_D","BAR","XJSE","ZA","ZAR","ordinary_share",date(2015,1,1),date(2020,1,1),
                None,"Africa/Johannesburg"),
    ]
    universe=discover_at(listings,now.date())
    assert set(universe)=={"US1234","JP1234","IN1234"}
    assert "ZA1234" in discover_at(listings,date(2019,1,1))
    assert "IN1234" in universe  # discovery does not silently drop unsupported broker stocks
    q=VenueObservation("US1234",now-timedelta(seconds=2),10.0,10.01,1000000,True,False,True)
    assert can_route_order(universe["US1234"],q,now)[0] is True
    assert can_route_order(universe["IN1234"],q,now)[0] is False
    assert can_route_order(universe["US1234"],VenueObservation("US1234",now-timedelta(minutes=3),10,10.01,1000000,True,False,True),now)[0] is False
    assert can_route_order(universe["US1234"],VenueObservation("US1234",now-timedelta(seconds=2),10,10.01,1000000,True,True,True),now)[0] is False
    print("GLOBAL_SECURITY_MASTER_TESTS_PASS: country-agnostic discovery, historical delisting, fail-closed trading")
if __name__=="__main__":tests()
