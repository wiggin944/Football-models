#!/usr/bin/env python3
"""Global Alpha v25: worldwide intraday, GBP-ledger PAPER execution kernel.

This is an event-driven EXECUTION SIMULATOR, not a forecasting strategy.
All target weights come from an EXTERNAL model/policy and can be revised any
minute. No broker API, network orders, API keys, or user funds.
Bid/ask, FX, fees, quote freshness, venue status, share lots and risk limits
apply to every simulated fill. GBP accounting uses local currency per GBP.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from datetime import datetime,timezone,timedelta,date
from zoneinfo import ZoneInfo
from math import floor,isfinite
from typing import Optional
import json

from global_security_master import Listing,VenueObservation,can_route_order,discover_at

@dataclass(frozen=True)
class Quote:
    security_id:str
    received_at:datetime
    bid:float
    ask:float
    bid_size:int
    ask_size:int
    daily_dollar_volume:float
    venue_open:bool
    halted:bool
    licensed_live:bool
    source_type:str  # 'simulated', 'licenced_quote'; recorded in every order

@dataclass(frozen=True)
class FXQuote:
    currency:str
    local_per_gbp:float
    received_at:datetime
    source_type:str  # audit, do not assume free public feed is real-time

@dataclass(frozen=True)
class Target:
    security_id:str
    model_id:str
    produced_at:datetime
    target_weight:float
    horizon_minutes:int
    expected_excess_return:Optional[float]=None

@dataclass(frozen=True)
class RiskLimits:
    max_weight_per_issuer:float=.12
    max_total_equity_weight:float=.70
    min_cash_fraction:float=.20
    max_spread_bps:float=100
    max_quote_age_seconds:int=15
    max_fx_age_seconds:int=60
    max_signal_age_seconds:int=180
    participation_fraction:float=.10
    fixed_fee_gbp:float=.50
    variable_fee_fraction:float=.0005
    fx_conversion_fee_fraction:float=.0010
    lot_size:int=1
    def __post_init__(self):
        if not (0<self.max_weight_per_issuer<=1
                and 0<self.max_total_equity_weight<=1
                and 0<=self.min_cash_fraction<1
                and 0<self.participation_fraction<=1
                and self.lot_size>=1):
            raise ValueError("Invalid paper-only risk limits")

@dataclass
class Mark:
    mid:float
    local_per_gbp:float
    seen_at:datetime
    issuer_id:str

@dataclass(frozen=True)
class Fill:
    at:str
    security_id:str
    issuer_id:str
    action:str
    shares:int
    local_price:float
    local_per_gbp:float
    gross_gbp:float
    charges_gbp:float
    cash_balance_gbp:float
    model_id:str
    quote_class:str

@dataclass
class PaperLedger:
    """No broker transport and no live-order method by design."""
    limits:RiskLimits=field(default_factory=RiskLimits)
    cash_gbp:float=0.
    shares:dict=field(default_factory=dict)
    issuer_of:dict=field(default_factory=dict)
    marks:dict=field(default_factory=dict)
    fills:list=field(default_factory=list)
    contributions:list=field(default_factory=list)
    rejections:list=field(default_factory=list)
    trading_enabled:bool=True  # internal PAPER engine switch ONLY

    def deposit(self,amount_gbp:float,now:datetime):
        if now.tzinfo is None or not isfinite(amount_gbp) or amount_gbp<=0:
            raise ValueError("Deposit requires positive GBP and timezone")
        self.cash_gbp+=amount_gbp
        self.contributions.append({"at":now.isoformat(),"amount_gbp":round(amount_gbp,6),
                                   "source":"paper_contribution"})
        return self.cash_gbp

    def nav(self,now:datetime)->float:
        nav=self.cash_gbp
        for symbol,qty in self.shares.items():
            if qty<=0:continue
            mark=self.marks.get(symbol)
            if mark is None:raise RuntimeError("UNKNOWN_MARK_FOR_ACTIVE_HOLDING")
            if now-mark.seen_at>timedelta(days=4):
                raise RuntimeError("STALE_PORTFOLIO_MARK")
            nav+=qty*mark.mid/mark.local_per_gbp
        return nav

    def reject(self,now,security,why):
        res={"at":now.isoformat(),"security_id":security,"status":"NO_TRADE",
             "reason":why}
        self.rejections.append(res)
        return res

    def _fee(self,notional:float,listing:Listing):
        return max(self.limits.fixed_fee_gbp,self.limits.variable_fee_fraction*notional)+(
            self.limits.fx_conversion_fee_fraction*notional if listing.currency!="GBP" else 0.0)

    def on_signal(self,target:Target,listing:Listing,quote:Quote,fx:FXQuote,now:datetime):
        """Single current model decision; fill only with immediately valid quotes.

        Stock selection, forecast horizon, alpha prediction and buy/sell threshold
        must be supplied by the future external *trained* decision model.
        """
        if now.tzinfo is None:raise ValueError("Timezone required")
        key=target.security_id
        if key!=listing.security_id or key!=quote.security_id:
            return self.reject(now,key,"INSTRUMENT_IDENTITY_MISMATCH")
        if not self.trading_enabled:return self.reject(now,key,"PAPER_KILL_SWITCH")
        if target.produced_at.tzinfo is None:
            return self.reject(now,key,"NAIVE_FORECAST_TIMESTAMP")
        age=(now-target.produced_at).total_seconds()
        if age<0 or age>self.limits.max_signal_age_seconds:
            return self.reject(now,key,"STALE_OR_FUTURE_FORECAST")
        if not target.model_id or target.horizon_minutes<1:
            return self.reject(now,key,"NO_VALID_MODEL_OR_HORIZON")
        if not isfinite(target.target_weight) or not 0<=target.target_weight<=1:
            return self.reject(now,key,"INVALID_TARGET_WEIGHT")
        if listing.exchange_time_zone is None:
            return self.reject(now,key,"MISSING_EXCHANGE_TIME_ZONE")
        # Correct market-local listing date, not UTC calendar date.
        local_date=now.astimezone(ZoneInfo(listing.exchange_time_zone)).date()
        if not listing.existed(local_date):
            return self.reject(now,key,"NOT_LISTED_IN_EXCHANGE_LOCAL_DATE")
        if fx.currency!=listing.currency or fx.received_at.tzinfo is None:
            return self.reject(now,key,"FX_CURRENCY_OR_TIMESTAMP_INVALID")
        fxage=(now-fx.received_at).total_seconds()
        if fxage<0 or fxage>self.limits.max_fx_age_seconds or not isfinite(fx.local_per_gbp) or fx.local_per_gbp<=0:
            return self.reject(now,key,"FX_STALE_OR_UNUSABLE")
        if listing.currency=="GBP" and abs(fx.local_per_gbp-1)>1e-6:
            return self.reject(now,key,"GBP_NATIVE_FX_NOT_ONE")
        if quote.received_at.tzinfo is None:
            return self.reject(now,key,"NAIVE_QUOTE_TIMESTAMP")
        obs=VenueObservation(
            key,quote.received_at,quote.bid,quote.ask,quote.daily_dollar_volume,
            quote.venue_open,quote.halted,quote.licensed_live)
        ready,why=can_route_order(listing,obs,now,
            max_stale_seconds=self.limits.max_quote_age_seconds)
        if not ready:return self.reject(now,key,",".join(why))
        if quote.bid_size<0 or quote.ask_size<0 or not all(
            isfinite(z) for z in (quote.bid,quote.ask)):
            return self.reject(now,key,"BAD_ORDER_BOOK")
        mid=(quote.ask+quote.bid)/2.
        spread_bps=10000*(quote.ask-quote.bid)/mid
        if spread_bps>self.limits.max_spread_bps:
            return self.reject(now,key,"SPREAD_TOO_WIDE")
        # Mark the instrument only when valid. Other held stocks cannot be
        # silently valued at future quotes; they retain prior as-of marks.
        self.marks[key]=Mark(mid,fx.local_per_gbp,now,listing.issuer_id)
        self.issuer_of[key]=listing.issuer_id
        try:equity=self.nav(now)
        except RuntimeError as e:return self.reject(now,key,str(e))
        if equity<=0:return self.reject(now,key,"NO_PAPER_CAPITAL")
        current=self.shares.get(key,0)
        existing_same_issuer=sum(
            q*self.marks[s].mid/self.marks[s].local_per_gbp
            for s,q in self.shares.items() if q>0 and
            self.issuer_of.get(s)==listing.issuer_id and s!=key)
        desired=min(target.target_weight,self.limits.max_weight_per_issuer)
        issuer_capacity=max(0.,equity*self.limits.max_weight_per_issuer-existing_same_issuer)
        desired_notional=min(equity*desired,issuer_capacity)
        desired_shares=self.limits.lot_size*floor(
            desired_notional*fx.local_per_gbp/quote.ask/self.limits.lot_size)
        difference=desired_shares-current
        if difference==0:return self.reject(now,key,"ALREADY_AT_TARGET")

        # Paper fill at offer for BUY, bid for SELL, with conservative
        # per-quote participation and fee/FX cash accounting.
        side="BUY" if difference>0 else "SELL"
        max_liq=int(floor(
            (quote.ask_size if side=="BUY" else quote.bid_size)*
            self.limits.participation_fraction/self.limits.lot_size
        ))*self.limits.lot_size
        qty=min(abs(difference),max_liq)
        if qty<self.limits.lot_size:
            return self.reject(now,key,"NO_SIZE_WITHIN_PARTICIPATION_LIMIT")
        exec_px=quote.ask if side=="BUY" else quote.bid
        unit_gbp=exec_px/fx.local_per_gbp

        if side=="BUY":
            equity_marked=sum(
                q*self.marks[s].mid/self.marks[s].local_per_gbp
                for s,q in self.shares.items() if q>0)
            total_available=max(0.,equity*self.limits.max_total_equity_weight-equity_marked)
            cash_available=max(0.,self.cash_gbp-equity*self.limits.min_cash_fraction)
            def affordable(k):
                x=k*unit_gbp
                return x+self._fee(x,listing)<=cash_available+1e-9 and x<=total_available+1e-9
            while qty>0 and not affordable(qty):
                qty-=self.limits.lot_size
            if qty<=0:
                return self.reject(now,key,"CASH_RESERVE_OR_EQUITY_RISK_CAP")
            gross=qty*unit_gbp
            charges=self._fee(gross,listing)
            self.cash_gbp-=gross+charges
            self.shares[key]=current+qty
        else:
            qty=min(qty,current)
            if qty<=0:return self.reject(now,key,"NO_SHARES_TO_SELL")
            gross=qty*unit_gbp
            charges=self._fee(gross,listing)
            self.cash_gbp+=gross-charges
            self.shares[key]=current-qty

        if self.cash_gbp< -1e-8:raise AssertionError("No leverage/negative cash allowed")
        execution=Fill(now.isoformat(),key,listing.issuer_id,side,qty,
            round(exec_px,8),fx.local_per_gbp,round(gross,6),
            round(charges,6),round(self.cash_gbp,6),
            target.model_id,quote.source_type)
        self.fills.append(execution)
        return vars(execution)

    def snapshot(self,now:datetime):
        return {"asof":now.isoformat(),"cash_gbp":round(self.cash_gbp,4),
                "nav_gbp":round(self.nav(now),4),
                "positions":self.shares,"paper_fills":len(self.fills),
                "rejected_orders":len(self.rejections),
                "external_model_fitted":False,
                "live_broker_connected":False}

def fixture_tests():
    """Deterministic tests across two equity venues and a third inaccessible one."""
    from datetime import timedelta
    now=datetime(2026,11,16,15,30,0,tzinfo=timezone.utc)
    us=Listing("US:EX:123","ISSUER-US","TEST","XNAS","US","USD",
        "ordinary_share",date(2015,1,1),None,"PAPER-CONID-US",
        "America/New_York")
    jp=Listing("JP:EX:456","ISSUER-JP","MOCK","XTKS","JP","JPY",
        "ordinary_share",date(2015,1,1),None,"PAPER-CONID-JP",
        "Asia/Tokyo")
    india=Listing("IN:EX:789","ISSUER-IN","EXAM","XNSE","IN","INR",
        "ordinary_share",date(2015,1,1),None,None,"Asia/Kolkata")
    portfolio=PaperLedger()
    portfolio.deposit(1000,now-timedelta(minutes=10))
    q=Quote(us.security_id,now-timedelta(seconds=2),99.95,100.05,
        1000,1000,2000000,True,False,True,"simulated_bid_ask")
    fx=FXQuote("USD",1.25,now-timedelta(seconds=2),"synthetic_fx")
    sig=Target(us.security_id,"fixture_forecast_v0",now,0.12,120,0.04)
    buy=portfolio.on_signal(sig,us,q,fx,now)
    assert buy["action"]=="BUY" and buy["shares"]==1,buy
    assert 800<portfolio.cash_gbp<1000
    # Even if the external signal is positive, a stale quote cannot trade.
    stale=Quote(us.security_id,now-timedelta(minutes=2),99,100,
        1000,1000,2000000,True,False,True,"simulated")
    assert portfolio.on_signal(sig,us,stale,fx,now)["reason"].find("STALE")>=0
    # Cannot place orders for a venue without a broker contract.
    iq=Quote(india.security_id,now-timedelta(seconds=2),99,100,100,100,
             2000000,True,False,True,"simulated")
    ifx=FXQuote("INR",115,now-timedelta(seconds=2),"synthetic_fx")
    assert "NO_BROKER_CONTRACT" in portfolio.on_signal(
        Target(india.security_id,"fixture",now,0.1,120),
        india,iq,ifx,now)["reason"]
    # Exchange closed even if the record has prices.
    jq=Quote(jp.security_id,now-timedelta(seconds=2),1000,1001,
            1000,1000,3000000,False,False,True,"simulated")
    jfx=FXQuote("JPY",190,now-timedelta(seconds=2),"synthetic_fx")
    assert "EXCHANGE_NOT_CONTINUOUSLY_OPEN" in portfolio.on_signal(
        Target(jp.security_id,"fixture",now,0.12,120),jp,jq,jfx,now)["reason"]
    later=now+timedelta(minutes=10)
    q2=Quote(us.security_id,later-timedelta(seconds=2),
        105,105.08,1000,1000,2000000,True,False,True,"simulated_bid_ask")
    f2=FXQuote("USD",1.25,later-timedelta(seconds=2),"synthetic_fx")
    sell=portfolio.on_signal(Target(us.security_id,"fixture_forecast_v1",
        later,0.0,120,-0.03),us,q2,f2,later)
    assert sell["action"]=="SELL" and portfolio.shares[us.security_id]==0,sell
    assert len(portfolio.fills)==2
    assert portfolio.cash_gbp>1000  # for these *synthetic* profitable prices
    portfolio.trading_enabled=False
    assert portfolio.on_signal(sig,us,q2,f2,later)["reason"]=="PAPER_KILL_SWITCH"
    # Multiple exchanges remain discovered, even if trading eligibility differs.
    universe=discover_at([us,jp,india],now.date())
    assert len(universe)==3
    result={"test_result":"PASS","market_examples":list(universe),
        "synthetic_open_close_actions":[x.action for x in portfolio.fills],
        "synthetic_final_cash_gbp":round(portfolio.cash_gbp,2),
        "blocked_reasons":[x["reason"] for x in portfolio.rejections],
        "forecast_source":"fixture / NOT a trained model",
        "broker_orders":0,"live_account_connected":False}
    print("V25_PAPER_ENGINE_TESTS_PASS",json.dumps(result,separators=(",",":")))
    return result

if __name__=="__main__":
    fixture_tests()
