#!/usr/bin/env python3
"""Global Alpha: portfolio-wide opportunity replacement prototype (PAPER ONLY).

One optimiser jointly evaluates existing holdings and new equity/gold/gilt/cash
opportunities, across multiple horizons on the same asset, after transaction
and dividend costs, correlated risk and diversification caps.

All examples are deliberately synthetic: no alpha, broker, market feed or live
orders. Model forecasts must already be calibrated to a COMMON next-30-day
decision window; horizon_days is the original model's forecast horizon, NOT
an invitation to add different-horizon cumulative profits together.
"""
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json
import numpy as np
from scipy.optimize import minimize

@dataclass(frozen=True)
class Instrument:
    security_id: str
    issuer_id: str
    asset_class: str      # equity, gold, gilt, etc.
    country: str
    currency: str
    sector: str
    roundtrip_buy_fraction: float
    roundtrip_sell_fraction: float   # include estimated realised-tax friction when relevant
    gross_dividend_30d: float = 0.0  # projected cash payout / current holding value
    dividend_withholding: float = 0.0
    return_basis: str = "price_only"  # or total_return, where cash dividends are included

@dataclass(frozen=True)
class HorizonForecast:
    security_id: str
    horizon_days: float
    predicted_return_next_30d: float  # already normalized/calibrated to shared 30d window
    confidence_weight: float
    observed_at: datetime
    validity_seconds: int
    model_id: str

@dataclass(frozen=True)
class PortfolioLimits:
    max_equity_weight: float = 0.50
    max_gold_weight: float = 0.25
    max_gilt_weight: float = 0.50
    min_cash_weight: float = 0.10
    issuer_cap: float = 0.30
    country_cap: float = 0.60
    sector_cap: float = 0.40
    currency_cap: float = 0.65
    stress_loss_cap: float = 0.14
    covariance_penalty: float = 0.40
    minimum_improvement_fraction: float = 0.0001

def validate_forecasts(instruments, forecasts, at):
    if at.tzinfo is None: raise ValueError("Timezone required")
    known={a.security_id:a for a in instruments}
    grouped=defaultdict(list)
    for f in forecasts:
        if f.security_id not in known: raise ValueError("Unknown instrument")
        if f.observed_at.tzinfo is None: raise ValueError("Naive forecast")
        age=(at-f.observed_at).total_seconds()
        if age<0 or age>f.validity_seconds: continue
        if (not np.isfinite(f.predicted_return_next_30d)
            or not np.isfinite(f.confidence_weight)
            or f.confidence_weight<=0
            or f.horizon_days<=0 or not f.model_id):
            raise ValueError("Invalid predictive distribution metadata")
        grouped[f.security_id].append(f)
    return grouped

def common_window_forecasts(instruments, forecasts, at, cash_return_30d):
    grouped=validate_forecasts(instruments,forecasts,at)
    score=[]
    trace={}
    for asset in instruments:
        signals=grouped.get(asset.security_id,[])
        if signals:
            # Reconcile *correlated*, cross-horizon model views by weighted mean.
            # Never add short/medium/long forecast returns as independent alpha.
            vals=np.array([f.predicted_return_next_30d for f in signals],float)
            w=np.array([f.confidence_weight for f in signals],float)
            predicted=float(np.average(vals,weights=w))
            disagreement=float(np.sqrt(np.average((vals-predicted)**2,weights=w)))
        else:
            predicted=0.0
            disagreement=None
        if asset.return_basis not in ("price_only","total_return"):
            raise ValueError("Return treatment must be explicit")
        if not (0<=asset.dividend_withholding<=1):
            raise ValueError("Invalid dividend withholding")
        if asset.gross_dividend_30d<0:
            raise ValueError("Bad future cash dividend forecast")
        # A total-return price/forecast has already incorporated distributions.
        # Never stack projected dividend cash on top a total-return model.
        dividend=(asset.gross_dividend_30d*(1-asset.dividend_withholding)
                  if asset.return_basis=="price_only" else 0.0)
        excess=predicted+dividend-cash_return_30d if signals else -cash_return_30d
        score.append(excess)
        trace[asset.security_id]={
            "horizons_days":[f.horizon_days for f in signals],
            "forecast_count":len(signals),
            "aggregated_price_or_total_return_30d":round(predicted,7),
            "after_withholding_dividend_return_30d":round(dividend,7),
            "disagreement_between_horizon_views":None if disagreement is None else round(disagreement,7),
            "excess_over_achievable_cash_30d":round(excess,7),
            "return_basis":asset.return_basis}
    return np.array(score,float),trace

def solve(instruments,forecasts,current,covariance,stress_scenarios,
          cash_return_30d,at,limits=PortfolioLimits()):
    n=len(instruments)
    if n<2 or len({a.security_id for a in instruments})!=n:
        raise ValueError("Need distinct canonical asset IDs")
    symbols=[a.security_id for a in instruments]
    old=np.array([current.get(s,0.0) for s in symbols],float)
    if not np.isfinite(old).all() or (old<0).any() or old.sum()>1+1e-8:
        raise ValueError("Bad current portfolio weights")
    if not np.isfinite(cash_return_30d) or cash_return_30d<=-1:
        raise ValueError("Invalid achievable cash return")
    sigma=np.asarray(covariance,float)
    stress=np.asarray(stress_scenarios,float)
    if sigma.shape!=(n,n) or stress.ndim!=2 or stress.shape[1]!=n:
        raise ValueError("Risk model dimension mismatch")
    if (not np.isfinite(sigma).all() or
        not np.allclose(sigma,sigma.T,atol=1e-10) or
        np.linalg.eigvalsh(sigma).min()<-1e-9 or not np.isfinite(stress).all()):
        raise ValueError("Bad covariance or stress returns")
    mu,audit=common_window_forecasts(instruments,forecasts,at,cash_return_30d)
    buys=np.array([a.roundtrip_buy_fraction for a in instruments],float)
    sells=np.array([a.roundtrip_sell_fraction for a in instruments],float)
    if (buys<0).any() or (sells<0).any() or not np.isfinite(buys+sells).all():
        raise ValueError("Bad execution/tax friction")
    # Source freshness: no fresh evidence -> no *new* buys on that instrument.
    for i,s in enumerate(symbols):
        if audit[s]["forecast_count"]==0 and old[i]<=0:
            # Unmodelled stocks cannot be purchased by this research policy.
            mu[i]=-1.
    risk=float(limits.covariance_penalty)
    def cost(w):
        delta=w-old
        # Smooth positive- and negative-part approximation keeps SLSQP stable.
        eps=1e-9
        smooth=np.sqrt(delta**2+eps**2)
        return float(.5*np.sum((buys+sells)*smooth+(buys-sells)*delta))
    def utility(w):
        return float(mu@w-.5*risk*(w@sigma@w)-cost(w))
    constraints=[]
    def upper(indices,limit):
        ix=np.array(indices,dtype=int)
        if len(ix):constraints.append({"type":"ineq",
            "fun":lambda w,ix=ix,limit=limit: float(limit-np.sum(w[ix]))})
    upper(range(n),1-limits.min_cash_weight)
    for category,cap in [("equity",limits.max_equity_weight),
                         ("gold",limits.max_gold_weight),
                         ("gilt",limits.max_gilt_weight)]:
        upper([i for i,a in enumerate(instruments) if a.asset_class==category],cap)
    for field,cap in [("issuer_id",limits.issuer_cap),("country",limits.country_cap),
                      ("sector",limits.sector_cap),("currency",limits.currency_cap)]:
        groups=defaultdict(list)
        for i,a in enumerate(instruments):groups[getattr(a,field)].append(i)
        for k,indices in groups.items():upper(indices,cap)
    for row in stress:
        # Stress scenario describes fractional returns over the same window,
        # cash scenario baseline is conservative zero for this constraint.
        constraints.append({"type":"ineq",
            "fun":lambda w,row=row:float(limits.stress_loss_cap+row@w)})
    # Start from the same portfolio (if valid); otherwise shrink to a feasible
    # cash-oriented seed and never manufacture risk-free profits.
    start=old.copy()
    if any(c["fun"](start)<-1e-8 for c in constraints):start*=.4
    result=minimize(lambda w:-utility(w),start,method="SLSQP",
        bounds=[(0.,1.)]*n,constraints=constraints,
        options={"maxiter":800,"ftol":1e-12})
    if not result.success:
        raise RuntimeError("Global allocation did not converge: "+result.message)
    proposal=np.where(result.x<1e-7,0.,result.x)
    if any(c["fun"](proposal)<-1e-6 for c in constraints):
        raise RuntimeError("Unsafe portfolio limits")
    # Conservative no-churn gate: only replace if the joint portfolio utility
    # improves by enough after estimated buy/sell/tax costs.
    valid_old=all(c["fun"](old)>=-1e-8 for c in constraints)
    advantage=utility(proposal)-utility(old)
    new=proposal if not valid_old or advantage>limits.minimum_improvement_fraction else old.copy()
    actions={s:round(float(w-o),6) for s,w,o in zip(symbols,new,old)
             if abs(w-o)>1e-5}
    return {
        "target_weights":{s:round(float(v),6) for s,v in zip(symbols,new)},
        "target_cash_weight":round(float(1-new.sum()),6),
        "trades_delta_weight":actions,
        "estimated_purchase_sale_cost_fraction":round(cost(new),7),
        "joint_utility_improvement_fraction":round(float(advantage),7),
        "same_capital_across_all_assets":True,
        "portfolio_stress_returns":[round(float(v),7) for v in stress@new],
        "forecast_reconciliation":audit,
        "broker_connected":False,
        "model_alpha_established":False,
        "paper_only":True}

def synthetic_tests():
    now=datetime(2026,11,16,12,tzinfo=timezone.utc)
    assets=[
        Instrument("OLD_US","OLD_ISSUER","equity","US","USD","TECH",.006,.006,
                   gross_dividend_30d=.001,dividend_withholding=.15),
        Instrument("NEW_JP","NEW_ISSUER","equity","JP","JPY","INDUSTRIAL",.006,.006),
        Instrument("GOLD_GBP","PHYSICAL_GOLD","gold","GB","GBP","GOLD",.002,.002),
        Instrument("UK_GILT","UK_GOV","gilt","GB","GBP","GOVERNMENT",.001,.001),
    ]
    base={
      "OLD_US":.30,"NEW_JP":0.,"GOLD_GBP":.10,"UK_GILT":.10}
    cov=np.diag([.001,.001,.0007,.00025])
    # Equity crash, inflation and gilt-rate shock scenarios.
    stress=np.array([[-.30,-.30,.08,.02],[-.10,-.07,.07,-.12]])
    limits=PortfolioLimits(max_equity_weight=.30,
        max_gold_weight=.25,max_gilt_weight=.35,min_cash_weight=.10,
        issuer_cap=.30,sector_cap=.40,country_cap=.65,
        covariance_penalty=.15,minimum_improvement_fraction=.0001)
    def signal(s,value,days=30):
        return HorizonForecast(s,days,value,1.,now,3600,"synthetic_calibrated_fixture")
    # First: a 30d old forecast and weak new 30d forecast.
    common=[signal("OLD_US",.019),signal("GOLD_GBP",.008),
            signal("UK_GILT",.005)]
    weak=solve(assets,common+[signal("NEW_JP",.0193)],base,cov,stress,.002,now,limits)
    assert weak["target_weights"]["OLD_US"]>.26,weak
    assert weak["target_weights"]["NEW_JP"]<.02,weak
    # Now something genuinely better appears: replace most of old position
    # because total equity budget is already consumed, after costs.
    strong=solve(assets,common+[signal("NEW_JP",.080)],base,cov,stress,.002,now,limits)
    assert strong["target_weights"]["NEW_JP"]>.22,strong
    assert strong["target_weights"]["OLD_US"]<.08,strong
    assert strong["target_weights"]["NEW_JP"]+strong["target_weights"]["OLD_US"]<=.300001
    assert all(v>=-.140001 for v in strong["portfolio_stress_returns"])
    # A second shorter-horizon view on same asset cannot be summed.
    both=common+[signal("NEW_JP",.08,1),signal("NEW_JP",.04,365)]
    blended=solve(assets,both,base,cov,stress,.002,now,limits)
    entry=blended["forecast_reconciliation"]["NEW_JP"]
    assert entry["forecast_count"]==2
    assert abs(entry["aggregated_price_or_total_return_30d"]-.06)<1e-9
    # Dividend return is not double counted if input is total return.
    trasset=Instrument("T","T_ISSUER","equity","GB","GBP","UTIL",.002,.002,
        gross_dividend_30d=.006,dividend_withholding=.25,return_basis="total_return")
    priceasset=Instrument("P","P_ISSUER","equity","GB","GBP","UTIL",.002,.002,
        gross_dividend_30d=.006,dividend_withholding=.25,return_basis="price_only")
    f1=signal("T",.02);f2=signal("P",.02)
    mt,dt=common_window_forecasts([trasset],[f1],now,.002)
    mp,dp=common_window_forecasts([priceasset],[f2],now,.002)
    assert abs(mt[0]-(.02-.002))<1e-9
    assert abs(mp[0]-(.02+.0045-.002))<1e-9
    assert dt["T"]["after_withholding_dividend_return_30d"]==0
    # Multi-issuer concentration cap blocks a fake duplicate dual listing.
    d2=Instrument("OLD_ADR","OLD_ISSUER","equity","US","USD","TECH",.002,.002)
    copy_assets=assets+[d2]
    copied=solve(copy_assets,common+[signal("OLD_ADR",.11)],
       base,np.diag([.001,.001,.0007,.00025,.001]),
       np.c_[stress,np.array([-.30,-.07])],.002,now,limits)
    assert copied["target_weights"]["OLD_US"]+copied["target_weights"]["OLD_ADR"]<=.300001
    results={
       "tests":"PASS",
       "weak_competitor_hold":{"old":weak["target_weights"]["OLD_US"],
                               "new":weak["target_weights"]["NEW_JP"]},
       "strong_competitor_sell_and_rotate":{"old":strong["target_weights"]["OLD_US"],
                                            "new":strong["target_weights"]["NEW_JP"],
                                            "actions":strong["trades_delta_weight"]},
       "same_asset_dual_horizon_aggregation":entry["aggregated_price_or_total_return_30d"],
       "dividend_price_only_aftertax":dp["P"]["after_withholding_dividend_return_30d"],
       "dividend_total_return_already_included":dt["T"]["after_withholding_dividend_return_30d"],
       "risk_constraint_respected":True,
       "issuer_duplicate_not_double_counted":True,
       "no_live_orders":True,
       "all_example_forecasts_and_returns_are_synthetic":True}
    print("GLOBAL_ALPHA_JOINT_REPLACEMENT_TESTS",json.dumps(results,separators=(",",":")),flush=True)
    return results
if __name__=="__main__":
    synthetic_tests()
