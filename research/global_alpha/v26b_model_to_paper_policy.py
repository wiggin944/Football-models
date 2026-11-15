#!/usr/bin/env python3
"""Global Alpha v26: conservative ML-forecast-to-GBP PAPER policy integration.

The original 11-country intraday ML study is a measured negative result.
When validation selects CASH, this adapter leaves real salary contributions
uninvested in a local-only simulator. A separate SYNTHETIC strong-forecast
fixture proves buy/sell and failsafe integration; it is NOT alpha.
There is no broker connection, secret, market subscription or live order route.
"""
from dataclasses import dataclass
from datetime import datetime,timezone,timedelta,date
from pathlib import Path
import json,math
from v25_global_intraday_paper_engine import PaperLedger,RiskLimits,Target,Quote,FXQuote
from global_security_master import Listing

SOURCE=Path("source/results/v26_intraday_ml_research.json")
OUT=Path("results")
OUT.mkdir(exist_ok=True,parents=True)

@dataclass(frozen=True)
class EconomicForecast:
    security_id:str
    source_model:str
    observed_at:datetime
    predicted_return:float       # log equity return over research holding horizon
    uncertainty:float            # forecast horizon standard deviation (NOT daily vol)
    estimated_roundtrip_bps:float
    horizon_minutes:int

@dataclass(frozen=True)
class PolicySettings:
    cost_uncertainty_multiplier:float=0.5
    reserve_excess_bps:float=10.0  # additional minimum expected gain beyond friction
    max_issuer_weight:float=0.08
    max_total_risk_budget_fraction:float=0.70

def choose_target(forecast:EconomicForecast,now:datetime,
                  settings:PolicySettings=PolicySettings(),
                  model_rejected_by_validation:bool=False)->Target:
    if now.tzinfo is None or forecast.observed_at.tzinfo is None:
        raise ValueError("Forecast timestamps must be timezone aware")
    age=(now-forecast.observed_at).total_seconds()
    if age<0 or age>60: # Adapter refuses stale intraday forecasts.
        raise ValueError("Stale or future model signal")
    if forecast.horizon_minutes<1 or not forecast.source_model:
        raise ValueError("Invalid model metadata")
    for key in ("predicted_return","uncertainty","estimated_roundtrip_bps"):
        if not math.isfinite(float(getattr(forecast,key))):
            raise ValueError("Nonfinite model risk inputs")
    if forecast.uncertainty<0 or forecast.estimated_roundtrip_bps<0:
        raise ValueError("Negative risk or trading-cost estimate")
    if model_rejected_by_validation or forecast.source_model=="cash":
        target=0.0
    else:
        # Only allocate if predicted gain exceeds *roundtrip* costs +
        # minimum excess hurdle + uncertainty penalty. All costs are research
        # assumptions and must eventually be calibrated to real spread/fees.
        excess=forecast.predicted_return*10000-forecast.estimated_roundtrip_bps
        evidence_hurdle=(settings.reserve_excess_bps+
              10000*settings.cost_uncertainty_multiplier*forecast.uncertainty)
        if excess<=evidence_hurdle:
            target=0.0
        else:
            # Risk-constrained size depends on forecast edge versus uncertainty,
            # bounded by issuer cap and portfolio engine total equity reserve.
            scaled=(excess-evidence_hurdle)/(10000*max(forecast.uncertainty,0.0005))
            target=min(settings.max_issuer_weight,max(0.0,scaled*0.05))
    return Target(forecast.security_id,"v26_paper_"+forecast.source_model,
                  forecast.observed_at,target,forecast.horizon_minutes,
                  forecast.predicted_return)

def fixture_quote(sid,now,price=100):
    return Quote(sid,now-timedelta(seconds=2),price-0.02,price+0.02,
                 5000,5000,2000000.,True,False,True,
                 "SYNTHETIC_ONLY_NO_LIVE_TOP_OF_BOOK")

def sample_listing():
    # Intentionally fictional listing: prevents mistaking the fixture
    # for a verified broker instrument even when a symbol resembles AAPL.
    return Listing("SIMULATED.US.XNAS.0001","SIM_ISSUER_0001","DEMO",
        "XNAS","US","USD","ordinary_share",date(2014,1,1),None,
        "SIMULATED_CONTRACT_ONLY","America/New_York")

def run_tests():
    date0=datetime(2026,11,16,15,30,tzinfo=timezone.utc)
    listing=sample_listing()
    fx=FXQuote("USD",1.25,date0-timedelta(seconds=1),"SYNTHETIC_FX")
    lp=PaperLedger()
    lp.deposit(1000.,date0-timedelta(days=1))
    zero=EconomicForecast(listing.security_id,"cash",date0,0.,.005,20.,30)
    target=choose_target(zero,date0,model_rejected_by_validation=True)
    assert target.target_weight==0
    assert lp.on_signal(target,listing,fixture_quote(listing.security_id,date0),fx,date0)["reason"]=="ALREADY_AT_TARGET"
    assert lp.shares.get(listing.security_id,0)==0
    assert lp.cash_gbp==1000.
    low=EconomicForecast(listing.security_id,"ridge",date0,.0010,.008,20.,30)
    assert choose_target(low,date0).target_weight==0
    # A purely ARTIFICIAL forecast, no trained model approval.
    strong=EconomicForecast(listing.security_id,"fixture_not_ml",date0,.05,.005,20.,30)
    buy_target=choose_target(strong,date0)
    assert 0<buy_target.target_weight<=.08
    buy=lp.on_signal(buy_target,listing,fixture_quote(listing.security_id,date0),fx,date0)
    assert buy["action"]=="BUY",buy
    later=date0+timedelta(minutes=10)
    close=EconomicForecast(listing.security_id,"fixture_not_ml",later,-.02,.008,20.,30)
    exit_target=choose_target(close,later)
    assert exit_target.target_weight==0
    closefx=FXQuote("USD",1.25,later-timedelta(seconds=1),"SYNTHETIC_FX")
    sell=lp.on_signal(exit_target,listing,fixture_quote(listing.security_id,later),closefx,later)
    assert sell["action"]=="SELL" and lp.shares[listing.security_id]==0
    lp.trading_enabled=False
    assert lp.on_signal(buy_target,listing,fixture_quote(listing.security_id,later),
        closefx,later)["reason"]=="PAPER_KILL_SWITCH"
    try:choose_target(strong,later+timedelta(minutes=4))
    except ValueError:pass
    else:raise AssertionError("Stale signal must be rejected")
    # Actual user can add cash whenever they like. Simulator ledger accepts
    # deposits, but does NOT broker-transfer, schedule or invest real funds.
    lp.deposit(300.,later+timedelta(days=1))
    return {"paper_fixture_buy_and_sell":True,
        "unpromoted_low_edge_forecast_abstains":True,
        "stale_signal_and_kill_switch_block":True,
        "GBP_contributions_tested":[1000,300],
        "synthetic_fills":len(lp.fills),"live_broker_orders":0}

def integrate_real_study():
    if not SOURCE.exists():
        raise RuntimeError("Missing reproducible v26 model-evaluation artifact")
    original=json.loads(SOURCE.read_text())
    selection=original["study"]["selected_on_validation"]["model"]
    rows=original["test_forecast_records"]
    assert len(rows)==original["training_valid_test_n"]["test"]
    # Test all *real* test-period model forecasts, without editing the model.
    # Don't use historical OHLC as an executable quote.
    signal_weights=[]
    for r in rows:
        obs=datetime.fromisoformat(r["decision_utc"])
        pred=float(r["forecasts"][selection])
        f=EconomicForecast("PAPER_UNIDENTIFIED:"+r["symbol"],selection,obs,
            pred,.008,20.,30)
        w=choose_target(f,obs,model_rejected_by_validation=(selection=="cash"))
        signal_weights.append(w.target_weight)
    # The current real run chose cash; fail closed if result artifact differs.
    if selection!="cash":
        raise RuntimeError("Re-test policy and approval gate before changing the negative v26 baseline")
    assert all(w==0.0 for w in signal_weights)
    paper=PaperLedger()
    when=datetime(2026,11,16,12,tzinfo=timezone.utc)
    paper.deposit(400,when)
    paper.deposit(250,when+timedelta(minutes=1))
    assert paper.cash_gbp==650 and not paper.fills and not paper.shares
    return {"selection":selection,"v26_test_signals":len(rows),
            "zero_buy_targets":len(rows),"GBP_cash_after_deposits":paper.cash_gbp,
            "model_decisions_executed_as_trades":0,"real_broker_orders":0}

def main():
    fixture=run_tests()
    real=integrate_real_study()
    output={"version":"v26b_fitted_intraday_model_to_local_paper_ledger",
       "model_source":"v26 frozen 11-market research run 43775407209",
       "integration":real,"synthetic_fixture":fixture,
       "research_only":True,"trained_model_has_alpha":False,
       "not_live_quotes":True,"not_connected_to_broker":True,
       "missing_live_prerequisites":["licensed live two-sided quotes and volume",
           "market-native broker permissions and exchange trading calendar",
           "historically exhaustive global security master",
           "full multi-horizon economic-context forecasting distribution",
           "approved portfolio-level sector/currency/issuer risks",
           "prospective positive net-risk-adjusted alpha and kill-switch monitoring"]}
    (OUT/"v26b_paper_decision_integration.json").write_text(json.dumps(output,indent=2))
    print("V26B_REAL_ML_TO_PAPER_INTEGRATION_PASS",json.dumps(output,separators=(",",":")))
if __name__=="__main__":
    main()
