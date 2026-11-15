#!/usr/bin/env python3
"""Global Alpha v26: honest multi-market intraday ML *research* pilot.

11 illustrative listed stocks, one per country. This is NOT an all-listed
universe and not remotely close to proven alpha. Fetch ~30 calendar days of
5-minute OHLC, freeze observed raw bars, construct 30-minute forward returns
with delayed entry, purge overlapping training/test periods by local session.
Fit only on earlier sessions, choose ML vs cash using validation dates, then
inspect untouched-within-this-run test dates. Hypothetical fills are NOT
executable: no bid/ask feed. Illustrative round-trip friction only.

There is NO broker transport, order submission, or real money in this module.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from v25_global_intraday_data_audit import SAMPLE

SEED=26091
BAR_MINUTES=5
WARMUP_BARS=24
ENTRY_DELAY_BARS=2
HORIZON_BARS=6           # 30-min exposure
STEP_BARS=6              # non-overlap same ticker within each session
HISTORY_PERIOD="30d"
RISK_FRICTION_BPS=(0,20,60)  # illustrative round trip, NOT a broker quote
RIDGE_ALPHA=50.0
RESULTS=Path("results")
FEATURE_NAMES=(
    "return_1_bar","return_3_bars","return_6_bars","return_12_bars",
    "bar_log_open_close","bar_high_low_range","bar_close_location",
    "realized_vol_last_12","avg_range_last_12","relative_volume_vs_prior_24",
    "bars_since_session_open","market_dummy_0","market_dummy_1",
    "market_dummy_2","market_dummy_3","market_dummy_4","market_dummy_5",
    "market_dummy_6","market_dummy_7","market_dummy_8",
    "market_dummy_9","market_dummy_10")
EXPECTED_COLS=len(FEATURE_NAMES)
SYMBOL_MARKETS={sym:i for i,(_,sym,_) in enumerate(SAMPLE)}

def safe_json(d):
    return json.dumps(d,indent=2,allow_nan=False,default=str)

def retrieve(entry):
    country,sym,mic=entry
    h=yf.Ticker(sym).history(
        period=HISTORY_PERIOD,interval="5m",actions=False,
        prepost=False,auto_adjust=False,raise_errors=True)
    if h.empty:raise ValueError("No 5m bars")
    cols=["Open","High","Low","Close","Volume"]
    h=h[cols].copy().dropna(subset=cols[:4])
    if h.index.tz is None:raise ValueError("Naive quote timestamps")
    if h.index.has_duplicates or not h.index.is_monotonic_increasing:
        raise ValueError("Nonmonotonic or duplicated bars")
    if not np.isfinite(h[cols].to_numpy(float)).all():
        raise ValueError("NaN/Inf price or volume")
    if (h[cols[:4]]<=0).any().any() or (h["Volume"]<0).any():
        raise ValueError("Invalid OHLCV values")
    if len(h)<250:raise ValueError("Insufficient recent 5m bars")
    h["symbol"]=sym
    h["country"]=country
    h["mic"]=mic
    h["session"]=np.array(h.index.date,dtype=str)
    h["bar_start_utc"]=h.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S%z")
    h=h.reset_index(drop=True)
    if h.session.nunique()<12:
        raise ValueError("Fewer than 12 local-market sessions")
    return h

def fetch_all():
    successes={}
    failures={}
    with ThreadPoolExecutor(max_workers=3) as ex:
        tasks={ex.submit(retrieve,entry):entry for entry in SAMPLE}
        for task,entry in tasks.items():
            try:
                f=task.result()
                successes[entry[1]]=f
                print("V26_MARKET_DATA",entry[0],entry[1],len(f),
                      f.session.nunique(),"SESSIONS",flush=True)
            except Exception as e:
                failures[entry[1]]=type(e).__name__+": "+str(e)[:190]
                print("V26_SOURCE_FAILURE",entry[1],failures[entry[1]],flush=True)
    if len(successes)<8:
        raise RuntimeError(f"Too few market samples for ML: {len(successes)}/11")
    return successes,failures

def encode_features(g,i,sym):
    """Feature vector uses bars up to index i (inclusive), never their future."""
    if i<WARMUP_BARS:raise ValueError("Insufficient same-session warmup")
    c=g.Close.to_numpy(float)
    o=g.Open.to_numpy(float)
    high=g.High.to_numpy(float)
    low=g.Low.to_numpy(float)
    volume=g.Volume.to_numpy(float)
    p=c[i]
    returns=np.diff(np.log(c[i-12:i+1]))
    rng=(high[i-11:i+1]-low[i-11:i+1])/c[i-11:i+1]
    # Volume baseline excludes the signal bar itself.
    reference=np.median(volume[i-24:i])+1.0
    last_high=max(high[i]-low[i],1e-10)
    v=np.array([
        np.log(p/c[i-1]),
        np.log(p/c[i-3]),
        np.log(p/c[i-6]),
        np.log(p/c[i-12]),
        np.log(p/o[i]),
        np.log(high[i]/low[i]),
        (p-low[i])/last_high-0.5,
        float(np.std(returns,ddof=1)),
        float(np.mean(rng)),
        float(np.log1p(volume[i])-np.log(reference)),
        float(i/100.0),
        *np.eye(len(SAMPLE),dtype=float)[SYMBOL_MARKETS[sym]]
    ],dtype=float)
    if v.shape!=(EXPECTED_COLS,) or not np.isfinite(v).all():
        raise RuntimeError("Broken 5-minute feature vector")
    return v

def build_events(symbol,h):
    events=[]
    sessions=list(dict.fromkeys(h.session.tolist()))
    if len(sessions)<12:raise ValueError("Too few sessions")
    n=len(sessions)
    # Test set is last 20% of SESSIONS (not random price bars).
    # Validation is preceding ~20%, earlier ~60% train.
    ntrain=max(8,int(math.floor(n*0.60)))
    nvalid=max(3,int(math.floor(n*0.20)))
    if ntrain+nvalid>=n:raise ValueError("Train/validation/test session separation impossible")
    phases={day:("train" if ix<ntrain else "valid" if ix<ntrain+nvalid else "test")
            for ix,day in enumerate(sessions)}
    for day,g in h.groupby("session",sort=False):
        g=g.reset_index(drop=True)
        if len(g)<(WARMUP_BARS+ENTRY_DELAY_BARS+HORIZON_BARS+1):continue
        opens=g.Open.to_numpy(float)
        utc=pd.to_datetime(g.bar_start_utc,utc=True)
        for i in range(WARMUP_BARS,len(g)-ENTRY_DELAY_BARS-HORIZON_BARS,STEP_BARS):
            entry=i+ENTRY_DELAY_BARS
            exit_ix=entry+HORIZON_BARS
            # A bar is identified by its open timestamp: bar i is only known at i+5m.
            signal_available=utc.iloc[i]+pd.Timedelta(minutes=BAR_MINUTES)
            entry_ts=utc.iloc[entry]
            exit_ts=utc.iloc[exit_ix]
            if not(signal_available<entry_ts<exit_ts):continue
            if (entry_ts-signal_available)>pd.Timedelta(minutes=10):continue
            if (exit_ts-entry_ts)>pd.Timedelta(minutes=40):continue
            realized=float(np.log(opens[exit_ix]/opens[entry]))
            if not np.isfinite(realized) or abs(realized)>.15:
                continue # source discontinuity / erroneous corporate-action intraday candle
            x=encode_features(g,i,symbol)
            events.append({"country":str(g.country.iloc[0]),
                "symbol":symbol,"session":str(day),"phase":phases[day],
                "decision_utc":signal_available.isoformat(),
                "hypothetical_entry_utc":entry_ts.isoformat(),
                "hypothetical_exit_utc":exit_ts.isoformat(),
                "bar_index":int(i),"realized_log_return":realized,
                "x":x})
    return events

def prepare_sources(prices):
    events=[]
    for symbol,data in prices.items():events.extend(build_events(symbol,data))
    if not events:raise RuntimeError("No eligible data after causal feature construction")
    events.sort(key=lambda r:(r["decision_utc"],r["symbol"]))
    ph={p:sum(r["phase"]==p for r in events) for p in ("train","valid","test")}
    print("V26_CAUSAL_EVENTS",safe_json(ph),flush=True)
    if min(ph.values())<55:raise RuntimeError("Insufficient temporally separated events")
    return events,ph

def models():
    return {
        "ridge":make_pipeline(StandardScaler(),Ridge(alpha=RIDGE_ALPHA)),
        "gbdt":HistGradientBoostingRegressor(max_iter=70,max_leaf_nodes=7,
            min_samples_leaf=22,learning_rate=.05,l2_regularization=15.,
            random_state=SEED,early_stopping=False),
    }

def losses(y,p):
    err=np.asarray(y,float)-np.asarray(p,float)
    return float(np.mean(err*err))

def forecast_quality(subset,pred):
    y=np.array([r["realized_log_return"] for r in subset])
    p=np.array(pred,float)
    if len(y)!=len(p) or len(y)<2:raise ValueError("Invalid test set")
    rank=float(pd.Series(p).corr(pd.Series(y),method="spearman"))
    direction=float(np.mean(np.sign(p)==np.sign(y)))
    return {"n":len(y),"mse":round(losses(y,p),10),
        "zero_predictor_mse":round(losses(y,np.zeros_like(y)),10),
        "pearson":float(np.corrcoef(p,y)[0,1]) if np.std(p)>1e-12 else None,
        "spearman":rank if np.isfinite(rank) else None,
        "direction_accuracy":direction}

def policy_score(subset,pred,cost_bps):
    cost=cost_bps/10000
    realised=np.array([r["realized_log_return"] for r in subset])
    pred=np.array(pred,float)
    chosen=pred>cost
    # Non-overlap per security; opportunity-level net return in local equity currency.
    net=np.where(chosen,np.expm1(realised)-cost,0.0)
    avg=10000*float(net.mean())
    return {"roundtrip_assumed_bps":cost_bps,"trade_candidates":int(chosen.sum()),
            "opportunities":len(subset),"mean_net_bps_per_opportunity":round(avg,3),
            "mean_net_bps_per_trade":round(10000*float(net[chosen].mean()),3) if chosen.any() else None,
            "win_rate_on_trades":float(np.mean(net[chosen]>0)) if chosen.any() else None,
            "no_quote_or_fill_validation":True}

def session_cluster_ci(subset,pred,cost_bps,seed=SEED,nboot=1200):
    if len(subset)<30:return None
    p=np.asarray(pred,float)
    outcome=np.array([r["realized_log_return"] for r in subset],float)
    payoffs=np.where(p>cost_bps/10000,np.expm1(outcome)-cost_bps/10000,0.0)
    bygroup={}
    for i,r in enumerate(subset):
        key=(r["symbol"],r["session"])
        bygroup.setdefault(key,[]).append(i)
    groups=list(bygroup.values())
    if len(groups)<12:return None
    rng=np.random.default_rng(seed)
    means=[]
    for _ in range(nboot):
        indices=[idx for j in rng.integers(0,len(groups),len(groups))
                    for idx in groups[int(j)]]
        means.append(float(np.mean(payoffs[indices])*10000))
    return [round(float(z),3) for z in np.percentile(means,[2.5,97.5])]

def run_study(events):
    train=[r for r in events if r["phase"]=="train"]
    valid=[r for r in events if r["phase"]=="valid"]
    test=[r for r in events if r["phase"]=="test"]
    xx=np.stack([r["x"] for r in train])
    yy=np.array([r["realized_log_return"] for r in train])
    xv=np.stack([r["x"] for r in valid])
    yv=np.array([r["realized_log_return"] for r in valid])
    xt=np.stack([r["x"] for r in test])
    ytest=np.array([r["realized_log_return"] for r in test])
    trained=models()
    validation={"cash":{"mse":losses(yv,np.zeros_like(yv))}}
    predicted={"cash":np.zeros(len(test))}
    for name,model in trained.items():
        model.fit(xx,yy)
        val_pred=model.predict(xv)
        validation[name]={"mse":losses(yv,val_pred),
            "mean_prediction_bps":round(float(np.mean(val_pred)*10000),4)}
        predicted[name]=model.predict(xt)
    # We require >5% validation MSE gain over always-zero; else no-trade model.
    valid_zero=validation["cash"]["mse"]
    valid_winner=min(trained,key=lambda name:validation[name]["mse"])
    selected=(valid_winner if validation[valid_winner]["mse"]<.95*valid_zero else "cash")
    mom=np.array([.25*r["x"][2] for r in test])
    predicted["momentum_naive"]=mom
    baseline=np.zeros(len(test))
    predicted["cash"]=baseline
    print("V26_VALIDATION_SELECTION",safe_json({"valid_mse":validation,
        "selected":selected,"rule":"ML must beat cash MSE by at least 5% on validation"}),flush=True)
    evaluation={}
    for name,p in predicted.items():
        evaluation[name]={
            "prediction_quality":forecast_quality(test,p),
            "cost_hurdles":{str(k):policy_score(test,p,k) for k in RISK_FRICTION_BPS}
        }
    evaluation["selected_on_validation"]={
        "model":selected,
        "test_quality":evaluation[selected]["prediction_quality"],
        "selected_policy_net_20bps_ci_by_symbol_session":
           session_cluster_ci(test,predicted[selected],20)}
    # Persist every test row and forecast, but note these are EXPLORATORY now.
    output_rows=[]
    for i,r in enumerate(test):
        output_rows.append({k:v for k,v in r.items() if k!="x"}|
            {"forecasts":{name:float(arr[i]) for name,arr in predicted.items()}})
    return evaluation,validation,trained,selected,output_rows

def selftest():
    from datetime import timedelta
    # Synthetic known and future-shock bar sources. No network required.
    dates=pd.bdate_range("2026-10-01",periods=18)
    country,sym,mic=SAMPLE[0]
    rows=[]
    for n,day in enumerate(dates):
        origin=pd.Timestamp(day.date(),tz="America/New_York")+pd.Timedelta(hours=9,minutes=30)
        for j in range(78):
            x=100*(1+.001*n+.00025*math.sin((j+n)/7))
            rows.append({"Open":x,"High":x*1.0005,"Low":x*.9995,
                  "Close":x*(1+0.00005*math.cos(j)),
                  "Volume":2000+j%17,
                  "symbol":sym,"country":country,"mic":mic,
                  "session":str(day.date()),
                  "bar_start_utc":(origin+pd.Timedelta(minutes=j*5)).tz_convert("UTC").isoformat()})
    h=pd.DataFrame(rows)
    ev=build_events(sym,h)
    assert len(ev)>30
    one=next(r for r in ev if r["bar_index"]==WARMUP_BARS)
    assert pd.Timestamp(one["decision_utc"])<pd.Timestamp(one["hypothetical_entry_utc"])
    assert pd.Timestamp(one["hypothetical_entry_utc"])<pd.Timestamp(one["hypothetical_exit_utc"])
    # Changing all bars *after* the decision bar must never alter features.
    day=one["session"]
    modified=h.copy()
    later=(modified.session==day)&(modified.bar_start_utc>
        (pd.Timestamp(one["decision_utc"])-pd.Timedelta(minutes=5)).isoformat())
    modified.loc[later,["Open","Close","High","Low"]]*=2
    before=one["x"]
    candidate=encode_features(h[h.session==day].reset_index(drop=True),WARMUP_BARS,sym)
    after=encode_features(modified[modified.session==day].reset_index(drop=True),WARMUP_BARS,sym)
    assert np.array_equal(before,candidate) and np.array_equal(before,after)
    assert {r["phase"] for r in ev}=={"train","valid","test"}
    assert all(pd.Timestamp(r["hypothetical_exit_utc"]).date()>=
               pd.Timestamp(r["decision_utc"]).date() for r in ev)
    print("V26_CAUSAL_PIT_SELFTEST_PASS",safe_json({
      "synthetic_sessions":len(dates),"synthetic_nonoverlap_events":len(ev),
      "no_future_bars_in_features":True,"delayed_entry":True,"no_live_orders":True}),flush=True)

def main():
    selftest()
    prices,failures=fetch_all()
    RESULTS.mkdir(exist_ok=True,parents=True)
    raw=pd.concat(prices.values(),ignore_index=True)
    raw.to_parquet(RESULTS/"v26_intraday_raw_snapshot.parquet",index=False,compression="zstd")
    snapshot_hash=hashlib.sha256((RESULTS/"v26_intraday_raw_snapshot.parquet").read_bytes()).hexdigest()
    events,counts=prepare_sources(prices)
    metrics,validation,trained,selected,test_forecasts=run_study(events)
    if selected!="cash":
        joblib.dump(trained[selected],RESULTS/"v26_selected_intraday_ml.joblib")
    report={"version":"v26_intraday_11_countries_exploratory",
       "retrieved_at_utc":datetime.now(timezone.utc).isoformat(),
       "one_sample_symbol_per_country":True,"worldwide_investable_universe_covered":False,
       "data":"Yahoo 30d 5min historic OHLC, unadjusted, no bid/ask, no broker entitlement",
       "snapshot_sha256":snapshot_hash,"successful_samples":list(prices),
       "failed_samples":failures,"observation_rows":len(raw),
       "training_valid_test_n":counts,
       "label":"return from open of bar i+2 to open of bar i+8, 30-minute holding interval, decision after bar i close",
       "no_same_session_label_overlap":True,
       "first_24_bars_for_features_only":True,
       "sample_days_split_per_symbol":"first 60% train, next 20% validation, final 20% test",
       "forecasts_selected_without_test_information":True,
       "validation":validation,"study":metrics,
       "test_forecast_records":test_forecasts,
       "research_only":True,"trades_executed":0,"alpha_verified":False,
       "limits":[
           "Only 11 convenient liquid stocks, one from each country; not all global markets or company sizes",
           "30 calendar days of intraday bars has too few market regimes to establish generalizable alpha",
           "5-minute OHLC bars have no contemporaneous bid/ask, market depth or order routing confirmations",
           "Execution delay is modelled as an open after the bar close, but latency and slippage not empirically verified",
           "20bps and 60bps roundtrip costs are scenarios, not actual broker charges",
           "Fees, FX conversion and taxes are excluded from the return labels",
           "Local currency mean opportunity return is NOT a UK GBP portfolio CAGR",
           "No universal market listing registry, broker permission, corporate action or delisted data",
           "Evaluation region/period is exploratory and cannot constitute an untouched future sample",
           "No live portfolio allocation, connection or financial recommendation"
       ]}
    (RESULTS/"v26_intraday_ml_research.json").write_text(safe_json(report))
    print("V26_COMPLETED",safe_json({
       "venue_success":len(prices),
       "raw_bars":len(raw),"counts":counts,"selected_model":selected,
       "test_prediction_mse":{k:v["prediction_quality"]["mse"] for k,v in metrics.items()
                               if "prediction_quality" in v},
       "20bps_net":{k:v["cost_hurdles"]["20"]["mean_net_bps_per_opportunity"]
                   for k,v in metrics.items() if "cost_hurdles" in v},
       "no_live_orders":True}),flush=True)

if __name__=="__main__":
    main()
