#!/usr/bin/env python3
"""Global Alpha v23 -- fixed-prediction v21 return recovery, no parameter tuning.

The original models, labels and forecasts are frozen. ONLY held-out realised
returns missing because S&P archive cuts off after index exit are supplemented.
Twelve stocks have Yahoo adjusted-price overlap audited in v22 (24 stock/month
pairs); QRVO uses separately disclosed 3 monthly first-trading-day opens. The
second source is lower assurance and MUST be flagged as a sensitivity estimate.

2024-25 history already inspected. EXPERIMENTAL, no validated alpha or trades.
"""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import v21_expanded_sec_signal_pilot as baseline
import v14_frozen_holdout as price

EXIT_AUDIT=Path("source_v22/results/v22_exit_price_continuity_audit.json")
BASELINE=Path("source_v21/results/v21_exploratory_ablation.json")
TTM=baseline.SRC
OUT=Path("results")
OUT.mkdir(parents=True,exist_ok=True)
QRVO_MONTHLY_OPEN={"2024-12":69.32,"2025-01":70.72,"2025-02":81.78}
QRVO_SOURCE="https://www.statmuse.com/money/ask/when-did-qorvo-go-public"
EXPECTED_MISSING_PAIRS=26

def read_bridge():
    audit=json.loads(EXIT_AUDIT.read_text())
    if audit["total_symbols"]!=13 or audit["missing_decision_pairs"]!=26:
        raise AssertionError("Different v22 source audit; fail closed")
    verified={}
    for key,value in audit["resolved_return_lookup"].items():
        symbol,month=key.split("|")
        if not audit["symbols"][symbol]["verified"]:
            raise AssertionError("Unverified source marked as passed")
        verified[(symbol,month)]=float(value)
    if len(verified)!=24:raise AssertionError("Unexpected audited return coverage")
    # The public monthly QRVO open series agrees to the cent with the original
    # 2024-12-02 first-day adjusted open in the pinned S&P source.
    q=price.load("sp500","QRVO")
    anchor=float(q.loc["2024-12","open"])
    if abs(anchor-QRVO_MONTHLY_OPEN["2024-12"])>0.02:
        raise AssertionError("QRVO independent monthly open fails December archive anchor")
    manual={("QRVO","2024-11"):QRVO_MONTHLY_OPEN["2025-01"]/QRVO_MONTHLY_OPEN["2024-12"]-1,
            ("QRVO","2024-12"):QRVO_MONTHLY_OPEN["2025-02"]/QRVO_MONTHLY_OPEN["2025-01"]-1}
    return verified,manual,anchor

def bootstrap_diff(a,b,seed=38117,repetitions=20000):
    if len(a)!=len(b):raise AssertionError("Unequal evaluation samples")
    z=np.array(a,float)-np.array(b,float)
    rng=np.random.default_rng(seed)
    values=np.empty(repetitions)
    for k in range(repetitions):
        idx=[]
        while len(idx)<len(z):
            t=int(rng.integers(0,len(z)))
            idx.extend(((t+i)%len(z) for i in range(3)))
        values[k]=12*np.mean(z[np.array(idx[:len(z)])])
    return {"annualized_arithmetic_mean_difference_pp":round(1200*float(np.mean(z)),2),
            "descriptive_95pct_three_month_block_interval_pp":[round(100*x,2)
                                for x in np.quantile(values,[0.025,0.975])],
            "wins_months":int((z>0).sum()),
            "months":len(z)}

def patched_performance():
    verified,manual,qrvo_anchor=read_bridge()
    # Read the same 709 original price series and 2022-25 SEC facts as v21.
    if not TTM.exists():raise RuntimeError("Exact v20B PIT original financial features missing")
    factsdata=pd.read_parquet(TTM)
    mapping={(r.ticker,r.month):r._asdict() for r in factsdata.itertuples(index=False)}
    stocks,index=baseline.load_prices()
    panels=baseline.panels(stocks,index,mapping)
    fixes={**verified,**manual}
    all_missing={(r["symbol"],str(p["date"])) for p in panels
        if baseline.E0<=p["date"]<=baseline.E1 for r in p["rows"]
        if r["realized"] is None}
    unknown=all_missing-set(fixes)
    if unknown:raise RuntimeError("Unresolved evaluation labels remain: "+str(sorted(unknown))[:500])
    # Critical: preserve original training labels and original forecasts.
    # Annotate only evaluation rows; wrappers create copies with corrected
    # observed returns after forecasting, not changed training observations.
    for panel in panels:
        for row in panel["rows"]:row["_decision_month"]=str(panel["date"])
    orig_holdings=baseline.holdings
    orig_rank_ic=baseline.rank_ic
    patched_seen=set()
    def evaluation_rows(rows):
        transformed=[]
        for row in rows:
            key=(row["symbol"],row["_decision_month"])
            if key in fixes and row["realized"] is None:
                changed=dict(row)
                changed["realized"]=fixes[key]
                patched_seen.add(key)
                transformed.append(changed)
            else:transformed.append(row)
        good=[r["realized"] for r in transformed if r["realized"] is not None
               and np.isfinite(r["realized"]) and r["realized"]>-1]
        mean=float(np.mean(good)) if good else None
        return [{**r,"label":r["realized"]-mean if mean is not None and
             r["realized"] is not None and np.isfinite(r["realized"]) and
             r["realized"]>-1 else None} for r in transformed]
    def fixed_holdings(rows,scores,prior,mode):
        return orig_holdings(evaluation_rows(rows),scores,prior,mode)
    def fixed_rank_ic(rows,scores):
        return orig_rank_ic(evaluation_rows(rows),scores)
    baseline.holdings=fixed_holdings
    baseline.rank_ic=fixed_rank_ic
    try:
        histories,qual=baseline.backtest(panels)
    finally:
        baseline.holdings=orig_holdings
        baseline.rank_ic=orig_rank_ic
    missing=dict((k,[r["month"] for r in v if r["ret"] is None])
                 for k,v in histories.items())
    if any(missing.values()):raise RuntimeError("Still have incomplete portfolios: "+str(missing))
    if len(patched_seen)!=len(all_missing):
        raise AssertionError(f"Not all observed outcome gaps were bridged: {len(patched_seen)}/{len(all_missing)}")
    # Assure the same predictions were selected as the earlier failed backtest,
    # by comparing every monthly turnover to the archived v21 model run.
    original=json.loads(BASELINE.read_text())
    for name in histories:
        for now,old in zip(histories[name],original["methods"][name]["monthly"]):
            if now["month"]!=old["month"] or abs(now["turnover"]-old["turnover"])>1e-10:
                raise AssertionError("A strategy holding changed; this was supposed to be return-only repair")
    stats={name:baseline.calc(vals) for name,vals in histories.items()}
    rank_stats={}
    for name,vals in histories.items():
        ic=[x["spearman_ic_observed"] for x in vals
            if x["spearman_ic_observed"] is not None and np.isfinite(x["spearman_ic_observed"])]
        rank_stats[name]={"rank_ic_n_months":len(ic),
                          "mean_rank_ic":round(float(np.mean(ic)),5) if ic else None,
                          "positive_ic_months":int((np.array(ic)>0).sum()) if ic else None}
    expected=len(histories["price_ridge"])
    if expected!=18:raise RuntimeError("Unexpected evaluation months "+str(expected))
    comparison={}
    for name in ("fund_ridge","fund_plus_peers","fund_plus_shuffled_peers","momentum"):
        comparison[name+"_vs_price"] = bootstrap_diff(
            [r["ret"] for r in histories[name]],
            [r["ret"] for r in histories["price_ridge"]])
    spy=original["SPY_total_return_research_benchmark"]
    report={
      "version":"v23_historical_return_source_bridge",
      "evaluated":"2024-01 to 2025-06, already inspected / NOT a fresh holdout",
      "not_independent_alpha_evidence":True,
      "model_fit_changed":False,"stock_selection_changed":False,
      "repaired_observed_label_pairs":len(patched_seen),
      "price_overlap_verified_return_pairs":len(verified),
      "manually_sourced_monthly_QRVO_pairs":len(manual),
      "price_archive_commit":price.PIN,
      "QRVO_monthly_open_prices":QRVO_MONTHLY_OPEN,
      "QRVO_citation_source":QRVO_SOURCE,
      "QRVO_archive_first_dec_2024_open":qrvo_anchor,
      "repaired_return_lookup":{s+"|"+m:float(v) for (s,m),v in fixes.items()},
      "methods":{name:{"stats":stats[name],"observed_ic":rank_stats[name],
                       "monthly":rows} for name,rows in histories.items()},
      "SPY_total_return_research_benchmark":spy,
      "paired_descriptive_arithmetic_advantage":comparison,
      "limitations":[
        "Only the original archive-missing realized trade outcomes were repaired, not training labels or any forecasts.",
        "All model predictions and position changes must match v21 exactly (asserted).",
        "12 tickers (24 decision pairs) were validated against independent Yahoo daily adjusted return overlaps.",
        "QRVO monthly first-open source was compared with archived December first-open; not 25 daily overlapping return checks.",
        "Jan and Feb 2025 QRVO monthly open values derive from a current external public page; not frozen point-in-time source.",
        "Archive ex-index gaps are not proof of delisting and training labels in 2022-23 can still have missing exits.",
        "Historical membership, old-to-new CIK/company identity, corporate actions, tax/ISA restrictions are not certified.",
        "Post-inspection repair and small 18-month sample cannot establish repeatable alpha.",
        "USD-only, assumed 20bps one-way turnover, SPY dividend-adjusted passive comparator.",
        "UK salary contributions and live trading deliberately disabled."
      ],
      "alpha_proven":False,"orders_sent":0}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"v23_patched_original_predictions.json").write_text(json.dumps(report,indent=2,default=str))
    print("V23_RETURN_REPAIR_RESULTS",json.dumps({
      "model_fit_changed":False,"patched":len(patched_seen),
      "QRVO_returns":{sym+"|"+mon:float(value) for (sym,mon),value in manual.items()},
      "strategies":{k:{"annualized_return_pct":v.get("cagr_usd_pct"),
                        "max_drawdown_pct":v.get("max_drawdown_pct"),
                        "meanIC":rank_stats[k]["mean_rank_ic"]} for k,v in stats.items()},
      "SPY":spy["metrics"],"paired":comparison},default=str,separators=(",",":")),flush=True)

if __name__=="__main__":
    patched_performance()
