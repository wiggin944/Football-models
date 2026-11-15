#!/usr/bin/env python3
"""Global Alpha v23 — v21 re-evaluation after independent source-verified exit returns.

STRICTLY exploratory on previously inspected January 2024–June 2025.
Every method uses identical decision-date universe and archived fundamentals.
Verified Yahoo price ratios are used ONLY to repair observed historical labels,
not for feature construction or to decide which stocks enter the test.
Unresolved QRVO exit returns stay MISSING and invalidate full CAGR claims.
"""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
import v21_expanded_sec_signal_pilot as baseline

AUDIT=Path("source/v22/results/v22_exit_price_continuity_audit.json")
FACTS=Path("source/v20/results/v20b_full_us_asfiled_ttm_peers.parquet")
RESULTS=Path("results")
METHODS=baseline.ALL_MODELS

def paired_diagnostic(a,b):
    joined=[]
    for x,y in zip(a,b):
        if x["month"]!=y["month"]:raise ValueError("Months mismatched")
        if x["ret"] is None or y["ret"] is None:continue
        joined.append((x["month"],float(x["ret"])-float(y["ret"])))
    if len(joined)<10:return {"comparable_months":len(joined),"insufficient":True}
    diffs=np.array([x[1] for x in joined])
    rng=np.random.default_rng(20032026)
    blocks=[]
    for rep in range(12000):
        idx=[]
        while len(idx)<len(diffs):
            k=int(rng.integers(0,len(diffs)))
            idx.extend((k+j)%len(diffs) for j in range(3))
        blocks.append(1200*float(np.mean(diffs[np.array(idx[:len(diffs)])])))
    return {"months_compared":[x[0] for x in joined],
            "comparable_months":len(joined),
            "winning_months":int(sum(x>0 for x in diffs)),
            "observed_arithmetic_annualised_pp":round(1200*float(np.mean(diffs)),3),
            "descriptive_95pct_block_interval_pp":[round(float(x),3) for x in np.quantile(blocks,[.025,.975])],
            "not_unbiased_inference":True}

def main():
    if not AUDIT.exists() or not FACTS.exists():
        raise RuntimeError("Required pinned verified v22 and v20 audits absent")
    audit=json.loads(AUDIT.read_text())
    if audit["verified_symbols"]<12 or audit["repaired_decision_pairs"]<24:
        raise RuntimeError("Invalid or incomplete verified audit source")
    fallback=audit["resolved_return_lookup"]
    if any(k.startswith("QRVO|") for k in fallback):
        raise RuntimeError("Do not pretend QRVO Yahoo quotes were validated")
    data=pd.read_parquet(FACTS)
    if data.duplicated(["ticker","month"]).any():raise AssertionError("Historical SEC identity duplicated")
    fundamentals={(r.ticker,r.month):r._asdict() for r in data.itertuples(index=False)}
    print("V23_LOADED_SECDATA",len(data),flush=True)
    assets,index=baseline.load_prices()
    idmap={id(series):name for name,series in assets.items()}
    original=baseline.price.holding
    repaired=0
    repair_log=[]
    def price_with_verified_exit(stock,month):
        nonlocal repaired
        v=original(stock,month)
        if v is not None:return v
        symbol=idmap.get(id(stock))
        key=f"{symbol}|{month}"
        if key not in fallback:return None
        # Only repair a specified, independently source-validated missing label.
        val=float(fallback[key])
        if not np.isfinite(val) or val<=-1:raise AssertionError("Invalid source return")
        repaired+=1
        repair_log.append({"ticker":symbol,"month":month,"source":"v22 daily adjusted-price overlap audit"})
        return val
    baseline.price.holding=price_with_verified_exit
    try:
        built=baseline.panels(assets,index,fundamentals)
        logs,cohorts=baseline.backtest(built)
    finally:
        baseline.price.holding=original
    RESULTS.mkdir(parents=True,exist_ok=True)
    comparable={}
    for name in ("fund_ridge","fund_plus_peers","fund_plus_shuffled_peers","price_ridge"):
        comparable[name+"_minus_momentum"]=paired_diagnostic(logs[name],logs["momentum"])
    for a,b in (("fund_ridge","price_ridge"),
                ("fund_plus_peers","fund_ridge"),
                ("fund_plus_peers","fund_plus_shuffled_peers")):
        comparable[a+"_minus_"+b]=paired_diagnostic(logs[a],logs[b])
    data={
        "version":"v23_post_index_departure_repaired_exploratory_replay",
        "parent_model":"v21 frozen ridge predictions (fitted from identical source and settings)",
        "source_v22_verified":audit["verified_symbols"],
        "source_v22_repaired_pairs":len(fallback),
        "unresolved_source":"QRVO Yahoo returns unavailable; independent Stooq blocked by JS gate",
        "repaired_label_accesses":repaired,
        "repair_log":repair_log,
        "metrics":{name:baseline.calc(rows) for name,rows in logs.items()},
        "month_by_month":logs,
        "pairwise_diagnostics":comparable,
        "cohort_eligibility":cohorts,
        "no_alpha_claim":True,"no_order_execution":True,
        "limitations":[
          "Historical labels repaired only for 12 independently verified securities.",
          "Index member exit does not necessarily imply delisting.",
          "No fabricated QRVO price; full-period ML CAGR must remain null if QRVO selected.",
          "Previously inspected 2024-2025 sample; bootstrap intervals are descriptive.",
          "Some 2024-25 historical labels remain undiscovered after universe archive termination.",
          "All stock-adjusted OHLC must be audited for security identity, splits, spin-offs and dividend semantics.",
          "No global factor model, UK account eligibility, allocation or live broker execution."
        ]}
    (RESULTS/"v23_repaired_exploratory_return_audit.json").write_text(json.dumps(data,indent=2,allow_nan=False))
    print("V23_EXIT_REPAIR_RESULTS",json.dumps({
      "repaired_label_accesses":repaired,
      "method_performance":data["metrics"],
      "paired":{k:{"months":v.get("comparable_months"),"annual_diff_pp":v.get("observed_arithmetic_annualised_pp"),
                  "interval":v.get("descriptive_95pct_block_interval_pp")}
                for k,v in comparable.items()}
    },allow_nan=False,separators=(",",":")),flush=True)

if __name__=="__main__":main()
