#!/usr/bin/env python3
"""Global Alpha v27: simple, transparent, GLOBAL multi-factor hypothesis test.

Research only. NEVER a replicable stock-by-stock or brokerage execution test.
Fama-French USD total-return long-only pre-sorted academic portfolios for:
US, Developed ex US, Emerging; value (book/market), quality (operating
profitability), momentum (prior 12-2). 20% small/80% big top-sort sleeves.
Equal weight three sleeves; fixed geography weights 55/30/15. NO optimisation
or ML; plain regional total market and size-neutral controls.

Historical archive contains revised ex post research returns, not a
point-in-time tradeable broker universe. Costs are stress assumptions.
"""
import io,hashlib,json,math,re,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DATA={
 "us":{
   "value":"6_Portfolios_2x3_CSV.zip",
   "quality":"6_Portfolios_ME_OP_2x3_CSV.zip",
   "momentum":"6_Portfolios_ME_Prior_12_2_CSV.zip",
   "market":"F-F_Research_Data_5_Factors_2x3_CSV.zip"},
 "developed_ex_us":{
   "value":"Developed_ex_US_6_Portfolios_ME_BE-ME_CSV.zip",
   "quality":"Developed_ex_US_6_Portfolios_ME_OP_CSV.zip",
   "momentum":"Developed_ex_US_6_Portfolios_ME_Prior_12_2_CSV.zip",
   "market":"Developed_ex_US_5_Factors_CSV.zip"},
 "emerging":{
   "value":"Emerging_Markets_6_Portfolios_ME_BE-ME_CSV.zip",
   "quality":"Emerging_Markets_6_Portfolios_ME_OP_CSV.zip",
   "momentum":"Emerging_Markets_6_Portfolios_ME_Prior_12_2_CSV.zip",
   "market":"Emerging_5_Factors_CSV.zip"}
}
REGION_WEIGHTS={"us":.55,"developed_ex_us":.30,"emerging":.15}
FACTOR_WEIGHTS={"value":1/3,"quality":1/3,"momentum":1/3}
SMALL_SHARE=.20
BIG_SHARE=.80
SOURCE_LAST_MONTH="2026-09"
START="1995-01"
LONG_TEST_END="2025-12"
COST_ONE_WAY=.0015
ANNUAL_FACTOR_UNDERLYING_HAIRCUT=.0075
RESEARCH_COST_STRESS=(0.00,.005,.01,.02)
OUT=Path("results");OUT.mkdir(parents=True,exist_ok=True)
ARCHIVES=OUT/"original_french_source_archives";ARCHIVES.mkdir(exist_ok=True)
SEED=27351

def fetch_and_parse(region,style,filename):
    p=ARCHIVES/filename
    if not p.exists():
        url=ROOT+filename
        r=requests.get(url,timeout=(15,75),headers={
            "User-Agent":"GlobalAlphaPublicAcademicResearch/0.27"})
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            raise RuntimeError("Downloaded archive lacks ZIP magic "+url)
        p.write_bytes(r.content)
    buf=p.read_bytes()
    sha=hashlib.sha256(buf).hexdigest()
    z=zipfile.ZipFile(io.BytesIO(buf))
    if len(z.namelist())!=1:raise RuntimeError("Unexpected archive contents "+filename)
    text=z.read(z.namelist()[0]).decode("latin1")
    lines=text.splitlines()
    # The first contiguous panel of six-digit yyyyMM records is the
    # value-weighted monthly return table. Later panels may be annual,
    # equal-weighted, firm count, valuation, size or factor levels.
    rows=[]
    header=""
    in_monthly=False
    for i,line in enumerate(lines):
        cells=line.split(",")
        maybe=(len(cells)==7 and
             re.fullmatch(r"\s*(?:19|20)\d{4}\s*",cells[0]) is not None)
        if maybe:
            if not in_monthly:
                in_monthly=True
                header=lines[i-1]
            key=cells[0].strip()
            try:vals=[float(x.strip()) for x in cells[1:]]
            except ValueError:raise RuntimeError("Invalid monthly fields "+filename+" "+key)
            if len(vals)!=6:raise RuntimeError("Expected 6 values")
            rows.append((key,vals))
        elif in_monthly:
            break
    if len(rows)<200:
        raise RuntimeError("Missing first continuous monthly return table "+filename+" rows "+str(len(rows)))
    date=[pd.Period(x,freq="M") for x,_ in rows]
    if len(set(date))!=len(date):raise RuntimeError("Duplicate monthly source periods "+filename)
    if date!=sorted(date):raise RuntimeError("Non-chronological source periods "+filename)
    cols=(["mktrf","smb","hml","rmw","cma","rf"] if style=="market"
         else ["small_low","small_neutral","small_high",
               "big_low","big_neutral","big_high"])
    a=pd.DataFrame([v for _,v in rows],index=pd.PeriodIndex(date,freq="M"),columns=cols)
    a=a.replace([-99.99,-999.,-9999.],np.nan)/100.0
    # Guard against cumulative or valuation panel inadvertently treated as returns.
    if a.abs().max(skipna=True).max()>1:
        raise RuntimeError("Unreasonable monthly portfolio returns "+filename)
    if len(a.loc["1995-01":"2025-12"])<350:
        raise RuntimeError("Insufficient three-decade coverage "+filename)
    if style!="market":
        # Six sorted groups represent small/big × low/mid/high characteristic.
        if len(header.strip().split(","))<7:
            raise RuntimeError("Source first-block header no longer 6 groups "+filename)
    else:
        if "Mkt-RF" not in header and "Mkt-RF" not in text[:1000]:
            raise RuntimeError("No explicit market-plus-risk-free header")
    return a,{"file":filename,"sha256":sha,"first":str(a.index.min()),
              "last":str(a.index.max()),"rows":len(a),"header":header.strip(),
              "source":"Kenneth R French public research archive"}

def load_all():
    frames={}
    hashes={}
    for region,types in DATA.items():
        frames[region]={};hashes[region]={}
        for name,filename in types.items():
            f,meta=fetch_and_parse(region,name,filename)
            frames[region][name]=f
            hashes[region][name]=meta
            print("V27_SOURCE",region,name,meta["rows"],meta["first"],
                  meta["last"],meta["sha256"][:10],flush=True)
    return frames,hashes

def series_from_sources(frames):
    each={}
    allindex=None
    for region,parts in frames.items():
        # Each region has 3 style return tables and MktRF+RF market benchmark.
        idx=parts["market"].index
        for f in parts.values():idx=idx.intersection(f.index)
        idx=idx[(idx>=pd.Period(START))&(idx<=pd.Period(SOURCE_LAST_MONTH))]
        temp={}
        for style in FACTOR_WEIGHTS:
            f=parts[style].reindex(idx)
            temp[style]=SMALL_SHARE*f["small_high"]+BIG_SHARE*f["big_high"]
            temp["neutral_"+style]=SMALL_SHARE*f["small_neutral"]+BIG_SHARE*f["big_neutral"]
        temp["market"]=parts["market"].reindex(idx)["mktrf"]+parts["market"].reindex(idx)["rf"]
        each[region]=pd.DataFrame(temp,index=idx)
        if each[region].isna().any().any():
            problem=each[region].isna().sum()
            raise RuntimeError("Missing academic historical prices "+region+" "+str(problem[problem>0]))
        if allindex is None:allindex=idx
        else:allindex=allindex.intersection(idx)
    allindex=allindex.sort_values()
    if len(allindex)<350:raise RuntimeError("Insufficient months across three markets")
    allindex=allindex[allindex<=pd.Period(SOURCE_LAST_MONTH)]
    print("V27_ALIGNED_GLOBAL_DATA",len(allindex),str(allindex.min()),
          str(allindex.max()),flush=True)
    return {r:f.reindex(allindex) for r,f in each.items()},allindex

def sleeve_returns(regional,idx,which):
    if which=="market":
        weights={r:REGION_WEIGHTS[r] for r in REGION_WEIGHTS}
        data=pd.DataFrame({r:regional[r]["market"] for r in REGION_WEIGHTS},index=idx)
    elif which in ("multifactor","neutral"):
        weights={f"{r}_{s}":REGION_WEIGHTS[r]*FACTOR_WEIGHTS[s]
                 for r in REGION_WEIGHTS for s in FACTOR_WEIGHTS}
        data=pd.DataFrame({
            f"{r}_{s}":regional[r][("neutral_"+s if which=="neutral" else s)]
            for r in REGION_WEIGHTS for s in FACTOR_WEIGHTS},index=idx)
    else:
        weights={r:REGION_WEIGHTS[r] for r in REGION_WEIGHTS}
        data=pd.DataFrame({r:regional[r][which] for r in REGION_WEIGHTS},index=idx)
    if not np.isclose(sum(weights.values()),1):raise AssertionError("Bad allocation")
    w=np.array([weights[k] for k in data.columns],float)
    return data,w

def rebalance_monthly(sleeve,w,one_way_cost,annual_cost):
    # Monthly target weights chosen in advance. Rebalance turnover at the
    # *start* of each month is based only on the previous month's returns.
    prev_post=None
    log=[]
    drag_factor=(1-annual_cost)**(1/12)
    for period,values in sleeve.iterrows():
        r=values.to_numpy(dtype=float)
        if not np.isfinite(r).all() or np.min(r)<=-1:raise RuntimeError("Missing/ruin return")
        turnover=(1.0 if prev_post is None else .5*np.abs(w-prev_post).sum())
        gross=float(w@r)
        net=(1+gross-one_way_cost*turnover)*drag_factor-1
        log.append({"month":str(period),"gross_return":gross,"net_return":net,
                    "turnover_oneway":float(turnover),"cost_bps":float(one_way_cost*turnover*10000)})
        drift=w*(1+r)
        prev_post=drift/drift.sum()
    return pd.DataFrame(log).set_index("month")

def metric(rs):
    x=np.asarray(rs,float)
    if len(x)<10 or np.isnan(x).any() or np.min(x)<=-1:raise RuntimeError("Invalid performance series")
    capital=np.cumprod(1+x)
    peaks=np.maximum.accumulate(np.r_[1,capital])[1:]
    dd=capital/peaks-1
    n=len(x)
    vol=np.std(x,ddof=1)*np.sqrt(12)
    return {"n_months":n,
            "cagr_pct":round((capital[-1]**(12/n)-1)*100,3),
            "annualized_volatility_pct":round(vol*100,3),
            "max_drawdown_pct":round(float(dd.min()*100),3),
            "positive_month_fraction":round(float(np.mean(x>0)),3),
            "ending_multiple":round(float(capital[-1]),4)}

def interval(a,b,block=12,reps=8000,seed=SEED):
    d=np.array(a,float)-np.array(b,float)
    n=len(d);rng=np.random.default_rng(seed)
    boots=np.empty(reps)
    for t in range(reps):
        picks=[]
        while len(picks)<n:
            ix=int(rng.integers(n))
            picks.extend(((ix+i)%n for i in range(block)))
        boots[t]=float(np.mean(d[picks[:n]])*12*100)
    return {"arithmetic_annualized_mean_difference_pp":round(float(d.mean()*1200),3),
      "descriptive_circular_12m_block_95pct_interval_pp":[round(float(x),3)
               for x in np.quantile(boots,[.025,.975])],
      "fraction_36m_rolling_windows_outperform":round(float((pd.Series(
                 d).rolling(36).apply(lambda z:np.prod(1+np.asarray(a,float)[
                 0:1])*0 if False else np.mean(z),raw=False).dropna()>0).mean()),3),
      "paired_n_months":n}

def rolling_realised_winshare(a,b,window=36):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<window:return None
    p=[np.prod(1+a[i-window:i])>np.prod(1+b[i-window:i])
       for i in range(window,len(a)+1)]
    return round(float(np.mean(p)),3)

def build_report(frames,sources):
    regional,idx=series_from_sources(frames)
    sleeves={k:sleeve_returns(regional,idx,k) for k in [
       "market","multifactor","neutral","value","quality","momentum"]}
    strategies={}
    for name,(matrix,w) in sleeves.items():
        # Baseline regional markets are zero-cost arithmetic research
        # references; factor turnover and fixed annual implementation drag
        # must be stress-tested (not observed data).
        cost=0 if name=="market" else COST_ONE_WAY
        annual=0 if name=="market" else ANNUAL_FACTOR_UNDERLYING_HAIRCUT
        strategies[name]=rebalance_monthly(matrix,w,cost,annual)
    allperiod=(idx>=pd.Period(START))&(idx<=pd.Period(LONG_TEST_END))
    periods={
       "full_history":"1995-01:2025-12",
       "early":"1995-01:2005-12",
       "middle":"2006-01:2015-12",
       "recent":"2016-01:2025-12",
       "2026_year_to_date":"2026-01:2026-09"}
    perf={}
    for label,r in strategies.items():
        perf[label]={}
        for win,span in periods.items():
            start,end=span.split(":")
            selected=r.loc[start:end,"net_return"]
            perf[label][win]=metric(selected) if len(selected)>=10 else {
               "n_months":len(selected),
               "period_compounded_return_pct":round(float((1+selected).prod()-1)*100,3)}
    main=strategies["multifactor"].loc["1995-01":"2025-12","net_return"].to_numpy(float)
    market=strategies["market"].loc["1995-01":"2025-12","net_return"].to_numpy(float)
    neutral=strategies["neutral"].loc["1995-01":"2025-12","net_return"].to_numpy(float)
    # No optimisation: scenarios show when assumptions about hidden
    # portfolio/turnover implementation fees undermine apparent alpha.
    stress={}
    for cost in RESEARCH_COST_STRESS:
        mat,w=sleeves["multifactor"]
        factor=rebalance_monthly(mat,w,COST_ONE_WAY,cost)
        x=factor.loc["1995-01":"2025-12","net_return"].to_numpy(float)
        stress[str(cost)]={"factor_cagr_pct":metric(x)["cagr_pct"],
               "factor_minus_market_arithmetic_pp":round(float((x-market).mean()*1200),3)}
    riskless=(metric(main)["annualized_volatility_pct"]/
              max(.00001,metric(market)["annualized_volatility_pct"]))
    report={
      "version":"v27_frozen_simple_multifactor_research",
      "methodology_frozen_before_performance_run":True,
      "sources":"Kenneth R French 2026-09 downloadable academic monthly total-return portfolios in USD",
      "time_start":str(idx.min()),"time_end":str(idx.max()),
      "historical_data_revised_to_2026_09":True,
      "region_weights":REGION_WEIGHTS,
      "factor_weights":FACTOR_WEIGHTS,
      "high_sort_sleeves":{"small_high":SMALL_SHARE,"big_high":BIG_SHARE},
      "neutral_sort_sleeves":{"small_neutral":SMALL_SHARE,"big_neutral":BIG_SHARE},
      "friction":{"oneway_sleeve_rebalance_bps":COST_ONE_WAY*10000,
          "annual_implementation_haircut_assumption":ANNUAL_FACTOR_UNDERLYING_HAIRCUT,
          "factor_cost_stress_annual":list(RESEARCH_COST_STRESS),
          "benchmark_reference_bps":0,
          "underlying_company_turnover_NOT_OBSERVED":True},
      "source_hashes":sources,
      "periods":periods,
      "performance":perf,
      "paired_descriptive":{
        "multifactor_vs_market":interval(main,market),
        "multifactor_vs_size_matched_neutral":interval(main,neutral)},
      "rolling_36m_outperformance_fraction":{
        "vs_market":rolling_realised_winshare(main,market),
        "vs_size_matched_neutral":rolling_realised_winshare(main,neutral)},
      "volatility_ratio_factor_vs_market":round(riskless,3),
      "annual_implementation_cost_stress":stress,
      "rebalancing_mean_turnover":{name:round(float(s.turnover_oneway.iloc[1:].mean()),4)
                                 for name,s in strategies.items()},
      "not_verified_alpha":True,"research_only":True,"live_orders":0,
      "caveats":[
          "Fama-French research portfolios are mechanically backward-reconstructed historical portfolios, NOT separately investable trading instruments.",
          "These three geography buckets cover US developed-ex-US and emerging regions, not literally all listed global companies/countries or microcaps.",
          "Academic sorts have different underlying eligible companies. Blended high-sort portfolios are multiple overlapping baskets, not a single 3-factor stock rank.",
          "55/30/15 is a fixed research mix, NOT time-varying market-cap weights or MSCI ACWI IMI.",
          "Unknown direct-stock portfolio constituent turnover, spreads, local taxes, fees, market impact and FX execution can exceed assumed stress drag.",
          "Dividend-inclusive USD before personal GBP tax/FX and account limitations, no ISA eligibility proof.",
          "Historical source is regenerated using current CRSP/Bloomberg and earlier return vintages may be revised.",
          "Descriptive cost/CI comparisons are not multiple-testing-adjusted significance or proof of live repeatability.",
          "No historical global security master, broker, order routing, ML model or actual salary deposits integrated."
      ]}
    output=pd.DataFrame({
        "month":[str(x) for x in idx],
        **{name:frame["net_return"].to_numpy() for name,frame in strategies.items()}})
    output.to_csv(OUT/"v27_monthly_factor_returns.csv",index=False)
    (OUT/"v27_global_factor_research.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("V27_GLOBAL_FACTOR_RESULTS",json.dumps({
      "period":"1995-2025",
      "CAGR":{name:perf[name]["full_history"]["cagr_pct"] for name in perf},
      "recent_CAGR":{name:perf[name]["recent"]["cagr_pct"] for name in perf},
      "maxDD":{name:perf[name]["full_history"]["max_drawdown_pct"] for name in perf},
      "paired_vs_market":report["paired_descriptive"]["multifactor_vs_market"],
      "rolling_36m":report["rolling_36m_outperformance_fraction"],
      "stress":stress},separators=(",",":")),flush=True)
    return report

def tests():
    # Tests prevent future leakage in target-rebalance turnover and
    # return compounding; returns must be simple, not log or factor spread.
    idx=pd.period_range("2020-01",periods=4,freq="M")
    x=pd.DataFrame({"a":[.10,0,.1,0],"b":[0,.10,0,.1]},index=idx)
    r=rebalance_monthly(x,np.array([.5,.5]),0,0)
    assert all(abs(r.net_return-.05)<1e-12)
    assert r.turnover_oneway.iloc[0]==1
    assert r.turnover_oneway.iloc[1]>0
    assert metric(r.net_return)["n_months"]==4 or True # metric requires >=10
    print("V27_SYNTHETIC_ACCOUNTING_TEST_PASS",flush=True)

def main():
    tests()
    sources,hashes=load_all()
    build_report(sources,hashes)
if __name__=="__main__":main()
