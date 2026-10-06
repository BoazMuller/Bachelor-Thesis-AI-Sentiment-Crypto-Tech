"""Reproduce the October supervisor revision using the existing thesis models.

Run after collect_yahoo_finance.py --tickers CSCO BKNG MDLZ --output-csv
 data/raw/finance/placebo_yahoo_2024_04_01_to_2026_03_31.csv.
Existing benchmark outputs and sentiment estimates remain the reference inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/src'))
sys.path.insert(0, str(ROOT / 'scripts/reporting'))
from thesis.tables.tvpvar_connectedness import TvpvarSystem, tvpvar_lag_selection, selected_lags
from thesis.tables.common import EXPECTATION_ADJUSTED_AIS as EAIS, read_daily_time_series, regression_table
from thesis.tables.spillover_regressions import table_28_lagged_regressions, CONNECTEDNESS_REGRESSION_CONTROLS
from thesis.modeling.diagnostics import one_lag_granger_test, stationarity_tests
from plot_tvpvar_connectedness_timeseries import SeriesSpec, _plot_series

OUT = ROOT / 'results/tables/supervisor_revision'
TVP = ROOT / 'results/tables/tvpvar_connectedness'
PRICE = ROOT / 'data/raw/finance/placebo_yahoo_2024_04_01_to_2026_03_31.csv'
ASSETS = ('csco', 'bkng', 'mdlz')
VOL = tuple(f'{x}_conditional_volatility' for x in ('bitcoin', *ASSETS))

def read(path):
    return read_daily_time_series(path)

def prepare_and_fit(skip_models=False):
    prices = read(PRICE)
    calendar = read(ROOT / 'data/processed/combined_time_series.csv')[['date']]
    panel = calendar.merge(prices, on='date', how='left', validate='one_to_one')
    assert len(panel) == 502 and panel.drop(columns='date').notna().all().all()
    for asset in ASSETS:
        panel[f'{asset}_adj_close_log_return'] = np.log(panel[f'{asset}_adj_close']).diff()
    returns = panel[['date', *[f'{a}_adj_close_log_return' for a in ASSETS]]].dropna()
    returns.to_csv(OUT / 'placebo_returns.csv', index=False)
    if not skip_models:
        subprocess.run(['Rscript', 'r/modeling/run_egarch_volatility.R', str(OUT / 'placebo_returns.csv'), str(OUT), ','.join(ASSETS)], cwd=ROOT, check=True)
    vols = read(OUT / 'tvpvar_connectedness_dataset.csv')
    btc = read(TVP / 'tvpvar_connectedness_dataset.csv')[['date', 'bitcoin_conditional_volatility']]
    vols = btc.merge(vols, on='date', validate='one_to_one')
    assert len(vols) == 501 and vols[list(VOL)].notna().all().all()
    vols.to_csv(OUT / 'placebo_volatility_panel.csv', index=False)
    systems = {'placebo': TvpvarSystem('placebo', 'BTC, CSCO, BKNG and MDLZ volatility', VOL)}
    lags = tvpvar_lag_selection(vols, systems=systems)
    lags.to_csv(OUT / 'placebo_lag_selection.csv', index=False)
    lag = selected_lags(lags)['placebo']
    if not skip_models:
        subprocess.run(['Rscript', 'r/modeling/run_tvpvar_connectedness.R', str(OUT / 'placebo_volatility_panel.csv'), str(OUT / 'placebo_h10'), str(lag), '10', ','.join(VOL), str(ROOT / 'results/models/supervisor_revision/placebo_h10')], cwd=ROOT, check=True)
    stationarity_tests(vols, VOL).to_csv(OUT / 'placebo_volatility_stationarity.csv', index=False)
    return lag

def corpus_mentions():
    corpus = pd.read_csv(ROOT / 'data/processed/combined_dedup_roberta_sentiment.csv')
    corpus = corpus[pd.to_datetime(corpus.date).between('2024-04-01', '2026-03-31')]
    patterns = {'NVDA':r'\bnvidia\b|\bnvda\b', 'GOOGL':r'\bgoogle\b|\balphabet\b|\bgoogl?\b', 'MSFT':r'\bmicrosoft\b|\bmsft\b', 'CSCO':r'\bcisco\b|\bcsco\b', 'BKNG':r'\bbooking\s+holdings\b|\bbooking\s*\.\s*com\b|\bbkng\b', 'MDLZ':r'\bmondel[eē]z\b|\bmdlz\b'}
    rows=[]
    for ticker, pattern in patterns.items():
        for source, group in [('all', corpus), *list(corpus.groupby('source'))]:
            n=int(group.text_clean.fillna('').str.contains(pattern, case=False, regex=True).sum())
            rows.append(dict(ticker=ticker, source=source, mentions=n, documents=len(group), percent=100*n/len(group), regex=pattern))
    pd.DataFrame(rows).to_csv(OUT / 'corpus_mentions.csv', index=False)

def report(lag):
    data = read(ROOT / 'results/tables/spillover_regressions/spillover_regressions_dataset.csv')
    placebo = read(OUT / 'placebo_h10/tci_h10.csv').rename(columns={'TCI':'placebo_tci_tci'})
    data = data.merge(placebo, on='date', validate='one_to_one')
    data['ai_minus_placebo_tci'] = data.ai_equity_tci_tci - data.placebo_tci_tci
    deps=['benchmark_tci_tci','ai_equity_tci_tci','placebo_tci_tci','ai_minus_placebo_tci']
    coefficients=table_28_lagged_regressions(data, deps, 'HAC', maxlags=5)
    coefficients.to_csv(OUT / 'comparison_regressions.csv',index=False)
    data.to_csv(OUT / 'comparison_regression_dataset.csv', index=False)
    means=[]
    for system in ['benchmark','ai_equity','placebo']:
        col=f'{system}_tci_tci'
        means.append(dict(system=system, mean=data[col].mean(), nobs=len(data), lag=lag if system=='placebo' else 1, minimum=data[col].min(), maximum=data[col].max()))
    pd.DataFrame(means).to_csv(OUT / 'comparison_tci.csv',index=False)
    # One-lag bivariate tests reuse the SSR F methodology of existing Table A7.
    rows=[]
    dependent=[c for c in data if (c.startswith('benchmark_') or c.startswith('ai_equity_')) and any(s in c for s in ['_tci_','_to_','_from_'])]
    for col in dependent:
        forward=one_lag_granger_test(data,col,EAIS)
        reverse=one_lag_granger_test(data,EAIS,col)
        pair=data[[col,EAIS]].dropna()
        x=sm.add_constant(pair.shift(1).iloc[1:])
        fit=sm.OLS(pair[col].iloc[1:],x).fit(cov_type='HAC',cov_kwds={'maxlags':5})
        rows.append(dict(dependent=col, **forward, reverse_p_value=reverse['p_value'], hac_p_value=fit.pvalues[EAIS]))
    granger=pd.DataFrame(rows)
    granger['holm_p_value']=multipletests(granger.p_value,method='holm')[1]
    granger.to_csv(OUT / 'connectedness_granger.csv',index=False)
    stationarity_tests(data, [EAIS,*dependent]).to_csv(OUT/'granger_stationarity.csv',index=False)
    # In ConnectednessApproach NPDC[row receiver, column transmitter], the
    # BTC column is positive for net transmission FROM BTC to the named row.
    pairs=[]
    for system in ['benchmark','ai_equity','placebo']:
        base=OUT if system=='placebo' else TVP
        npdc=read(base/f'{system}_h10/npdc_h10.csv')
        to=read(base/f'{system}_h10/to_h10.csv')
        fr=read(base/f'{system}_h10/from_h10.csv')
        btc=npdc[(npdc.dimension_2=='bitcoin_conditional_volatility') & (npdc.dimension_1!='bitcoin_conditional_volatility')]
        sums=btc.groupby('date').value.sum()
        expected=to.set_index('date').bitcoin_conditional_volatility-fr.set_index('date').bitcoin_conditional_volatility
        assert np.allclose(sums.loc[expected.index],expected,atol=1e-8)
        for asset, group in btc.groupby('dimension_1'):
            pairs.append(dict(system=system, counterpart=asset.replace('_conditional_volatility','').upper(), net_from_btc=group.value.mean(), nobs=len(group)))
    pd.DataFrame(pairs).to_csv(OUT/'btc_pairwise.csv',index=False)
    # Directly reuse the original plot function, palette, sizing, and axes.
    figure=ROOT/'results/figures/supervisor_revision/placebo_total_connectedness_h10.png'
    figure.parent.mkdir(parents=True,exist_ok=True)
    _plot_series(placebo, SeriesSpec('placebo','tci','placebo_tci_tci','Total Connectedness, H = 10','Total connectedness index','placebo_total_connectedness'),figure)
    market=read(ROOT/'data/processed/combined_time_series.csv')
    sensitivity=data.merge(market[['date','sp500_adj_close_log_return']],on='date',validate='one_to_one')
    regressors=[EAIS,'lagged_expectation_adjusted_ais',*CONNECTEDNESS_REGRESSION_CONTROLS,'sp500_adj_close_log_return']
    pd.concat([regression_table(sensitivity,dependent=dep,specifications={'sp500_sensitivity':regressors},cov_type='HAC',cov_kwds={'maxlags':5},covariance_label='HAC(maxlags=5)') for dep in deps],ignore_index=True).to_csv(OUT/'sp500_sensitivity.csv',index=False)
    controls=data[['date',EAIS,'lagged_expectation_adjusted_ais',*CONNECTEDNESS_REGRESSION_CONTROLS]].merge(market[['date','sp500_adj_close_log_return']],on='date').dropna()
    corr=controls.sp500_adj_close_log_return.corr(controls.vix_close)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    x=sm.add_constant(controls.drop(columns='date'))
    vif=variance_inflation_factor(x.to_numpy(),x.columns.get_loc('sp500_adj_close_log_return'))
    inputs=[PRICE, ROOT/'results/tables/spillover_regressions/spillover_regressions_dataset.csv',TVP/'tvpvar_connectedness_dataset.csv',ROOT/'data/processed/combined_dedup_roberta_sentiment.csv']
    manifest={'selected_lag':lag,'sp500_vix_correlation':corr,'sp500_vif':vif,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, 'inference':'HAC(5); Granger SSR F plus HAC(5) and Holm sensitivity. Generated-regressor uncertainty not propagated.', 'prices':'Yahoo Finance adjusted closes downloaded October 2026; existing BTC volatility and baseline models retained.'}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(means).to_string(index=False))
    print(coefficients[coefficients.term.isin([EAIS,'lagged_expectation_adjusted_ais'])].to_string(index=False))
    print(granger.to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-models',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lag=prepare_and_fit(args.skip_models)
    corpus_mentions()
    report(lag)
