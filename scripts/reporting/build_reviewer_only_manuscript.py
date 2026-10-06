"""Rebuild from the original manuscript, limiting revisions to feedback DG.

No model is re-estimated. Existing, validated revision outputs are reused.
The clean source and review copy share the same substantive paper text.
"""
from pathlib import Path
import json
import re
import shutil

import pandas as pd
from revision_highlights import (
    LOCAL_PREAMBLE, LOCAL_PDFSTRINGS, LEGEND, add_original_paragraphs,
    original_paragraph, remove_review_markup,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'thesis/revision_2026_10'
OUT = ROOT / 'results/tables/supervisor_revision'
original = (PAPER / 'main_original.tex').read_text()
s = original
operations = []
provenance = []


def mark(text, colour='orange'):
    return '\\' + {'orange': 'reviewadd', 'yellow': 'reviewchange',
                   'green': 'reviewclarify'}[colour] + '{' + text + '}'


def change(old, new, reason):
    global s
    assert s.count(old) == 1, (reason, old[:100], s.count(old))
    s = s.replace(old, new, 1)
    operations.append({'reason': reason, 'old': old, 'new': new})


def edit(old, new, reason, colour='yellow'):
    if colour == 'yellow':
        provenance.append((new, original_paragraph(original, old)))
    change(old, mark(new, colour), reason)


def para(prefix):
    start = s.index(prefix)
    end = s.index('\n\n', start)
    return s[start:end]


def after(prefix, text, reason):
    old = para(prefix)
    change(old, old + '\n\n' + text, reason)


def table(caption, label, body, notes, width):
    return (r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n'
            + r'\caption{' + mark(caption) + '}\n' + r'\label{' + label + '}\n'
            + r'\reviewaddbox{\resizebox{' + width + r'\textwidth}{!}{%' + '\n'
            + body + '\n}}\n' + r'\begin{flushleft}\footnotesize' + '\n'
            + mark(r'\textit{Notes:} ' + notes) + '\n' + r'\end{flushleft}'
            + '\n' + r'\end{table}')


# Preserve the sole user edit made since the last verified Overleaf snapshot:
# the original opening sentence, including the user's line break.
online = (PAPER / 'overleaf_before_restart.tex').read_text()
previous = (PAPER / 'overleaf_verified_clean.tex').read_text()
prior_opening = 'Investment narratives may be associated with changing cross-market risk relationships.'
user_opening = ('Investment narratives increasingly shape how investors position\n'
                'across asset classes, raising the question of whether a common narrative can '
                'weaken diversification by altering the way volatility is transmitted across otherwise distinct markets.')
assert previous.replace(prior_opening, user_opening, 1) == online, 'New user edits need reconciliation'
change(user_opening.replace('\n', ' '), user_opening, 'Preserve user-edited abstract opening')

# 1. Trading-day treatment: addition in 3.1, plus exact implementation corrections.
edit('Textual observations are first aggregated by calendar date, after which weekend and market-holiday observations are rolled forward to the next trading day.',
     'Textual observations are assigned to US equity trading days before source-specific sentiment averages are calculated, as detailed in Section 3.1.',
     'R1.1: keep the existing calendar description consistent with the requested clarification', 'green')
after('Textual data are collected from Reddit and GDELT', mark(
    "Because US equity markets are closed on weekends and public holidays, textual observations collected on non-trading days are pooled with the following trading day's observations. Each text is scored individually and assigned to the next available equity trading date, including its own date when it is a trading day; scores are then averaged across all assigned texts within each source. This rolling-forward procedure retains the in-sample textual observations while synchronizing the sentiment series with the daily asset-return and volatility data. Weekend and holiday texts therefore receive the same observation-level weight as texts collected on the following trading day."), 'R1.1: requested weekend/holiday paragraph')
edit('The scores are aggregated separately by source $l$ and calendar day:',
     'The scores are aggregated separately by source $l$ and assigned equity trading day $d$:',
     'R1.1: align appendix terminology with the implemented calendar treatment', 'green')
edit('where $N_d^{(l)}$ is the number of observations from source $l$ on day $d$.',
     'where $N_d^{(l)}$ counts all source-$l$ texts assigned to trading day $d$, including those from preceding non-trading days.',
     'R1.1: clarify the existing aggregation denominator', 'green')

# 2. Expand the existing description, using the actual contract and transformation.
old = para('Forward-looking AI expectations are measured using prediction-market data')
new = (r"""Forward-looking AI expectations are measured using the Kalshi contract price and Metaculus community forecasts. The Kalshi variable is the price, in cents, of the ``Before 2030'' contract on OpenAI announcing that it has attained AGI. For a one-dollar binary payoff, the price is interpretable as a market-implied probability in percentage points, subject to trading frictions. The Metaculus variable uses the recency-weighted median forecast for the date on which the first general AI system will be devised, tested and publicly announced (question 5121). The code measures the number of days from each forecast timestamp to its predicted date and changes the sign, so that higher values indicate nearer-term expected developments; ``inverted'' means negative days, rather than a reciprocal. The last daily forecast is retained and both expectation series are carried forward before alignment with equity trading dates. Both series are first-differenced because their levels are highly persistent. Kalshi aggregates financially backed positions, whereas this Metaculus series aggregates community forecasts without a traded contract price. They concern different events and are included to capture expectation changes from these distinct sources. Descriptive statistics and diagnostic tests are reported in Table \ref{tab:control_descriptives}, which shows mixed evidence of stationarity for $\ln(EPU)$.""")
edit(old, new, 'R1.2: prediction-market description expanded in Section 3.3')
after(r'\reviewchange{Forward-looking AI expectations',
      mark('Contract and question definitions:', 'green') + r' \href{https://www.cftc.gov/filings/ptc/ptc01052410467.pdf}{Kalshi/CFTC} and \href{https://www.metaculus.com/questions/5121/date-of-artificial-general-intelligence/}{Metaculus}.',
      'Factual source links for the expanded measure definitions')

# 3. Explain the control choice; do not repeat the unsupported collinearity claim.
after('where $Spillover_{i,t}$ denotes', mark(
    r'S\&P 500 returns enter the orthogonalization regression (Equation \eqref{eq:orthogonalization}) to remove linear co-movement between raw AI sentiment and the broad equity market at construction. The connectedness regression (Equation \eqref{eq:spillover_regression}) serves a different purpose: it relates the adjusted sentiment measure to spillovers, conditional on the existing parsimonious set of dollar and uncertainty controls. VIX captures broad market uncertainty in this second equation, whereas S\&P 500 returns capture realized equity-market movements. Their inclusion in different stages reflects this specification choice; it does not imply that they are interchangeable or nearly collinear.'),
    'R1.3: requested explanation only; omit the extra S&P regression')

# 4. Add the requested limitation, leaving the original limitations intact.
after('Several limitations should be considered when interpreting these findings.', mark(
    r'A further limitation concerns the roughly two-year sample. It reflects both the selected post-ChatGPT period and the availability of the Kalshi series from April 2024 onward. The effective sample contains 501 return observations and 487 observations in the main lagged connectedness regressions. The TVP-VAR is estimated using Kalman filtering with forgetting factors, following \citet{Koop2014}, so it does not require a fixed rolling-window length. This does not remove the limitations of a short sample or guarantee statistical power. A longer sample would help assess the stability of the estimates across episodes and increase the information available for inference.'),
    'R1.4: add sample-length limitation, without unverified literature sample-size comparisons')

# 5. Reuse the requested comparison estimates and the original table style.
after('Financial data consist of Bitcoin', mark(
    r'The same-size comparison adds Cisco Systems (CSCO), Booking Holdings (BKNG) and Mondelez International (MDLZ), using Yahoo Finance adjusted prices and the identical trading calendar. These firms span networking, travel services and packaged foods and have substantially fewer mentions in this study\textquotesingle s AI-filtered text corpus than NVIDIA, Alphabet and Microsoft. They provide a comparison basket rather than a market-capitalization- or sector-matched control group; Cisco also has AI-infrastructure exposure. Section \ref{sec:placebo} reports the selection check and comparison results.'.replace(r'\textquotesingle s', "'s")),
    'R2.5: requested comparison firms in Section 3.2')
after(r'Table \ref{tab:average_connectedness_table} shows that', mark(
    r'To address the possibility that the higher connectedness in the BTC--AI-equity system reflects its larger number of variables, Section \ref{sec:placebo} reports a comparison system of identical size using Bitcoin, Cisco, Booking and Mondelez. Panel C reports its average connectedness alongside the two original systems.'),
    'R2.5: forward-reference, preserving the original results paragraph')
ct = pd.read_csv(OUT / 'placebo_h10/connectedness_table_h10.csv')
rows = [r'\begin{tabular}{lrrrrr}', r'\hline', r'\multicolumn{6}{l}{\textit{Panel C: BTC--comparison-equity system}} \\', r'\hline', r' & BTC & CSCO & BKNG & MDLZ & $FROM$ \\', r'\hline']
for i, label in enumerate(['BTC', 'CSCO', 'BKNG', 'MDLZ']):
    rows.append(label + ' & ' + ' & '.join(f'${float(v):.2f}$' for v in ct.iloc[i]) + r' \\')
rows += [r'\hline', r'$TO$ & ' + ' & '.join(f'${float(v):.2f}$' for v in ct.iloc[4, :4]) + r' & \\', r'$NET$ & ' + ' & '.join(f'${float(v):.2f}$' for v in ct.iloc[6, :4]) + r' & \\', r'\hline', r'\multicolumn{5}{l}{$TCI$} & $19.15$ \\', r'\hline', r'\end{tabular}']
anchor = r'\textit{Notes:} The table reports average connectedness measures from the TVP-VAR model using forecast horizon $H = 10$.'
at = s.rfind(r'\begin{flushleft}', 0, s.index(anchor))
table_start = s.rfind(r'\begin{table}', 0, at)
old = s[table_start:s.index(anchor) + len(anchor)]
panel = r'\vspace{0.4cm}' + '\n' + r'\reviewaddbox{\resizebox{0.55\textwidth}{!}{%' + '\n' + '\n'.join(rows) + '\n}}\n'
change(old, s[table_start:at] + panel + s[at:s.index(anchor) + len(anchor)], 'R2.5: add Panel C using existing Table 4 layout')

pairs = pd.read_csv(OUT / 'btc_pairwise.csv')
rows = [r'\begin{tabular}{llr}', r'\toprule', r'System & Counterpart & Net transmission from BTC \\', r'\midrule']
for system, display in [('benchmark', 'BTC--NDX'), ('ai_equity', 'BTC--AI equities'), ('placebo', 'BTC--comparison equities')]:
    for _, row in pairs[pairs.system.eq(system)].iterrows():
        rows.append(f'{display} & {row.counterpart} & ${row.net_from_btc:.2f}$' + r' \\')
rows += [r'\bottomrule', r'\end{tabular}']
pairtable = table('Bitcoin-Specific Net Pairwise Connectedness, $H=10$', 'tab:btc_pairwise', '\n'.join(rows), r'Means over 501 dates, in percentage points. Positive: net transmission from BTC; negative: net reception. Estimates are conditional on each system.', '.75')
after(r'Figure \ref{fig:npdc_network_h10} confirms', mark(
    r'Table \ref{tab:btc_pairwise} reports Bitcoin\textquotesingle s average pairwise net directional connectedness with each counterpart. Bitcoin receives net spillovers from NDX (4.96 percentage points), NVIDIA (4.43) and Microsoft (3.44), and transmits a smaller net amount to Alphabet (1.41). The three AI-system components sum to Bitcoin\textquotesingle s net connectedness of $-6.46$ percentage points. Reporting the individual links avoids comparing sums over different numbers of counterparties, although each estimate still depends on the dynamics and variance decomposition of its multivariate system.'.replace(r'\textquotesingle s', "'s")) + '\n\n' + pairtable,
    'R2.5: requested BTC-specific pairwise table and interpretation')

coefs = pd.read_csv(OUT / 'comparison_regressions.csv')
columns = ['ai_equity_tci_tci', 'placebo_tci_tci']
terms = [('const', 'Constant'), ('expectation_adjusted_ais', 'EAIS'), ('lagged_expectation_adjusted_ais', 'L. EAIS'), ('dxy_close_log_return', 'DXY (Log Return)'), ('vix_close', 'VIX'), ('log_gpr', r'$\ln(GPR)$'), ('log_epu', r'$\ln(EPU)$')]
def stars(p):
    return '^{***}' if p < .01 else '^{**}' if p < .05 else '^{*}' if p < .1 else ''
rows = [r'\begin{tabular}{lcc}', r'\hline', r' & AI equities & Comparison equities \\', r'\hline']
for term, label in terms:
    rs = [coefs[(coefs.dependent_variable == col) & (coefs.term == term)].iloc[0] for col in columns]
    rows += [label + ' & ' + ' & '.join(f'${r.coefficient:.3f}{stars(r.p_value)}$' for r in rs) + r' \\', ' & ' + ' & '.join(f'$({r.standard_error:.3f})$' for r in rs) + r' \\']
for col, label, fmt in [('nobs', '$N$', '.0f'), ('r_squared', '$R^2$', '.3f'), ('adjusted_r_squared', 'Adjusted $R^2$', '.3f')]:
    rows.append(label + ' & ' + ' & '.join('$' + format(coefs[coefs.dependent_variable.eq(c)].iloc[0][col], fmt) + '$' for c in columns) + r' \\')
rows += [r'\hline', r'\end{tabular}']
regtable = table('EAIS and Total Connectedness in Equally Sized Systems', 'tab:placebo_regressions', '\n'.join(rows), r'HAC(5) standard errors in parentheses. ***, ** and * denote 1\%, 5\% and 10\% significance.', '.72')
prior_clean = (PAPER / 'before_reviewer_only_restart/main_clean.tex').read_text()
fig_at = prior_clean.index(r'\label{fig:placebo_tci}')
fig_start = prior_clean.rfind(r'\begin{figure}', 0, fig_at)
fig_end = prior_clean.index(r'\end{figure}', fig_at) + len(r'\end{figure}')
figure = prior_clean[fig_start:fig_end]
figure = re.sub(r'\\caption\{([^{}]*)\}', lambda m: r'\caption{' + mark(m.group(1)) + '}', figure)
note = r'\textit{Notes:} Both systems contain BTC and three equities; the comparison basket comprises CSCO, BKNG and MDLZ.'
figure = figure.replace(note, mark(note))
section = r'\subsubsection{' + mark('Same-size comparison system: Cisco, Booking and Mondelez') + '}\n' + r'\label{sec:placebo}' + '\n\n'
section += mark(r"""To assess whether the higher connectedness in the BTC--AI-equity system reflects system size, we estimate an equally sized system comprising BTC, CSCO, BKNG and MDLZ. The comparison firms span different degrees of sectoral proximity to AI. A case-insensitive search of the 92,407 in-sample texts identifies 34 Cisco documents, seven Booking documents and four Mondelez documents, compared with 2,167 for NVIDIA, 7,464 for Google and Alphabet, and 3,775 for Microsoft. The search uses company names and tickers, including Booking Holdings and Booking.com rather than the ambiguous word ``booking''; documents are counted once per firm and can mention more than one firm. This is a corpus-specific measure of narrative salience, not a comprehensive measure of AI business exposure. The firms are not matched in market capitalization or sector, and Cisco has AI-infrastructure exposure, so the comparison does not isolate AI exposure from every other firm characteristic.""") + '\n\n'
section += mark(r'We estimate the three additional EGARCH(1,1) volatility models and the TVP-VAR using the same procedure as in Sections 4.1--4.2, retaining the existing Bitcoin volatility series. All three EGARCH fits converge over 501 trading-day returns. BIC selects TVP-VAR(1); the forecast horizon is $H=10$ and both forgetting factors are 0.99.') + '\n\n'
section += mark(r'Panel C of Table \ref{tab:average_connectedness_table} reports average comparison-system TCI of 19.15\%, close to 18.79\% for BTC--NDX and below 27.43\% for the equally sized AI-equity basket. Thus, increasing the system to four variables does not by itself reproduce the higher connectedness observed for the AI equities. The result addresses the system-size concern, while differences in firm and sector composition remain relevant. Bitcoin is a net receiver in the comparison system ($-12.55$ percentage points), predominantly from Booking (Table \ref{tab:btc_pairwise}). Figure \ref{fig:placebo_tci} compares the two four-variable TCI paths.') + '\n\n' + figure + '\n\n'
section += mark(r'Table \ref{tab:placebo_regressions} repeats the main sentiment-connectedness regression for the comparison basket. Its contemporaneous EAIS coefficient is positive and significant at 4.636 (HAC standard error 0.949), compared with 2.135 (0.410) for the AI basket; both lagged coefficients are negative. Although average connectedness is lower for the comparison basket, the sentiment association is not absent from it. The comparison therefore supports the conclusion that the higher AI-basket TCI is not merely a consequence of system size, but it does not support interpreting the sentiment-connectedness association as exclusive to the selected AI firms.') + '\n\n' + regtable
change(r'\section{Discussion \& Conclusion}', section + '\n\n' + r'\section{Discussion \& Conclusion}', 'R2.5: requested new robustness subsection, figure and regression table')

# 6. Target the two existing EAIS interpretations; preserve the wider narrative.
edit(r'This residual is used as a proxy for the component of AI sentiment that is not explained by market expectations. Following the logic of \citet{Baker2006}, this residual can be interpreted as a more speculative, or potentially irrational, sentiment component.',
     r'This residual is used as a macro- and expectation-adjusted measure of AI sentiment (EAIS). Following the filtering logic of \citet{Baker2006}, it removes the variation explained by the included controls, but does not by itself identify speculative or irrational beliefs.',
     'R2.6: requested EAIS interpretation in Section 3.1')
old = para(r'Table \ref{tab:orthogonalization_regression} shows the complete orthogonalization results')
edit(old, r'Table \ref{tab:orthogonalization_regression} shows that the prediction-market variables, DXY and S\&P 500 returns are individually insignificant, while GPR, VIX and EPU are significant at conventional levels. The adjustment therefore operates primarily through macro-financial, geopolitical and policy-uncertainty conditions, rather than through a separately demonstrated prediction-market contribution. EAIS is interpreted accordingly as a macro- and expectation-adjusted measure. One possible explanation for the insignificant prediction-market coefficients is that AGI-timing expectations contain relatively little information about day-to-day fluctuations in textual tone. EAIS remains the main sentiment measure and AIS is retained for the robustness checks.', 'R2.6: requested interpretation of existing orthogonalization results')

# 7. Two additions, no expanded validation exercise or changed conceptual diagram.
after('The validation results are reported in Table', mark(r'The manual validation is based on 100 observations and should be interpreted as an indicative, rather than definitive, check on classifier performance. The 72\% agreement rate and the small number of polarity reversals support its use as a noisy directional measure, but a larger independently labeled sample would permit a more precise estimate of classification accuracy.'), 'R2.7: requested validation sample-size limitation')
after('Based on the literature discussed above', mark(r'Positive sentiment toward AI as a technology does not necessarily imply positive investor sentiment toward AI-exposed equities. A text about capabilities, risks or societal implications can have a different sentiment orientation from its implications for stock valuations. An attention channel offers one possible interpretation, consistent with the behavioral motivation in \citet{DeLong1990}: AI-related discussion may draw attention to exposed assets and affect trading behavior. The signed sentiment index does not directly measure attention volume or investor positions, however, so the conceptual transmission mechanism remains a hypothesis rather than an identified causal sequence.'), 'R2.7: requested technology-tone versus equity-sentiment clarification')

# 8. Narrow sentence/phrase edits to the specified Results and Discussion sections.
causal_edits = [
    ('a one-unit increase in lagged EAIS lowers connectedness by 1.98 percentage points.', 'a one-unit increase in lagged EAIS is associated with 1.98 percentage points lower connectedness.'),
    ('This suggests that the sentiment effect on firm-level AI-equity connectedness is strong on the same trading day, but partly reverses on the next trading day.', 'This suggests that the sentiment association with firm-level AI-equity connectedness is strong on the same trading day, with an opposite-signed coefficient on lagged sentiment.'),
    ('clarify the transmission mechanism.', 'describe the directional associations.'),
    ("a one-unit increase in EAIS lowers Bitcoin's transmitted connectedness by 3.45 percentage points", "a one-unit increase in EAIS is associated with 3.45 percentage points lower Bitcoin transmitted connectedness"),
    ("A one-unit increase in EAIS raises Bitcoin's received connectedness by 5.57 percentage points while lowering its transmitted connectedness by 3.15 percentage points.", "A one-unit increase in EAIS is associated with 5.57 percentage points higher Bitcoin received connectedness and 3.15 percentage points lower transmitted connectedness."),
    ("At the same time, NVIDIA's transmitted connectedness rises by 6.08 percentage points and Alphabet's by 9.23 percentage points. Microsoft's transmitted connectedness decreases by 3.63 percentage points, while its received connectedness increases by 7.24 percentage points.", "The corresponding transmitted-connectedness coefficients are 6.08 for NVIDIA, 9.23 for Alphabet and $-3.63$ for Microsoft; Microsoft's received-connectedness coefficient is 7.24."),
    ('elevated AI sentiment shifts volatility dominance away from Bitcoin and toward AI-exposed technology firms', 'elevated AI sentiment is associated with a shift in volatility dominance away from Bitcoin and toward AI-exposed technology firms'),
    ('but that these effects partly reverse over time.', 'with opposite-signed lagged associations conditional on contemporaneous sentiment.'),
    ('The main result is not that AI sentiment uniformly increases all connectedness, but that it changes the direction and timing of volatility transmission.', 'The main result is not a uniform positive association between AI sentiment and connectedness, but differences in the direction and timing of the estimated associations.'),
    ('Moreover, the lagged coefficients show evidence of subsequent corrections, suggesting that the connectedness effects of AI sentiment are strongest in the short run.', 'Moreover, the opposite-signed lagged coefficients suggest a short-run pattern in the conditional associations; they do not establish a causal correction following a sentiment shock.'),
    ('The central finding of the analysis is that narrative-driven sentiment does not simply increase cross-market connectedness uniformly; instead, it changes the structure and direction of risk transmission across assets.', 'The central finding of the analysis is that narrative-driven sentiment is not uniformly associated with higher cross-market connectedness; instead, its associations differ across the structure and direction of risk transmission.'),
    ('This contrast suggests that the impact of an investment narrative depends on the economic and thematic exposure of the assets involved.', 'This contrast suggests that the association with an investment narrative can differ across the economic and thematic exposure of the assets involved.'),
    ('elevated AI sentiment appears to strengthen the risk linkage between Bitcoin and firms that are more directly exposed to the AI investment theme, even though it does not increase connectedness between Bitcoin and the broader technology market.', 'elevated AI sentiment is associated with a stronger risk linkage between Bitcoin and firms that are more directly exposed to the AI investment theme, without a corresponding positive association for Bitcoin and the broader technology market.'),
    ("higher AI sentiment reduces Bitcoin's role as a transmitter while increasing its role as a receiver, whereas NVIDIA and Alphabet become stronger transmitters of volatility shocks.", "higher AI sentiment is associated with a smaller transmitter role and a larger receiver role for Bitcoin, alongside stronger transmission from NVIDIA and Alphabet."),
    ('narrative-driven sentiment can reshape the architecture of cross-market risk, potentially creating concentrated sources of portfolio exposure', 'narrative-driven sentiment is associated with differences in the architecture of cross-market risk and potentially concentrated sources of portfolio exposure'),
]
for old, new in causal_edits:
    edit(old, new, 'R2.8: soften causal wording without replacing the paragraph or its focus')
granger_text = mark(r'As an additional check on timing, Table \ref{tab:connectedness_granger} reports one-trading-day Granger tests of EAIS on TCI, TO and FROM connectedness, conditional on one own lag. The tests do not reject the absence of predictive content for TCI in either BTC--NDX ($p=0.846$) or BTC--AI equities ($p=0.449$). One of the 14 directional and total-connectedness tests rejects at 5\%: Bitcoin received connectedness in the AI system ($p=0.012$). The tests complement the main regressions but do not identify a causal behavioral mechanism. In particular, sentiment and markets may react to the same underlying news, so the same-day coefficients are interpreted as conditional associations.')
after('Overall, the connectedness regressions suggest', granger_text, 'R2.8: requested Granger results and contemporaneous-information caveat')
after('The directional results provide further evidence', mark(r'The Granger results in Table \ref{tab:connectedness_granger} provide limited evidence that EAIS precedes connectedness: neither TCI test rejects, and the directional rejection is isolated among 14 tests. Thus, the directional patterns remain relevant to describing cross-market risk, but do not by themselves demonstrate that sentiment causes its transmission. The portfolio implications are interpretations of the observed associations.'), 'R2.8: requested discussion caveat, added to existing framing')
g = pd.read_csv(OUT / 'connectedness_granger.csv')
rows = [r'\begin{tabular}{llrr}', r'\toprule', r'System & Measure & $F$ & $p$ \\', r'\midrule']
for _, row in g.iterrows():
    system = 'BTC--NDX' if row.dependent.startswith('benchmark') else 'BTC--AI equities'
    if '_tci_' in row.dependent:
        label = 'TCI'
    else:
        component = 'TO' if '_to_' in row.dependent else 'FROM'
        asset = row.dependent.split('_' + component.lower() + '_')[1].replace('_conditional_volatility', '').upper().replace('BITCOIN', 'BTC')
        label = asset + ' ' + component
    rows.append(f'{system} & {label} & {row.statistic:.3f} & {row.p_value:.3f}' + r' \\')
rows += [r'\bottomrule', r'\end{tabular}']
gtable = table('One-Day Granger Tests of EAIS on Connectedness', 'tab:connectedness_granger', '\n'.join(rows), r'Null: no predictive contribution from EAIS, conditional on one own lag and a constant. $N=487$; $F(1,484)$. Unadjusted $p$-values.', '.60')
pos = s.index(r'\label{tab:granger_tests}')
end = s.index(r'\end{table}', pos) + len(r'\end{table}')
old = s[s.rfind(r'\begin{table}', 0, pos):end]
change(old, old + '\n\n' + gtable, 'R2.8: Table A8, preserving the requested resizebox treatment')

# 9. Keep the original robustness subsection, correcting only its endorsement.
edit('which supports the orthogonalization procedure used to filter the raw sentiment index.', 'which limits the interpretation of this robustness check.', 'R2.9: avoid describing raw-model instability as supporting evidence')
edit('Nevertheless, the evidence is more concentrated and less systematic than in the EAIS specification, supporting the broader conclusion that AI-related sentiment is more relevant for firm-level AI-equity connectedness than for the broad BTC--NDX benchmark system.', 'The evidence is less systematic than in the EAIS specification, indicating that the main findings depend on how the sentiment measure is constructed.', 'R2.9: targeted reframing of the existing raw-AIS robustness conclusion')
edit('The robustness checks broadly support this interpretation, but also qualify it.', 'The robustness checks qualify this interpretation, particularly because the raw-sentiment specifications do not consistently reproduce the main findings.', 'R2.9: retain the discussion paragraph but correct its opening')
edit('When raw AI sentiment is used in the ARMA-EGARCH models, the return and volatility effects become insignificant,', 'When raw AI sentiment is used in the ARMA-EGARCH models, the NASDAQ-100 mean and variance coefficients are insignificant and the BTC specification is unstable,', 'R2.9: distinguish the insignificant NASDAQ estimates from the unstable BTC model')
edit('supporting the value of filtering sentiment using prediction-market and macro-financial information.', 'showing sensitivity to filtering sentiment using prediction-market and macro-financial information.', 'R2.9: avoid treating divergent results as validation')
after('The robustness checks', mark(r'The divergence between AIS and EAIS deserves separate consideration. Table \ref{tab:orthogonalization_regression} shows that raw AI sentiment contains variation associated with macro-financial, geopolitical and policy-uncertainty conditions. Removing that variation can therefore change the estimated relationship with connectedness. This helps explain why the two measures need not yield the same results, but does not validate EAIS as a purified speculative component. The main findings should be understood as conditional on the specified filtering procedure rather than as a general property of every measure of AI-related sentiment.'), 'R2.9: requested additional interpretive discussion')

# Report the comparison qualification where relevant, without replacing conclusions.
after('The TVP-VAR results reveal important differences', mark(r'The same-size comparison in Section \ref{sec:placebo} supports this connectedness ranking: its average TCI is 19.15\%, below 27.43\% for the AI-equity basket. However, its sentiment coefficient is also positive and significant. The comparison addresses the mechanical system-size concern, while showing that the sentiment association is not exclusive to the selected AI firms.'), 'R2.5: concise discussion of the requested comparison findings')

# Formatting explicitly requested in previous turns. No broad prose cleanup.
change(r'\begin{table}[H]' + '\n' + r'\centering' + '\n' + r'\caption{Average TVP-VAR Connectedness Table, $H=10$}', r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n' + r'\caption{Average TVP-VAR Connectedness Table, $H=10$}', 'Keep the expanded table on a suitable float page')
change(r'\begin{table}[htbp]' + '\n' + r'\centering' + '\n' + r'\caption{' + mark('EAIS and Total Connectedness in Equally Sized Systems') + '}', r'\begin{table}[H]' + '\n' + r'\centering' + '\n' + r'\caption{' + mark('EAIS and Total Connectedness in Equally Sized Systems') + '}', 'Keep comparison table within the requested robustness subsection')

# Reversibility check: every edit has an exact original and a feedback rationale.
restored = s
for op in reversed(operations):
    assert op['new'] in restored
    restored = restored.replace(op['new'], op['old'], 1)
assert restored == original

assert not any(ord(c) < 32 and c not in '\n\t' for c in s)
clean = remove_review_markup(s)
clean = clean.replace(r'\begin{document}', r'\setlength{\emergencystretch}{2em}\begin{document}', 1)
assert not re.search(r'\\(?:review\w*|rev|hl|sethlcolor|colorbox)\b', clean)
# Confirm restoration of the entire original introduction and abstract wording.
def section_text(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]
assert section_text(clean, r'\section{Introduction}', r'\section{Literature review}') == section_text(original, r'\section{Introduction}', r'\section{Literature review}')
assert ' '.join(section_text(clean, r'\begin{abstract}', r'\end{abstract}').split()) == ' '.join(section_text(original, r'\begin{abstract}', r'\end{abstract}').split())

local = s.replace(r'\usepackage{tikz}', LOCAL_PREAMBLE + r'\usepackage{tikz}', 1)
local = local.replace(r'\begin{document}', LOCAL_PDFSTRINGS + '\n' + r'\setlength{\emergencystretch}{2em}\begin{document}', 1)
legend = LEGEND.replace('Aqua: original paragraph before the yellow revisions', 'Blue/aqua: original paragraph before the yellow revisions')
anchor = r'    {\normalsize \thesisdate}'
assert anchor in local
local = local.replace(anchor, anchor + '\n' + legend, 1)
local, originals = add_original_paragraphs(local, provenance, expected_count=None)

(PAPER / 'main.tex').write_text(local)
(PAPER / 'main_clean.tex').write_text(clean)
(PAPER / 'overleaf/main.tex').write_text(clean)
(PAPER / 'reviewer_only_change_audit.json').write_text(json.dumps(operations, indent=2, ensure_ascii=False))
(PAPER / 'highlight_categories.txt').write_text(
    'REVIEWER-ONLY RESTART\nOrange: requested additions. Yellow: targeted reviewer-requested wording edits.\n'
    'Green: factual/implementation clarifications. Blue/aqua: exact original below each yellow paragraph.\n'
    'Original abstract and introduction restored; user opening preserved. Broad editorial rewrites removed.\n'
    'Optional S&P sensitivity and AI-minus-comparison regression omitted; no new analyses run.\n\n'
    + '\n'.join(op['reason'] for op in operations) + '\n\n' + '\n\n'.join(originals))
shutil.copy2(ROOT / 'results/figures/supervisor_revision/placebo_total_connectedness_h10.png', PAPER / 'placebo_total_connectedness_h10.png')
print(f'Rebuilt from original: {len(operations)} targeted operations; {len(originals)} original comparisons.')
