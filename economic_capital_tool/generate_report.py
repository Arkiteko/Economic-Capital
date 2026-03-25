"""
Generate a standalone interactive HTML report from the Economic Capital simulation.
All Plotly charts are embedded — opens in any browser with no server required.
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from engine.simulation import run_simulation
from data.generator import (
    generate_all_data, MIGRATION_MATRIX, RATINGS,
    GICS_GROUP_TO_SECTOR, GICS_GROUP_DISPLAY_NAMES,
)

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Economic_Capital_Report.html')

def fmt(val):
    if abs(val) >= 1e9: return f"${val/1e9:,.2f}B"
    elif abs(val) >= 1e6: return f"${val/1e6:,.2f}M"
    elif abs(val) >= 1e3: return f"${val/1e3:,.1f}K"
    return f"${val:,.0f}"

# ── Generate Data & Run Simulations ──
print("Generating portfolio data (500 counterparties, 2000 instruments)...")
data = generate_all_data(500, 2000, seed=42)
cp = data['counterparties']
inst = data['instruments']
pd_vals = np.array([c['pd_1y'] for c in cp])
cp_map = {c['counterparty_id']: i for i, c in enumerate(cp)}
o_lgd = np.zeros(len(cp)); o_cnt = np.zeros(len(cp))
for ins in inst:
    idx = cp_map.get(ins['counterparty_id'], 0)
    o_lgd[idx] += ins['lgd']; o_cnt[idx] += 1
o_lgd = np.where(o_cnt > 0, o_lgd / o_cnt, 0.45)

N_SIMS = 100_000

# ── Baseline Run ──
print(f"Running baseline simulation ({N_SIMS:,} trials)...")
t0 = time.time()
results = run_simulation(cp, inst, pd_vals, o_lgd, n_simulations=N_SIMS, seed=42)
baseline_time = time.time() - t0
print(f"  Baseline done in {baseline_time:.1f}s")
m = results['metrics']
ead = results['ead']

# ── Stress Scenarios ──
scenarios = data['scenarios']
scenario_metrics = {'Baseline': m}
for sc in scenarios[1:]:
    print(f"  Running scenario: {sc['scenario_name']}...")
    sr = run_simulation(cp, inst, pd_vals, o_lgd, n_simulations=N_SIMS, seed=42, scenario=sc)
    scenario_metrics[sc['scenario_name']] = sr['metrics']

print("All simulations complete. Building report...")

# ── Precompute aggregations ──
losses = results['portfolio_losses']
var_contrib = m['var_contributions']
es_contrib = m['es_contributions']
inst_el = m['instrument_el']

# Counterparty-level aggregation
cp_agg = {}
for j, ins in enumerate(inst):
    cid = ins['counterparty_id']
    if cid not in cp_agg:
        cp_agg[cid] = {'var': 0, 'es': 0, 'el': 0, 'ead': 0}
    cp_agg[cid]['var'] += var_contrib[j]
    cp_agg[cid]['es'] += es_contrib[j]
    cp_agg[cid]['el'] += inst_el[j]
    cp_agg[cid]['ead'] += ead[j]

cp_lookup = {c['counterparty_id']: c for c in cp}
cp_rows = []
for cid, v in cp_agg.items():
    c = cp_lookup[cid]
    cp_rows.append({
        'Name': c['legal_name'],
        'GICS_Group': GICS_GROUP_DISPLAY_NAMES.get(c['sector_code'], c['sector_code']),
        'GICS_Sector': c.get('gics_sector', GICS_GROUP_TO_SECTOR.get(c['sector_code'], '')),
        'Sector': c.get('gics_sector', GICS_GROUP_TO_SECTOR.get(c['sector_code'], c['sector_code'])),
        'Country': c['country_code'],
        'Rating': c['rating'], 'EAD': v['ead'], 'EL': v['el'],
        'VaR_Contrib': v['var'], 'ES_Contrib': v['es'], 'EC_Contrib': v['var'] - v['el']
    })
cp_df = pd.DataFrame(cp_rows).sort_values('VaR_Contrib', ascending=False)

# Sector / Country / Type / Rating aggregations
sector_ead, country_ead, type_ead, rating_ead = {}, {}, {}, {}
for j, ins in enumerate(inst):
    c = cp_lookup[ins['counterparty_id']]
    gics_sector = c.get('gics_sector', GICS_GROUP_TO_SECTOR.get(c['sector_code'], c['sector_code']))
    sector_ead[gics_sector] = sector_ead.get(gics_sector, 0) + ead[j]
    country_ead[c['country_code']] = country_ead.get(c['country_code'], 0) + ead[j]
    type_ead[ins['instrument_type']] = type_ead.get(ins['instrument_type'], 0) + ead[j]
    rating_ead[ins.get('rating', 'NR')] = rating_ead.get(ins.get('rating', 'NR'), 0) + ead[j]

# ═══════════════════════════════════════════
# BUILD CHARTS
# ═══════════════════════════════════════════

# 1. Loss Distribution
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=losses/1e6, nbinsx=200, name='Loss Distribution',
    marker_color='rgba(55,83,109,0.7)',
    hovertemplate='Loss: $%{x:.1f}M<br>Count: %{y}<extra></extra>'))
for label, val, color in [
    ('EL', m['expected_loss'], '#11998e'), ('VaR 95%', m['var_95'], 'orange'),
    ('VaR 99%', m['var_99'], '#f45c43'), ('VaR 99.9%', m['var_999'], '#eb3349'),
    ('ES 99.9%', m['es_999'], '#764ba2')]:
    fig_dist.add_vline(x=val/1e6, line_dash="dash", line_color=color,
        annotation_text=f"{label}: {fmt(val)}", annotation_font_size=10)
fig_dist.update_layout(title='Portfolio Loss Distribution (100,000 Simulations)',
    xaxis_title='Loss ($M)', yaxis_title='Frequency', template='plotly_white', height=500)

# 2. Exceedance Curve
sorted_l = np.sort(losses)
exc = 1 - np.arange(1, len(sorted_l)+1) / len(sorted_l)
idx = np.linspace(0, len(sorted_l)-1, 2000, dtype=int)
fig_exc = go.Figure()
fig_exc.add_trace(go.Scatter(x=sorted_l[idx]/1e6, y=exc[idx], mode='lines',
    line=dict(color='#eb3349', width=2)))
fig_exc.update_layout(title='Loss Exceedance Curve', xaxis_title='Loss ($M)',
    yaxis_title='Exceedance Probability', yaxis_type='log', template='plotly_white', height=400)

# 3. Top 20 Counterparties
top20 = cp_df.head(20)
fig_top = go.Figure()
fig_top.add_trace(go.Bar(x=top20['Name'], y=top20['VaR_Contrib']/1e6,
    name='VaR Contribution', marker_color='#eb3349'))
fig_top.add_trace(go.Bar(x=top20['Name'], y=top20['EL']/1e6,
    name='Expected Loss', marker_color='#2193b0'))
fig_top.update_layout(title='Top 20 Counterparties by Risk Contribution',
    yaxis_title='Amount ($M)', barmode='group', template='plotly_white',
    height=500, xaxis_tickangle=-45)

# 4. GICS Sector/Group Treemap
tree_data = cp_df[cp_df['VaR_Contrib'] > 0].copy()
fig_tree = px.treemap(tree_data, path=['GICS_Sector', 'GICS_Group'], values='VaR_Contrib',
    color='EC_Contrib', color_continuous_scale='RdYlGn_r', title='Risk by GICS Sector → Industry Group')
fig_tree.update_layout(height=450)

# 5. Concentration Pies
fig_sector = go.Figure(data=[go.Pie(labels=list(sector_ead.keys()),
    values=list(sector_ead.values()), hole=0.4, textinfo='label+percent')])
fig_sector.update_layout(title='EAD by Sector', height=400)

fig_country = go.Figure(data=[go.Pie(labels=list(country_ead.keys()),
    values=list(country_ead.values()), hole=0.4, textinfo='label+percent')])
fig_country.update_layout(title='EAD by Country', height=400)

# 6. Product Type Bar
fig_type = go.Figure(data=[go.Bar(x=list(type_ead.keys()),
    y=[v/1e6 for v in type_ead.values()], marker_color='#667eea')])
fig_type.update_layout(title='EAD by Product Type', yaxis_title='EAD ($M)',
    template='plotly_white', height=400)

# 7. Rating Distribution
ordered = [r for r in RATINGS if r in rating_ead]
colors = ['#11998e','#38ef7d','#2193b0','#6dd5ed','#f7971e','#eb3349','#f45c43','#333']
fig_rating = go.Figure(data=[go.Bar(x=ordered,
    y=[rating_ead.get(r,0)/1e6 for r in ordered], marker_color=colors[:len(ordered)])])
fig_rating.update_layout(title='EAD by Rating Grade', yaxis_title='EAD ($M)',
    template='plotly_white', height=400)

# 8. Scenario Comparison
sc_rows = []
for name, sm in scenario_metrics.items():
    sc_rows.append({'Scenario': name, 'EL': sm['expected_loss']/1e6,
        'VaR999': sm['var_999']/1e6, 'ES999': sm['es_999']/1e6,
        'EC': sm['economic_capital']/1e6, 'EC_pct': sm['ec_as_pct_ead'],
        'Div': sm['diversification_benefit']*100})
sc_df = pd.DataFrame(sc_rows)

fig_sc = go.Figure()
fig_sc.add_trace(go.Bar(name='Expected Loss', x=sc_df['Scenario'], y=sc_df['EL'], marker_color='#2193b0'))
fig_sc.add_trace(go.Bar(name='Economic Capital', x=sc_df['Scenario'], y=sc_df['EC'], marker_color='#eb3349'))
fig_sc.add_trace(go.Bar(name='ES 99.9%', x=sc_df['Scenario'], y=sc_df['ES999'], marker_color='#764ba2'))
fig_sc.update_layout(title='Scenario Comparison', barmode='group', yaxis_title='Amount ($M)',
    template='plotly_white', height=450)

# 9. Scenario Radar
categories = ['EL', 'VaR', 'ES', 'EC', 'EC%EAD']
fig_radar = go.Figure()
b = sc_df.iloc[0]
for _, row in sc_df.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row['EL']/(b['EL'] or 1), row['VaR999']/(b['VaR999'] or 1),
           row['ES999']/(b['ES999'] or 1), row['EC']/(b['EC'] or 1),
           row['EC_pct']/(b['EC_pct'] or 1)],
        theta=categories, fill='toself', name=row['Scenario']))
fig_radar.update_layout(title='Scenario Severity (vs Baseline)',
    polar=dict(radialaxis=dict(visible=True)), template='plotly_white', height=450)

# 10. Migration Matrix Heatmap
fig_mig = go.Figure(data=go.Heatmap(z=MIGRATION_MATRIX*100, x=RATINGS, y=RATINGS,
    colorscale='RdYlGn_r', text=np.round(MIGRATION_MATRIX*100, 2),
    texttemplate='%{text:.2f}%', textfont={"size":9},
    hovertemplate='From %{y} to %{x}: %{z:.3f}%<extra></extra>'))
fig_mig.update_layout(title='1-Year Rating Migration Probabilities (%)',
    xaxis_title='To Rating', yaxis_title='From Rating', height=450, template='plotly_white')

# ═══════════════════════════════════════════
# BUILD HTML
# ═══════════════════════════════════════════

def chart_html(fig, div_id):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id)

# Concentration metrics
hhi_sector = sum((v/m['total_ead'])**2 for v in sector_ead.values())
hhi_country = sum((v/m['total_ead'])**2 for v in country_ead.values())

# Top 50 counterparty table
top50 = cp_df.head(50)
cp_table_rows = ""
for _, r in top50.iterrows():
    cp_table_rows += f"""<tr>
        <td>{r['Name']}</td><td>{r['GICS_Sector']}</td><td>{r['GICS_Group']}</td><td>{r['Country']}</td><td>{r['Rating']}</td>
        <td class="num">{fmt(r['EAD'])}</td><td class="num">{fmt(r['EL'])}</td>
        <td class="num">{fmt(r['VaR_Contrib'])}</td><td class="num">{fmt(r['ES_Contrib'])}</td>
        <td class="num">{fmt(r['EC_Contrib'])}</td></tr>"""

# Scenario table
sc_table_rows = ""
for _, r in sc_df.iterrows():
    sc_table_rows += f"""<tr>
        <td><b>{r['Scenario']}</b></td><td class="num">${r['EL']:,.1f}M</td>
        <td class="num">${r['VaR999']:,.1f}M</td><td class="num">${r['ES999']:,.1f}M</td>
        <td class="num">${r['EC']:,.1f}M</td><td class="num">{r['EC_pct']:.2f}%</td>
        <td class="num">{r['Div']:.1f}%</td></tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Economic Capital Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root {{ --bg: #f5f6fa; --card: #fff; --primary: #1a1a2e; --accent: #667eea; --red: #eb3349; --green: #11998e; --blue: #2193b0; --purple: #764ba2; }}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: #333; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
header {{ background: linear-gradient(135deg, var(--primary) 0%, #16213e 100%); color: white; padding: 40px 0; text-align: center; }}
header h1 {{ font-size: 2.5rem; margin-bottom: 8px; }}
header p {{ opacity: 0.8; font-size: 1.1rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 16px; margin: 30px 0; }}
.metric {{ border-radius: 12px; padding: 20px; color: white; text-align: center; }}
.metric .val {{ font-size: 1.6rem; font-weight: 700; }}
.metric .lbl {{ font-size: 0.8rem; opacity: 0.85; }}
.metric .sub {{ font-size: 0.75rem; opacity: 0.7; margin-top: 4px; }}
.m1 {{ background: linear-gradient(135deg, #667eea, #764ba2); }}
.m2 {{ background: linear-gradient(135deg, #eb3349, #f45c43); }}
.m3 {{ background: linear-gradient(135deg, #2193b0, #6dd5ed); }}
.m4 {{ background: linear-gradient(135deg, #11998e, #38ef7d); }}
.m5 {{ background: linear-gradient(135deg, #f7971e, #ffd200); }}
.m6 {{ background: linear-gradient(135deg, #4776E6, #8E54E9); }}
.section {{ margin: 30px 0; }}
.section h2 {{ font-size: 1.5rem; color: var(--primary); border-bottom: 3px solid var(--accent); padding-bottom: 8px; margin-bottom: 20px; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
.card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th {{ background: var(--primary); color: white; padding: 10px 8px; text-align: left; position: sticky; top: 0; }}
td {{ padding: 8px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f8f9ff; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.table-wrap {{ max-height: 500px; overflow-y: auto; border-radius: 8px; border: 1px solid #e0e0e0; }}
.conc-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
.conc-card {{ background: white; border-radius: 10px; padding: 16px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
.conc-card .val {{ font-size: 1.4rem; font-weight: 700; color: var(--primary); }}
.conc-card .lbl {{ font-size: 0.8rem; color: #666; }}
.model-doc {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); font-size: 0.95rem; }}
.model-doc h3 {{ color: var(--primary); margin: 16px 0 8px; }}
.model-doc p {{ margin-bottom: 10px; }}
code {{ background: #f0f0f5; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
.run-info {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin: 20px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; font-size: 0.9rem; }}
.run-info div {{ padding: 8px; background: #f8f9ff; border-radius: 8px; }}
.run-info .k {{ font-weight: 600; color: var(--primary); }}
@media (max-width: 900px) {{
    .metrics {{ grid-template-columns: repeat(3, 1fr); }}
    .grid2 {{ grid-template-columns: 1fr; }}
    .conc-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
</style>
</head>
<body>

<header>
<h1>Economic Capital Report</h1>
<p>Monte Carlo Credit Portfolio Risk Simulation &mdash; Multi-Factor Merton Model with Gaussian Copula</p>
</header>

<div class="container">

<div class="run-info">
<div><span class="k">Simulations:</span> {N_SIMS:,}</div>
<div><span class="k">Counterparties:</span> {len(cp):,}</div>
<div><span class="k">Instruments:</span> {len(inst):,}</div>
<div><span class="k">Product Types:</span> {len(set(i['instrument_type'] for i in inst))}</div>
<div><span class="k">GICS Groups:</span> {len(set(c['sector_code'] for c in cp))}</div>
<div><span class="k">GICS Sectors:</span> {len(set(c.get('gics_sector', '') for c in cp))}</div>
<div><span class="k">Countries:</span> {len(set(c['country_code'] for c in cp))}</div>
<div><span class="k">Random Seed:</span> 42</div>
<div><span class="k">Runtime:</span> {baseline_time:.1f}s</div>
</div>

<div class="metrics">
<div class="metric m1"><div class="lbl">Expected Loss</div><div class="val">{fmt(m['expected_loss'])}</div><div class="sub">{m['el_as_pct_ead']:.2f}% of EAD</div></div>
<div class="metric m2"><div class="lbl">Economic Capital</div><div class="val">{fmt(m['economic_capital'])}</div><div class="sub">{m['ec_as_pct_ead']:.2f}% of EAD</div></div>
<div class="metric m3"><div class="lbl">Credit VaR (99.9%)</div><div class="val">{fmt(m['var_999'])}</div><div class="sub">1-year horizon</div></div>
<div class="metric m4"><div class="lbl">Expected Shortfall (99.9%)</div><div class="val">{fmt(m['es_999'])}</div><div class="sub">Tail conditional expectation</div></div>
<div class="metric m5"><div class="lbl">Diversification Benefit</div><div class="val">{m['diversification_benefit']*100:.1f}%</div><div class="sub">vs standalone sum</div></div>
<div class="metric m6"><div class="lbl">Total EAD</div><div class="val">{fmt(m['total_ead'])}</div><div class="sub">{len(inst):,} instruments</div></div>
</div>

<div class="section">
<h2>1. Portfolio Loss Distribution</h2>
<div class="card">{chart_html(fig_dist, 'dist')}</div>
<div class="grid2" style="margin-top:20px">
<div class="card">{chart_html(fig_exc, 'exc')}</div>
<div class="card">
<h3 style="margin-bottom:12px; color:#1a1a2e;">Distribution Statistics</h3>
<table>
<tr><td>Expected Loss</td><td class="num"><b>{fmt(m['expected_loss'])}</b></td></tr>
<tr><td>Unexpected Loss (Std Dev)</td><td class="num">{fmt(m['unexpected_loss'])}</td></tr>
<tr><td>VaR 95%</td><td class="num">{fmt(m['var_95'])}</td></tr>
<tr><td>VaR 99%</td><td class="num">{fmt(m['var_99'])}</td></tr>
<tr><td>VaR 99.9%</td><td class="num"><b>{fmt(m['var_999'])}</b></td></tr>
<tr><td>VaR 99.99%</td><td class="num">{fmt(m['var_9999'])}</td></tr>
<tr><td>ES 95%</td><td class="num">{fmt(m['es_95'])}</td></tr>
<tr><td>ES 99%</td><td class="num">{fmt(m['es_99'])}</td></tr>
<tr><td>ES 99.9%</td><td class="num"><b>{fmt(m['es_999'])}</b></td></tr>
<tr><td>Economic Capital</td><td class="num"><b style="color:#eb3349">{fmt(m['economic_capital'])}</b></td></tr>
<tr><td>Skewness</td><td class="num">{m['skewness']:.2f}</td></tr>
<tr><td>Kurtosis</td><td class="num">{m['kurtosis']:.2f}</td></tr>
<tr><td>Max Single-Trial Loss</td><td class="num">{fmt(m['max_loss'])}</td></tr>
<tr><td>Portfolio Default Rate</td><td class="num">{m['portfolio_default_rate']*100:.3f}%</td></tr>
<tr><td>HHI (Obligor)</td><td class="num">{m['hhi']:.4f}</td></tr>
</table>
</div>
</div>
</div>

<div class="section">
<h2>2. Risk Contributions</h2>
<div class="grid2">
<div class="card">{chart_html(fig_top, 'top20')}</div>
<div class="card">{chart_html(fig_tree, 'tree')}</div>
</div>
<div class="card" style="margin-top:20px">
<h3 style="margin-bottom:12px; color:#1a1a2e;">Top 50 Counterparties by Risk Contribution</h3>
<div class="table-wrap"><table>
<thead><tr><th>Name</th><th>GICS Sector</th><th>Industry Group</th><th>Country</th><th>Rating</th><th>EAD</th><th>EL</th><th>VaR Contrib</th><th>ES Contrib</th><th>EC Contrib</th></tr></thead>
<tbody>{cp_table_rows}</tbody>
</table></div>
</div>
</div>

<div class="section">
<h2>3. Concentration Analysis</h2>
<div class="conc-grid">
<div class="conc-card"><div class="lbl">HHI (Obligor)</div><div class="val">{m['hhi']:.4f}</div></div>
<div class="conc-card"><div class="lbl">HHI (Sector)</div><div class="val">{hhi_sector:.4f}</div></div>
<div class="conc-card"><div class="lbl">HHI (Country)</div><div class="val">{hhi_country:.4f}</div></div>
<div class="conc-card"><div class="lbl">Effective # of Names</div><div class="val">{1/m['hhi']:.0f}</div></div>
</div>
<div class="grid2">
<div class="card">{chart_html(fig_sector, 'sec')}</div>
<div class="card">{chart_html(fig_country, 'cty')}</div>
</div>
<div class="grid2" style="margin-top:20px">
<div class="card">{chart_html(fig_type, 'typ')}</div>
<div class="card">{chart_html(fig_rating, 'rat')}</div>
</div>
</div>

<div class="section">
<h2>4. Stress Scenario Analysis</h2>
<div class="grid2">
<div class="card">{chart_html(fig_sc, 'scbar')}</div>
<div class="card">{chart_html(fig_radar, 'radar')}</div>
</div>
<div class="card" style="margin-top:20px">
<h3 style="margin-bottom:12px; color:#1a1a2e;">Scenario Results Summary</h3>
<table>
<thead><tr><th>Scenario</th><th>Expected Loss</th><th>VaR 99.9%</th><th>ES 99.9%</th><th>Economic Capital</th><th>EC % of EAD</th><th>Diversification</th></tr></thead>
<tbody>{sc_table_rows}</tbody>
</table>
</div>
</div>

<div class="section">
<h2>5. Migration Matrix</h2>
<div class="card">{chart_html(fig_mig, 'mig')}</div>
</div>

<div class="section">
<h2>6. Model Documentation</h2>
<div class="model-doc">
<h3>Model Framework</h3>
<p>Merton asset-value model with multi-factor systematic risk decomposition.</p>
<h3>Correlation Structure</h3>
<p>GCorr-style decomposition with 15 country factors and 30 GICS industry group factors (45 total). Intra-sector groups have higher correlation. Each obligor's asset return:</p>
<p><code>r_i = sqrt(RSQ_i) &times; Z_systematic + sqrt(1 - RSQ_i) &times; &epsilon;_i</code></p>
<p>where <code>Z_systematic = &Sigma; w_ij &times; F_j</code> and factors <code>F_j</code> are jointly normal with a Cholesky-decomposed correlation matrix.</p>
<h3>Default Determination</h3>
<p>Default occurs when the asset return falls below the Merton threshold: <code>&Phi;<sup>-1</sup>(PD)</code>.</p>
<h3>Loss Given Default</h3>
<p>Stochastic LGD with PD-LGD correlation through the systematic factor. LGD is drawn from a normal distribution centered at the seniority-adjusted mean, shifted by the systematic component to capture downturn effects.</p>
<h3>Economic Capital</h3>
<p><code>EC = VaR(99.9%) - EL</code> over a 1-year horizon, consistent with regulatory and ICAAP frameworks.</p>
<h3>Risk Contributions</h3>
<p>Euler allocation via conditional expectation in the tail (ES-based). Contributions are additive and sum to portfolio-level risk.</p>
<h3>Scenario Conditioning</h3>
<p>GCorr Macro-style: factor draws are shifted by scenario-specified standard deviations, with cross-factor propagation via the correlation matrix.</p>
<h3>Product Coverage</h3>
<p>Term Loans, Revolving Credit Facilities, Interest Rate Derivatives, FX Derivatives, Credit Default Swaps, Trade Finance, and Guarantees. EAD calculation is product-specific (drawn amounts, CCF for undrawn, MTM + add-on for derivatives).</p>
<h3>Governance</h3>
<p>Deterministic seeding ensures full reproducibility. Chunked processing supports up to 1M+ simulations within memory constraints. All run parameters are recorded for audit trail compliance (SR 11-7, BCBS 239).</p>
</div>
</div>

<footer style="text-align:center; padding:30px; color:#999; font-size:0.85rem;">
Economic Capital Calculator &mdash; Generated {time.strftime('%Y-%m-%d %H:%M:%S')} &mdash; Seed: 42 &mdash; {N_SIMS:,} Simulations
</footer>

</div>
</body>
</html>"""

with open(OUTPUT, 'w') as f:
    f.write(html)

print(f"\nReport saved to: {OUTPUT}")
print(f"File size: {os.path.getsize(OUTPUT) / 1024:.0f} KB")
