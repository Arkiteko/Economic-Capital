"""
RiskMatrix - EC Engine - Streamlit Application
====================================================
Dark-themed professional credit portfolio risk management tool.
Monte Carlo simulation with multi-factor Merton model and Gaussian copula.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.simulation import run_simulation
from data.generator import (
    generate_all_data, MIGRATION_MATRIX, RATINGS,
    GICS_INDUSTRY_GROUPS, GICS_GROUP_TO_SECTOR, GICS_GROUP_DISPLAY_NAMES, GICS_SECTORS,
)
from data.upload import parse_uploaded_excel

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RiskMatrix - EC Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Dark Theme CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #1c2333;
        --bg-card: #1a2030;
        --border: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-blue: #58a6ff;
        --accent-cyan: #79c0ff;
        --accent-green: #3fb950;
        --accent-red: #f85149;
        --accent-orange: #ffa657;
        --accent-purple: #d2a8ff;
        --accent-yellow: #e3b341;
        --glow-blue: rgba(88, 166, 255, 0.15);
        --glow-red: rgba(248, 81, 73, 0.15);
        --glow-green: rgba(63, 185, 80, 0.15);
    }

    .stApp, .main .block-container { background-color: var(--bg-primary) !important; color: var(--text-primary) !important; }

    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    section[data-testid="stSidebar"] .stMarkdown p { color: var(--text-secondary) !important; }
    section[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0; background: var(--bg-secondary); border-radius: 8px; padding: 4px;
        border: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px; border-radius: 6px; font-weight: 500; font-size: 0.85rem;
        color: var(--text-secondary) !important; background: transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent-blue) !important; color: #fff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    .stDataFrame, .stDataFrame * { color: var(--text-primary) !important; }

    /* Dark inputs for sidebar selectbox, number input, multiselect */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    section[data-testid="stSidebar"] input {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] span,
    section[data-testid="stSidebar"] [data-baseweb="select"] div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] input { color: var(--text-primary) !important; }
    /* Dropdown menu styling */
    [data-baseweb="popover"], [data-baseweb="menu"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
    }
    [data-baseweb="menu"] li { color: var(--text-primary) !important; }
    [data-baseweb="menu"] li:hover { background-color: var(--bg-tertiary) !important; }
    /* Main area inputs too */
    [data-baseweb="select"] > div { background-color: var(--bg-tertiary) !important; color: var(--text-primary) !important; border-color: var(--border) !important; }
    [data-baseweb="select"] span { color: var(--text-primary) !important; }
    [data-baseweb="input"] > div { background-color: var(--bg-tertiary) !important; border-color: var(--border) !important; }
    [data-baseweb="input"] input { color: var(--text-primary) !important; }
    /* Multiselect tags */
    [data-baseweb="tag"] { background-color: var(--accent-blue) !important; color: #fff !important; }
    /* Dark buttons */
    section[data-testid="stSidebar"] button[kind="secondary"],
    section[data-testid="stSidebar"] button[kind="secondaryFormSubmit"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover,
    section[data-testid="stSidebar"] button[kind="secondaryFormSubmit"]:hover {
        background-color: var(--bg-card) !important;
        border-color: var(--accent-blue) !important;
    }
    /* Download buttons in main area */
    button[kind="secondary"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }
    button[kind="secondary"]:hover {
        border-color: var(--accent-blue) !important;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px;
    }
    div[data-testid="stMetric"] label { color: var(--text-secondary) !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--accent-cyan) !important; }

    .hero-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a1a3e 100%);
        border: 1px solid var(--border); border-radius: 12px;
        padding: 32px 40px; margin-bottom: 24px;
        position: relative; overflow: hidden;
    }
    .hero-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
    }
    .hero-header h1 { font-family: 'Inter', sans-serif; font-size: 2rem; font-weight: 700; color: #fff; margin: 0 0 6px 0; letter-spacing: -0.02em; }
    .hero-header p { font-family: 'Inter', sans-serif; font-size: 0.95rem; color: var(--text-secondary); margin: 0; }

    .metric-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin: 20px 0; }
    .metric-card {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
        padding: 18px 16px; text-align: center; position: relative; overflow: hidden;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: var(--accent-blue); }
    .metric-card .accent-bar { position: absolute; top: 0; left: 0; right: 0; height: 3px; }
    .metric-card .label { font-family: 'Inter', sans-serif; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 8px; }
    .metric-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 700; color: #fff; line-height: 1.2; }
    .metric-card .sub { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-muted); margin-top: 4px; }

    .mc-blue .accent-bar { background: var(--accent-blue); } .mc-blue .value { color: var(--accent-cyan); }
    .mc-red .accent-bar { background: var(--accent-red); } .mc-red .value { color: var(--accent-red); }
    .mc-green .accent-bar { background: var(--accent-green); } .mc-green .value { color: var(--accent-green); }
    .mc-purple .accent-bar { background: var(--accent-purple); } .mc-purple .value { color: var(--accent-purple); }
    .mc-orange .accent-bar { background: var(--accent-orange); } .mc-orange .value { color: var(--accent-orange); }
    .mc-cyan .accent-bar { background: var(--accent-cyan); } .mc-cyan .value { color: var(--accent-cyan); }

    .section-title {
        font-family: 'Inter', sans-serif; font-size: 1.15rem; font-weight: 600;
        color: var(--text-primary); border-left: 3px solid var(--accent-blue);
        padding-left: 12px; margin: 28px 0 16px 0;
    }

    .run-badge {
        display: inline-block; background: var(--bg-card); border: 1px solid var(--accent-green);
        border-radius: 6px; padding: 6px 14px; font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem; color: var(--accent-green); margin-bottom: 16px;
    }

    .sidebar-section {
        font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent-blue) !important;
        margin: 16px 0 8px 0; padding-bottom: 4px;
        border-bottom: 1px solid var(--border);
    }

    .upload-box {
        background: var(--bg-card); border: 1px dashed var(--border); border-radius: 10px;
        padding: 16px; margin: 10px 0; text-align: center;
    }

    @media (max-width: 1200px) { .metric-grid { grid-template-columns: repeat(3, 1fr); } }
    @media (max-width: 768px) { .metric-grid { grid-template-columns: repeat(2, 1fr); } }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Dark Plotly Template
# ──────────────────────────────────────────────
DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,17,23,0.95)',
        font=dict(family='Inter, system-ui, sans-serif', color='#c9d1d9', size=12),
        title=dict(font=dict(size=14, color='#e6edf3'), x=0.02, xanchor='left'),
        xaxis=dict(
            gridcolor='rgba(48,54,61,0.5)', gridwidth=1,
            zerolinecolor='rgba(48,54,61,0.7)',
            tickfont=dict(color='#8b949e', size=11),
            title_font=dict(color='#8b949e', size=12),
            linecolor='#30363d', linewidth=1, mirror=False,
        ),
        yaxis=dict(
            gridcolor='rgba(48,54,61,0.5)', gridwidth=1,
            zerolinecolor='rgba(48,54,61,0.7)',
            tickfont=dict(color='#8b949e', size=11),
            title_font=dict(color='#8b949e', size=12),
            linecolor='#30363d', linewidth=1, mirror=False,
        ),
        colorway=['#58a6ff', '#3fb950', '#f85149', '#d2a8ff', '#ffa657', '#79c0ff', '#e3b341', '#f778ba'],
        legend=dict(
            bgcolor='rgba(0,0,0,0)', font=dict(color='#8b949e', size=11),
            bordercolor='rgba(0,0,0,0)',
        ),
        hoverlabel=dict(
            bgcolor='#1c2333', bordercolor='#30363d',
            font=dict(color='#e6edf3', family='JetBrains Mono, monospace', size=12),
        ),
        margin=dict(l=50, r=20, t=50, b=50),
        bargap=0.15,
    )
)

# Chart color constants
C_BLUE = '#58a6ff'
C_GREEN = '#3fb950'
C_RED = '#f85149'
C_PURPLE = '#d2a8ff'
C_ORANGE = '#ffa657'
C_CYAN = '#79c0ff'
C_YELLOW = '#e3b341'
C_PINK = '#f778ba'
C_HIST = 'rgba(88,166,255,0.55)'
C_HIST_LINE = 'rgba(88,166,255,0.9)'

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'run_history' not in st.session_state:
    st.session_state.run_history = []
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'generate'


def format_currency(val):
    if abs(val) >= 1e9:
        return f"${val/1e9:,.2f}B"
    elif abs(val) >= 1e6:
        return f"${val/1e6:,.2f}M"
    elif abs(val) >= 1e3:
        return f"${val/1e3:,.1f}K"
    return f"${val:,.0f}"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio("Portfolio Data", ['Generate Sample', 'Upload Excel'],
                            label_visibility='collapsed', horizontal=True)

    if data_source == 'Upload Excel':
        uploaded_file = st.file_uploader("Upload .xlsx", type=['xlsx'], label_visibility='collapsed')
        if uploaded_file is not None:
            with st.spinner("Parsing..."):
                data, errs, warns = parse_uploaded_excel(uploaded_file.read())
            if errs:
                for e in errs:
                    st.error(e, icon="🚫")
            if warns:
                with st.expander(f"⚠️ {len(warns)} warnings", expanded=False):
                    for w in warns:
                        st.warning(w, icon="⚠️")
            if data:
                st.session_state.data = data
                st.session_state.results = None
                st.session_state.data_source = 'upload'
                n_cp = len(data['counterparties'])
                n_inst = len(data['instruments'])
                st.success(f"Loaded {n_cp} counterparties, {n_inst} instruments", icon="✅")

    st.markdown('<div class="sidebar-section">Simulation</div>', unsafe_allow_html=True)

    n_simulations = st.select_slider(
        "Simulations",
        options=[10000, 50000, 100000, 250000, 500000, 1000000],
        value=100000,
        help="Number of Monte Carlo trials"
    )

    confidence_level = st.selectbox("Confidence Level", options=[95.0, 99.0, 99.5, 99.9, 99.97], index=3)
    seed = st.number_input("Random Seed", value=42, min_value=1, max_value=99999)

    if data_source == 'Generate Sample':
        st.markdown('<div class="sidebar-section">Portfolio</div>', unsafe_allow_html=True)
        n_counterparties = st.slider("Counterparties", 50, 1000, 500, 50)
        n_instruments = st.slider("Instruments", 100, 5000, 2000, 100)

    st.markdown('<div class="sidebar-section">Model</div>', unsafe_allow_html=True)
    pd_lgd_corr = st.slider("PD-LGD Correlation", 0.0, 0.6, 0.3, 0.05)
    lgd_volatility = st.slider("LGD Volatility", 0.05, 0.40, 0.15, 0.05)

    st.markdown('<div class="sidebar-section">Scenario</div>', unsafe_allow_html=True)
    scenario_options = ['Baseline', 'Mild Downturn', 'Severe Recession',
                        'Emerging Market Crisis', 'Interest Rate Spike']
    selected_scenario = st.selectbox("Stress Scenario", scenario_options, label_visibility='collapsed')

    st.markdown("---")

    if data_source == 'Generate Sample':
        if st.button("🔄 Generate Data", use_container_width=True, type="secondary"):
            with st.spinner("Generating..."):
                st.session_state.data = generate_all_data(n_counterparties, n_instruments, seed)
                st.session_state.results = None
                st.session_state.data_source = 'generate'
            st.success(f"Generated {n_counterparties} CP, {n_instruments} instruments")

    if st.button("▶ Run Simulation", use_container_width=True, type="primary"):
        if st.session_state.data is None:
            if data_source == 'Generate Sample':
                st.session_state.data = generate_all_data(n_counterparties, n_instruments, seed)
            else:
                st.warning("Please upload portfolio data first")
                st.stop()
        st.session_state.run_trigger = True


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏦 RiskMatrix — EC Engine</h1>
    <p>Monte Carlo Credit Portfolio Simulation — Multi-Factor Merton Model with Gaussian Copula • GICS Industry Classification</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Run Simulation
# ──────────────────────────────────────────────
if getattr(st.session_state, 'run_trigger', False):
    st.session_state.run_trigger = False
    data = st.session_state.data

    scenario_map = {
        'Baseline': data['scenarios'][0],
        'Mild Downturn': data['scenarios'][1],
        'Severe Recession': data['scenarios'][2],
        'Emerging Market Crisis': data['scenarios'][3],
        'Interest Rate Spike': data['scenarios'][4],
    }
    scenario = scenario_map.get(selected_scenario, data['scenarios'][0])

    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    def update_progress(pct, msg):
        progress_bar.progress(pct)
        status_text.text(f"⏳ {msg} ({pct*100:.0f}%)")

    counterparties = data['counterparties']
    instruments = data['instruments']
    pd_values = np.array([cp['pd_1y'] for cp in counterparties])

    cp_id_to_idx = {cp['counterparty_id']: i for i, cp in enumerate(counterparties)}
    obligor_lgd = np.zeros(len(counterparties))
    obligor_lgd_count = np.zeros(len(counterparties))
    for inst in instruments:
        idx = cp_id_to_idx.get(inst['counterparty_id'], 0)
        obligor_lgd[idx] += inst['lgd']
        obligor_lgd_count[idx] += 1
    obligor_lgd = np.where(obligor_lgd_count > 0, obligor_lgd / obligor_lgd_count, 0.45)

    results = run_simulation(
        counterparties=counterparties,
        instruments=instruments,
        pd_values=pd_values,
        lgd_values=obligor_lgd,
        migration_matrix=MIGRATION_MATRIX,
        current_ratings=[RATINGS.index(cp.get('rating', 'BBB')) for cp in counterparties],
        n_simulations=n_simulations,
        seed=seed,
        scenario=scenario if selected_scenario != 'Baseline' else None,
        progress_callback=update_progress
    )

    elapsed = time.time() - start_time
    progress_bar.progress(1.0)
    status_text.empty()

    st.session_state.results = results
    st.session_state.run_history.append({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_simulations': n_simulations,
        'scenario': selected_scenario,
        'seed': seed,
        'economic_capital': results['metrics']['economic_capital'],
        'var_999': results['metrics']['var_999'],
        'expected_loss': results['metrics']['expected_loss'],
        'elapsed_seconds': elapsed,
    })

    st.markdown(f'<div class="run-badge">✓ {n_simulations:,} trials · {elapsed:.1f}s · {selected_scenario}</div>',
                unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Display Results
# ──────────────────────────────────────────────
if st.session_state.results is not None:
    results = st.session_state.results
    metrics = results['metrics']
    data = st.session_state.data

    # ── Metric Cards ──
    cards = [
        ('Expected Loss', format_currency(metrics['expected_loss']), f"{metrics['el_as_pct_ead']:.2f}% of EAD", 'mc-blue'),
        ('Economic Capital', format_currency(metrics['economic_capital']), f"{metrics['ec_as_pct_ead']:.2f}% of EAD", 'mc-red'),
        ('Credit VaR 99.9%', format_currency(metrics['var_999']), '1-year horizon', 'mc-orange'),
        ('Expected Shortfall', format_currency(metrics['es_999']), '99.9% tail', 'mc-green'),
        ('Diversification', f"{metrics['diversification_benefit']*100:.1f}%", 'vs standalone', 'mc-purple'),
        ('Total EAD', format_currency(metrics['total_ead']), f"{len(data['instruments']):,} instruments", 'mc-cyan'),
    ]

    cards_html = '<div class="metric-grid">'
    for label, value, sub, cls in cards:
        cards_html += f'''<div class="metric-card {cls}">
            <div class="accent-bar"></div>
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
        </div>'''
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs([
        "📊 Loss Distribution",
        "🎯 Risk Contributions",
        "🏢 Concentration",
        "📈 Scenarios",
        "💳 CDS Analysis",
        "🔍 Instruments",
        "📋 Portfolio",
        "📜 Audit Log",
        "📖 Upload Guide",
        "📥 Export",
    ])

    # Use the expanded instrument list from simulation (includes CVA instruments)
    all_instruments = results.get('instruments', data['instruments'])
    original_instruments = data['instruments']
    n_original = results.get('n_original_instruments', len(original_instruments))
    n_cva = results.get('n_cva_instruments', 0)

    instruments_list = all_instruments
    var_contrib = metrics['var_contributions']
    es_contrib = metrics['es_contributions']
    inst_el = metrics['instrument_el']
    ead = results['ead']

    # Counterparty aggregation (shared across tabs)
    cp_contributions = {}
    for j, inst in enumerate(instruments_list):
        cp_id = inst['counterparty_id']
        if cp_id not in cp_contributions:
            cp_contributions[cp_id] = {'var_contrib': 0, 'es_contrib': 0, 'el': 0, 'ead': 0}
        cp_contributions[cp_id]['var_contrib'] += var_contrib[j]
        cp_contributions[cp_id]['es_contrib'] += es_contrib[j]
        cp_contributions[cp_id]['el'] += inst_el[j]
        cp_contributions[cp_id]['ead'] += ead[j]

    cp_lookup = {cp['counterparty_id']: cp for cp in data['counterparties']}
    cp_df = pd.DataFrame([
        {
            'Name': cp_lookup[cid]['legal_name'],
            'GICS Group': GICS_GROUP_DISPLAY_NAMES.get(cp_lookup[cid]['sector_code'], cp_lookup[cid]['sector_code']),
            'GICS Sector': cp_lookup[cid].get('gics_sector', GICS_GROUP_TO_SECTOR.get(cp_lookup[cid]['sector_code'], '')),
            'Country': cp_lookup[cid]['country_code'],
            'Rating': cp_lookup[cid]['rating'],
            'EAD': v['ead'],
            'Expected Loss': v['el'],
            'VaR Contribution': v['var_contrib'],
            'ES Contribution': v['es_contrib'],
            'EC Contribution': v['var_contrib'] - v['el'],
        }
        for cid, v in cp_contributions.items()
    ]).sort_values('VaR Contribution', ascending=False)

    # ── TAB 1: Loss Distribution ──
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        with col1:
            losses = results['portfolio_losses']
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=losses / 1e6, nbinsx=250, name='Loss Distribution',
                marker=dict(
                    color=C_HIST,
                    line=dict(color=C_HIST_LINE, width=0.5),
                ),
                hovertemplate='Loss: $%{x:.1f}M<br>Count: %{y:,}<extra></extra>'
            ))
            vlines = [
                ('EL', metrics['expected_loss'], C_GREEN, 'dot'),
                ('VaR 95%', metrics['var_95'], C_ORANGE, 'dash'),
                ('VaR 99%', metrics['var_99'], C_RED, 'dash'),
                ('VaR 99.9%', metrics['var_999'], '#ff7b72', 'dashdot'),
                ('ES 99.9%', metrics['es_999'], C_PURPLE, 'dashdot'),
            ]
            for label, val, color, dash in vlines:
                fig.add_vline(x=val/1e6, line_dash=dash, line_color=color, line_width=1.5,
                              annotation_text=f"{label}: {format_currency(val)}",
                              annotation_font=dict(color=color, size=10),
                              annotation_bgcolor='rgba(13,17,23,0.7)')
            fig.update_layout(template=DARK_TEMPLATE, title='Portfolio Loss Distribution',
                              xaxis_title='Loss ($M)', yaxis_title='Frequency',
                              height=500, showlegend=False, bargap=0.02)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Distribution Statistics</div>', unsafe_allow_html=True)
            stats_df = pd.DataFrame({
                'Metric': ['Expected Loss', 'Unexpected Loss', 'VaR 95%', 'VaR 99%', 'VaR 99.9%',
                           'VaR 99.99%', 'ES 95%', 'ES 99%', 'ES 99.9%',
                           'Economic Capital', 'Skewness', 'Kurtosis', 'Max Loss', 'Default Rate'],
                'Value': [
                    format_currency(metrics['expected_loss']),
                    format_currency(metrics['unexpected_loss']),
                    format_currency(metrics['var_95']),
                    format_currency(metrics['var_99']),
                    format_currency(metrics['var_999']),
                    format_currency(metrics['var_9999']),
                    format_currency(metrics['es_95']),
                    format_currency(metrics['es_99']),
                    format_currency(metrics['es_999']),
                    format_currency(metrics['economic_capital']),
                    f"{metrics['skewness']:.2f}",
                    f"{metrics['kurtosis']:.2f}",
                    format_currency(metrics['max_loss']),
                    f"{metrics['portfolio_default_rate']*100:.3f}%",
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        col3, col4 = st.columns(2)
        with col3:
            sorted_losses = np.sort(losses)
            exceedance = 1 - np.arange(1, len(sorted_losses)+1) / len(sorted_losses)
            sample_idx = np.linspace(0, len(sorted_losses)-1, 2000, dtype=int)
            fig_exc = go.Figure()
            fig_exc.add_trace(go.Scatter(
                x=sorted_losses[sample_idx]/1e6, y=exceedance[sample_idx],
                mode='lines', name='Exceedance',
                line=dict(color=C_RED, width=2.5),
                fill='tozeroy', fillcolor='rgba(248,81,73,0.08)',
            ))
            fig_exc.update_layout(template=DARK_TEMPLATE, title='Loss Exceedance Curve',
                                  xaxis_title='Loss ($M)', yaxis_title='Exceedance Probability',
                                  yaxis_type='log', height=400, showlegend=False)
            st.plotly_chart(fig_exc, use_container_width=True)

        with col4:
            theoretical = np.random.default_rng(42).standard_normal(min(5000, len(losses)))
            theoretical.sort()
            sample_q = np.sort(np.random.default_rng(42).choice(losses, min(5000, len(losses)), replace=False))
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=theoretical, y=sample_q/1e6,
                mode='markers', marker=dict(size=3, color=C_CYAN, opacity=0.6),
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: $%{y:.1f}M<extra></extra>'
            ))
            # Reference line
            min_t, max_t = theoretical[0], theoretical[-1]
            mean_s, std_s = np.mean(sample_q/1e6), np.std(sample_q/1e6)
            fig_qq.add_trace(go.Scatter(
                x=[min_t, max_t], y=[mean_s + min_t*std_s, mean_s + max_t*std_s],
                mode='lines', line=dict(color=C_ORANGE, width=1.5, dash='dot'),
                showlegend=False,
            ))
            fig_qq.update_layout(template=DARK_TEMPLATE, title='Q-Q Plot vs Normal',
                                 xaxis_title='Theoretical Quantiles', yaxis_title='Sample ($M)',
                                 height=400, showlegend=False)
            st.plotly_chart(fig_qq, use_container_width=True)

    # ── TAB 2: Risk Contributions ──
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            top20 = cp_df.head(20)
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                x=top20['Name'], y=top20['VaR Contribution']/1e6,
                name='VaR Contribution',
                marker=dict(color=C_RED, line=dict(width=0)),
            ))
            fig_top.add_trace(go.Bar(
                x=top20['Name'], y=top20['Expected Loss']/1e6,
                name='Expected Loss',
                marker=dict(color=C_BLUE, line=dict(width=0)),
            ))
            fig_top.update_layout(template=DARK_TEMPLATE, title='Top 20 Counterparties by Risk',
                                  yaxis_title='Amount ($M)', barmode='group', height=500,
                                  xaxis_tickangle=-45, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'))
            st.plotly_chart(fig_top, use_container_width=True)

        with col2:
            tree_data = cp_df.copy()
            tree_data = tree_data[tree_data['VaR Contribution'] > 0]
            fig_tree = px.treemap(
                tree_data, path=['GICS Sector', 'GICS Group'], values='VaR Contribution',
                color='EC Contribution',
                color_continuous_scale=[[0, '#0d4429'], [0.3, '#1a6b3c'], [0.5, '#e3b341'], [0.8, '#da3633'], [1, '#8b1a1a']],
                title='Risk by GICS Sector → Industry Group'
            )
            fig_tree.update_layout(template=DARK_TEMPLATE, height=500,
                                   coloraxis_colorbar=dict(title='EC', tickfont=dict(color='#8b949e')))
            st.plotly_chart(fig_tree, use_container_width=True)

        st.markdown('<div class="section-title">Counterparty Risk Detail (Top 50)</div>', unsafe_allow_html=True)
        display_df = cp_df.head(50).copy()
        for col in ['EAD', 'Expected Loss', 'VaR Contribution', 'ES Contribution', 'EC Contribution']:
            display_df[col] = display_df[col].apply(lambda x: f"${x/1e6:,.2f}M")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── TAB 3: Concentration ──
    with tabs[2]:
        # Build lookup for EAD by various dimensions
        sector_ead, group_ead, country_ead, type_ead, rating_ead = {}, {}, {}, {}, {}
        for j, inst in enumerate(instruments_list):
            cp = cp_lookup.get(inst['counterparty_id'])
            if cp is None:
                continue
            gics_sector = cp.get('gics_sector', GICS_GROUP_TO_SECTOR.get(cp['sector_code'], 'Other'))
            gics_group = GICS_GROUP_DISPLAY_NAMES.get(cp['sector_code'], cp['sector_code'])
            sector_ead[gics_sector] = sector_ead.get(gics_sector, 0) + ead[j]
            group_ead[gics_group] = group_ead.get(gics_group, 0) + ead[j]
            country_ead[cp['country_code']] = country_ead.get(cp['country_code'], 0) + ead[j]
            type_ead[inst['instrument_type']] = type_ead.get(inst['instrument_type'], 0) + ead[j]
            rating_ead[inst.get('rating', 'NR')] = rating_ead.get(inst.get('rating', 'NR'), 0) + ead[j]

        _pie_colors = [C_BLUE, C_GREEN, C_PURPLE, C_ORANGE, C_CYAN, C_YELLOW, C_RED, C_PINK,
                       '#56d4dd', '#a5d6a7', '#ce93d8', '#ffab91', '#80cbc4', '#b0bec5']

        col1, col2 = st.columns(2)
        with col1:
            fig_sector = go.Figure(data=[go.Pie(
                labels=list(sector_ead.keys()), values=list(sector_ead.values()),
                hole=0.5, textinfo='label+percent', textfont=dict(size=11, color='#e6edf3'),
                marker=dict(colors=_pie_colors, line=dict(color='#0d1117', width=2)),
                hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>',
            )])
            fig_sector.update_layout(template=DARK_TEMPLATE, title='EAD by GICS Sector', height=420,
                                     legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_sector, use_container_width=True)
        with col2:
            fig_country = go.Figure(data=[go.Pie(
                labels=list(country_ead.keys()), values=list(country_ead.values()),
                hole=0.5, textinfo='label+percent', textfont=dict(size=11, color='#e6edf3'),
                marker=dict(colors=_pie_colors[3:] + _pie_colors[:3], line=dict(color='#0d1117', width=2)),
                hovertemplate='%{label}<br>$%{value:,.0f}<br>%{percent}<extra></extra>',
            )])
            fig_country.update_layout(template=DARK_TEMPLATE, title='EAD by Country', height=420,
                                      legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_country, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            sorted_types = sorted(type_ead.items(), key=lambda x: x[1], reverse=True)
            fig_type = go.Figure(data=[go.Bar(
                x=[t[0] for t in sorted_types], y=[t[1]/1e6 for t in sorted_types],
                marker=dict(color=C_BLUE, line=dict(width=0)),
                hovertemplate='%{x}<br>$%{y:,.1f}M<extra></extra>',
            )])
            fig_type.update_layout(template=DARK_TEMPLATE, title='EAD by Product Type',
                                   yaxis_title='EAD ($M)', height=400)
            st.plotly_chart(fig_type, use_container_width=True)
        with col4:
            ordered_ratings = [r for r in RATINGS if r in rating_ead]
            rating_colors = [C_GREEN, C_BLUE, C_CYAN, C_PURPLE, C_ORANGE, C_RED, '#ff7b72', '#8b949e']
            fig_rating = go.Figure(data=[go.Bar(
                x=ordered_ratings,
                y=[rating_ead.get(r, 0)/1e6 for r in ordered_ratings],
                marker_color=rating_colors[:len(ordered_ratings)],
                hovertemplate='%{x}<br>$%{y:,.1f}M<extra></extra>',
            )])
            fig_rating.update_layout(template=DARK_TEMPLATE, title='EAD by Rating',
                                     yaxis_title='EAD ($M)', height=400)
            st.plotly_chart(fig_rating, use_container_width=True)

        # GICS Industry Group breakdown — horizontal for readability
        st.markdown('<div class="section-title">EAD by GICS Industry Group</div>', unsafe_allow_html=True)
        sorted_groups = sorted(group_ead.items(), key=lambda x: x[1], reverse=False)
        top_groups = sorted_groups[-20:]  # last 20 = largest
        fig_groups = go.Figure(data=[go.Bar(
            y=[g[0] for g in top_groups],
            x=[g[1]/1e6 for g in top_groups],
            orientation='h',
            marker=dict(
                color=[g[1]/1e6 for g in top_groups],
                colorscale=[[0, '#1a3a5c'], [0.5, '#58a6ff'], [1, '#d2a8ff']],
                line=dict(width=0),
            ),
            hovertemplate='%{y}<br>$%{x:,.1f}M<extra></extra>',
        )])
        fig_groups.update_layout(template=DARK_TEMPLATE, title='Top 20 GICS Industry Groups by EAD',
                                 xaxis_title='EAD ($M)', height=500, showlegend=False,
                                 margin=dict(l=220))
        st.plotly_chart(fig_groups, use_container_width=True)

        # Concentration metrics
        st.markdown('<div class="section-title">Concentration Metrics</div>', unsafe_allow_html=True)
        hhi_sector = sum((v/metrics['total_ead'])**2 for v in sector_ead.values())
        hhi_country = sum((v/metrics['total_ead'])**2 for v in country_ead.values())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HHI (Obligor)", f"{metrics['hhi']:.4f}")
        c2.metric("HHI (GICS Sector)", f"{hhi_sector:.4f}")
        c3.metric("HHI (Country)", f"{hhi_country:.4f}")
        c4.metric("Effective Names", f"{1/metrics['hhi']:.0f}" if metrics['hhi'] > 0 else "N/A")

    # ── TAB 4: Scenarios ──
    with tabs[3]:
        st.markdown('<div class="section-title">Multi-Scenario Comparison</div>', unsafe_allow_html=True)
        if st.button("🔁 Run All Scenarios", key="run_all_scenarios"):
            scenario_results = {}
            progress = st.progress(0)
            for i, sc in enumerate(data['scenarios']):
                progress.progress(i / len(data['scenarios']), text=f"Running: {sc['scenario_name']}...")
                sc_result = run_simulation(
                    counterparties=data['counterparties'],
                    instruments=data['instruments'],
                    pd_values=np.array([cp['pd_1y'] for cp in data['counterparties']]),
                    lgd_values=np.array([
                        np.mean([inst['lgd'] for inst in data['instruments']
                                 if inst['counterparty_id']==cp['counterparty_id']] or [0.45])
                        for cp in data['counterparties']
                    ]),
                    n_simulations=min(n_simulations, 100000),
                    seed=seed,
                    scenario=sc if sc['scenario_type'] != 'baseline' else None,
                )
                scenario_results[sc['scenario_name']] = sc_result['metrics']
            progress.progress(1.0, text="All scenarios complete!")
            st.session_state.scenario_results = scenario_results

        if 'scenario_results' in st.session_state:
            sc_res = st.session_state.scenario_results
            sc_df = pd.DataFrame([
                {
                    'Scenario': name,
                    'Expected Loss ($M)': m['expected_loss']/1e6,
                    'VaR 99.9% ($M)': m['var_999']/1e6,
                    'ES 99.9% ($M)': m['es_999']/1e6,
                    'Economic Capital ($M)': m['economic_capital']/1e6,
                    'EC % of EAD': m['ec_as_pct_ead'],
                    'Diversification': m['diversification_benefit']*100,
                }
                for name, m in sc_res.items()
            ])

            col1, col2 = st.columns(2)
            with col1:
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Bar(name='Expected Loss', x=sc_df['Scenario'],
                                        y=sc_df['Expected Loss ($M)'], marker_color='#58a6ff'))
                fig_sc.add_trace(go.Bar(name='Economic Capital', x=sc_df['Scenario'],
                                        y=sc_df['Economic Capital ($M)'], marker_color='#f85149'))
                fig_sc.add_trace(go.Bar(name='ES 99.9%', x=sc_df['Scenario'],
                                        y=sc_df['ES 99.9% ($M)'], marker_color='#d2a8ff'))
                fig_sc.update_layout(template=DARK_TEMPLATE, title='Scenario Comparison',
                                     barmode='group', yaxis_title='Amount ($M)', height=450)
                st.plotly_chart(fig_sc, use_container_width=True)

            with col2:
                categories = ['EL', 'VaR', 'ES', 'EC', 'EC%EAD']
                fig_radar = go.Figure()
                b = sc_df.iloc[0]
                for _, row in sc_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row['Expected Loss ($M)']/(b['Expected Loss ($M)'] or 1),
                           row['VaR 99.9% ($M)']/(b['VaR 99.9% ($M)'] or 1),
                           row['ES 99.9% ($M)']/(b['ES 99.9% ($M)'] or 1),
                           row['Economic Capital ($M)']/(b['Economic Capital ($M)'] or 1),
                           row['EC % of EAD']/(b['EC % of EAD'] or 1)],
                        theta=categories, fill='toself', name=row['Scenario']
                    ))
                fig_radar.update_layout(template=DARK_TEMPLATE, title='Scenario Severity (vs Baseline)',
                                        polar=dict(radialaxis=dict(visible=True, gridcolor='#30363d'),
                                                   angularaxis=dict(gridcolor='#30363d')),
                                        height=450)
                st.plotly_chart(fig_radar, use_container_width=True)

            st.dataframe(sc_df.round(2), use_container_width=True, hide_index=True)

    # ── TAB 5: CDS Analysis ──
    with tabs[4]:
        cds_list = [inst for inst in instruments_list if inst.get('instrument_type') in ('CDS', 'CDS_CVA')]
        if cds_list:
            cds_bought = [i for i in cds_list if i.get('cds_direction') == 'Protection_Bought' and i.get('instrument_type') == 'CDS']
            cds_sold = [i for i in cds_list if i.get('cds_direction') == 'Protection_Sold']
            cds_cva = [i for i in cds_list if i.get('instrument_type') == 'CDS_CVA']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CDS Count", f"{len(cds_bought) + len(cds_sold)}")
            col2.metric("Protection Bought", f"{len(cds_bought)}")
            col3.metric("Protection Sold", f"{len(cds_sold)}")
            col4.metric("CVA Exposures", f"{len(cds_cva)}")

            st.markdown('<div class="section-title">CDS Position Summary</div>', unsafe_allow_html=True)

            # Build CDS detail table
            cds_rows = []
            for j, inst in enumerate(instruments_list):
                if inst.get('instrument_type') not in ('CDS', 'CDS_CVA'):
                    continue
                cp_info = cp_lookup.get(inst['counterparty_id'], {})
                direction = inst.get('cds_direction', '')
                itype = inst.get('instrument_type', 'CDS')
                row_data = {
                    'ID': inst['instrument_id'],
                    'Type': 'CDS (Sold)' if direction == 'Protection_Sold' else ('CDS (Bought)' if direction == 'Protection_Bought' else 'CVA'),
                    'Reference Entity': cp_info.get('legal_name', inst['counterparty_id']),
                    'Seller': inst.get('cds_seller_name', inst.get('cds_seller_id', '')),
                    'Notional': abs(ead[j]),
                    'EAD (Signed)': ead[j],
                    'Spread (bps)': float(inst.get('cds_spread_bps', 0)) if inst.get('cds_spread_bps') else 0.0,
                    'EL': inst_el[j],
                    'VaR Contrib': var_contrib[j],
                    'EC Contrib': var_contrib[j] - inst_el[j],
                }
                cds_rows.append(row_data)

            if cds_rows:
                cds_df = pd.DataFrame(cds_rows).sort_values('EC Contrib', ascending=True)

                col1, col2 = st.columns(2)
                with col1:
                    # EAD by direction
                    bought_ead = sum(abs(r['EAD (Signed)']) for r in cds_rows if 'Bought' in r['Type'])
                    sold_ead = sum(abs(r['EAD (Signed)']) for r in cds_rows if 'Sold' in r['Type'])
                    cva_ead = sum(abs(r['EAD (Signed)']) for r in cds_rows if 'CVA' in r['Type'])
                    fig_cds_bar = go.Figure(data=[go.Bar(
                        x=['Protection Sold', 'Protection Bought', 'CVA (Seller Risk)'],
                        y=[sold_ead/1e6, bought_ead/1e6, cva_ead/1e6],
                        marker_color=['#f85149', '#3fb950', '#ffa657']
                    )])
                    fig_cds_bar.update_layout(template=DARK_TEMPLATE, title='CDS EAD by Direction',
                                              yaxis_title='EAD ($M)', height=400)
                    st.plotly_chart(fig_cds_bar, use_container_width=True)

                with col2:
                    # EC contribution by direction
                    bought_ec = sum(r['EC Contrib'] for r in cds_rows if 'Bought' in r['Type'])
                    sold_ec = sum(r['EC Contrib'] for r in cds_rows if 'Sold' in r['Type'])
                    cva_ec = sum(r['EC Contrib'] for r in cds_rows if 'CVA' in r['Type'])
                    fig_cds_ec = go.Figure(data=[go.Bar(
                        x=['Protection Sold', 'Protection Bought (Hedge)', 'CVA (Seller Risk)'],
                        y=[sold_ec/1e6, bought_ec/1e6, cva_ec/1e6],
                        marker_color=['#f85149', '#3fb950', '#ffa657']
                    )])
                    fig_cds_ec.update_layout(template=DARK_TEMPLATE,
                                             title='CDS Economic Capital Contribution',
                                             yaxis_title='EC ($M)', height=400)
                    st.plotly_chart(fig_cds_ec, use_container_width=True)

                st.markdown('<div class="section-title">CDS Position Detail</div>', unsafe_allow_html=True)
                disp = cds_df.copy()
                for c in ['Notional', 'EAD (Signed)', 'EL', 'VaR Contrib', 'EC Contrib']:
                    disp[c] = disp[c].apply(lambda x: f"${x/1e6:,.3f}M" if isinstance(x, (int, float)) else x)
                st.dataframe(disp, use_container_width=True, hide_index=True)
        else:
            st.info("No CDS instruments in the portfolio.")

    # ── TAB 6: Instruments ──
    with tabs[5]:
        inst_detail = pd.DataFrame([
            {
                'ID': inst['instrument_id'],
                'Type': inst['instrument_type'],
                'Counterparty': inst['counterparty_id'],
                'Name': cp_lookup.get(inst['counterparty_id'], {}).get('legal_name', ''),
                'Rating': inst.get('rating', ''),
                'Currency': inst.get('currency', ''),
                'Seniority': inst.get('seniority', ''),
                'EAD': ead[j],
                'LGD': inst['lgd'],
                'Expected Loss': inst_el[j],
                'VaR Contrib': var_contrib[j],
                'ES Contrib': es_contrib[j],
                'EC Contrib': var_contrib[j] - inst_el[j],
            }
            for j, inst in enumerate(instruments_list)
        ])

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            type_filter = st.multiselect("Product Type", options=inst_detail['Type'].unique(),
                                          default=inst_detail['Type'].unique())
        with fc2:
            rating_filter = st.multiselect("Rating", options=sorted(inst_detail['Rating'].unique()),
                                            default=sorted(inst_detail['Rating'].unique()))
        with fc3:
            sort_by = st.selectbox("Sort By", ['EC Contrib', 'VaR Contrib', 'EAD', 'Expected Loss'])

        filtered = inst_detail[
            (inst_detail['Type'].isin(type_filter)) &
            (inst_detail['Rating'].isin(rating_filter))
        ].sort_values(sort_by, ascending=False)

        st.caption(f"Showing {len(filtered):,} of {len(inst_detail):,} instruments")
        display = filtered.head(100).copy()
        for col in ['EAD', 'Expected Loss', 'VaR Contrib', 'ES Contrib', 'EC Contrib']:
            display[col] = display[col].apply(lambda x: f"${x/1e6:,.3f}M")
        display['LGD'] = display['LGD'].apply(lambda x: f"{x:.1%}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── TAB 7: Portfolio ──
    with tabs[6]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">Portfolio Summary</div>', unsafe_allow_html=True)
            summary = pd.DataFrame({
                'Metric': [
                    'Counterparties', 'Instruments', 'Total EAD', 'Avg Instrument EAD',
                    'Wtd Avg PD', 'Wtd Avg LGD', 'GICS Sectors', 'Industry Groups',
                    'Countries', 'Simulations', 'Seed',
                ],
                'Value': [
                    f"{len(data['counterparties']):,}",
                    f"{len(data['instruments']):,}",
                    format_currency(metrics['total_ead']),
                    format_currency(metrics['total_ead'] / len(data['instruments'])),
                    f"{np.average([cp['pd_1y'] for cp in data['counterparties']]):.4%}",
                    f"{np.mean([inst['lgd'] for inst in data['instruments']]):.2%}",
                    str(len(set(cp.get('gics_sector', '') for cp in data['counterparties']))),
                    str(len(set(cp['sector_code'] for cp in data['counterparties']))),
                    str(len(set(cp['country_code'] for cp in data['counterparties']))),
                    f"{results['n_simulations']:,}",
                    str(results['seed']),
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

        with col2:
            maturities = [inst.get('maturity_date', '2028-01-01') for inst in data['instruments']]
            mat_years = [(pd.to_datetime(m) - pd.Timestamp.now()).days / 365.25 for m in maturities]
            fig_mat = go.Figure(data=[go.Histogram(x=mat_years, nbinsx=20, marker_color='#58a6ff')])
            fig_mat.update_layout(template=DARK_TEMPLATE, title='Maturity Profile',
                                  xaxis_title='Years to Maturity', yaxis_title='Count', height=350)
            st.plotly_chart(fig_mat, use_container_width=True)

        st.markdown('<div class="section-title">Credit Migration Matrix</div>', unsafe_allow_html=True)
        fig_mig = go.Figure(data=go.Heatmap(
            z=MIGRATION_MATRIX * 100, x=RATINGS, y=RATINGS,
            colorscale=[[0, '#0d1117'], [0.005, '#161b22'], [0.02, '#1a3a5c'],
                        [0.1, '#58a6ff'], [0.5, '#e3b341'], [1, '#f85149']],
            text=np.round(MIGRATION_MATRIX * 100, 2), texttemplate='%{text:.2f}%',
            textfont={"size": 11, "color": "#e6edf3"},
            hovertemplate='From %{y} → %{x}: %{z:.3f}%<extra></extra>',
            colorbar=dict(title='%', tickfont=dict(color='#8b949e')),
        ))
        fig_mig.update_layout(template=DARK_TEMPLATE, title='1-Year Rating Migration Probabilities (%)',
                              xaxis_title='To Rating', yaxis_title='From Rating', height=450,
                              xaxis=dict(side='top'))
        st.plotly_chart(fig_mig, use_container_width=True)

    # ── TAB 8: Audit Log ──
    with tabs[7]:
        st.markdown('<div class="section-title">Simulation Run History</div>', unsafe_allow_html=True)
        if st.session_state.run_history:
            audit_df = pd.DataFrame(st.session_state.run_history)
            audit_df['economic_capital'] = audit_df['economic_capital'].apply(format_currency)
            audit_df['var_999'] = audit_df['var_999'].apply(format_currency)
            audit_df['expected_loss'] = audit_df['expected_loss'].apply(format_currency)
            audit_df['elapsed_seconds'] = audit_df['elapsed_seconds'].apply(lambda x: f"{x:.1f}s")
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
        else:
            st.info("No runs recorded yet.")

        st.markdown('<div class="section-title">Model Documentation</div>', unsafe_allow_html=True)
        st.markdown("""
        **Model:** Merton asset-value model with multi-factor systematic risk

        **Factors:** 15 country + 30 GICS industry group factors (45 total). GCorr-style decomposition with
        intra-sector correlation boost for groups within the same GICS sector.

        **Asset Return:** `r_i = √RSQ_i × Z_systematic + √(1-RSQ_i) × ε_i`

        **Default:** When asset return < Φ⁻¹(PD)

        **LGD:** Stochastic beta-distributed with PD-LGD correlation through systematic factor

        **Economic Capital:** VaR(99.9%) - Expected Loss, 1-year horizon

        **Risk Contributions:** Euler allocation via conditional expectation in the tail

        **Scenario Conditioning:** GCorr Macro-style factor shifting with cross-factor propagation

        **Classification:** GICS (Global Industry Classification Standard) at Industry Group level
        """)

    # ── TAB 9: Upload Guide ──
    with tabs[8]:
        st.markdown('<div class="section-title">Portfolio Upload Guide</div>', unsafe_allow_html=True)
        st.markdown("""
Upload an Excel workbook (`.xlsx`) via the sidebar to run the simulation on your own portfolio data.
The workbook must contain at least a **Counterparties** and an **Instruments** sheet.

---

#### Counterparties Sheet (Required)

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| Counterparty ID | Yes | Unique identifier | `CP0001` |
| Legal Name | Yes | Company name | `Acme Corp` |
| Sector / GICS Industry Group | Yes | GICS Industry Group code | `Banks`, `OilGasFuels`, `SoftwareServices` |
| Country | Yes | ISO 2-letter country code | `US`, `UK`, `DE`, `JP` |
| Rating | Yes | Credit rating | `AAA`, `AA`, `A`, `BBB`, `BB`, `B`, `CCC` |
| PD (1Y) | Yes | 1-year probability of default (decimal or %) | `0.0025` or `0.25%` |
| RSQ | No | Asset correlation R² (default: 0.25) | `0.30` |
| Parent ID | No | Parent entity ID for group linkage | `CP0010` |
| Revenue ($mm) | No | Annual revenue | `5000` |
| Total Assets ($mm) | No | Total assets | `25000` |

**Accepted GICS Industry Group codes:**
`EnergyEquipSvc`, `OilGasFuels`, `Chemicals`, `ConstructionMaterials`, `ContainersPkg`,
`MetalsMining`, `PaperForest`, `CapitalGoods`, `CommercialProfSvc`, `Transportation`,
`AutosComponents`, `ConsumerDurablesApparel`, `ConsumerServices`, `Retailing`,
`FoodStaplesRetail`, `FoodBevTobacco`, `HouseholdProducts`, `HealthCareEquipSvc`,
`PharmaBiotech`, `Banks`, `DiversifiedFinancials`, `Insurance`, `SoftwareServices`,
`TechHardware`, `Semiconductors`, `MediaEntertainment`, `TelecomServices`, `Utilities`,
`EquityREITs`, `REMgmtDev`

Full display names (e.g., "Oil, Gas & Consumable Fuels") are also accepted.

**Supported countries:** `US`, `UK`, `DE`, `FR`, `JP`, `CN`, `BR`, `IN`, `CA`, `AU`, `SG`, `CH`, `KR`, `MX`, `ZA`

---

#### Instruments Sheet (Required)

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| Instrument ID | Yes | Unique identifier | `INS00001` |
| Type | Yes | Product type | `TermLoan`, `Revolver`, `CDS`, etc. |
| Counterparty ID | Yes | Links to Counterparties sheet | `CP0001` |
| LGD | Yes | Loss Given Default (decimal or %) | `0.45` or `45%` |
| Drawn Amount | Depends | Outstanding balance (loans) | `5000000` |
| Undrawn Amount | No | Undrawn commitment | `2000000` |
| Notional | Depends | Notional amount (derivatives/CDS) | `10000000` |
| MTM Value | No | Mark-to-market value | `150000` |
| CCF | No | Credit Conversion Factor (revolvers) | `0.75` |
| Add-on Factor | No | SA-CCR add-on (derivatives) | `0.01` |
| Currency | No | ISO currency code (default: USD) | `USD`, `EUR` |
| Maturity Date | No | Maturity date | `2028-06-15` |
| Seniority | No | Debt seniority | `Senior Secured` |
| CDS Direction | CDS only | Protection direction | `Protection_Bought` or `Protection_Sold` |
| CDS Spread (bps) | No | CDS spread in basis points | `125` |
| CDS Seller ID | CDS Bought | Counterparty ID of protection seller | `CP0050` |

**Product types:** `TermLoan`, `Revolver`, `Derivative_IR`, `Derivative_FX`, `CDS`, `TradeFinance`, `Guarantee`

**CDS Direction values:** `Protection_Bought` (or `Bought`/`Buy`/`Long`), `Protection_Sold` (default)

**CDS notes:**
- **Protection Sold**: Generates positive EC (you lose if reference entity defaults)
- **Protection Bought**: Generates negative EC (hedge — you gain if reference entity defaults)
- **CVA**: If `CDS Seller ID` is provided, counterparty risk to the seller is automatically computed

---

#### Migration Matrix Sheet (Optional)

An 8×8 matrix of annual rating transition probabilities (AAA through D). Each row must sum to 1.0. If omitted, the default Moody's-calibrated matrix is used.

---

#### Scenarios Sheet (Optional)

| Column | Description |
|--------|-------------|
| Scenario Name | Name of the stress scenario |
| Type | `baseline` or `stress` |
| Description | Scenario description |
| GDP Shock | GDP shock in standard deviations |
| Rate Shock | Interest rate shock in σ |
| Spread Shock | Credit spread shock in σ |
| Factor Shocks | Semicolon-separated: `US: -2.5; Banks: -3.0` |
        """)

    # ── TAB 10: Export ──
    with tabs[9]:
        st.markdown('<div class="section-title">Export Results</div>', unsafe_allow_html=True)

        import io

        # Build export dataframes
        export_summary = pd.DataFrame({
            'Metric': ['Expected Loss', 'Unexpected Loss', 'VaR 95%', 'VaR 99%', 'VaR 99.9%',
                       'VaR 99.99%', 'ES 95%', 'ES 99%', 'ES 99.9%', 'Economic Capital',
                       'EC % of EAD', 'EL % of EAD', 'Diversification Benefit',
                       'Total EAD', 'HHI', 'Max Loss', 'Skewness', 'Kurtosis',
                       'Portfolio Default Rate', 'Simulations', 'Seed'],
            'Value': [
                metrics['expected_loss'], metrics['unexpected_loss'],
                metrics['var_95'], metrics['var_99'], metrics['var_999'], metrics['var_9999'],
                metrics['es_95'], metrics['es_99'], metrics['es_999'],
                metrics['economic_capital'], metrics['ec_as_pct_ead'], metrics['el_as_pct_ead'],
                metrics['diversification_benefit'] * 100, metrics['total_ead'], metrics['hhi'],
                metrics['max_loss'], metrics['skewness'], metrics['kurtosis'],
                metrics['portfolio_default_rate'], results['n_simulations'], results['seed'],
            ]
        })

        export_cp = cp_df.copy()
        for c in ['EAD', 'Expected Loss', 'VaR Contribution', 'ES Contribution', 'EC Contribution']:
            export_cp[c] = cp_df[c]  # Keep numeric for Excel

        export_inst = pd.DataFrame([
            {
                'ID': inst['instrument_id'],
                'Type': inst['instrument_type'],
                'Direction': inst.get('cds_direction', ''),
                'Counterparty': inst['counterparty_id'],
                'Name': cp_lookup.get(inst['counterparty_id'], {}).get('legal_name', ''),
                'EAD': ead[j],
                'LGD': inst.get('lgd', 0),
                'Expected Loss': inst_el[j],
                'VaR Contrib': var_contrib[j],
                'ES Contrib': es_contrib[j],
                'EC Contrib': var_contrib[j] - inst_el[j],
            }
            for j, inst in enumerate(instruments_list)
        ])

        # Write to Excel buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_summary.to_excel(writer, sheet_name='Risk Metrics', index=False)
            export_cp.to_excel(writer, sheet_name='Counterparty Risk', index=False)
            export_inst.to_excel(writer, sheet_name='Instrument Risk', index=False)

            if st.session_state.run_history:
                pd.DataFrame(st.session_state.run_history).to_excel(writer, sheet_name='Audit Log', index=False)

        buffer.seek(0)

        st.download_button(
            label="📥 Download EC Results (.xlsx)",
            data=buffer,
            file_name=f"EC_Results_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            type="primary"
        )

        st.caption("Exports: Risk Metrics, Counterparty Risk Contributions, Instrument-Level Detail, and Audit Log.")

        # Also offer CSV downloads
        st.markdown('<div class="section-title">Individual CSV Downloads</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("Risk Metrics CSV", export_summary.to_csv(index=False),
                               "ec_risk_metrics.csv", "text/csv", use_container_width=True)
        with col2:
            st.download_button("Counterparty CSV", export_cp.to_csv(index=False),
                               "ec_counterparty_risk.csv", "text/csv", use_container_width=True)
        with col3:
            st.download_button("Instrument CSV", export_inst.to_csv(index=False),
                               "ec_instrument_risk.csv", "text/csv", use_container_width=True)

else:
    # ── Landing Page ──
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; color: var(--text-secondary);">
        <p style="font-size: 3rem; margin-bottom: 8px;">📊</p>
        <p style="font-size: 1.1rem; color: var(--text-primary); font-weight: 600;">Configure parameters and click Run Simulation</p>
        <p style="font-size: 0.9rem; margin-top: 8px;">Or upload your own portfolio via the sidebar</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="section-title">What This Tool Does</div>', unsafe_allow_html=True)
        st.markdown("Computes economic capital for a credit portfolio using Monte Carlo simulation "
                    "with a multi-factor Merton model. Covers loans, revolvers, derivatives, CDS, and more.")
    with col2:
        st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
        st.markdown("Correlated asset-value model with Gaussian copula, "
                    "stochastic LGD with PD-LGD correlation, rating migration, and GICS-based industry factors.")
    with col3:
        st.markdown('<div class="section-title">Key Outputs</div>', unsafe_allow_html=True)
        st.markdown("Expected Loss, Credit VaR, Expected Shortfall, Economic Capital, "
                    "risk contributions by counterparty/GICS sector/country, stress scenarios, and audit trail.")
